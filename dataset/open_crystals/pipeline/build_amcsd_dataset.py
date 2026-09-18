"""Build AMCSD structure artifacts compatible with the MP/COD pipeline.

The dataset entry point passes --dedup_scope none, then applies joint
MP/COD/AMCSD approximate deduplication during reconstruction. AMCSD records
are assigned to training. Optional deduplication modes are separate recipes."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from pymatgen.analysis.structure_matcher import StructureMatcher

from get_data import (
    CONFIG,
    configure_warning_filters,
    load_cod_structure,
    process_structure_to_record,
    strip_oxidation_states_safe,
)


LOGGER = logging.getLogger("build_amcsd_dataset")


ID_RE = re.compile(r"__(\d+)$")
AMCSD_CODE_RE = re.compile(r"_database_code_amcsd\s+['\"]?([^'\"\s]+)", re.IGNORECASE)
CIF_TAG_RE = re.compile(r"^(_[A-Za-z0-9_.-]+)\s+(.*)$")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deduplicated AMCSD add-on dataset.",
    )
    parser.add_argument("--amcsd_dir", default="AMCSD")
    parser.add_argument("--output_dir", default="AMCSD_dataset")
    parser.add_argument(
        "--raw_metadata",
        default=None,
        help=(
            "Existing raw AMCSD metadata CSV to reuse. Defaults to "
            "<output_dir>/metadata_amcsd_raw.csv when it exists."
        ),
    )
    parser.add_argument(
        "--rebuild_raw",
        action="store_true",
        help="Rebuild metadata_amcsd_raw.csv from AMCSD CIF files even if it already exists.",
    )
    parser.add_argument(
        "--restore_missing_artifacts",
        action="store_true",
        help="After deduplication, rebuild only missing base_peaks/labels/structures artifacts.",
    )
    parser.add_argument(
        "--dedup_scope",
        choices=["rruff", "dataset_rruff", "dataset", "none"],
        default="rruff",
        help=(
            "Deduplication scope. Default rruff keeps AMCSD even if it "
            "duplicates COD+MP, and only removes overlaps with RRUFF test."
        ),
    )
    parser.add_argument(
        "--reference_metadata",
        default="dataset/postprocess/metadata_filtered.csv",
        help="COD+MP postprocessed metadata; only used when dedup_scope includes dataset.",
    )
    parser.add_argument(
        "--rruff_test_csv",
        default="RRUFF_dataset/test_rruff.csv",
        help="RRUFF held-out/test CSV; used when dedup_scope includes rruff.",
    )
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min((os.cpu_count() or 2) - 1, 8)),
    )
    parser.add_argument("--no_parallel", action="store_true")
    parser.add_argument(
        "--dedup_internal_amcsd",
        action="store_true",
        help="Also remove StructureMatcher duplicates within AMCSD itself.",
    )
    parser.add_argument(
        "--keep_excluded_artifacts",
        action="store_true",
        help="Deprecated compatibility flag; excluded artifacts are kept by default.",
    )
    parser.add_argument(
        "--prune_excluded_artifacts",
        action="store_true",
        help="Delete artifacts for excluded AMCSD rows. Off by default.",
    )
    parser.add_argument("--volume_per_atom_tol", type=float, default=None)
    parser.add_argument("--rruff_length_rtol", type=float, default=0.02)
    parser.add_argument("--rruff_angle_atol", type=float, default=1.0)
    parser.add_argument("--rruff_volume_rtol", type=float, default=0.03)
    return parser.parse_args()


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    stream_exists = any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)
    if not stream_exists:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root_logger.addHandler(sh)

    log_path = output_dir / "amcsd_build.log"
    if not any(
        isinstance(h, logging.FileHandler)
        and Path(getattr(h, "baseFilename", "")).resolve() == log_path.resolve()
        for h in root_logger.handlers
    ):
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        root_logger.addHandler(fh)


def iter_amcsd_cifs(amcsd_dir: Path, max_files: int | None = None) -> list[Path]:
    files = sorted(amcsd_dir.glob("*.cif"))
    if max_files is not None:
        files = files[: int(max_files)]
    return files


def _strip_cif_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value


def extract_amcsd_metadata(cif_path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    try:
        lines = cif_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        lines = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = CIF_TAG_RE.match(line)
        if not m:
            i += 1
            continue

        tag = m.group(1).lower()
        value = m.group(2).strip()
        if value == ";" and i + 1 < len(lines):
            block = []
            i += 1
            while i < len(lines) and lines[i].strip() != ";":
                block.append(lines[i])
                i += 1
            value = " ".join(x.strip() for x in block if x.strip())
        meta[tag] = _strip_cif_value(value)
        i += 1

    stem = cif_path.stem
    id_match = ID_RE.search(stem)
    amcsd_id = meta.get("_database_code_amcsd", "")
    if not amcsd_id:
        code_match = AMCSD_CODE_RE.search("\n".join(lines[:80]))
        amcsd_id = code_match.group(1) if code_match else ""
    if not amcsd_id and id_match:
        amcsd_id = id_match.group(1)
    if not amcsd_id:
        amcsd_id = slugify(stem)

    stem_name = stem[: id_match.start()] if id_match else stem
    name = meta.get("_chemical_name_mineral") or stem_name.replace("_", " ").strip()

    return {
        "amcsd_id": str(amcsd_id).strip(),
        "amcsd_name": name,
        "amcsd_formula_sum": meta.get("_chemical_formula_sum", ""),
    }


def slugify(value: str) -> str:
    value = NON_ALNUM_RE.sub("-", value.lower()).strip("-")
    return value or "unknown"


def normalize_name(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return NON_ALNUM_RE.sub("", text)


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = -1) -> int:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(float(value))
    except Exception:
        return default


def dedup_key(row: pd.Series | dict[str, Any], volume_per_atom_tol: float) -> str:
    getter = row.get
    formula = (
        getter("reduced_formula", "")
        or getter("formula", "")
        or getter("anonymous_formula", "")
        or ""
    )
    sg = safe_int(getter("space_group_number", -1), -1)
    vpa = safe_float(getter("volume_per_atom", float("nan")))
    vbin = -1 if math.isnan(vpa) else int(round(vpa / volume_per_atom_tol))
    return f"{formula}|{sg}|{vbin}"


def add_dedup_key_column(df: pd.DataFrame, volume_per_atom_tol: float) -> pd.DataFrame:
    out = df.copy()
    formula = pd.Series("", index=out.index, dtype=object)
    for col in ("reduced_formula", "formula", "anonymous_formula"):
        if col in out.columns:
            values = out[col].fillna("").astype(str)
            formula = formula.mask(formula.eq(""), values)

    if "space_group_number" in out.columns:
        sg = pd.to_numeric(out["space_group_number"], errors="coerce").fillna(-1).astype(int)
    else:
        sg = pd.Series(-1, index=out.index, dtype=int)

    if "volume_per_atom" in out.columns:
        vpa = pd.to_numeric(out["volume_per_atom"], errors="coerce")
        vbin = (vpa / float(volume_per_atom_tol)).round()
        vbin = vbin.where(vbin.notna(), -1).astype(int)
    else:
        vbin = pd.Series(-1, index=out.index, dtype=int)

    out["_dedup_key"] = formula + "|" + sg.astype(str) + "|" + vbin.astype(str)
    return out


def resolve_existing_path(value: Any, workspace_root: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return workspace_root / path


def resolve_amcsd_path(value: Any, output_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return output_dir / path


def load_structure_cached(path: Path, cache: dict[Path, Any]) -> Any:
    path = path.resolve()
    if path not in cache:
        cache[path] = load_cod_structure(path)
    return cache[path]


def build_matcher(cfg: dict[str, Any]) -> StructureMatcher:
    dedup_cfg = cfg.get("dedup", {})
    return StructureMatcher(
        ltol=float(dedup_cfg.get("ltol", 0.2)),
        stol=float(dedup_cfg.get("stol", 0.3)),
        angle_tol=float(dedup_cfg.get("angle_tol", 5)),
        primitive_cell=True,
        scale=True,
        attempt_supercell=False,
    )


def process_one_cif(task: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    cif_path_s, output_dir_s, cfg = task
    cif_path = Path(cif_path_s)
    output_dir = Path(output_dir_s)
    try:
        metadata = extract_amcsd_metadata(cif_path)
        amcsd_id = str(metadata["amcsd_id"]).strip()
        material_id = f"amcsd-{amcsd_id.lower()}"

        structure = load_cod_structure(cif_path)
        structure, _ = strip_oxidation_states_safe(structure)

        extra = {
            "amcsd_id": amcsd_id,
            "amcsd_name": metadata.get("amcsd_name", ""),
            "amcsd_formula_sum": metadata.get("amcsd_formula_sum", ""),
            "cif_path": cif_path.as_posix(),
        }
        row = process_structure_to_record(
            structure=structure,
            material_id=material_id,
            source="AMCSD",
            output_dir=output_dir,
            cfg=cfg,
            extra_metadata=extra,
        )
        row["_ok"] = True
        return row
    except Exception as exc:
        metadata = extract_amcsd_metadata(cif_path)
        amcsd_id = metadata.get("amcsd_id") or cif_path.stem
        return {
            "_ok": False,
            "material_id": f"amcsd-{str(amcsd_id).lower()}",
            "amcsd_id": amcsd_id,
            "amcsd_name": metadata.get("amcsd_name", ""),
            "cif_path": cif_path.as_posix(),
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_raw_amcsd(
    cif_files: list[Path],
    output_dir: Path,
    cfg: dict[str, Any],
    workers: int,
    no_parallel: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = [(str(path), str(output_dir), cfg) for path in cif_files]
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    if no_parallel or workers <= 1:
        iterator = (process_one_cif(task) for task in tasks)
        for result in tqdm(iterator, total=len(tasks), desc="AMCSD CIFs"):
            if result.pop("_ok", False):
                rows.append(result)
            else:
                errors.append(result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process_one_cif, task) for task in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="AMCSD CIFs"):
                result = fut.result()
                if result.pop("_ok", False):
                    rows.append(result)
                else:
                    errors.append(result)

    raw_df = pd.DataFrame(rows)
    if not raw_df.empty:
        raw_df = raw_df.sort_values("material_id").drop_duplicates("material_id", keep="first")
        raw_df = raw_df.reset_index(drop=True)
    err_df = pd.DataFrame(errors)
    return raw_df, err_df


def remove_against_structure_reference(
    amcsd_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    workspace_root: Path,
    output_dir: Path,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    stats = {
        "before": int(len(amcsd_df)),
        "removed": 0,
        "after": int(len(amcsd_df)),
        "reference_candidates": 0,
        "structure_load_errors": 0,
    }
    if amcsd_df.empty or reference_df.empty or "structure_path" not in reference_df.columns:
        return amcsd_df, pd.DataFrame(), stats

    dedup_cfg = cfg.get("dedup", {})
    tol = float(dedup_cfg.get("volume_per_atom_tol", 0.05))
    amcsd_with_keys = add_dedup_key_column(amcsd_df, tol)
    wanted_keys = set(amcsd_with_keys["_dedup_key"].dropna().astype(str))
    ref = add_dedup_key_column(reference_df, tol)
    ref = ref[ref["_dedup_key"].isin(wanted_keys)].copy()
    ref_groups = {
        key: group.to_dict("records")
        for key, group in ref.groupby("_dedup_key", sort=False)
    }

    matcher = build_matcher(cfg)
    cache: dict[Path, Any] = {}
    keep_mask = np.ones(len(amcsd_df), dtype=bool)
    excluded: list[dict[str, Any]] = []

    for pos, (_, row) in enumerate(
        tqdm(amcsd_with_keys.iterrows(), total=len(amcsd_with_keys), desc="Dedup COD+MP")
    ):
        key = str(row["_dedup_key"])
        candidates = ref_groups.get(key, [])
        if not candidates:
            continue
        stats["reference_candidates"] += len(candidates)
        try:
            amcsd_structure = load_structure_cached(
                resolve_amcsd_path(row["structure_path"], output_dir),
                cache,
            )
        except Exception:
            stats["structure_load_errors"] += 1
            continue

        matched: dict[str, Any] | None = None
        for ref_row in candidates:
            try:
                ref_structure = load_structure_cached(
                    resolve_existing_path(ref_row["structure_path"], workspace_root),
                    cache,
                )
                if matcher.fit(amcsd_structure, ref_structure):
                    matched = ref_row
                    break
            except Exception:
                stats["structure_load_errors"] += 1
                continue

        if matched is not None:
            keep_mask[pos] = False
            out = row.to_dict()
            out["exclude_reason"] = "duplicate_of_dataset_postprocess"
            out["matched_material_id"] = matched.get("material_id", "")
            out["matched_source"] = matched.get("source", "")
            excluded.append(out)

    kept = amcsd_df.loc[keep_mask].reset_index(drop=True)
    excluded_df = pd.DataFrame(excluded)
    stats["removed"] = int(len(excluded_df))
    stats["after"] = int(len(kept))
    return kept, excluded_df, stats


def remove_internal_amcsd_duplicates(
    amcsd_df: pd.DataFrame,
    output_dir: Path,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    stats = {"before": int(len(amcsd_df)), "removed": 0, "after": int(len(amcsd_df))}
    if amcsd_df.empty:
        return amcsd_df, pd.DataFrame(), stats

    dedup_cfg = cfg.get("dedup", {})
    tol = float(dedup_cfg.get("volume_per_atom_tol", 0.05))
    work = add_dedup_key_column(amcsd_df, tol)
    matcher = build_matcher(cfg)
    cache: dict[Path, Any] = {}
    keep_indices: set[int] = set()
    excluded: list[dict[str, Any]] = []

    for _, group in tqdm(work.groupby("_dedup_key", sort=False), desc="Dedup AMCSD internal"):
        representatives: list[tuple[int, str, Any]] = []
        for idx, row in group.iterrows():
            try:
                structure = load_structure_cached(
                    resolve_amcsd_path(row["structure_path"], output_dir),
                    cache,
                )
            except Exception:
                keep_indices.add(idx)
                continue

            matched: tuple[int, str] | None = None
            for rep_idx, rep_mid, rep_structure in representatives:
                try:
                    if matcher.fit(structure, rep_structure):
                        matched = (rep_idx, rep_mid)
                        break
                except Exception:
                    continue

            if matched is None:
                keep_indices.add(idx)
                representatives.append((idx, str(row["material_id"]), structure))
            else:
                out = row.drop(labels=["_dedup_key"]).to_dict()
                out["exclude_reason"] = "duplicate_within_amcsd"
                out["matched_material_id"] = matched[1]
                out["matched_source"] = "AMCSD"
                excluded.append(out)

    kept = work.loc[sorted(keep_indices)].drop(columns=["_dedup_key"]).reset_index(drop=True)
    excluded_df = pd.DataFrame(excluded)
    stats["removed"] = int(len(excluded_df))
    stats["after"] = int(len(kept))
    return kept, excluded_df, stats


def cell_close(
    left: pd.Series | dict[str, Any],
    right: pd.Series | dict[str, Any],
    length_rtol: float,
    angle_atol: float,
    volume_rtol: float,
) -> bool:
    for key in ("a", "b", "c"):
        lv = safe_float(left.get(key))
        rv = safe_float(right.get(key))
        if math.isnan(lv) or math.isnan(rv):
            return False
        if abs(lv - rv) > length_rtol * max(abs(lv), abs(rv), 1e-9):
            return False
    for key in ("alpha", "beta", "gamma"):
        lv = safe_float(left.get(key))
        rv = safe_float(right.get(key))
        if math.isnan(lv) or math.isnan(rv):
            return False
        if abs(lv - rv) > angle_atol:
            return False
    lv = safe_float(left.get("volume"))
    rv = safe_float(right.get("volume"))
    if math.isnan(lv) or math.isnan(rv):
        return False
    return abs(lv - rv) <= volume_rtol * max(abs(lv), abs(rv), 1e-9)


def symmetry_compatible(left: pd.Series | dict[str, Any], right: pd.Series | dict[str, Any]) -> bool:
    lsg = safe_int(left.get("space_group_number", -1), -1)
    rsg = safe_int(right.get("space_group_number", -1), -1)
    if lsg > 0 and rsg > 0:
        return lsg == rsg
    lcs = str(left.get("crystal_system", "") or "").lower()
    rcs = str(right.get("crystal_system", "") or "").lower()
    return bool(lcs and rcs and lcs == rcs)


def remove_against_rruff_test(
    amcsd_df: pd.DataFrame,
    rruff_df: pd.DataFrame,
    length_rtol: float,
    angle_atol: float,
    volume_rtol: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    stats = {"before": int(len(amcsd_df)), "removed": 0, "after": int(len(amcsd_df))}
    if amcsd_df.empty or rruff_df.empty or "name" not in rruff_df.columns:
        return amcsd_df, pd.DataFrame(), stats

    rruff = rruff_df.copy()
    rruff["_name_key"] = rruff["name"].map(normalize_name)
    rruff_groups = {
        key: group.to_dict("records")
        for key, group in rruff.groupby("_name_key", sort=False)
        if key
    }

    keep_mask = np.ones(len(amcsd_df), dtype=bool)
    excluded: list[dict[str, Any]] = []

    for pos, (_, row) in enumerate(tqdm(amcsd_df.iterrows(), total=len(amcsd_df), desc="Dedup RRUFF test")):
        name_key = normalize_name(row.get("amcsd_name", ""))
        candidates = rruff_groups.get(name_key, [])
        if not candidates:
            continue
        matched: dict[str, Any] | None = None
        for candidate in candidates:
            if not symmetry_compatible(row, candidate):
                continue
            if cell_close(row, candidate, length_rtol, angle_atol, volume_rtol):
                matched = candidate
                break

        if matched is not None:
            keep_mask[pos] = False
            out = row.to_dict()
            out["exclude_reason"] = "duplicate_of_rruff_test"
            out["matched_material_id"] = matched.get("material_id", "")
            out["matched_source"] = "RRUFF"
            out["matched_rruff_name"] = matched.get("name", "")
            excluded.append(out)

    kept = amcsd_df.loc[keep_mask].reset_index(drop=True)
    excluded_df = pd.DataFrame(excluded)
    stats["removed"] = int(len(excluded_df))
    stats["after"] = int(len(kept))
    return kept, excluded_df, stats


def remove_excluded_artifacts(excluded_df: pd.DataFrame, output_dir: Path) -> int:
    if excluded_df.empty:
        return 0
    output_root = output_dir.resolve()
    removed = 0
    for _, row in excluded_df.iterrows():
        for col in ("base_peaks_path", "labels_path", "structure_path"):
            value = row.get(col)
            if not value or pd.isna(value):
                continue
            path = resolve_amcsd_path(value, output_dir).resolve()
            try:
                path.relative_to(output_root)
            except ValueError:
                continue
            if path.exists() and path.is_file():
                path.unlink()
                removed += 1
    return removed


def artifact_paths_for_row(row: pd.Series | dict[str, Any], output_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for col in ("base_peaks_path", "labels_path", "structure_path"):
        value = row.get(col)
        if value is None or pd.isna(value):
            continue
        paths[col] = resolve_amcsd_path(value, output_dir)
    return paths


def find_missing_artifacts(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(columns=["material_id", "missing_artifacts"])

    for _, row in df.iterrows():
        missing = [
            col
            for col, path in artifact_paths_for_row(row, output_dir).items()
            if not path.exists()
        ]
        if missing:
            rows.append({
                "material_id": row.get("material_id", ""),
                "amcsd_id": row.get("amcsd_id", ""),
                "amcsd_name": row.get("amcsd_name", ""),
                "cif_path": row.get("cif_path", ""),
                "missing_artifacts": ";".join(missing),
            })
    return pd.DataFrame(rows)


def merge_rows_by_material_id(base_df: pd.DataFrame, replacement_df: pd.DataFrame) -> pd.DataFrame:
    if base_df.empty or replacement_df.empty or "material_id" not in replacement_df.columns:
        return base_df
    base = base_df.copy()
    replacements = {
        str(row["material_id"]): row.to_dict()
        for _, row in replacement_df.iterrows()
        if row.get("material_id")
    }
    for idx, row in base.iterrows():
        material_id = str(row.get("material_id", ""))
        if material_id not in replacements:
            continue
        for key, value in replacements[material_id].items():
            if key in base.columns:
                base.at[idx, key] = value
            else:
                base[key] = pd.NA
                base.at[idx, key] = value
    return base


def restore_missing_artifacts(
    final_df: pd.DataFrame,
    output_dir: Path,
    cfg: dict[str, Any],
    workers: int,
    no_parallel: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing_df = find_missing_artifacts(final_df, output_dir)
    if missing_df.empty:
        return final_df, pd.DataFrame(), missing_df

    cif_files: list[Path] = []
    for value in missing_df["cif_path"].dropna().astype(str).unique():
        path = Path(value)
        if path.exists():
            cif_files.append(path)

    rebuilt_df, rebuild_errors_df = build_raw_amcsd(
        cif_files=cif_files,
        output_dir=output_dir,
        cfg=cfg,
        workers=workers,
        no_parallel=no_parallel,
    )
    final_df = merge_rows_by_material_id(final_df, rebuilt_df)
    missing_after_df = find_missing_artifacts(final_df, output_dir)
    return final_df, rebuild_errors_df, missing_after_df


def write_empty_split(path: Path, columns: list[str]) -> None:
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    workspace_root = Path.cwd().resolve()
    amcsd_dir = (workspace_root / args.amcsd_dir).resolve()
    output_dir = (workspace_root / args.output_dir).resolve()
    configure_logging(output_dir)
    configure_warning_filters()

    if not amcsd_dir.exists():
        raise FileNotFoundError(f"AMCSD directory not found: {amcsd_dir}")

    cfg = copy.deepcopy(CONFIG)
    if args.volume_per_atom_tol is not None:
        cfg.setdefault("dedup", {})["volume_per_atom_tol"] = float(args.volume_per_atom_tol)

    for subdir in ("base_peaks", "labels", "structures"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    if args.raw_metadata:
        raw_path = Path(args.raw_metadata)
        if not raw_path.is_absolute():
            raw_path = (workspace_root / raw_path).resolve()
    else:
        raw_path = output_dir / "metadata_amcsd_raw.csv"
    err_path = output_dir / "metadata_amcsd_errors.csv"

    if raw_path.exists() and not args.rebuild_raw:
        raw_df = pd.read_csv(raw_path, low_memory=False)
        if err_path.exists():
            try:
                err_df = pd.read_csv(err_path, low_memory=False)
            except pd.errors.EmptyDataError:
                err_df = pd.DataFrame()
        else:
            err_df = pd.DataFrame()
        cif_files: list[Path] = []
        raw_source = "existing_csv"
        LOGGER.info("Reusing raw AMCSD metadata: %s (%d rows)", raw_path, len(raw_df))
    else:
        cif_files = iter_amcsd_cifs(amcsd_dir, args.max_files)
        raw_source = "rebuilt_from_cif"
        LOGGER.info("AMCSD CIF files: %d", len(cif_files))
        raw_df, err_df = build_raw_amcsd(
            cif_files=cif_files,
            output_dir=output_dir,
            cfg=cfg,
            workers=int(args.workers),
            no_parallel=bool(args.no_parallel),
        )
        raw_df.to_csv(raw_path, index=False)
        err_df.to_csv(err_path, index=False)
        LOGGER.info("Raw AMCSD metadata: %s (%d rows)", raw_path, len(raw_df))
        LOGGER.info("AMCSD parse/build errors: %s (%d rows)", err_path, len(err_df))

    dedup_scope = str(args.dedup_scope).lower()
    use_dataset_dedup = dedup_scope in {"dataset", "dataset_rruff"}
    use_rruff_dedup = dedup_scope in {"rruff", "dataset_rruff"}
    LOGGER.info("Dedup scope: %s", dedup_scope)

    if use_dataset_dedup:
        reference_path = (workspace_root / args.reference_metadata).resolve()
        if reference_path.exists():
            reference_df = pd.read_csv(reference_path, low_memory=False)
            LOGGER.info("Loaded reference metadata: %s (%d rows)", reference_path, len(reference_df))
        else:
            reference_df = pd.DataFrame()
            LOGGER.warning("Reference metadata not found: %s", reference_path)
    else:
        reference_df = pd.DataFrame()
        LOGGER.info("Skipping COD+MP dataset dedup.")

    if use_rruff_dedup:
        rruff_path = (workspace_root / args.rruff_test_csv).resolve()
        if rruff_path.exists():
            rruff_df = pd.read_csv(rruff_path, low_memory=False)
            LOGGER.info("Loaded RRUFF test CSV: %s (%d rows)", rruff_path, len(rruff_df))
        else:
            rruff_df = pd.DataFrame()
            LOGGER.warning("RRUFF test CSV not found: %s", rruff_path)
    else:
        rruff_df = pd.DataFrame()
        LOGGER.info("Skipping RRUFF test dedup.")

    if use_dataset_dedup:
        kept_df, excluded_dataset_df, stats_dataset = remove_against_structure_reference(
            amcsd_df=raw_df,
            reference_df=reference_df,
            workspace_root=workspace_root,
            output_dir=output_dir,
            cfg=cfg,
        )
    else:
        kept_df = raw_df
        excluded_dataset_df = pd.DataFrame()
        stats_dataset = {
            "before": int(len(raw_df)),
            "removed": 0,
            "after": int(len(raw_df)),
            "skipped": True,
        }

    if args.dedup_internal_amcsd:
        kept_df, excluded_internal_df, stats_internal = remove_internal_amcsd_duplicates(
            amcsd_df=kept_df,
            output_dir=output_dir,
            cfg=cfg,
        )
    else:
        excluded_internal_df = pd.DataFrame()
        stats_internal = {"before": int(len(kept_df)), "removed": 0, "after": int(len(kept_df))}

    if use_rruff_dedup:
        kept_df, excluded_rruff_df, stats_rruff = remove_against_rruff_test(
            amcsd_df=kept_df,
            rruff_df=rruff_df,
            length_rtol=float(args.rruff_length_rtol),
            angle_atol=float(args.rruff_angle_atol),
            volume_rtol=float(args.rruff_volume_rtol),
        )
    else:
        excluded_rruff_df = pd.DataFrame()
        stats_rruff = {
            "before": int(len(kept_df)),
            "removed": 0,
            "after": int(len(kept_df)),
            "skipped": True,
        }

    excluded_df = pd.concat(
        [excluded_dataset_df, excluded_internal_df, excluded_rruff_df],
        ignore_index=True,
        sort=False,
    )
    excluded_path = output_dir / "metadata_amcsd_excluded.csv"
    excluded_df.to_csv(excluded_path, index=False)

    if args.prune_excluded_artifacts and not args.keep_excluded_artifacts:
        removed_artifacts = remove_excluded_artifacts(excluded_df, output_dir)
    else:
        removed_artifacts = 0

    final_df = kept_df.sort_values("material_id").reset_index(drop=True)
    rebuild_errors_df = pd.DataFrame()
    if args.restore_missing_artifacts:
        final_df, rebuild_errors_df, missing_artifacts_df = restore_missing_artifacts(
            final_df=final_df,
            output_dir=output_dir,
            cfg=cfg,
            workers=int(args.workers),
            no_parallel=bool(args.no_parallel),
        )
        final_df = final_df.sort_values("material_id").reset_index(drop=True)
    else:
        missing_artifacts_df = find_missing_artifacts(final_df, output_dir)

    missing_artifacts_path = output_dir / "metadata_amcsd_missing_artifacts.csv"
    missing_artifacts_df.to_csv(missing_artifacts_path, index=False)
    rebuild_errors_path = output_dir / "metadata_amcsd_rebuild_errors.csv"
    rebuild_errors_df.to_csv(rebuild_errors_path, index=False)
    if not missing_artifacts_df.empty:
        LOGGER.warning(
            "Final AMCSD metadata has %d rows with missing artifacts. See %s",
            len(missing_artifacts_df),
            missing_artifacts_path,
        )

    final_path = output_dir / "metadata_amcsd_dedup.csv"
    metadata_path = output_dir / "metadata.csv"
    train_path = output_dir / "train.csv"
    final_df.to_csv(final_path, index=False)
    final_df.to_csv(metadata_path, index=False)
    final_df.to_csv(train_path, index=False)
    write_empty_split(output_dir / "val.csv", list(final_df.columns))
    write_empty_split(output_dir / "test.csv", list(final_df.columns))

    stats = {
        "amcsd_dir": str(amcsd_dir),
        "output_dir": str(output_dir),
        "raw_metadata_path": str(raw_path),
        "raw_metadata_source": raw_source,
        "n_cif_files": int(len(cif_files)) if raw_source == "rebuilt_from_cif" else None,
        "n_raw_built": int(len(raw_df)),
        "n_build_errors": int(len(err_df)),
        "dedup_scope": dedup_scope,
        "dedup_dataset": stats_dataset,
        "dedup_internal_amcsd": stats_internal,
        "dedup_rruff_test": stats_rruff,
        "n_final_train": int(len(final_df)),
        "n_excluded_total": int(len(excluded_df)),
        "n_removed_excluded_artifacts": int(removed_artifacts),
        "n_missing_artifact_rows": int(len(missing_artifacts_df)),
        "n_rebuild_errors": int(len(rebuild_errors_df)),
    }
    stats_path = output_dir / "amcsd_build_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    LOGGER.info("Final AMCSD train CSV: %s (%d rows)", train_path, len(final_df))
    LOGGER.info("Excluded AMCSD rows: %s (%d rows)", excluded_path, len(excluded_df))
    LOGGER.info("Stats: %s", stats_path)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
