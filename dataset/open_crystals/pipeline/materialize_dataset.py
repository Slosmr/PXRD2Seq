from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.io.cif import CifParser


ROOT = Path(__file__).resolve().parent
DATASET_CODE_DIR = ROOT
if str(DATASET_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_CODE_DIR))

from get_data import (  # noqa: E402
    CONFIG,
    configure_warning_filters,
    compute_base_peaks,
    generate_structure_labels,
    get_bravais_lattice,
    process_structure_to_record,
    save_standardized_structure,
    strip_oxidation_states_safe,
    _as_float_d_spacing,
    _composition_summary,
)


CORE_ARTIFACT_COLUMNS = ("base_peaks_path", "labels_path", "structure_path")
CORE_OUTPUT_COLUMNS = [
    "source",
    "material_id",
    "base_material_id",
    "formula",
    "reduced_formula",
    "anonymous_formula",
    "crystal_system",
    "space_group",
    "space_group_number",
    "laue_class",
    "bravais_lattice",
    "a",
    "b",
    "c",
    "alpha",
    "beta",
    "gamma",
    "volume",
    "n_elements",
    "n_sites",
    "volume_per_atom",
    "n_base_peaks",
    "base_peaks_path",
    "labels_path",
    "structure_path",
    "cod_id",
    "cif_path",
    "energy_above_hull",
    "original_split",
    "sample_weight",
    "review_reason",
    "hard_exclude_reason",
    "dedup_cluster_id",
    "dedup_key",
    "family",
    "mineral_name",
    "compound_source",
    "title",
    "risk_flags",
    "risk_score",
    "source_unknown",
    "parse_status",
    "icsd_id",
    "formula_sum",
    "cod_risk_flags",
    "cod_risk_score",
    "cod_metadata_status",
]


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_int(value: Any, default: int = -1) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        text = str(value).strip()
        if not text:
            return default
        return int(float(text))
    except Exception:
        return default


def first_nonempty(*values: Any, default: str = "") -> str:
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return default


def workspace_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def resolve_path(root: Path, value: Any) -> Path | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return root / path


def ensure_output_dirs(output_dir: Path) -> None:
    for name in ("base_peaks", "labels", "structures"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def verify_inside_output(path: Path, output_dir: Path) -> None:
    path.resolve().relative_to(output_dir.resolve())


def reset_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for child in ("base_peaks", "labels", "structures"):
        target = output_dir / child
        if not target.exists():
            continue
        verify_inside_output(target, output_dir)
        shutil.rmtree(target)


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    if mode == "reference":
        return "referenced"
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copied"
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlinked"
        except OSError:
            shutil.copy2(src, dst)
            return "copied_fallback"
    raise ValueError(f"unknown artifact mode: {mode}")


def destination_paths(output_dir: Path, material_id: str) -> dict[str, Path]:
    mid = str(material_id).strip().lower()
    return {
        "base_peaks_path": output_dir / "base_peaks" / f"{mid}.npz",
        "labels_path": output_dir / "labels" / f"{mid}.json",
        "structure_path": output_dir / "structures" / f"{mid}.cif",
    }


def artifact_paths_exist(output_dir: Path, material_id: str) -> bool:
    paths = destination_paths(output_dir, material_id)
    return all(path.exists() for path in paths.values())


def update_row_paths(row: dict[str, Any], output_dir: Path, root: Path, reference: bool = False) -> dict[str, Any]:
    out = dict(row)
    if reference:
        for col in CORE_ARTIFACT_COLUMNS:
            value = out.get(col, "")
            path = resolve_path(root, value)
            out[col] = workspace_rel(path, root) if path else value
        return out

    paths = destination_paths(output_dir, out["material_id"])
    for col, path in paths.items():
        out[col] = workspace_rel(path, root)
    base_path = paths["base_peaks_path"]
    if base_path.exists():
        try:
            with np.load(base_path) as peaks:
                if "mus" in peaks.files:
                    out["n_base_peaks"] = int(len(peaks["mus"]))
        except Exception:
            pass
    return out


def copy_existing_artifacts(
    row: pd.Series,
    root: Path,
    output_dir: Path,
    artifact_mode: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    row_dict = row.to_dict()
    material_id = str(row_dict["material_id"]).strip().lower()
    stats: Counter[str] = Counter()

    if artifact_mode == "reference":
        stats["referenced_rows"] += 1
        return update_row_paths(row_dict, output_dir, root, reference=True), dict(stats)

    dst_paths = destination_paths(output_dir, material_id)
    for col in CORE_ARTIFACT_COLUMNS:
        src = resolve_path(root, row_dict.get(col))
        if src is None:
            raise FileNotFoundError(f"{material_id}: missing {col}")
        result = link_or_copy(src, dst_paths[col], artifact_mode)
        stats[result] += 1

    return update_row_paths(row_dict, output_dir, root), dict(stats)


def load_icsd_structure(cif_path: Path):
    last_exc: Exception | None = None
    for check_occu in (True, False):
        try:
            with warnings.catch_warnings():
                configure_warning_filters()
                parser = CifParser(
                    str(cif_path),
                    occupancy_tolerance=1.0,
                    check_cif=False,
                    comp_tol=0.05,
                )
                structures = parser.parse_structures(
                    primitive=False,
                    symmetrized=False,
                    check_occu=check_occu,
                    on_error="ignore",
                )
            if structures:
                return structures[0], ("strict" if check_occu else "no_occu_check")
        except Exception as exc:
            last_exc = exc
    if last_exc:
        raise last_exc
    raise ValueError("empty_structure")


def build_icsd_artifacts(
    row: pd.Series,
    root: Path,
    output_dir: Path,
    cfg: dict[str, Any],
    skip_existing: bool,
) -> dict[str, Any]:
    row_dict = row.to_dict()
    material_id = str(row_dict["material_id"]).strip().lower()
    if skip_existing and artifact_paths_exist(output_dir, material_id):
        return update_row_paths(row_dict, output_dir, root)

    cif_path = resolve_path(root, row_dict.get("cif_path"))
    if cif_path is None or not cif_path.exists():
        raise FileNotFoundError(f"{material_id}: missing ICSD CIF {row_dict.get('cif_path')}")

    structure, load_mode = load_icsd_structure(cif_path)
    structure, _ = strip_oxidation_states_safe(structure)

    extra = {
        key: value
        for key, value in row_dict.items()
        if key not in set(CORE_OUTPUT_COLUMNS) | set(CORE_ARTIFACT_COLUMNS)
    }
    for key in (
        "cif_path",
        "icsd_id",
        "formula_sum",
        "family",
        "mineral_name",
        "compound_source",
        "title",
        "risk_flags",
        "risk_score",
        "source_unknown",
        "sample_weight",
        "review_reason",
        "parse_status",
        "dedup_cluster_id",
        "dedup_key",
    ):
        if key in row_dict:
            extra[key] = row_dict.get(key)
    extra["icsd_load_mode"] = load_mode

    try:
        built = process_structure_to_record(
            structure=structure,
            material_id=material_id,
            source="ICSD",
            output_dir=output_dir,
            cfg=cfg,
            extra_metadata=extra,
        )
    except RuntimeError as exc:
        if not str(exc).startswith("symmetry_failed"):
            raise
        built = process_icsd_with_manifest_symmetry(
            structure=structure,
            row_dict=row_dict,
            material_id=material_id,
            output_dir=output_dir,
            cfg=cfg,
            extra_metadata=extra,
        )
    return update_row_paths(built, output_dir, root)


def process_icsd_with_manifest_symmetry(
    structure,
    row_dict: dict[str, Any],
    material_id: str,
    output_dir: Path,
    cfg: dict[str, Any],
    extra_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build ICSD artifacts when spglib cannot infer symmetry.

    Several ICSD mineral CIFs contain partial occupancies or disordered sites.
    Pymatgen can parse the structure and XRDCalculator can compute a pattern,
    but SpacegroupAnalyzer may fail. In that case the manifest's CIF-tag-derived
    symmetry labels are safer than dropping the sample.
    """
    paths = {
        "root": output_dir,
        "base_peaks": output_dir / "base_peaks",
        "labels": output_dir / "labels",
        "structures": output_dir / "structures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    try:
        structure_out = structure.get_reduced_structure(reduction_algo="niggli")
    except Exception:
        structure_out = structure

    try:
        base_peaks = compute_base_peaks(structure_out, cfg)
    except Exception:
        if structure_out is not structure:
            base_peaks = compute_base_peaks(structure, cfg)
            structure_out = structure
        else:
            raise
    if not base_peaks:
        raise RuntimeError("xrd_failed: no_base_peaks")

    mus = np.array([float(p[0]) for p in base_peaks], dtype=np.float32)
    intensities = np.array([float(p[1]) for p in base_peaks], dtype=np.float32)
    d_hkls = np.array([_as_float_d_spacing(p[2]) for p in base_peaks], dtype=np.float32)

    base_peaks_path = paths["base_peaks"] / f"{material_id}.npz"
    labels_path = paths["labels"] / f"{material_id}.json"
    structure_path = paths["structures"] / f"{material_id}.cif"
    np.savez_compressed(str(base_peaks_path), mus=mus, intensities=intensities, d_hkls=d_hkls)
    save_standardized_structure(structure_out, structure_path)

    summary = _composition_summary(structure_out)
    sg_number = safe_int(row_dict.get("space_group_number"), -1)
    sg_symbol = first_nonempty(row_dict.get("space_group"), default="unknown")
    crystal_system = first_nonempty(row_dict.get("crystal_system"), default="unknown")
    laue_class = first_nonempty(row_dict.get("laue_class"), default="unknown")
    bravais = first_nonempty(
        row_dict.get("bravais_lattice"),
        default=get_bravais_lattice(crystal_system, sg_symbol),
    )
    formula = first_nonempty(row_dict.get("formula"), row_dict.get("reduced_formula"), summary["formula"])

    labels = generate_structure_labels(
        structure_out,
        material_id,
        formula,
        crystal_system,
        sg_symbol,
        laue_class,
        bravais,
        sg_number=sg_number,
        source="ICSD",
    )
    with labels_path.open("w", encoding="utf-8") as handle:
        json.dump(labels, handle, ensure_ascii=False, indent=2)

    lattice = structure_out.lattice
    volume_per_atom = float(lattice.volume) / max(1, len(structure_out))
    built = {
        "source": "ICSD",
        "material_id": material_id,
        "base_material_id": material_id,
        "formula": formula,
        "reduced_formula": first_nonempty(row_dict.get("reduced_formula"), summary["reduced_formula"]),
        "anonymous_formula": summary["anonymous_formula"],
        "crystal_system": crystal_system,
        "space_group": sg_symbol,
        "space_group_number": sg_number,
        "laue_class": laue_class,
        "bravais_lattice": bravais,
        "a": labels["a"],
        "b": labels["b"],
        "c": labels["c"],
        "alpha": labels["alpha"],
        "beta": labels["beta"],
        "gamma": labels["gamma"],
        "volume": labels["volume"],
        "n_elements": summary["n_elements"],
        "n_sites": summary["n_sites"],
        "volume_per_atom": round(volume_per_atom, 6),
        "n_base_peaks": int(len(mus)),
        "base_peaks_path": base_peaks_path.as_posix(),
        "labels_path": labels_path.as_posix(),
        "structure_path": structure_path.as_posix(),
    }
    for key, value in extra_metadata.items():
        if key not in built:
            built[key] = value
    built["icsd_symmetry_source"] = "manifest_fallback"
    return built


def assign_splits(df: pd.DataFrame) -> pd.Series:
    source = df["source"].fillna("").astype(str).str.upper()
    original = df.get("original_split", pd.Series("train", index=df.index)).fillna("train").astype(str).str.lower()
    split = original.where(original.isin(["train", "val", "test"]), "train")
    split = split.mask(source.isin(["ICSD", "AMCSD"]), "train")
    return split


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in CORE_OUTPUT_COLUMNS if col in df.columns]
    rest = [col for col in df.columns if col not in cols]
    return df[cols + rest]


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "filter_keep" in df.columns:
        df = df[df["filter_keep"].map(safe_bool)].copy()
    if "dedup_keep" in df.columns:
        df = df[df["dedup_keep"].map(safe_bool)].copy()
    df["source"] = df["source"].fillna("").astype(str).str.upper()
    df["material_id"] = df["material_id"].fillna("").astype(str).str.lower()
    df = df[df["material_id"].ne("")]
    return df.reset_index(drop=True)


def write_csvs(output_dir: Path, rows: list[dict[str, Any]], errors: list[dict[str, Any]]) -> dict[str, Any]:
    df = reorder_columns(pd.DataFrame(rows))
    if df.empty:
        metadata = pd.DataFrame(columns=CORE_OUTPUT_COLUMNS)
        train = metadata.copy()
        val = metadata.copy()
        test = metadata.copy()
    else:
        df["_split"] = assign_splits(df)
        train = df[df["_split"].eq("train")].drop(columns=["_split"]).copy()
        val = df[df["_split"].eq("val")].drop(columns=["_split"]).copy()
        test = df[df["_split"].eq("test")].drop(columns=["_split"]).copy()
        metadata = df.drop(columns=["_split"]).copy()

    metadata.to_csv(output_dir / "metadata.csv", index=False)
    metadata.to_csv(output_dir / "metadata_v2.csv", index=False)
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "materialize_errors.csv", index=False)

    summary = {
        "metadata_rows": int(len(metadata)),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "source_counts": (
            {str(k): int(v) for k, v in metadata["source"].value_counts().to_dict().items()}
            if "source" in metadata.columns
            else {}
        ),
        "train_source_counts": (
            {str(k): int(v) for k, v in train["source"].value_counts().to_dict().items()}
            if "source" in train.columns
            else {}
        ),
        "val_source_counts": (
            {str(k): int(v) for k, v in val["source"].value_counts().to_dict().items()}
            if "source" in val.columns
            else {}
        ),
        "test_source_counts": (
            {str(k): int(v) for k, v in test["source"].value_counts().to_dict().items()}
            if "source" in test.columns
            else {}
        ),
        "rruff_leakage_excluded": 0,
        "materialize_errors": int(len(errors)),
    }
    (output_dir / "materialize_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize dataset from an audited reconstruction manifest.")
    parser.add_argument("--manifest", type=Path, default=ROOT.parent / "build" / "dataset_plan" / "manifest_dedup_keep.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "build" / "dataset")
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--artifact-mode", choices=["hardlink", "copy", "reference"], default="hardlink")
    parser.add_argument("--reset-output", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-icsd", type=int, default=None)
    parser.add_argument("--only-sources", type=str, default="")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = args.workspace.resolve()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()

    if args.reset_output:
        reset_output_dir(output_dir)
    ensure_output_dirs(output_dir)

    cfg = copy.deepcopy(CONFIG)
    cfg["output_dir"] = str(output_dir)

    manifest = load_manifest(manifest_path)
    if args.only_sources.strip():
        wanted = {part.strip().upper() for part in args.only_sources.split(",") if part.strip()}
        manifest = manifest[manifest["source"].isin(wanted)].copy()
    if args.max_icsd is not None:
        is_icsd = manifest["source"].eq("ICSD")
        keep_icsd = manifest[is_icsd].head(args.max_icsd)
        manifest = pd.concat([manifest[~is_icsd], keep_icsd], ignore_index=True)
    if args.max_rows is not None:
        manifest = manifest.head(args.max_rows).copy()

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    artifact_stats: Counter[str] = Counter()
    source_seen: Counter[str] = Counter()
    skip_existing = not args.no_skip_existing

    total = len(manifest)
    for pos, (_, row) in enumerate(manifest.iterrows(), start=1):
        source = str(row.get("source", "")).upper()
        material_id = str(row.get("material_id", "")).lower()
        source_seen[source] += 1
        if pos == 1 or pos % 5000 == 0 or (source == "ICSD" and source_seen[source] % 100 == 0):
            print(
                f"Materializing {pos}/{total}: {material_id} ({source}, source #{source_seen[source]})",
                flush=True,
            )
        try:
            if source == "ICSD":
                out_row = build_icsd_artifacts(
                    row=row,
                    root=root,
                    output_dir=output_dir,
                    cfg=cfg,
                    skip_existing=skip_existing,
                )
                artifact_stats["icsd_built_or_reused"] += 1
            else:
                out_row, stats = copy_existing_artifacts(
                    row=row,
                    root=root,
                    output_dir=output_dir,
                    artifact_mode=args.artifact_mode,
                )
                artifact_stats.update(stats)
            rows.append(out_row)
        except Exception as exc:
            err = {
                "material_id": material_id,
                "source": source,
                "error": f"{type(exc).__name__}: {exc}",
                "cif_path": row.get("cif_path", ""),
            }
            errors.append(err)
            print(f"ERROR {material_id}: {err['error']}", flush=True)
            if args.strict:
                raise

    summary = write_csvs(output_dir, rows, errors)
    summary["artifact_stats"] = {str(k): int(v) for k, v in artifact_stats.items()}
    (output_dir / "materialize_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 1 if errors and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
