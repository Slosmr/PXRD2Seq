from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from pymatgen.core import Composition
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


ROOT = Path(__file__).resolve().parent
warnings.filterwarnings("ignore", message="No Pauling electronegativity.*")

SOURCE_PRIORITY = {
    "ICSD": 0,
    "AMCSD": 1,
    "COD": 2,
    "MP": 3,
}

RISK_TERMS = {
    "synthetic": ["synthetic", "synthesized", "synthesised", "synthesis"],
    "pressure": [
        "high pressure",
        "high-pressure",
        "pressure",
        "diamond anvil",
        "diamond-anvil",
        "gpa",
        "kbar",
        "mpa",
        "megapascal",
        "megapascals",
    ],
    "thermal": [
        "heated",
        "heating",
        "heat-treated",
        "quenched",
        "quench",
        "annealed",
        "annealing",
        "melt",
        "dehydrat",
        "calcined",
    ],
    "growth": [
        "hydrothermal",
        "crystal growth",
        "grown",
        "flux",
        "glass",
        "ceramic",
        "solid-state",
        "solid state",
        "sinter",
        "sol-gel",
        "combustion",
        "verneuil",
    ],
}

DEFAULT_COD_HARD_RISK_FLAGS = ("pressure", "high_temperature")
DEFAULT_COD_REVIEW_RISK_FLAGS = ("synthetic", "thermal", "growth", "low_temperature")
TEMPERATURE_TAGS = (
    "_diffrn_ambient_temperature",
    "_cell_measurement_temperature",
    "_cell_measurement_temperature_min",
    "_cell_measurement_temperature_max",
)

TEMP_RE = re.compile(
    r"\b(?:[7-9]\d{2}|1\d{3}|2\d{3})\s*(?:k|kelvin|c|deg|degrees|grad celsius)\b",
    re.IGNORECASE,
)
TAG_RE = re.compile(r"^(_[A-Za-z0-9_.-]+)\s+(.*)$")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


try:
    from organic_filter import looks_organic_v2 as _looks_organic_v2
except Exception:  # pragma: no cover - fallback keeps this script standalone.
    _looks_organic_v2 = None


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "?", "."}:
        return None
    text = re.sub(r"\([0-9]+\)$", "", text)
    try:
        out = float(text)
    except Exception:
        return None
    if math.isnan(out):
        return None
    return out


def safe_int(value: Any, default: int = -1) -> int:
    number = safe_float(value)
    if number is None:
        return default
    return int(round(number))


def crystal_system_from_sg(sg: int) -> str:
    if 1 <= sg <= 2:
        return "triclinic"
    if 3 <= sg <= 15:
        return "monoclinic"
    if 16 <= sg <= 74:
        return "orthorhombic"
    if 75 <= sg <= 142:
        return "tetragonal"
    if 143 <= sg <= 167:
        return "trigonal"
    if 168 <= sg <= 194:
        return "hexagonal"
    if 195 <= sg <= 230:
        return "cubic"
    return "unknown"


def laue_from_sg(sg: int) -> str:
    if 1 <= sg <= 2:
        return "-1"
    if 3 <= sg <= 15:
        return "2/m"
    if 16 <= sg <= 74:
        return "mmm"
    if 75 <= sg <= 88:
        return "4/m"
    if 89 <= sg <= 142:
        return "4/mmm"
    if 143 <= sg <= 148:
        return "-3"
    if 149 <= sg <= 167:
        return "-3m"
    if 168 <= sg <= 176:
        return "6/m"
    if 177 <= sg <= 194:
        return "6/mmm"
    if 195 <= sg <= 206:
        return "m-3"
    if 207 <= sg <= 230:
        return "m-3m"
    return "unknown"


def normalize_name(value: Any) -> str:
    return NON_ALNUM_RE.sub("", str(value or "").lower())


def normalize_formula(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return Composition(text).reduced_formula
    except Exception:
        return re.sub(r"\s+", "", text)


def normalize_rruff_formula(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if any(ch in text for ch in "[]_^"):
        return ""
    if re.search(r"\d+(?:\.\d+)?\s*-\s*\d", text):
        return ""
    return normalize_formula(text)


def risk_flags_for_text(text: str, include_numeric_temperature: bool = True) -> list[str]:
    lowered = text.lower()
    flags: list[str] = []
    for flag, terms in RISK_TERMS.items():
        if any(term in lowered for term in terms):
            flags.append(flag)
    if include_numeric_temperature and TEMP_RE.search(lowered):
        flags.append("high_temperature")
    return sorted(set(flags))


def parse_cif_tags(text: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match = TAG_RE.match(line)
        if not match:
            i += 1
            continue
        tag, value = match.group(1).lower(), match.group(2).strip()
        if value == ";":
            block: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith(";"):
                block.append(lines[i].strip())
                i += 1
            tags[tag] = " ".join(part for part in block if part)
        else:
            tags[tag] = value.strip().strip("'").strip('"')
        i += 1
    return tags


def first_tag(tags: dict[str, str], names: list[str], default: str = "") -> str:
    for name in names:
        value = tags.get(name.lower())
        if value not in (None, "", "?"):
            return str(value)
    return default


def rel_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return str(path)


def resolve_workspace_path(root: Path, value: Any) -> Path | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return root / path


def append_reason(current: Any, reason: str) -> str:
    text = str(current or "").strip()
    return reason if not text else f"{text};{reason}"


def split_flag_list(value: Any) -> set[str]:
    return {
        part.strip()
        for part in str(value or "").split(",")
        if part.strip()
    }


def judge_cod_organic(formula: str, c_frac_threshold: float) -> tuple[bool, str]:
    if _looks_organic_v2 is not None:
        return _looks_organic_v2(formula, c_frac_threshold=c_frac_threshold)


    tokens = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", formula or "")
    counts: dict[str, float] = {}
    for sym, num in tokens:
        counts[sym] = counts.get(sym, 0.0) + (float(num) if num else 1.0)
    syms = set(counts)
    metals = {
        "Li", "Na", "K", "Rb", "Cs", "Fr", "Be", "Mg", "Ca", "Sr", "Ba", "Ra",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Zr",
        "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "Hf", "Ta", "W", "Re",
        "Os", "Ir", "Pt", "Au", "Hg", "Al", "Ga", "In", "Sn", "Tl", "Pb",
        "Bi", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
        "Er", "Tm", "Yb", "Lu", "Th", "U",
    }
    metalloids = {"B", "Si", "Ge", "As", "Se", "Te"}
    hetero = {"N", "O", "S", "P", "F", "Cl", "Br", "I"}
    if syms & metals:
        return False, "not_organic_metal"
    if (syms & metalloids) and "H" not in syms:
        return False, "not_organic_metalloid_no_H"
    if "C" not in syms:
        return False, "not_organic_no_C"
    if "H" in syms and (syms & hetero):
        return True, "organic_v1_CH_hetero"
    total = sum(counts.values())
    c_frac = counts.get("C", 0.0) / total if total else 0.0
    if c_frac >= c_frac_threshold:
        return True, "organic_v2_C_dominant"
    return False, "not_organic_other"


def structure_summary_from_cif(
    path: Path,
    symprec: float,
    parse_structure: bool,
) -> tuple[dict[str, Any], str]:
    if not parse_structure:
        return {}, "skipped"

    last_error = "no_structure"
    for check_occu, ok_status in ((True, "ok"), (False, "ok_no_occu_check")):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parser = CifParser(
                    str(path),
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
            if not structures:
                last_error = "no_structure"
                continue
        except Exception as exc:
            last_error = f"error:{type(exc).__name__}"
            continue

        structure = structures[0]
        lattice = structure.lattice
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            formula = structure.composition.formula
            reduced_formula = structure.composition.reduced_formula
        summary: dict[str, Any] = {
            "formula": formula,
            "reduced_formula": reduced_formula,
            "a": lattice.a,
            "b": lattice.b,
            "c": lattice.c,
            "alpha": lattice.alpha,
            "beta": lattice.beta,
            "gamma": lattice.gamma,
            "volume": lattice.volume,
            "n_sites": len(structure),
            "volume_per_atom": lattice.volume / max(1, len(structure)),
            "n_elements": len(structure.composition.elements),
        }
        try:
            analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
            summary["space_group_number_calc"] = analyzer.get_space_group_number()
            summary["space_group_calc"] = analyzer.get_space_group_symbol()
        except Exception:
            pass
        return summary, ok_status

    return {}, last_error


def load_current_dataset(root: Path, dataset_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in ("train", "val", "test"):
        path = dataset_dir / f"{split}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        df["original_split"] = split
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Fresh get_data.py outputs artifact paths relative to dataset_dir
    # (base_peaks/..., labels/..., structures/...). Normalize them to the
    # reconstruction workspace so materialization can resolve every source.
    artifact_dirs = {"base_peaks", "labels", "structures"}
    for col in ("base_peaks_path", "labels_path", "structure_path"):
        if col not in out.columns:
            continue
        normalized: list[str] = []
        for value in out[col]:
            if value is None or pd.isna(value) or not str(value).strip():
                normalized.append("")
                continue
            path = Path(str(value).strip())
            if not path.is_absolute():
                first = path.parts[0].lower() if path.parts else ""
                path = (dataset_dir if first in artifact_dirs else root) / path
            normalized.append(rel_path(path, root))
        out[col] = normalized
    out["source"] = out["source"].fillna("").astype(str).str.upper()
    out["family"] = ""
    out["mineral_name"] = ""
    out["compound_source"] = ""
    out["title"] = ""
    out["risk_flags"] = ""
    out["risk_score"] = 0
    out["source_unknown"] = False
    out["parse_status"] = "existing_artifact"
    out["source_priority"] = out["source"].map(SOURCE_PRIORITY).fillna(99).astype(int)
    return out


def normalize_replacement_amcsd_paths(
    root: Path,
    amcsd_root: Path,
    df: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()
    for col in ("base_peaks_path", "labels_path", "structure_path"):
        if col not in out.columns:
            raise ValueError(f"replacement AMCSD metadata is missing required column: {col}")
        values: list[str] = []
        for value in out[col]:
            if value is None or pd.isna(value) or not str(value).strip():
                values.append("")
                continue
            path = Path(str(value).strip())
            if not path.is_absolute():
                path = amcsd_root / path
            values.append(rel_path(path, root))
        out[col] = values

    if "cif_path" in out.columns:
        values = []
        for value in out["cif_path"]:
            if value is None or pd.isna(value) or not str(value).strip():
                values.append("")
                continue
            path = Path(str(value).strip())
            values.append(rel_path(path, root) if path.is_absolute() else str(value).strip())
        out["cif_path"] = values
    return out


def load_replacement_amcsd(
    root: Path,
    metadata_path: Path,
    amcsd_root: Path | None,
) -> pd.DataFrame:
    metadata_path = metadata_path.resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(f"replacement AMCSD metadata not found: {metadata_path}")
    base_dir = (amcsd_root or metadata_path.parent).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"replacement AMCSD root not found: {base_dir}")

    df = pd.read_csv(metadata_path, low_memory=False)
    if df.empty:
        return df
    if "source" not in df.columns:
        df["source"] = "AMCSD"
    df["source"] = df["source"].fillna("").astype(str).str.upper()
    non_amcsd = df[~df["source"].eq("AMCSD")]
    if not non_amcsd.empty:
        raise ValueError(
            "replacement AMCSD metadata must contain only AMCSD rows; "
            f"found sources: {sorted(non_amcsd['source'].dropna().unique().tolist())}"
        )

    if "original_split" not in df.columns:
        df["original_split"] = "train"
    else:
        df["original_split"] = df["original_split"].fillna("train").replace("", "train")

    defaults: dict[str, Any] = {
        "family": "",
        "mineral_name": "",
        "compound_source": "",
        "title": "",
        "risk_flags": "",
        "risk_score": 0,
        "source_unknown": False,
        "parse_status": "replacement_amcsd_artifact",
        "sample_weight": 1.0,
        "review_reason": "",
        "hard_exclude_reason": "",
        "cod_risk_flags": "",
        "cod_risk_score": 0,
        "cod_metadata_status": "",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    if "amcsd_name" in df.columns:
        empty_name = df["mineral_name"].fillna("").astype(str).str.strip().eq("")
        df.loc[empty_name, "mineral_name"] = df.loc[empty_name, "amcsd_name"].fillna("")
        empty_family = df["family"].fillna("").astype(str).str.strip().eq("")
        df.loc[empty_family, "family"] = df.loc[empty_family, "mineral_name"].map(normalize_name)
    if "amcsd_formula_sum" in df.columns and "formula_sum" not in df.columns:
        df["formula_sum"] = df["amcsd_formula_sum"]

    df = normalize_replacement_amcsd_paths(root, base_dir, df)
    df["source_priority"] = SOURCE_PRIORITY["AMCSD"]
    return df


def enrich_current_amcsd(root: Path, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "source" not in df.columns:
        return df
    out = df.copy()
    mask = out["source"].astype(str).str.upper().eq("AMCSD")
    if not mask.any():
        return out

    cache: dict[str, dict[str, Any]] = {}
    for idx, row in out.loc[mask].iterrows():
        structure_path = resolve_workspace_path(root, row.get("structure_path"))
        key = str(structure_path or "")
        if key not in cache:
            meta: dict[str, Any] = {
                "mineral_name": "",
                "compound_source": "",
                "title": "",
                "risk_flags": "",
                "risk_score": 0,
                "source_unknown": True,
                "family": "",
            }
            if structure_path and structure_path.exists():
                try:
                    tags = parse_cif_tags(structure_path.read_text(encoding="utf-8", errors="ignore"))
                except OSError:
                    tags = {}
                name = first_tag(tags, ["_chemical_name_mineral", "_chemical_name_common"])
                source = first_tag(tags, ["_chemical_compound_source"])
                title = first_tag(tags, ["_publ_section_title"])
                risk_text = " | ".join(
                    [
                        source,
                        title,
                        first_tag(tags, ["_chemical_name_systematic"]),
                        first_tag(tags, ["_chemical_formula_structural"]),
                    ]
                )
                flags = risk_flags_for_text(risk_text)
                meta.update(
                    {
                        "mineral_name": name,
                        "family": normalize_name(name),
                        "compound_source": source,
                        "title": title,
                        "risk_flags": ";".join(flags),
                        "risk_score": len(flags),
                        "source_unknown": not bool(source.strip()),
                    }
                )
            cache[key] = meta
        for col, value in cache[key].items():
            out.at[idx, col] = value
    return out


def resolve_cod_raw_cif(root: Path, cod_raw_root: Path | None, row: pd.Series) -> Path | None:
    candidates: list[Path] = []
    cif_path = resolve_workspace_path(root, row.get("cif_path"))
    if cif_path is not None:
        candidates.append(cif_path)

    if cod_raw_root is not None:
        raw_value = str(row.get("cif_path") or "").strip()
        if raw_value:
            raw_path = Path(raw_value)
            if not raw_path.is_absolute():
                parts = raw_path.parts
                if parts and parts[0].lower() == "cod":
                    candidates.append(cod_raw_root.joinpath(*parts[1:]))
                candidates.append(cod_raw_root / raw_path)
        cod_id = str(row.get("cod_id") or "").strip()
        if cod_id:
            candidates.append(cod_raw_root / f"{cod_id}.cif")
            candidates.extend(cod_paths_from_id(cod_raw_root, cod_id))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def cod_paths_from_id(cod_raw_root: Path, cod_id: Any) -> list[Path]:
    raw = str(cod_id or "").strip()
    if re.fullmatch(r"\d+(?:\.0+)?", raw):
        text = raw.split(".", 1)[0]
    else:
        text = re.sub(r"\D", "", raw)
    if len(text) < 5:
        return []
    filename = f"{text}.cif"
    first = text[0]
    second = text[1:3]
    third = text[3:5]
    roots = [cod_raw_root]
    if cod_raw_root.name.lower() != "cif":
        roots.insert(0, cod_raw_root / "cif")
    return [base / first / second / third / filename for base in roots]


def cod_temperature_flags(tags: dict[str, str], low_lt: float, high_gt: float) -> list[str]:
    flags: list[str] = []
    for tag in TEMPERATURE_TAGS:
        value = safe_float(tags.get(tag))
        if value is None:
            continue
        if value < low_lt:
            flags.append("low_temperature")
        elif value > high_gt:
            flags.append("high_temperature")
    return sorted(set(flags))


def cod_risk_from_tags(tags: dict[str, str]) -> tuple[list[str], str, str, str]:
    title = first_tag(tags, ["_publ_section_title"])
    source_text = first_tag(tags, ["_chemical_compound_source", "_exptl_crystal_description"])
    chemical_name = " | ".join(
        part
        for part in [
            first_tag(tags, ["_chemical_name_mineral"]),
            first_tag(tags, ["_chemical_name_common"]),
            first_tag(tags, ["_chemical_name_systematic"]),
        ]
        if part
    )
    experimental_text = " | ".join(
        str(value)
        for key, value in tags.items()
        if key.startswith("_exptl") or key.startswith("_diffrn") or key.startswith("_publ")
    )
    risk_text = " | ".join([title, source_text, chemical_name, experimental_text])
    flags = risk_flags_for_text(risk_text, include_numeric_temperature=False)
    return flags, title, source_text, chemical_name


def enrich_current_cod(
    root: Path,
    df: pd.DataFrame,
    cod_raw_root: Path | None,
    read_structure_fallback: bool,
    low_temperature_lt: float,
    high_temperature_gt: float,
) -> pd.DataFrame:
    if df.empty or "source" not in df.columns:
        return df
    out = df.copy()
    mask = out["source"].astype(str).str.upper().eq("COD")
    if not mask.any():
        return out

    out.loc[mask, "cod_risk_flags"] = ""
    out.loc[mask, "cod_risk_score"] = 0
    out.loc[mask, "cod_metadata_status"] = "metadata_unavailable"
    out.loc[mask, "cod_risk_title"] = ""
    out.loc[mask, "cod_risk_source_text"] = ""
    out.loc[mask, "cod_risk_chemical_name"] = ""

    cache: dict[str, dict[str, Any]] = {}
    for idx, row in out.loc[mask].iterrows():
        raw_path = resolve_cod_raw_cif(root, cod_raw_root, row)
        parse_path = raw_path
        status = "raw_cif" if raw_path else "metadata_unavailable"
        if parse_path is None and read_structure_fallback:
            parse_path = resolve_workspace_path(root, row.get("structure_path"))
            if parse_path and parse_path.exists():
                status = "structure_cif_fallback"
            else:
                parse_path = None

        cache_key = str(parse_path or f"missing:{idx}")
        if cache_key not in cache:
            meta: dict[str, Any] = {
                "cod_risk_flags": "",
                "cod_risk_score": 0,
                "cod_metadata_status": status,
                "cod_risk_title": "",
                "cod_risk_source_text": "",
                "cod_risk_chemical_name": "",
            }
            if parse_path and parse_path.exists():
                try:
                    tags = parse_cif_tags(parse_path.read_text(encoding="utf-8", errors="ignore"))
                except OSError:
                    tags = {}
                flags, title, source_text, chemical_name = cod_risk_from_tags(tags)
                flags = sorted(set(flags + cod_temperature_flags(tags, low_temperature_lt, high_temperature_gt)))
                meta.update(
                    {
                        "cod_risk_flags": ";".join(flags),
                        "cod_risk_score": len(flags),
                        "cod_metadata_status": status,
                        "cod_risk_title": title,
                        "cod_risk_source_text": source_text,
                        "cod_risk_chemical_name": chemical_name,
                    }
                )
            cache[cache_key] = meta

        for col, value in cache[cache_key].items():
            out.at[idx, col] = value
    return out


def parse_icsd_cif(path: Path, root: Path, symprec: float, parse_structure: bool) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        text = ""
    tags = parse_cif_tags(text)
    summary, parse_status = structure_summary_from_cif(path, symprec, parse_structure)

    icsd_id = first_tag(tags, ["_database_code_icsd"], path.stem)
    icsd_id = re.sub(r"^icsd[_-]?", "", icsd_id, flags=re.IGNORECASE)
    family = path.parent.name
    formula_sum = first_tag(tags, ["_chemical_formula_sum", "_chemical_formula_structural"])
    formula = summary.get("formula") or formula_sum
    reduced_formula = summary.get("reduced_formula") or normalize_formula(formula)

    sg_tag = first_tag(
        tags,
        [
            "_symmetry_int_tables_number",
            "_space_group_it_number",
            "_space_group.it_number",
        ],
    )
    sg = safe_int(sg_tag)
    sg_calc = safe_int(summary.get("space_group_number_calc"), -1)
    if sg <= 0 and sg_calc > 0:
        sg = sg_calc

    mineral = first_tag(tags, ["_chemical_name_mineral", "_chemical_name_common"])
    compound_source = first_tag(tags, ["_chemical_compound_source"])
    title = first_tag(tags, ["_publ_section_title"])
    risk_text = " | ".join(
        [
            compound_source,
            title,
            first_tag(tags, ["_chemical_name_systematic"]),
            first_tag(tags, ["_chemical_formula_structural"]),
        ]
    )
    flags = risk_flags_for_text(risk_text)

    a = summary.get("a", safe_float(first_tag(tags, ["_cell_length_a"])))
    b = summary.get("b", safe_float(first_tag(tags, ["_cell_length_b"])))
    c = summary.get("c", safe_float(first_tag(tags, ["_cell_length_c"])))
    alpha = summary.get("alpha", safe_float(first_tag(tags, ["_cell_angle_alpha"])))
    beta = summary.get("beta", safe_float(first_tag(tags, ["_cell_angle_beta"])))
    gamma = summary.get("gamma", safe_float(first_tag(tags, ["_cell_angle_gamma"])))
    volume = summary.get("volume", safe_float(first_tag(tags, ["_cell_volume"])))
    n_sites = summary.get("n_sites", "")
    volume_per_atom = summary.get("volume_per_atom", "")

    return {
        "source": "ICSD",
        "material_id": f"icsd-{icsd_id}",
        "base_material_id": f"icsd-{icsd_id}",
        "formula": formula,
        "reduced_formula": reduced_formula,
        "anonymous_formula": "",
        "crystal_system": crystal_system_from_sg(sg),
        "space_group": summary.get("space_group_calc")
        or first_tag(tags, ["_symmetry_space_group_name_h-m", "_space_group_name_h-m_alt"]),
        "space_group_number": sg,
        "laue_class": laue_from_sg(sg),
        "bravais_lattice": "",
        "a": a,
        "b": b,
        "c": c,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "volume": volume,
        "n_elements": summary.get("n_elements", ""),
        "n_sites": n_sites,
        "volume_per_atom": volume_per_atom,
        "n_base_peaks": "",
        "base_peaks_path": "",
        "labels_path": "",
        "structure_path": "",
        "cod_id": "",
        "cif_path": rel_path(path, root),
        "energy_above_hull": "",
        "original_split": "new_candidate",
        "family": family,
        "mineral_name": mineral,
        "compound_source": compound_source,
        "title": title,
        "risk_flags": ";".join(flags),
        "risk_score": len(flags),
        "source_unknown": not bool(compound_source.strip()),
        "parse_status": parse_status,
        "source_priority": SOURCE_PRIORITY["ICSD"],
        "icsd_id": icsd_id,
        "formula_sum": formula_sum,
    }


def load_icsd_manifest(
    root: Path,
    icsd_dir: Path,
    symprec: float,
    parse_structure: bool,
    max_icsd: int | None,
) -> pd.DataFrame:
    files = sorted(icsd_dir.rglob("*.cif"))
    if max_icsd is not None:
        files = files[:max_icsd]

    rows: list[dict[str, Any]] = []
    for pos, path in enumerate(files, start=1):
        if pos % 500 == 0:
            print(f"Parsed ICSD CIFs: {pos}/{len(files)}", flush=True)
        rows.append(parse_icsd_cif(path, root, symprec, parse_structure))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    duplicated = df["material_id"].duplicated(keep=False)
    if duplicated.any():
        for idx, row in df.loc[duplicated].iterrows():
            digest = hashlib.sha1(str(row.get("cif_path", "")).encode("utf-8")).hexdigest()[:8]
            material_id = f"icsd-{row.get('family', 'unknown')}-{row.get('icsd_id', idx)}-{digest}"
            df.at[idx, "material_id"] = material_id
            df.at[idx, "base_material_id"] = material_id
    return df


def apply_filters(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out["hard_exclude_reason"] = ""
    out["review_reason"] = ""
    out["sample_weight"] = 1.0

    source = out["source"].fillna("").astype(str).str.upper()

    cod_mask = source.eq("COD")
    cod_hard_flags = split_flag_list(args.cod_hard_risk_flags)
    cod_review_flags = split_flag_list(args.cod_review_risk_flags)
    for idx, formula in out.loc[cod_mask, "reduced_formula"].fillna("").astype(str).items():
        if not formula:
            formula = str(out.at[idx, "formula"] or "")
        is_organic, reason = judge_cod_organic(formula, args.cod_c_frac_threshold)
        out.at[idx, "organic_filter_reason"] = reason
        if is_organic:
            out.at[idx, "hard_exclude_reason"] = append_reason(
                out.at[idx, "hard_exclude_reason"],
                f"cod_organic:{reason}",
            )
        cod_flags = {
            flag
            for flag in str(out.at[idx, "cod_risk_flags"] if "cod_risk_flags" in out.columns else "").split(";")
            if flag
        }
        hard_hits = sorted(cod_flags & cod_hard_flags)
        review_hits = sorted(cod_flags & cod_review_flags)
        if hard_hits:
            out.at[idx, "hard_exclude_reason"] = append_reason(
                out.at[idx, "hard_exclude_reason"],
                f"cod_non_ambient:{'+'.join(hard_hits)}",
            )
        elif review_hits:
            out.at[idx, "sample_weight"] = min(float(out.at[idx, "sample_weight"]), args.cod_risk_review_weight)
            out.at[idx, "review_reason"] = append_reason(
                out.at[idx, "review_reason"],
                f"cod_non_ambient_review:{'+'.join(review_hits)}",
            )

    energy = pd.to_numeric(out.get("energy_above_hull", pd.Series(index=out.index)), errors="coerce")
    mp_mask = source.eq("MP")
    mp_gt_exclude = mp_mask & energy.gt(args.mp_exclude_energy_gt)
    out.loc[mp_gt_exclude, "hard_exclude_reason"] = out.loc[mp_gt_exclude, "hard_exclude_reason"].map(
        lambda value: append_reason(value, f"mp_energy_above_hull_gt_{args.mp_exclude_energy_gt:g}")
    )
    mp_missing = mp_mask & energy.isna()
    out.loc[mp_missing, "review_reason"] = out.loc[mp_missing, "review_reason"].map(
        lambda value: append_reason(value, "mp_missing_energy")
    )
    out.loc[mp_missing, "sample_weight"] = out.loc[mp_missing, "sample_weight"].clip(upper=args.mp_missing_weight)

    mp_mid = mp_mask & energy.gt(0.05) & energy.le(0.10)
    out.loc[mp_mid, "sample_weight"] = out.loc[mp_mid, "sample_weight"].clip(upper=args.mp_weight_005_010)
    out.loc[mp_mid, "review_reason"] = out.loc[mp_mid, "review_reason"].map(
        lambda value: append_reason(value, "mp_energy_0.05_0.10")
    )
    mp_high = mp_mask & energy.gt(0.10) & energy.le(args.mp_exclude_energy_gt)
    out.loc[mp_high, "sample_weight"] = out.loc[mp_high, "sample_weight"].clip(upper=args.mp_weight_010_020)
    out.loc[mp_high, "review_reason"] = out.loc[mp_high, "review_reason"].map(
        lambda value: append_reason(value, "mp_energy_0.10_0.20")
    )

    mineral_mask = source.isin(["ICSD", "AMCSD"])
    risk = pd.to_numeric(out.get("risk_score", pd.Series(index=out.index)), errors="coerce").fillna(0)
    risk_exclude = mineral_mask & risk.ge(args.exclude_risk_ge)
    out.loc[risk_exclude, "hard_exclude_reason"] = out.loc[risk_exclude, "hard_exclude_reason"].map(
        lambda value: append_reason(value, f"non_ambient_risk_ge_{args.exclude_risk_ge}")
    )
    risk_review = mineral_mask & risk.gt(0) & ~risk_exclude
    out.loc[risk_review, "sample_weight"] = out.loc[risk_review, "sample_weight"].clip(upper=args.risk_review_weight)
    out.loc[risk_review, "review_reason"] = out.loc[risk_review, "review_reason"].map(
        lambda value: append_reason(value, "non_ambient_risk_review")
    )

    source_unknown = out.get("source_unknown", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    unknown_review = mineral_mask & source_unknown & ~risk_exclude
    out.loc[unknown_review, "sample_weight"] = out.loc[unknown_review, "sample_weight"].clip(
        upper=args.source_unknown_weight
    )
    out.loc[unknown_review, "review_reason"] = out.loc[unknown_review, "review_reason"].map(
        lambda value: append_reason(value, "mineral_source_unknown")
    )

    out["filter_keep"] = out["hard_exclude_reason"].fillna("").astype(str).eq("")
    return out


def make_dedup_key(row: pd.Series, volume_per_atom_bin: float) -> str:
    formula = normalize_formula(row.get("reduced_formula") or row.get("formula"))
    sg = safe_int(row.get("space_group_number"), -1)
    vpa = safe_float(row.get("volume_per_atom"))
    if not formula or sg <= 0 or vpa is None:
        return f"unique:{row.get('material_id', '')}"
    vpa_bin = int(math.floor(vpa / volume_per_atom_bin))
    return f"{formula}|sg={sg}|vpa_bin={vpa_bin}"


def deduplicate_manifest(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    work = df[df["filter_keep"]].copy()
    work["dedup_key"] = work.apply(lambda row: make_dedup_key(row, args.volume_per_atom_bin), axis=1)
    work["dedup_cluster_id"] = ""
    work["dedup_keep"] = True
    work["dedup_exclude_reason"] = ""
    work["matched_keeper_id"] = ""

    duplicate_rows: list[pd.Series] = []
    cluster_counter = 0
    rank = work.copy()
    rank["_rank_source"] = pd.to_numeric(rank["source_priority"], errors="coerce").fillna(99)
    rank["_rank_risk"] = pd.to_numeric(rank.get("risk_score", 0), errors="coerce").fillna(0)
    rank["_rank_source_unknown"] = rank.get("source_unknown", False).fillna(False).astype(bool).astype(int)
    energy = rank.get("energy_above_hull", pd.Series(float("nan"), index=rank.index))
    rank["_rank_energy"] = pd.to_numeric(energy, errors="coerce").fillna(1e9)
    non_mp = ~rank["source"].fillna("").astype(str).str.upper().eq("MP")
    rank.loc[non_mp, "_rank_energy"] = 0.0
    rank["_rank_weight"] = -pd.to_numeric(rank.get("sample_weight", 1.0), errors="coerce").fillna(1.0)
    rank["_rank_input"] = range(len(rank))

    keep_indices: set[int] = set()
    update_values: dict[int, dict[str, Any]] = {}
    for key, group in rank.groupby("dedup_key", sort=False):
        if str(key).startswith("unique:") or len(group) == 1:
            idx = int(group.index[0])
            keep_indices.add(idx)
            update_values[idx] = {
                "dedup_cluster_id": "",
                "dedup_keep": True,
                "dedup_exclude_reason": "",
                "matched_keeper_id": "",
            }
            continue

        cluster_counter += 1
        cluster_id = f"dup-{cluster_counter:07d}"
        ordered = group.sort_values(
            [
                "_rank_source",
                "_rank_risk",
                "_rank_source_unknown",
                "_rank_energy",
                "_rank_weight",
                "_rank_input",
            ],
            ascending=True,
        )
        keeper_idx = int(ordered.index[0])
        keeper_id = str(ordered.iloc[0].get("material_id", ""))
        keep_indices.add(keeper_idx)
        update_values[keeper_idx] = {
            "dedup_cluster_id": cluster_id,
            "dedup_keep": True,
            "dedup_exclude_reason": "",
            "matched_keeper_id": "",
        }
        for idx, row in ordered.iloc[1:].iterrows():
            idx = int(idx)
            duplicate = row.copy()
            duplicate["dedup_cluster_id"] = cluster_id
            duplicate["dedup_keep"] = False
            duplicate["dedup_exclude_reason"] = "duplicate_lower_priority_same_formula_sg_vpa"
            duplicate["matched_keeper_id"] = keeper_id
            duplicate_rows.append(duplicate)
            update_values[idx] = {
                "dedup_cluster_id": cluster_id,
                "dedup_keep": False,
                "dedup_exclude_reason": "duplicate_lower_priority_same_formula_sg_vpa",
                "matched_keeper_id": keeper_id,
            }

    for idx, values in update_values.items():
        for col, value in values.items():
            work.at[idx, col] = value

    keep_df = work.loc[sorted(keep_indices)].copy().reset_index(drop=True)
    dup_df = pd.DataFrame(duplicate_rows).reset_index(drop=True)
    drop_cols = [col for col in keep_df.columns if col.startswith("_rank")]
    if drop_cols:
        keep_df = keep_df.drop(columns=drop_cols)
    if not dup_df.empty:
        dup_df = dup_df.drop(columns=[col for col in dup_df.columns if col.startswith("_rank")], errors="ignore")

    stats = {
        "input_after_filters": int(len(work)),
        "kept": int(len(keep_df)),
        "duplicate_excluded": int(len(dup_df)),
        "duplicate_clusters": int(cluster_counter),
        "kept_by_source": counter_dict(keep_df, "source"),
        "duplicates_by_source": counter_dict(dup_df, "source"),
    }
    return keep_df, dup_df, stats


def keep_without_dedup(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    work = df[df["filter_keep"]].copy()
    work["dedup_key"] = work.apply(lambda row: make_dedup_key(row, args.volume_per_atom_bin), axis=1)
    work["dedup_cluster_id"] = ""
    work["dedup_keep"] = True
    work["dedup_exclude_reason"] = ""
    work["matched_keeper_id"] = ""
    stats = {
        "enabled": False,
        "method": "disabled",
        "input_after_filters": int(len(work)),
        "kept": int(len(work)),
        "duplicate_excluded": 0,
        "duplicate_clusters": 0,
        "kept_by_source": counter_dict(work, "source"),
        "duplicates_by_source": {},
    }
    return work.reset_index(drop=True), pd.DataFrame(columns=work.columns), stats


def counter_dict(df: pd.DataFrame, column: str) -> dict[str, int]:
    if df.empty or column not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[column].fillna("unknown").astype(str).value_counts().to_dict().items()}


def load_rruff(root: Path, path_override: Path | None = None) -> pd.DataFrame:
    path = path_override.resolve() if path_override else root / "RRUFF" / "RRUFF_dataset" / "test_rruff_sg.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df["_name_key"] = df.get("name", pd.Series(dtype=str)).fillna("").astype(str).map(normalize_name)
    df["_reduced_formula"] = df.get("formula", pd.Series(dtype=str)).fillna("").astype(str).map(normalize_rruff_formula)
    df["_sg"] = df.get("space_group_number", pd.Series(dtype=str)).map(lambda value: safe_int(value, -1))
    return df


def values_close(left: float | None, right: float | None, rtol: float) -> bool:
    if left is None or right is None:
        return False
    scale = max(abs(left), abs(right), 1e-12)
    return abs(left - right) / scale <= rtol


def lengths_close(row: pd.Series, candidate: dict[str, Any], rtol: float) -> bool:
    left = [safe_float(row.get(k)) for k in ("a", "b", "c")]
    right = [safe_float(candidate.get(k)) for k in ("a", "b", "c")]
    if any(v is None for v in left + right):
        return False
    direct = all(values_close(left[i], right[i], rtol) for i in range(3))
    ordered = all(values_close(sorted(left)[i], sorted(right)[i], rtol) for i in range(3))
    return direct or ordered


def angles_close(row: pd.Series, candidate: dict[str, Any], atol: float) -> bool:
    left = [safe_float(row.get(k)) for k in ("alpha", "beta", "gamma")]
    right = [safe_float(candidate.get(k)) for k in ("alpha", "beta", "gamma")]
    if any(v is None for v in left + right):
        return False
    direct = all(abs(left[i] - right[i]) <= atol for i in range(3))
    ordered = all(abs(sorted(left)[i] - sorted(right)[i]) <= atol for i in range(3))
    return direct or ordered


def cell_close(row: pd.Series, candidate: dict[str, Any], length_rtol: float, angle_atol: float, volume_rtol: float) -> bool:
    vol_ok = values_close(safe_float(row.get("volume")), safe_float(candidate.get("volume")), volume_rtol)
    return vol_ok and lengths_close(row, candidate, length_rtol) and angles_close(row, candidate, angle_atol)


def name_keys_for_row(row: pd.Series) -> set[str]:
    keys = {
        normalize_name(row.get("family", "")),
        normalize_name(row.get("mineral_name", "")),
    }
    return {key for key in keys if key}


def names_overlap(row_keys: set[str], rruff_key: str) -> bool:
    if not row_keys or not rruff_key:
        return False
    return any(key == rruff_key or key in rruff_key or rruff_key in key for key in row_keys)


def find_rruff_leakage_candidates(keep_df: pd.DataFrame, rruff_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if keep_df.empty or rruff_df.empty:
        return pd.DataFrame()
    by_sg: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in rruff_df.to_dict("records"):
        sg = safe_int(record.get("space_group_number"), -1)
        if sg > 0:
            by_sg[sg].append(record)

    rows: list[dict[str, Any]] = []
    for _, row in keep_df.iterrows():
        sg = safe_int(row.get("space_group_number"), -1)
        if sg <= 0:
            continue
        formula = normalize_formula(row.get("reduced_formula") or row.get("formula"))
        row_names = name_keys_for_row(row)
        for candidate in by_sg.get(sg, []):
            name_hit = names_overlap(row_names, str(candidate.get("_name_key", "")))
            formula_hit = bool(formula and formula == candidate.get("_reduced_formula"))
            if not name_hit and not formula_hit:
                continue
            volume_hit = values_close(
                safe_float(row.get("volume")),
                safe_float(candidate.get("volume")),
                args.rruff_volume_rtol,
            )
            cell_hit = cell_close(
                row,
                candidate,
                args.rruff_length_rtol,
                args.rruff_angle_atol,
                args.rruff_volume_rtol,
            )
            if name_hit and cell_hit:
                reason = "name_sg_cell"
            elif formula_hit and volume_hit:
                reason = "formula_sg_volume"
            elif name_hit:
                reason = "name_sg"
            else:
                continue

            rows.append(
                {
                    "material_id": row.get("material_id", ""),
                    "source": row.get("source", ""),
                    "family": row.get("family", ""),
                    "mineral_name": row.get("mineral_name", ""),
                    "formula": row.get("reduced_formula", ""),
                    "space_group_number": row.get("space_group_number", ""),
                    "volume": row.get("volume", ""),
                    "leakage_reason": reason,
                    "matched_rruff_material_id": candidate.get("material_id", ""),
                    "matched_rruff_id": candidate.get("rruff_id", ""),
                    "matched_rruff_name": candidate.get("name", ""),
                    "matched_rruff_formula": candidate.get("formula", ""),
                    "matched_rruff_volume": candidate.get("volume", ""),
                }
            )
            break
    return pd.DataFrame(rows)


def summarize_outputs(
    manifest: pd.DataFrame,
    keep_df: pd.DataFrame,
    duplicate_df: pd.DataFrame,
    leakage_df: pd.DataFrame,
    dedup_stats: dict[str, Any],
) -> dict[str, Any]:
    excluded = manifest[~manifest["filter_keep"]]
    review = manifest[manifest["review_reason"].fillna("").astype(str).ne("")]
    icsd = manifest[manifest["source"].astype(str).str.upper().eq("ICSD")]
    cod = manifest[manifest["source"].astype(str).str.upper().eq("COD")]
    cod_flag_counts = Counter(
        flag
        for value in cod.get("cod_risk_flags", pd.Series(dtype=str)).fillna("").astype(str)
        for flag in value.split(";")
        if flag
    )
    summary = {
        "raw_rows": int(len(manifest)),
        "raw_by_source": counter_dict(manifest, "source"),
        "filter_excluded_rows": int(len(excluded)),
        "filter_excluded_by_source": counter_dict(excluded, "source"),
        "filter_excluded_by_reason": dict(
            Counter(
                reason
                for value in excluded["hard_exclude_reason"].fillna("").astype(str)
                for reason in value.split(";")
                if reason
            )
        ),
        "review_rows": int(len(review)),
        "review_by_reason": dict(
            Counter(
                reason
                for value in review["review_reason"].fillna("").astype(str)
                for reason in value.split(";")
                if reason
            )
        ),
        "dedup": dedup_stats,
        "final_rows": int(len(keep_df)),
        "final_by_source": counter_dict(keep_df, "source"),
        "duplicates_by_source": counter_dict(duplicate_df, "source"),
        "rruff_leakage_candidates": int(len(leakage_df)),
        "rruff_leakage_by_source": counter_dict(leakage_df, "source"),
        "rruff_leakage_by_reason": counter_dict(leakage_df, "leakage_reason"),
        "icsd": {
            "rows": int(len(icsd)),
            "parse_status": counter_dict(icsd, "parse_status"),
            "risk_score": {
                str(k): int(v)
                for k, v in icsd["risk_score"].fillna(0).astype(int).value_counts().sort_index().to_dict().items()
            },
            "crystal_system": counter_dict(icsd, "crystal_system"),
            "family_top30": {
                str(k): int(v)
                for k, v in icsd["family"].fillna("").astype(str).value_counts().head(30).to_dict().items()
            },
        },
        "cod": {
            "rows": int(len(cod)),
            "metadata_status": counter_dict(cod, "cod_metadata_status"),
            "risk_flag_counts": {str(k): int(v) for k, v in cod_flag_counts.items()},
            "risk_score": {
                str(k): int(v)
                for k, v in pd.to_numeric(
                    cod.get("cod_risk_score", pd.Series(dtype=int)),
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
                .value_counts()
                .sort_index()
                .to_dict()
                .items()
            },
        },
    }
    return summary


def render_report(summary: dict[str, Any], output_dir: Path) -> str:
    lines: list[str] = []
    lines.append("# Dataset Reconstruction Plan\n")
    lines.append("## Outputs")
    for name in [
        "manifest_raw.csv",
        "manifest_filtered.csv",
        "manifest_dedup_keep.csv",
        "manifest_excluded.csv",
        "manifest_dedup_excluded.csv",
        "rruff_leakage_candidates.csv",
        "summary.json",
    ]:
        lines.append(f"- `{(output_dir / name).as_posix()}`")

    lines.append("\n## Source Counts")
    lines.append(f"- raw rows: {summary['raw_rows']}")
    lines.append(f"- raw by source: {summary['raw_by_source']}")
    dedup_enabled = summary.get("dedup", {}).get("enabled", True)
    if dedup_enabled:
        lines.append(f"- final rows after filters + approximate dedup: {summary['final_rows']}")
    else:
        lines.append(f"- final rows after filters, dedup disabled: {summary['final_rows']}")
    lines.append(f"- final by source: {summary['final_by_source']}")

    lines.append("\n## Filters")
    lines.append(f"- excluded rows: {summary['filter_excluded_rows']}")
    lines.append(f"- excluded by source: {summary['filter_excluded_by_source']}")
    lines.append(f"- excluded by reason: {summary['filter_excluded_by_reason']}")
    lines.append(f"- review/downweight rows: {summary['review_rows']}")
    lines.append(f"- review/downweight reasons: {summary['review_by_reason']}")

    lines.append("\n## Dedup")
    dedup = summary["dedup"]
    if dedup.get("enabled", True):
        lines.append(
            f"- method: formula + SG + volume_per_atom bin, keeper priority ICSD > AMCSD > COD > MP"
        )
        lines.append(
            f"- clusters: {dedup['duplicate_clusters']}; duplicate excluded: {dedup['duplicate_excluded']}"
        )
    else:
        lines.append("- method: disabled; all rows surviving filters are kept.")
    lines.append(f"- duplicates by source: {summary['duplicates_by_source']}")

    lines.append("\n## RRUFF Leakage Candidates")
    lines.append(f"- candidates: {summary['rruff_leakage_candidates']}")
    lines.append(f"- by source: {summary['rruff_leakage_by_source']}")
    lines.append(f"- by reason: {summary['rruff_leakage_by_reason']}")

    lines.append("\n## ICSD")
    icsd = summary["icsd"]
    lines.append(f"- rows: {icsd['rows']}")
    lines.append(f"- parse status: {icsd['parse_status']}")
    lines.append(f"- risk score: {icsd['risk_score']}")
    lines.append(f"- crystal system: {icsd['crystal_system']}")
    lines.append(f"- top families: {icsd['family_top30']}")

    lines.append("\n## COD Ambient-Risk Metadata")
    cod = summary["cod"]
    lines.append(f"- rows: {cod['rows']}")
    lines.append(f"- metadata status: {cod['metadata_status']}")
    lines.append(f"- risk score: {cod['risk_score']}")
    lines.append(f"- risk flags: {cod['risk_flag_counts']}")

    lines.append("\n## Next Step")
    lines.append(
        "- Review `manifest_excluded.csv`, `manifest_dedup_excluded.csv`, and "
        "`rruff_leakage_candidates.csv`. Once the policy looks right, the next script can "
        "materialize `dataset` artifacts from `manifest_dedup_keep.csv`."
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    manifest: pd.DataFrame,
    keep_df: pd.DataFrame,
    duplicate_df: pd.DataFrame,
    leakage_df: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "manifest_raw.csv", index=False)
    filtered = manifest[manifest["filter_keep"]].copy()
    excluded = manifest[~manifest["filter_keep"]].copy()
    filtered.to_csv(output_dir / "manifest_filtered.csv", index=False)
    excluded.to_csv(output_dir / "manifest_excluded.csv", index=False)
    keep_df.to_csv(output_dir / "manifest_dedup_keep.csv", index=False)
    duplicate_df.to_csv(output_dir / "manifest_dedup_excluded.csv", index=False)
    leakage_df.to_csv(output_dir / "rruff_leakage_candidates.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "rebuild_plan.md").write_text(
        render_report(summary, output_dir),
        encoding="utf-8",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an auditable manifest for dataset reconstruction.",
    )
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--icsd-dir", type=Path, default=None)
    parser.add_argument(
        "--replace-amcsd-metadata",
        type=Path,
        default=None,
        help=(
            "Optional AMCSD metadata.csv to use instead of AMCSD rows from --dataset-dir. "
            "Useful after rebuilding AMCSD with a different RRUFF dedup policy."
        ),
    )
    parser.add_argument(
        "--replace-amcsd-root",
        type=Path,
        default=None,
        help=(
            "Root directory for relative artifact paths in --replace-amcsd-metadata. "
            "Defaults to the metadata CSV parent directory."
        ),
    )
    parser.add_argument(
        "--cod-raw-root",
        type=Path,
        default=None,
        help="Optional root containing original COD CIF files. Enables COD ambient-risk metadata screening.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-icsd", type=int, default=None)
    parser.add_argument(
        "--skip-icsd",
        action="store_true",
        help="Do not load ICSD CIFs. Useful for building COD+MP+AMCSD-only manifests.",
    )
    parser.add_argument("--skip-icsd-structure-parse", action="store_true")
    parser.add_argument("--symprec", type=float, default=0.1)
    parser.add_argument("--cod-c-frac-threshold", type=float, default=0.3)
    parser.add_argument(
        "--cod-hard-risk-flags",
        type=str,
        default=",".join(DEFAULT_COD_HARD_RISK_FLAGS),
        help="COD ambient-risk flags that trigger hard exclusion.",
    )
    parser.add_argument(
        "--cod-review-risk-flags",
        type=str,
        default=",".join(DEFAULT_COD_REVIEW_RISK_FLAGS),
        help="COD ambient-risk flags that keep the row but downweight/review it.",
    )
    parser.add_argument("--cod-risk-review-weight", type=float, default=0.5)
    parser.add_argument(
        "--cod-read-structure-fallback",
        action="store_true",
        help="Also inspect generated dataset/structures COD CIFs when original COD CIFs are unavailable.",
    )
    parser.add_argument("--cod-low-temperature-lt", type=float, default=250.0)
    parser.add_argument("--cod-high-temperature-gt", type=float, default=400.0)
    parser.add_argument("--mp-exclude-energy-gt", type=float, default=0.20)
    parser.add_argument("--mp-weight-005-010", type=float, default=0.5)
    parser.add_argument("--mp-weight-010-020", type=float, default=0.1)
    parser.add_argument("--mp-missing-weight", type=float, default=0.5)
    parser.add_argument("--exclude-risk-ge", type=int, default=2)
    parser.add_argument("--risk-review-weight", type=float, default=0.3)
    parser.add_argument("--source-unknown-weight", type=float, default=0.7)
    parser.add_argument("--volume-per-atom-bin", type=float, default=0.05)
    parser.add_argument(
        "--skip-dedup",
        action="store_true",
        help="Skip approximate internal/cross-source deduplication after filtering.",
    )
    parser.add_argument("--rruff-length-rtol", type=float, default=0.02)
    parser.add_argument("--rruff-angle-atol", type=float, default=1.0)
    parser.add_argument("--rruff-volume-rtol", type=float, default=0.03)
    parser.add_argument(
        "--rruff-test-csv",
        type=Path,
        default=None,
        help="Optional RRUFF test metadata used only to report leakage candidates; rows are never removed here.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = args.workspace.resolve()
    dataset_dir = (args.dataset_dir or (root / "dataset")).resolve()
    icsd_dir = (args.icsd_dir or (root / "icsd")).resolve()
    if args.cod_raw_root:
        cod_raw_root = args.cod_raw_root.resolve()
    elif (root / "COD" / "cif").exists():
        cod_raw_root = (root / "COD").resolve()
    else:
        cod_raw_root = None
    output_dir = (args.output_dir or (root / "dataset_plan")).resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset directory not found: {dataset_dir}")
    if not args.skip_icsd and not icsd_dir.exists():
        raise FileNotFoundError(f"ICSD directory not found: {icsd_dir}")

    print("Loading current dataset splits...", flush=True)
    current = load_current_dataset(root, dataset_dir)
    if args.replace_amcsd_metadata:
        print("Replacing AMCSD rows from external metadata...", flush=True)
        replacement_amcsd = load_replacement_amcsd(
            root=root,
            metadata_path=args.replace_amcsd_metadata,
            amcsd_root=args.replace_amcsd_root,
        )
        old_count = int(current["source"].astype(str).str.upper().eq("AMCSD").sum())
        current = current[~current["source"].astype(str).str.upper().eq("AMCSD")].copy()
        current = pd.concat([current, replacement_amcsd], ignore_index=True, sort=False)
        print(
            f"Replaced AMCSD rows: removed {old_count}, added {len(replacement_amcsd)}.",
            flush=True,
        )
    current = enrich_current_amcsd(root, current)
    current = enrich_current_cod(
        root=root,
        df=current,
        cod_raw_root=cod_raw_root,
        read_structure_fallback=args.cod_read_structure_fallback,
        low_temperature_lt=args.cod_low_temperature_lt,
        high_temperature_gt=args.cod_high_temperature_gt,
    )

    if args.skip_icsd:
        print("Skipping ICSD CIF manifest.", flush=True)
        icsd = pd.DataFrame()
    else:
        print("Loading ICSD CIF manifest...", flush=True)
        icsd = load_icsd_manifest(
            root=root,
            icsd_dir=icsd_dir,
            symprec=args.symprec,
            parse_structure=not args.skip_icsd_structure_parse,
            max_icsd=args.max_icsd,
        )

    print("Applying filters and sample weights...", flush=True)
    manifest = pd.concat([current, icsd], ignore_index=True, sort=False)
    manifest["source"] = manifest["source"].fillna("").astype(str).str.upper()
    manifest["source_priority"] = manifest["source"].map(SOURCE_PRIORITY).fillna(99).astype(int)
    manifest = apply_filters(manifest, args)

    if args.skip_dedup:
        print("Skipping approximate source-priority dedup.", flush=True)
        keep_df, duplicate_df, dedup_stats = keep_without_dedup(manifest, args)
    else:
        print("Running approximate source-priority dedup...", flush=True)
        keep_df, duplicate_df, dedup_stats = deduplicate_manifest(manifest, args)

    print("Finding RRUFF leakage candidates...", flush=True)
    rruff = load_rruff(root, args.rruff_test_csv)
    leakage_df = find_rruff_leakage_candidates(keep_df, rruff, args)

    summary = summarize_outputs(manifest, keep_df, duplicate_df, leakage_df, dedup_stats)
    write_outputs(output_dir, manifest, keep_df, duplicate_df, leakage_df, summary)

    print(render_report(summary, output_dir), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
