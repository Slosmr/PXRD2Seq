"""Build a strict RRUFF classification cohort.

DIF records must provide valid lattice, space-group, and wavelength metadata.
Paired XY and DIF records must agree on crystal system and cell parameters.
Experimental angles are converted by Bragg's law to a 20 keV equivalent;
patterns must cover 5-20 degrees on that grid. Signals are interpolated to
8,500 points and normalized to [0, 100]. This grid differs from the base
RRUFF grid. The integrated pipeline intersects candidates with the base set.
Regression-proxy construction is implemented in prepare_regression.py."""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from prepare_base import (
    build_report,
    check_symmetry_consistency,
    check_xy_dif_match,
    detect_peaks_from_signal,
    expand_peak_matrix_with_observed_mask,
    is_calculated_profile,
)
from rruff_parser import (
    get_bravais,
    get_laue_from_sg_number,
    is_cu_ka,
    niggli_reduce_cell,
    parse_dif_file,
    parse_xy_file,
    sg_symbol_to_info,
)


LOGGER = logging.getLogger("rruff_classification")
warnings.filterwarnings("ignore", message="Full symbol not available.*")

HC_KEV_ANGSTROM = 12.398419843320026
TARGET_ENERGY_KEV = 20.0
TARGET_WAVELENGTH_ANG = HC_KEV_ANGSTROM / TARGET_ENERGY_KEV

CONFIG = {
    "theta_range": (5.0, 20.0),
    "signal_length": 8500,
    "max_peaks": 20,
    "peak_dim": 14,
    "wavelength_ang": TARGET_WAVELENGTH_ANG,
    "peak_prominence_pct": 3.0,
    "peak_min_distance_deg": 0.09,
}

CS_ALIAS = {
    "rhombohedral": "trigonal",
}


def normalize_crystal_system(value: str | None) -> str:
    value = (value or "").strip().lower()
    return CS_ALIAS.get(value, value)


def crystal_systems_compatible(xy_crystal_system: str, dif_crystal_system: str) -> bool:
    if xy_crystal_system == dif_crystal_system:
        return True
    # RRUFF often labels trigonal structures in the hexagonal cell setting.
    return {xy_crystal_system, dif_crystal_system} == {"hexagonal", "trigonal"}


def dif_cell_to_dict(dif_cell: tuple[float, float, float, float, float, float],
                     crystal_system: str) -> dict:
    a, b, c, alpha, beta, gamma = dif_cell
    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "volume": float(lattice_volume(dif_cell)),
        "crystal_system": normalize_crystal_system(crystal_system),
    }


def lattice_volume(cell: tuple[float, float, float, float, float, float]) -> float:
    a, b, c, alpha, beta, gamma = cell
    ca = np.cos(np.radians(alpha))
    cb = np.cos(np.radians(beta))
    cg = np.cos(np.radians(gamma))
    factor = 1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg
    if factor <= 0:
        return float("nan")
    return float(a * b * c * np.sqrt(factor))


def validate_pymatgen_lattice(cell: tuple[float, float, float, float, float, float]) -> bool:
    try:
        from pymatgen.core import Lattice
    except ImportError as exc:
        raise RuntimeError("pymatgen is required for RRUFF classification curation") from exc

    try:
        Lattice.from_parameters(
            a=cell[0],
            b=cell[1],
            c=cell[2],
            alpha=cell[3],
            beta=cell[4],
            gamma=cell[5],
        )
        return True
    except Exception:
        return False


@lru_cache(maxsize=None)
def cached_sg_info(space_group: str) -> tuple[int | None, str | None]:
    return sg_symbol_to_info(space_group)


def bragg_convert_two_theta(two_theta: np.ndarray,
                            source_wavelength: float,
                            target_wavelength: float = TARGET_WAVELENGTH_ANG) -> np.ndarray | None:
    if source_wavelength <= 0 or target_wavelength <= 0:
        return None
    source_theta = np.radians(two_theta / 2.0)
    sin_target_theta = (target_wavelength / source_wavelength) * np.sin(source_theta)
    if np.any(sin_target_theta < -1.0) or np.any(sin_target_theta > 1.0):
        return None
    return np.degrees(2.0 * np.arcsin(sin_target_theta))


def resample_converted_pattern(two_theta_20kev: np.ndarray,
                               intensity: np.ndarray,
                               target_range: tuple[float, float],
                               target_length: int) -> np.ndarray:
    order = np.argsort(two_theta_20kev)
    x = two_theta_20kev[order]
    y = intensity[order]

    unique_x, unique_idx = np.unique(x, return_index=True)
    unique_y = y[unique_idx]

    x_target = np.linspace(target_range[0], target_range[1], target_length)
    y_target = np.interp(x_target, unique_x, unique_y).astype(np.float32)

    y_target = y_target - float(np.nanmin(y_target))
    max_val = float(np.nanmax(y_target))
    if max_val > 0:
        y_target = y_target / max_val * 100.0
    return y_target.astype(np.float32)


def converted_range_fully_covers(two_theta_20kev: np.ndarray,
                                 target_range: tuple[float, float]) -> bool:
    lo, hi = target_range
    return float(np.nanmin(two_theta_20kev)) <= lo and float(np.nanmax(two_theta_20kev)) >= hi


def index_dif_files(dif_dir: Path, require_cu_ka: bool = False) -> tuple[dict, dict, Counter]:
    dif_files = sorted(
        set(dif_dir.rglob("*.txt"))
        | set(dif_dir.rglob("*.dif"))
        | set(dif_dir.rglob("*.rtf"))
    )
    by_full = {}
    by_base = defaultdict(list)
    stats = Counter(total_dif_files=len(dif_files))

    for path in tqdm(dif_files, desc="Index DIF"):
        dif = parse_dif_file(path)
        if dif is None or dif.get("rruff_id_base") is None:
            stats["dif_parse_failed"] += 1
            continue

        stats["dif_parsed"] += 1
        if dif.get("wavelength") is None:
            stats["dif_missing_wavelength"] += 1
            continue
        if require_cu_ka and not is_cu_ka(dif["wavelength"]):
            stats["dif_non_cu_ka"] += 1
            continue

        sg_number, sg_crystal_system = cached_sg_info(dif["space_group"])
        if not sg_number or not sg_crystal_system:
            stats["dif_sg_unresolved"] += 1
            continue

        if not validate_pymatgen_lattice(dif["cell"]):
            stats["dif_lattice_unparseable"] += 1
            continue

        volume = lattice_volume(dif["cell"])
        if not np.isfinite(volume) or volume >= 100000.0:
            stats["dif_bad_volume"] += 1
            continue

        dif_cell = dif_cell_to_dict(dif["cell"], sg_crystal_system)
        if not check_symmetry_consistency(dif_cell):
            stats["dif_symmetry_inconsistent"] += 1
            continue

        record = {
            **dif,
            "path": str(path),
            "space_group_number": int(sg_number),
            "crystal_system": normalize_crystal_system(sg_crystal_system),
            "cell_dict": dif_cell,
            "volume": volume,
        }
        by_base[dif["rruff_id_base"]].append(record)
        if dif["rruff_id_full"] and dif["rruff_id_full"] not in by_full:
            by_full[dif["rruff_id_full"]] = record
        stats["dif_accepted"] += 1

    for records in by_base.values():
        records.sort(key=lambda item: (item.get("rruff_id_full") or "", item["path"]))

    stats["dif_unique_full"] = len(by_full)
    stats["dif_unique_base"] = len(by_base)
    return by_full, dict(by_base), stats


def candidate_difs(xy: dict, by_full: dict, by_base: dict) -> list[dict]:
    full = xy.get("rruff_id_full")
    base = xy.get("rruff_id_base")
    if full in by_full:
        return [by_full[full]]
    return list(by_base.get(base, []))


def select_matching_dif(xy: dict,
                        candidates: list[dict],
                        length_rtol: float,
                        angle_atol: float) -> tuple[dict | None, str]:
    xy_cs = normalize_crystal_system(xy["cell"].get("crystal_system"))
    for dif in candidates:
        if not crystal_systems_compatible(xy_cs, dif["crystal_system"]):
            continue
        if not check_xy_dif_match(
            xy["cell"],
            dif["cell"],
            length_rtol=length_rtol,
            angle_atol=angle_atol,
        ):
            continue
        return dif, ""
    if not candidates:
        return None, "xy_no_matching_dif"
    if all(not crystal_systems_compatible(xy_cs, dif["crystal_system"]) for dif in candidates):
        return None, "xy_dif_crystal_system_mismatch"
    return None, "xy_dif_lattice_mismatch"


def build_rruff_classification_candidates(
    xy_dir: Path,
    dif_dir: Path,
    output_dir: Path,
    csv_name: str = "_rruff_classification_candidates.csv",
    length_rtol: float = 0.001,
    angle_atol: float = 0.5,
    require_cu_ka: bool = False,
    signal_length: int = 8500,
    dry_run: bool = False,
) -> tuple[pd.DataFrame, dict]:
    cfg = {**CONFIG, "signal_length": int(signal_length)}
    output_dir = Path(output_dir)
    if not dry_run:
        (output_dir / "signals").mkdir(parents=True, exist_ok=True)
        (output_dir / "peaks").mkdir(parents=True, exist_ok=True)
        (output_dir / "reports").mkdir(parents=True, exist_ok=True)

    by_full, by_base, dif_stats = index_dif_files(dif_dir, require_cu_ka=require_cu_ka)
    xy_files = sorted(Path(xy_dir).rglob("*.txt"))

    stats = Counter(dif_stats)
    stats["total_xy_files"] = len(xy_files)
    rows = []
    seen_material_ids = set()
    x_grid = np.linspace(*cfg["theta_range"], cfg["signal_length"])

    for path in tqdm(xy_files, desc="Build RRUFF classification candidates"):
        xy = parse_xy_file(path)
        if xy is None:
            stats["xy_parse_failed"] += 1
            continue
        stats["xy_parsed"] += 1

        if xy.get("cell") is None:
            stats["xy_missing_cell"] += 1
            continue

        candidates = candidate_difs(xy, by_full, by_base)
        dif, reason = select_matching_dif(
            xy,
            candidates,
            length_rtol=length_rtol,
            angle_atol=angle_atol,
        )
        if dif is None:
            stats[reason] += 1
            continue
        stats["xy_dif_matched"] += 1

        converted_two_theta = bragg_convert_two_theta(
            xy["two_theta"],
            source_wavelength=float(dif["wavelength"]),
        )
        if converted_two_theta is None:
            stats["xy_bragg_conversion_failed"] += 1
            continue
        if not converted_range_fully_covers(converted_two_theta, cfg["theta_range"]):
            stats["xy_converted_range_incomplete"] += 1
            continue

        material_id = f"RRUFF_{xy['rruff_id_full']}"
        if material_id in seen_material_ids:
            stats["xy_duplicate_material_id"] += 1
            continue
        seen_material_ids.add(material_id)

        y_resampled = resample_converted_pattern(
            converted_two_theta,
            xy["intensity"],
            target_range=cfg["theta_range"],
            target_length=cfg["signal_length"],
        )
        peak_matrix = detect_peaks_from_signal(y_resampled, x_grid, cfg)
        peak_matrix = expand_peak_matrix_with_observed_mask(
            peak_matrix,
            peak_dim=cfg["peak_dim"],
        )
        n_valid_peaks = int((peak_matrix[:, 7] == 1.0).sum())

        dif_cell_niggli = niggli_reduce_cell(dif["cell_dict"])
        laue_class = get_laue_from_sg_number(dif["space_group_number"])
        bravais = get_bravais(dif["crystal_system"], dif["space_group"])

        signal_rel = f"signals/{material_id}.npy"
        peak_rel = f"peaks/{material_id}.npy"
        report_rel = f"reports/{material_id}.json"

        if not dry_run:
            np.save(output_dir / signal_rel, y_resampled.astype(np.float32))
            np.save(output_dir / peak_rel, peak_matrix.astype(np.float32))
            report = build_report(
                name=material_id,
                formula=xy["formula"] or xy["name"],
                cell=dif_cell_niggli,
                sg_symbol=dif["space_group"],
                sg_number=dif["space_group_number"],
                laue_class=laue_class,
                bravais=bravais,
                peak_matrix=peak_matrix,
            )
            report.update(
                {
                    "source_wavelength_ang": float(dif["wavelength"]),
                    "target_energy_kev": TARGET_ENERGY_KEV,
                    "target_wavelength_ang": TARGET_WAVELENGTH_ANG,
                    "signal_length": cfg["signal_length"],
                    "source_xy_path": str(path),
                    "source_dif_path": dif["path"],
                }
            )
            with open(output_dir / report_rel, "w", encoding="utf-8") as fh:
                json.dump(report, fh, ensure_ascii=False, indent=2)

        rows.append(
            {
                "material_id": material_id,
                "base_material_id": material_id,
                "rruff_id": xy["rruff_id_full"],
                "rruff_base_id": xy["rruff_id_base"],
                "name": xy["name"],
                "formula": xy["formula"],
                "is_calculated": is_calculated_profile(xy),
                "xy_crystal_system": normalize_crystal_system(xy["cell"].get("crystal_system")),
                "crystal_system": dif["crystal_system"],
                "space_group": dif["space_group"],
                "space_group_number": dif["space_group_number"],
                "laue_class": laue_class,
                "bravais_lattice": bravais,
                "a": dif_cell_niggli["a"],
                "b": dif_cell_niggli["b"],
                "c": dif_cell_niggli["c"],
                "alpha": dif_cell_niggli["alpha"],
                "beta": dif_cell_niggli["beta"],
                "gamma": dif_cell_niggli["gamma"],
                "volume": dif_cell_niggli["volume"],
                "source_a": dif["cell_dict"]["a"],
                "source_b": dif["cell_dict"]["b"],
                "source_c": dif["cell_dict"]["c"],
                "source_alpha": dif["cell_dict"]["alpha"],
                "source_beta": dif["cell_dict"]["beta"],
                "source_gamma": dif["cell_dict"]["gamma"],
                "source_volume": dif["volume"],
                "source_wavelength_ang": float(dif["wavelength"]),
                "target_energy_kev": TARGET_ENERGY_KEV,
                "target_wavelength_ang": TARGET_WAVELENGTH_ANG,
                "target_two_theta_min": cfg["theta_range"][0],
                "target_two_theta_max": cfg["theta_range"][1],
                "signal_length": cfg["signal_length"],
                "converted_two_theta_min": float(np.nanmin(converted_two_theta)),
                "converted_two_theta_max": float(np.nanmax(converted_two_theta)),
                "n_valid_peaks": n_valid_peaks,
                "signal_path": signal_rel,
                "peak_path": peak_rel,
                "report_path": report_rel,
                "source_xy_path": str(path),
                "source_dif_path": dif["path"],
            }
        )
        stats["success"] += 1

    df = pd.DataFrame(rows)
    stats["calculated_profiles_kept"] = int(df["is_calculated"].sum()) if not df.empty else 0
    stats["signal_length"] = int(cfg["signal_length"])

    if not dry_run:
        df.to_csv(output_dir / csv_name, index=False)
        stats_path = output_dir / "classification_candidates_stats.json"
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(dict(stats), fh, ensure_ascii=False, indent=2)

    return df, dict(stats)


def write_local_intersection(
    candidate_df: pd.DataFrame,
    local_csv: Path,
    output_dir: Path,
    csv_name: str = "rruff_classification.csv",
) -> tuple[pd.DataFrame | None, dict | None]:
    """Intersect classification candidates with the labeled RRUFF base cohort."""
    if not local_csv.is_file():
        LOGGER.warning("Local CSV not found, skip intersection: %s", local_csv)
        return None, None

    local_df = pd.read_csv(local_csv)
    local_ids = set(local_df["rruff_id"].dropna().astype(str))
    candidate_ids = set(candidate_df["rruff_id"].dropna().astype(str))
    combined = candidate_df[
        candidate_df["rruff_id"].astype(str).isin(local_ids)
    ].copy()

    combined_path = output_dir / csv_name
    combined.to_csv(combined_path, index=False)

    stats = {
        "local_csv": str(local_csv),
        "local_rows": int(len(local_df)),
        "candidate_rows": int(len(candidate_df)),
        "combined_rows": int(len(combined)),
        "base_only_rows": int(len(local_ids - candidate_ids)),
        "candidate_only_rows": int(len(candidate_ids - local_ids)),
        "calculated_profiles": int(combined["is_calculated"].sum()) if not combined.empty else 0,
        "signal_length": int(combined["signal_length"].iloc[0]) if not combined.empty else None,
        "crystal_system_distribution": {
            k: int(v) for k, v in combined["crystal_system"].value_counts().items()
        } if not combined.empty else {},
    }
    with open(output_dir / "rruff_classification_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    LOGGER.info("RRUFF base/candidate intersection rows: %d", len(combined))
    LOGGER.info("Intersection CSV: %s", combined_path)
    return combined, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xy_dir", default="XY_Processed")
    parser.add_argument("--dif_dir", default="DIF")
    parser.add_argument("--output_dir", default="datasets/classification")
    parser.add_argument("--csv_name", default="_rruff_classification_candidates.csv")
    parser.add_argument("--local_csv", default="datasets/base/rruff_labeled.csv",
                        help="Existing local strict RRUFF CSV used to write the 730-row intersection.")
    parser.add_argument("--combined_csv_name", default="rruff_classification.csv")
    parser.add_argument("--length_rtol", type=float, default=0.001)
    parser.add_argument("--angle_atol", type=float, default=0.5)
    parser.add_argument("--require_cu_ka", action="store_true")
    parser.add_argument("--signal_length", type=int, default=8500)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df, stats = build_rruff_classification_candidates(
        xy_dir=Path(args.xy_dir),
        dif_dir=Path(args.dif_dir),
        output_dir=Path(args.output_dir),
        csv_name=args.csv_name,
        length_rtol=args.length_rtol,
        angle_atol=args.angle_atol,
        require_cu_ka=args.require_cu_ka,
        signal_length=args.signal_length,
        dry_run=args.dry_run,
    )

    LOGGER.info("RRUFF classification rows: %d", len(df))
    LOGGER.info("Stats:\n%s", json.dumps(stats, ensure_ascii=False, indent=2))
    if not df.empty:
        LOGGER.info("Crystal-system distribution:\n%s", df["crystal_system"].value_counts().to_string())
        LOGGER.info("Calculated profiles kept: %d", int(df["is_calculated"].sum()))

    if not args.dry_run:
        write_local_intersection(
            candidate_df=df,
            local_csv=Path(args.local_csv),
            output_dir=Path(args.output_dir),
            csv_name=args.combined_csv_name,
        )


if __name__ == "__main__":
    main()
