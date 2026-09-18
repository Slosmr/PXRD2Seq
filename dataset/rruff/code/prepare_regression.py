"""
Build a pymatgen-validated regression proxy subset from the strict RRUFF set.

The proxy validates reconstructed reference cells using DIF metadata and
simulated diffraction peak positions:

1. Start from the 730-row strict classification set.
2. Require a parseable RRUFF DIF atom table.
3. Build a pymatgen Structure from the DIF cell, space group, and atom sites.
4. Require Niggli reduction to succeed.
5. Simulate PXRD peak positions with pymatgen and require the strongest DIF
   peaks to be position-consistent with simulated peaks.

The output is not a GSAS-II-validated subset, so the CSV records the validation
method and thresholds explicitly.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import warnings
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


LOGGER = logging.getLogger("rruff_regression_proxy")

NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
ATOM_HEADER_RE = re.compile(
    r"^\s*ATOM\s+X\s+Y\s+Z\s+OCCUPANCY\s+ISO\(B\)", re.IGNORECASE
)
ATOM_ROW_RE = re.compile(
    rf"^\s*(\S+)\s+({NUMBER_RE})\s+({NUMBER_RE})\s+({NUMBER_RE})\s+"
    rf"({NUMBER_RE})\s+({NUMBER_RE})\s*$"
)
PEAK_HEADER_RE = re.compile(r"2-THETA\s+INTENSITY\s+D-SPACING", re.IGNORECASE)
PEAK_ROW_RE = re.compile(
    rf"^\s*({NUMBER_RE})\s+({NUMBER_RE})\s+({NUMBER_RE})\s+"
    r"([-+]?\d+)\s+([-+]?\d+)\s+([-+]?\d+)"
)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


@lru_cache(maxsize=None)
def is_valid_element(symbol: str) -> bool:
    from pymatgen.core.periodic_table import Element

    return Element.is_valid_symbol(symbol)


def element_from_label(label: str) -> str | None:
    """Map RRUFF atom labels such as Oa/OH/Wa to chemical elements."""
    letters = re.sub(r"[^A-Za-z]", "", label)
    lowered = letters.lower()
    if lowered in {"wa", "wat", "water", "ow", "oh"}:
        return "O"
    if lowered == "d":
        return "H"

    for n_chars in (2, 1):
        candidate = letters[:n_chars].capitalize()
        if candidate and is_valid_element(candidate):
            return candidate
    return None


def parse_atom_table(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    atoms: list[dict[str, Any]] = []
    errors: list[str] = []
    in_table = False

    for line in read_text(path).splitlines():
        if not in_table:
            if ATOM_HEADER_RE.search(line):
                in_table = True
            continue

        if not line.strip():
            if atoms:
                break
            continue
        upper = line.upper()
        if "X-RAY WAVELENGTH" in upper or "2-THETA" in upper:
            break

        match = ATOM_ROW_RE.match(line)
        if not match:
            errors.append(f"malformed atom row: {line.strip()}")
            continue

        label, x, y, z, occupancy, b_iso = match.groups()
        element = element_from_label(label)
        if element is None:
            errors.append(f"unresolved atom label: {label}")
            continue

        atoms.append(
            {
                "label": label,
                "element": element,
                "x": float(x) % 1.0,
                "y": float(y) % 1.0,
                "z": float(z) % 1.0,
                "occupancy": float(occupancy),
                "b_iso": float(b_iso),
            }
        )

    return atoms, errors


def parse_dif_peaks(path: Path) -> list[dict[str, Any]]:
    peaks: list[dict[str, Any]] = []
    in_table = False

    for line in read_text(path).splitlines():
        if not in_table:
            if PEAK_HEADER_RE.search(line):
                in_table = True
            continue

        match = PEAK_ROW_RE.match(line)
        if match:
            two_theta, intensity, d_spacing, h, k, l = match.groups()
            peaks.append(
                {
                    "two_theta": float(two_theta),
                    "intensity": float(intensity),
                    "d": float(d_spacing),
                    "h": int(h),
                    "k": int(k),
                    "l": int(l),
                }
            )
        elif line.strip().startswith("="):
            break

    return peaks


def merge_close_peaks(
    peaks: list[dict[str, Any]],
    two_theta_atol: float,
) -> list[dict[str, Any]]:
    """Merge nearly coincident DIF hkl rows into one observed peak."""
    if not peaks:
        return []

    sorted_peaks = sorted(peaks, key=lambda item: item["two_theta"])
    clusters: list[list[dict[str, Any]]] = []
    current = [sorted_peaks[0]]
    for peak in sorted_peaks[1:]:
        if abs(peak["two_theta"] - current[-1]["two_theta"]) <= two_theta_atol:
            current.append(peak)
        else:
            clusters.append(current)
            current = [peak]
    clusters.append(current)

    merged = []
    for cluster in clusters:
        weights = np.array(
            [max(0.0, float(peak["intensity"])) for peak in cluster],
            dtype=float,
        )
        if float(weights.sum()) <= 0.0:
            weights = np.ones(len(cluster), dtype=float)
        merged.append(
            {
                "two_theta": float(np.average([p["two_theta"] for p in cluster], weights=weights)),
                "d": float(np.average([p["d"] for p in cluster], weights=weights)),
                "intensity": float(sum(p["intensity"] for p in cluster)),
                "merged_hkl_count": int(len(cluster)),
            }
        )
    return merged


def group_disordered_sites(
    atoms: list[dict[str, Any]],
    coord_decimals: int,
    occupancy_sum_tolerance: float,
) -> tuple[list[Any], list[list[float]], dict[str, Any]]:
    groups: dict[tuple[float, float, float], dict[str, float]] = {}
    order: list[tuple[float, float, float]] = []

    for atom in atoms:
        key = (
            round(float(atom["x"]), coord_decimals),
            round(float(atom["y"]), coord_decimals),
            round(float(atom["z"]), coord_decimals),
        )
        if key not in groups:
            groups[key] = defaultdict(float)
            order.append(key)
        groups[key][atom["element"]] += float(atom["occupancy"])

    species: list[Any] = []
    coords: list[list[float]] = []
    max_occupancy_sum = 0.0
    normalized_sites = 0

    for key in order:
        site_species = {
            element: occupancy
            for element, occupancy in dict(groups[key]).items()
            if occupancy > 1e-6
        }
        occupancy_sum = float(sum(site_species.values()))
        max_occupancy_sum = max(max_occupancy_sum, occupancy_sum)

        if occupancy_sum > 1.0 and occupancy_sum <= 1.0 + occupancy_sum_tolerance:
            site_species = {
                element: occupancy / occupancy_sum
                for element, occupancy in site_species.items()
            }
            occupancy_sum = 1.0
            normalized_sites += 1

        if occupancy_sum > 1.0 + occupancy_sum_tolerance:
            raise ValueError(f"site occupancy sum {occupancy_sum:.4f} exceeds tolerance")

        if len(site_species) == 1 and abs(occupancy_sum - 1.0) <= 1e-6:
            species.append(next(iter(site_species)))
        else:
            species.append(site_species)
        coords.append(list(key))

    return species, coords, {
        "site_count": len(order),
        "max_occupancy_sum": max_occupancy_sum,
        "normalized_occupancy_sites": normalized_sites,
    }


def symmetrized_lattice(row: pd.Series):
    """Make tiny metric corrections so pymatgen accepts the stated symmetry."""
    from pymatgen.core import Lattice

    a = float(row["source_a"])
    b = float(row["source_b"])
    c = float(row["source_c"])
    alpha = float(row["source_alpha"])
    beta = float(row["source_beta"])
    gamma = float(row["source_gamma"])
    crystal_system = str(row["crystal_system"]).lower()

    original = np.array([a, b, c, alpha, beta, gamma], dtype=float)

    if crystal_system == "cubic":
        mean_length = (a + b + c) / 3.0
        a = b = c = mean_length
        alpha = beta = gamma = 90.0
    elif crystal_system == "tetragonal":
        mean_ab = (a + b) / 2.0
        a = b = mean_ab
        alpha = beta = gamma = 90.0
    elif crystal_system in {"hexagonal", "trigonal"}:
        mean_ab = (a + b) / 2.0
        a = b = mean_ab
        alpha = beta = 90.0
        gamma = 120.0
    elif crystal_system == "orthorhombic":
        alpha = beta = gamma = 90.0
    elif crystal_system == "monoclinic":
        alpha = 90.0
        gamma = 90.0

    adjusted = np.array([a, b, c, alpha, beta, gamma], dtype=float)
    return Lattice.from_parameters(a, b, c, alpha, beta, gamma), float(np.max(np.abs(adjusted - original)))


def build_structure(row: pd.Series, species: list[Any], coords: list[list[float]]):
    from pymatgen.core import Structure

    lattice, max_adjustment = symmetrized_lattice(row)
    structure = Structure.from_spacegroup(
        int(row["space_group_number"]),
        lattice,
        species,
        coords,
        coords_are_cartesian=False,
        tol=1e-3,
    )
    return structure, max_adjustment


def lattice_to_dict(lattice) -> dict[str, float]:
    return {
        "a": float(lattice.a),
        "b": float(lattice.b),
        "c": float(lattice.c),
        "alpha": float(lattice.alpha),
        "beta": float(lattice.beta),
        "gamma": float(lattice.gamma),
        "volume": float(lattice.volume),
    }


def validate_xrd_peaks(
    row: pd.Series,
    structure,
    dif_peaks: list[dict[str, Any]],
    merged_peaks: list[dict[str, Any]],
    top_n: int,
    min_matches: int,
    d_rtol: float,
    two_theta_atol: float,
) -> tuple[bool, dict[str, Any]]:
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    top_peaks = sorted(merged_peaks, key=lambda item: item["intensity"], reverse=True)[:top_n]
    two_theta_min = min(peak["two_theta"] for peak in dif_peaks) - 0.5
    two_theta_max = max(peak["two_theta"] for peak in dif_peaks) + 0.5

    calculator = XRDCalculator(wavelength=float(row["source_wavelength_ang"]), symprec=0)
    pattern = calculator.get_pattern(
        structure,
        two_theta_range=(two_theta_min, two_theta_max),
    )
    sim_two_theta = np.array(pattern.x, dtype=float)
    sim_d = np.array(pattern.d_hkls, dtype=float)

    matches: list[dict[str, Any]] = []
    for exp_peak in top_peaks:
        d_rel_errors = np.abs(sim_d - exp_peak["d"]) / exp_peak["d"]
        two_theta_errors = np.abs(sim_two_theta - exp_peak["two_theta"])
        matched = (d_rel_errors <= d_rtol) | (two_theta_errors <= two_theta_atol)
        if not bool(matched.any()):
            continue

        score = np.where(
            matched,
            np.minimum(d_rel_errors / d_rtol, two_theta_errors / two_theta_atol),
            np.inf,
        )
        sim_index = int(score.argmin())
        matches.append(
            {
                "exp_two_theta": float(exp_peak["two_theta"]),
                "exp_d": float(exp_peak["d"]),
                "sim_two_theta": float(sim_two_theta[sim_index]),
                "sim_d": float(sim_d[sim_index]),
                "d_relative_error": float(d_rel_errors[sim_index]),
                "two_theta_error": float(two_theta_errors[sim_index]),
            }
        )

    d_errors = [match["d_relative_error"] for match in matches]
    two_theta_errors = [match["two_theta_error"] for match in matches]
    result = {
        "dif_peak_count": int(len(dif_peaks)),
        "merged_dif_peak_count": int(len(merged_peaks)),
        "sim_peak_count": int(len(sim_two_theta)),
        "xrd_top_n": int(top_n),
        "xrd_min_matches": int(min_matches),
        "xrd_matched_top_n": int(len(matches)),
        "xrd_match_fraction": float(len(matches) / top_n),
        "xrd_mean_d_relative_error": float(np.mean(d_errors)) if d_errors else np.nan,
        "xrd_max_d_relative_error": float(np.max(d_errors)) if d_errors else np.nan,
        "xrd_mean_two_theta_error": float(np.mean(two_theta_errors)) if two_theta_errors else np.nan,
        "xrd_max_two_theta_error": float(np.max(two_theta_errors)) if two_theta_errors else np.nan,
    }
    return len(matches) >= min_matches, result


def validate_row(
    row: pd.Series,
    base_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    dif_path = base_dir / str(row["source_dif_path"])
    if not dif_path.is_file():
        return None, {"rruff_id": row["rruff_id"], "reason": "source_dif_missing"}

    atoms, atom_errors = parse_atom_table(dif_path)
    if not atoms:
        return None, {"rruff_id": row["rruff_id"], "reason": "no_atom_table"}
    if atom_errors:
        return None, {
            "rruff_id": row["rruff_id"],
            "reason": "malformed_atom_table",
            "detail": "; ".join(atom_errors[:3]),
        }

    try:
        species, coords, site_info = group_disordered_sites(
            atoms,
            coord_decimals=args.coord_decimals,
            occupancy_sum_tolerance=args.occupancy_sum_tolerance,
        )
    except Exception as exc:
        return None, {
            "rruff_id": row["rruff_id"],
            "reason": "bad_occupancy",
            "detail": str(exc),
        }

    try:
        structure, lattice_adjustment = build_structure(row, species, coords)
    except Exception as exc:
        return None, {
            "rruff_id": row["rruff_id"],
            "reason": "structure_build_failed",
            "detail": f"{type(exc).__name__}: {exc}",
        }

    try:
        reduced_lattice = structure.lattice.get_niggli_reduced_lattice()
    except Exception as exc:
        return None, {
            "rruff_id": row["rruff_id"],
            "reason": "niggli_reduction_failed",
            "detail": f"{type(exc).__name__}: {exc}",
        }

    dif_peaks = parse_dif_peaks(dif_path)
    merged_peaks = merge_close_peaks(dif_peaks, args.merge_two_theta_atol)
    if len(merged_peaks) < args.top_n:
        return None, {
            "rruff_id": row["rruff_id"],
            "reason": "not_enough_dif_peaks",
            "dif_peak_count": len(dif_peaks),
            "merged_dif_peak_count": len(merged_peaks),
        }

    try:
        xrd_passed, xrd_info = validate_xrd_peaks(
            row=row,
            structure=structure,
            dif_peaks=dif_peaks,
            merged_peaks=merged_peaks,
            top_n=args.top_n,
            min_matches=args.min_matches,
            d_rtol=args.d_rtol,
            two_theta_atol=args.two_theta_atol,
        )
    except Exception as exc:
        return None, {
            "rruff_id": row["rruff_id"],
            "reason": "xrd_simulation_failed",
            "detail": f"{type(exc).__name__}: {exc}",
        }

    if not xrd_passed:
        return None, {
            "rruff_id": row["rruff_id"],
            "reason": "peak_mismatch",
            **xrd_info,
        }

    reduced = lattice_to_dict(reduced_lattice)
    accepted = row.to_dict()
    accepted.update(
        {
            "regression_a": reduced["a"],
            "regression_b": reduced["b"],
            "regression_c": reduced["c"],
            "regression_alpha": reduced["alpha"],
            "regression_beta": reduced["beta"],
            "regression_gamma": reduced["gamma"],
            "regression_volume": reduced["volume"],
            "regression_label_source": "pymatgen_niggli_from_dif_atom_table",
            "validation_method": "pymatgen_xrd_peak_position_proxy",
            "xrd_d_rtol": float(args.d_rtol),
            "xrd_two_theta_atol": float(args.two_theta_atol),
            "atom_count": int(len(atoms)),
            "unique_site_count": int(site_info["site_count"]),
            "expanded_site_count": int(len(structure)),
            "max_site_occupancy_sum": float(site_info["max_occupancy_sum"]),
            "normalized_occupancy_sites": int(site_info["normalized_occupancy_sites"]),
            "lattice_symmetry_adjustment_max_abs": float(lattice_adjustment),
            **xrd_info,
        }
    )
    return accepted, {"rruff_id": row["rruff_id"], "reason": "accepted"}


def build_regression_proxy(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base_dir = Path(args.base_dir)
    input_csv = base_dir / args.input_csv
    output_csv = base_dir / args.output_csv
    rejection_csv = base_dir / args.rejection_csv
    stats_json = base_dir / args.stats_json

    input_df = pd.read_csv(input_csv)
    accepted_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []

    for _, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Build regression proxy"):
        accepted, audit = validate_row(row, base_dir=base_dir, args=args)
        if accepted is None:
            rejection = row.to_dict()
            rejection.update(audit)
            rejection_rows.append(rejection)
        else:
            accepted_rows.append(accepted)

    accepted_df = pd.DataFrame(accepted_rows)
    rejection_df = pd.DataFrame(rejection_rows)
    accepted_df.to_csv(output_csv, index=False)
    rejection_df.to_csv(rejection_csv, index=False)

    reason_counts = Counter(rejection_df["reason"]) if not rejection_df.empty else Counter()
    stats = {
        "input_csv": str(input_csv),
        "input_rows": int(len(input_df)),
        "accepted_rows": int(len(accepted_df)),
        "rejected_rows": int(len(rejection_df)),
        "rejection_reasons": {reason: int(count) for reason, count in reason_counts.items()},
        "top_n": int(args.top_n),
        "min_matches": int(args.min_matches),
        "d_rtol": float(args.d_rtol),
        "two_theta_atol": float(args.two_theta_atol),
        "merge_two_theta_atol": float(args.merge_two_theta_atol),
        "coord_decimals": int(args.coord_decimals),
        "occupancy_sum_tolerance": float(args.occupancy_sum_tolerance),
        "crystal_system_distribution": (
            {k: int(v) for k, v in accepted_df["crystal_system"].value_counts().items()}
            if not accepted_df.empty
            else {}
        ),
        "space_group_count": int(accepted_df["space_group"].nunique()) if not accepted_df.empty else 0,
    }
    with open(stats_json, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    return accepted_df, rejection_df, stats


def build_regression_dataset(
    base_dir: Path = Path("."),
    input_csv: str = "datasets/classification/rruff_classification.csv",
    output_csv: str = "datasets/classification/rruff_regression.csv",
    rejection_csv: str = "datasets/classification/rruff_regression_rejections.csv",
    stats_json: str = "datasets/classification/rruff_regression_stats.json",
    top_n: int = 10,
    min_matches: int = 7,
    d_rtol: float = 0.005,
    two_theta_atol: float = 0.2,
    merge_two_theta_atol: float = 0.05,
    coord_decimals: int = 5,
    occupancy_sum_tolerance: float = 0.06,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Entry point for regression-proxy construction from an input cohort."""
    args = argparse.Namespace(
        base_dir=str(base_dir),
        input_csv=input_csv,
        output_csv=output_csv,
        rejection_csv=rejection_csv,
        stats_json=stats_json,
        top_n=top_n,
        min_matches=min_matches,
        d_rtol=d_rtol,
        two_theta_atol=two_theta_atol,
        merge_two_theta_atol=merge_two_theta_atol,
        coord_decimals=coord_decimals,
        occupancy_sum_tolerance=occupancy_sum_tolerance,
    )
    return build_regression_proxy(args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=".")
    parser.add_argument(
        "--input_csv",
        default="datasets/classification/rruff_classification.csv",
    )
    parser.add_argument(
        "--output_csv",
        default="datasets/classification/rruff_regression.csv",
    )
    parser.add_argument(
        "--rejection_csv",
        default="datasets/classification/rruff_regression_rejections.csv",
    )
    parser.add_argument(
        "--stats_json",
        default="datasets/classification/rruff_regression_stats.json",
    )
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--min_matches", type=int, default=7)
    parser.add_argument("--d_rtol", type=float, default=0.005)
    parser.add_argument("--two_theta_atol", type=float, default=0.2)
    parser.add_argument("--merge_two_theta_atol", type=float, default=0.05)
    parser.add_argument("--coord_decimals", type=int, default=5)
    parser.add_argument("--occupancy_sum_tolerance", type=float, default=0.06)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    warnings.filterwarnings("ignore", message="Full symbol not available.*")
    warnings.filterwarnings("ignore", message=".*occupancy.*")

    accepted_df, rejection_df, stats = build_regression_proxy(args)
    LOGGER.info("Regression proxy rows: %d", len(accepted_df))
    LOGGER.info("Rejected rows: %d", len(rejection_df))
    LOGGER.info("Stats:\n%s", json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
