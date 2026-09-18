"""Build RRUFF base, adaptation, classification, and regression-proxy datasets."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from prepare_base import CONFIG as BASE_CONFIG
from prepare_base import process_rruff
from prepare_classification import (
    build_rruff_classification_candidates,
    write_local_intersection,
)
from prepare_regression import build_regression_dataset


LOGGER = logging.getLogger("rruff_pipeline")
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

RAW_XY_DIR = Path("raw_data/XY_Processed")
RAW_DIF_DIR = Path("raw_data/DIF")
BASE_OUTPUT_DIR = Path("datasets/base")
CLASSIFICATION_OUTPUT_DIR = Path("datasets/classification")

LABELED_CSV = BASE_OUTPUT_DIR / "rruff_labeled.csv"
ADAPT_CSV = BASE_OUTPUT_DIR / "rruff_adapt.csv"
CLASSIFICATION_CANDIDATES_CSV = (
    CLASSIFICATION_OUTPUT_DIR / "_rruff_classification_candidates.csv"
)
CLASSIFICATION_CSV = CLASSIFICATION_OUTPUT_DIR / "rruff_classification.csv"
REGRESSION_CSV = CLASSIFICATION_OUTPUT_DIR / "rruff_regression.csv"


def ensure_layout() -> None:
    for path in (BASE_OUTPUT_DIR, CLASSIFICATION_OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)
    if not RAW_XY_DIR.is_dir() or not RAW_DIF_DIR.is_dir():
        raise FileNotFoundError(
            "Input raw_data/XY_Processed and raw_data/DIF directories are required"
        )


def run_base_stage(signal_length: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    LOGGER.info("Stage 1/3: labeled base dataset and unlabeled adapt dataset")
    config = {**BASE_CONFIG, "signal_length": int(signal_length)}
    labeled, adapt, _ = process_rruff(
        xy_dir=RAW_XY_DIR,
        dif_dir=RAW_DIF_DIR,
        output_dir=BASE_OUTPUT_DIR,
        cfg=config,
        strict_cu_ka=True,
        require_dif=True,
        check_symmetry=True,
        check_xy_dif=True,
        exclude_calculated=True,
        labeled_csv_name=LABELED_CSV.name,
        adapt_csv_name=ADAPT_CSV.name,
    )
    return labeled, adapt


def run_classification_stage(signal_length: int) -> pd.DataFrame:
    if not LABELED_CSV.is_file():
        raise FileNotFoundError(
            f"Missing {LABELED_CSV}; run the base stage before classification"
        )

    LOGGER.info("Stage 2/3: strict classification dataset")
    candidates, _ = build_rruff_classification_candidates(
        xy_dir=RAW_XY_DIR,
        dif_dir=RAW_DIF_DIR,
        output_dir=CLASSIFICATION_OUTPUT_DIR,
        csv_name=CLASSIFICATION_CANDIDATES_CSV.name,
        length_rtol=0.001,
        angle_atol=0.5,
        require_cu_ka=False,
        signal_length=signal_length,
        dry_run=False,
    )
    classification, _ = write_local_intersection(
        candidate_df=candidates,
        local_csv=LABELED_CSV,
        output_dir=CLASSIFICATION_OUTPUT_DIR,
        csv_name=CLASSIFICATION_CSV.name,
    )
    if classification is None:
        raise RuntimeError("Classification intersection was not generated")
    return classification


def run_regression_stage(args: argparse.Namespace) -> pd.DataFrame:
    if not CLASSIFICATION_CSV.is_file():
        raise FileNotFoundError(
            f"Missing {CLASSIFICATION_CSV}; run the classification stage first"
        )

    LOGGER.info("Stage 3/3: pymatgen-validated regression proxy dataset")
    regression, _, _ = build_regression_dataset(
        base_dir=Path("."),
        input_csv=CLASSIFICATION_CSV.as_posix(),
        output_csv=REGRESSION_CSV.as_posix(),
        rejection_csv=(
            CLASSIFICATION_OUTPUT_DIR / "rruff_regression_rejections.csv"
        ).as_posix(),
        stats_json=(
            CLASSIFICATION_OUTPUT_DIR / "rruff_regression_stats.json"
        ).as_posix(),
        top_n=args.top_n,
        min_matches=args.min_matches,
        d_rtol=args.d_rtol,
        two_theta_atol=args.two_theta_atol,
        merge_two_theta_atol=args.merge_two_theta_atol,
        coord_decimals=args.coord_decimals,
        occupancy_sum_tolerance=args.occupancy_sum_tolerance,
    )
    return regression


def csv_row_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    return int(len(pd.read_csv(path)))


def write_build_manifest(args: argparse.Namespace) -> None:
    manifest = {
        "pipeline_version": 1,
        "built_at": datetime.now().astimezone().isoformat(),
        "signal_length": int(args.signal_length),
        "raw_data": {
            "xy_processed_files": sum(1 for p in RAW_XY_DIR.rglob("*") if p.is_file()),
            "dif_files": sum(1 for p in RAW_DIF_DIR.rglob("*") if p.is_file()),
        },
        "datasets": {
            str(LABELED_CSV): csv_row_count(LABELED_CSV),
            str(ADAPT_CSV): csv_row_count(ADAPT_CSV),
            str(CLASSIFICATION_CSV): csv_row_count(CLASSIFICATION_CSV),
            str(REGRESSION_CSV): csv_row_count(REGRESSION_CSV),
        },
        "regression_peak_validation": {
            "top_n": int(args.top_n),
            "min_matches": int(args.min_matches),
            "d_rtol": float(args.d_rtol),
            "two_theta_atol": float(args.two_theta_atol),
            "merge_two_theta_atol": float(args.merge_two_theta_atol),
        },
    }
    with open("datasets/build_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build RRUFF base, adapt, classification, and regression-proxy datasets"
    )
    parser.add_argument(
        "--stage",
        choices=("all", "base", "classification", "regression"),
        default="all",
    )
    parser.add_argument("--root", type=Path, default=PACKAGE_ROOT,
                        help="Working directory containing raw_data/; results are written to datasets/ beneath it.")
    parser.add_argument("--signal_length", type=int, default=8500)
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--min_matches", type=int, default=7)
    parser.add_argument("--d_rtol", type=float, default=0.005)
    parser.add_argument("--two_theta_atol", type=float, default=0.2)
    parser.add_argument("--merge_two_theta_atol", type=float, default=0.05)
    parser.add_argument("--coord_decimals", type=int, default=5)
    parser.add_argument("--occupancy_sum_tolerance", type=float, default=0.06)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    os.chdir(args.root.resolve())
    ensure_layout()

    if args.stage in {"all", "base"}:
        run_base_stage(args.signal_length)
    if args.stage in {"all", "classification"}:
        run_classification_stage(args.signal_length)
    if args.stage in {"all", "regression"}:
        run_regression_stage(args)

    write_build_manifest(args)
    LOGGER.info(
        "Build complete: labeled=%s, adapt=%s, classification=%s, regression=%s",
        csv_row_count(LABELED_CSV),
        csv_row_count(ADAPT_CSV),
        csv_row_count(CLASSIFICATION_CSV),
        csv_row_count(REGRESSION_CSV),
    )


if __name__ == "__main__":
    main()
