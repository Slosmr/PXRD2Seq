from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
PIPELINE_DIR = PACKAGE_ROOT / "pipeline"
STAGES = ("sources", "amcsd", "plan", "materialize")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild dataset from MP, COD, and AMCSD source data.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_ROOT / "config.json",
        help="JSON configuration file. Relative paths are resolved from this file.",
    )
    parser.add_argument("--from-stage", choices=STAGES, default="sources")
    parser.add_argument("--to-stage", choices=STAGES, default="materialize")
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Remove final dataset artifact directories before materialization.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview commands without requiring source data or credentials and without writing output.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a JSON object")
    return config


def resolve_path(value: Any, base: Path, default: Path | None = None) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return default.resolve() if default else None
    path = Path(os.path.expandvars(os.path.expanduser(text)))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def require_dir(path: Path | None, label: str) -> Path:
    if path is None:
        raise ValueError(f"{label} is not configured")
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {path}")
    return path


def require_file(path: Path | None, label: str) -> Path:
    if path is None:
        raise ValueError(f"{label} is not configured")
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return path


def add_option(command: list[str], flag: str, value: Any) -> None:
    if value not in (None, ""):
        command.extend([flag, str(value)])


def printable_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def run(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print(f"\n> {printable_command(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PACKAGE_ROOT, env=env, check=True)


def selected_stages(first: str, last: str) -> list[str]:
    start = STAGES.index(first)
    end = STAGES.index(last)
    if start > end:
        raise ValueError("--from-stage must not come after --to-stage")
    return list(STAGES[start : end + 1])


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    config_dir = config_path.parent

    paths_cfg = config.get("paths", {})
    mp_cfg = config.get("mp", {})
    cod_cfg = config.get("cod", {})
    amcsd_cfg = config.get("amcsd", {})
    policy = config.get("policy", {})
    build_cfg = config.get("build", {})

    build_root = resolve_path(paths_cfg.get("build_root"), config_dir, PACKAGE_ROOT / "build")
    assert build_root is not None
    source_dataset = build_root / "source_dataset"
    amcsd_dataset = build_root / "amcsd_no_rruff"
    plan_dir = build_root / "dataset_plan"
    output_dir = resolve_path(paths_cfg.get("output_dir"), config_dir, build_root / "dataset")
    assert output_dir is not None

    cod_root = resolve_path(paths_cfg.get("cod_cif_root"), config_dir)
    amcsd_root = resolve_path(paths_cfg.get("amcsd_cif_root"), config_dir)
    rruff_test_csv = resolve_path(paths_cfg.get("rruff_test_csv"), config_dir)
    stages = selected_stages(args.from_stage, args.to_stage)
    if "sources" in stages and not args.dry_run:
        cod_root = require_dir(cod_root, "paths.cod_cif_root")
    if "amcsd" in stages and not args.dry_run:
        amcsd_root = require_dir(amcsd_root, "paths.amcsd_cif_root")
    if "plan" in stages and not args.dry_run:
        cod_root = require_dir(cod_root, "paths.cod_cif_root")
        if "sources" not in stages:
            require_file(source_dataset / "train.csv", "source dataset train split")
        if "amcsd" not in stages:
            require_file(amcsd_dataset / "metadata.csv", "AMCSD metadata")
        if rruff_test_csv is not None:
            require_file(rruff_test_csv, "paths.rruff_test_csv")
    if "materialize" in stages and "plan" not in stages and not args.dry_run:
        require_file(plan_dir / "manifest_dedup_keep.csv", "dataset manifest")

    env = os.environ.copy()
    if mp_cfg.get("api_key"):
        raise ValueError("Remove mp.api_key from the configuration; use MP_API_KEY instead.")
    api_key = env.get("MP_API_KEY", "").strip()
    if "sources" in stages:
        if not api_key and not args.dry_run:
            raise ValueError(
                "Materials Project API key is missing. Set the MP_API_KEY environment variable."
            )
        env["MP_API_KEY"] = api_key

    python = sys.executable
    workers = positive_int(build_cfg.get("workers"), min(max((os.cpu_count() or 2) - 1, 1), 8))
    seed = int(build_cfg.get("random_seed", 42))
    commands: list[tuple[str, list[str]]] = []

    if "sources" in stages:
        command = [
            python,
            str(PIPELINE_DIR / "get_data.py"),
            "--output_dir",
            str(source_dataset),
            "--include_mp",
            "--include_cod",
            "--cod_root",
            str(cod_root),
            "--num_workers",
            str(workers),
            "--no_dedup",
            "--split",
            "--seed",
            str(seed),
        ]
        command.append("--resume" if mp_cfg.get("resume", True) else "--no_resume")
        add_option(command, "--cod_max_files", optional_int(cod_cfg.get("max_files")))
        add_option(command, "--cod_batch_size", cod_cfg.get("batch_size", 64))
        add_option(command, "--mp_max_records", optional_int(mp_cfg.get("max_records")))
        add_option(command, "--mp_batch_size", mp_cfg.get("batch_size", 64))
        add_option(command, "--mp_checkpoint_every", mp_cfg.get("checkpoint_every", 512))
        commands.append(("sources", command))

    if "amcsd" in stages:
        command = [
            python,
            str(PIPELINE_DIR / "build_amcsd_dataset.py"),
            "--amcsd_dir",
            str(amcsd_root),
            "--output_dir",
            str(amcsd_dataset),
            "--dedup_scope",
            "none",
            "--workers",
            str(positive_int(amcsd_cfg.get("workers"), workers)),
        ]
        add_option(command, "--max_files", optional_int(amcsd_cfg.get("max_files")))
        if amcsd_cfg.get("rebuild_raw", False):
            command.append("--rebuild_raw")
        commands.append(("amcsd", command))

    if "plan" in stages:
        command = [
            python,
            str(PIPELINE_DIR / "reconstruct_dataset.py"),
            "--workspace",
            str(build_root),
            "--dataset-dir",
            str(source_dataset),
            "--skip-icsd",
            "--replace-amcsd-metadata",
            str(amcsd_dataset / "metadata.csv"),
            "--replace-amcsd-root",
            str(amcsd_dataset),
            "--cod-raw-root",
            str(cod_root),
            "--output-dir",
            str(plan_dir),
            "--cod-hard-risk-flags",
            str(policy.get("cod_hard_risk_flags", "pressure,high_temperature")),
            "--cod-review-risk-flags",
            str(policy.get("cod_review_risk_flags", "synthetic,thermal,growth,low_temperature")),
            "--cod-low-temperature-lt",
            str(policy.get("cod_low_temperature_lt", 250.0)),
            "--cod-high-temperature-gt",
            str(policy.get("cod_high_temperature_gt", 400.0)),
            "--mp-exclude-energy-gt",
            str(policy.get("mp_exclude_energy_gt", 0.2)),
            "--mp-weight-005-010",
            str(policy.get("mp_weight_005_010", 0.5)),
            "--mp-weight-010-020",
            str(policy.get("mp_weight_010_020", 0.1)),
            "--volume-per-atom-bin",
            str(policy.get("volume_per_atom_bin", 0.05)),
        ]
        if rruff_test_csv is not None:
            command.extend(["--rruff-test-csv", str(rruff_test_csv)])
        commands.append(("plan", command))

    if "materialize" in stages:
        command = [
            python,
            str(PIPELINE_DIR / "materialize_dataset.py"),
            "--workspace",
            str(build_root),
            "--manifest",
            str(plan_dir / "manifest_dedup_keep.csv"),
            "--output-dir",
            str(output_dir),
            "--artifact-mode",
            str(build_cfg.get("artifact_mode", "hardlink")),
        ]
        if args.reset_output or build_cfg.get("reset_output", False):
            command.append("--reset-output")


        commands.append(("materialize", command))

    if not args.dry_run:
        build_root.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    run_record: dict[str, Any] = {
        "started_utc": started.isoformat(),
        "config": str(config_path),
        "stages": stages,
        "commands": [{"stage": stage, "command": command} for stage, command in commands],
        "rruff_policy": "audit_only_never_excluded",
        "status": "dry_run" if args.dry_run else "running",
    }

    try:
        for stage, command in commands:
            print(f"\n[{stage}]", flush=True)
            run(command, env, args.dry_run)
    except Exception as exc:
        run_record["status"] = "failed"
        run_record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    else:
        if not args.dry_run:
            run_record["status"] = "complete"
    finally:
        run_record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        if not args.dry_run:
            with (build_root / "pipeline_run.json").open("w", encoding="utf-8") as handle:
                json.dump(run_record, handle, indent=2, ensure_ascii=False)

    print(f"\nPipeline status: {run_record['status']}", flush=True)
    print(f"Final dataset: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
