"""Build MP and COD structure artifacts and material-level data splits.

Each record includes an ideal diffraction peak table, crystallographic labels,
and a standardized CIF. Online signal synthesis belongs to the model pipeline
and is not included in this data construction release."""

import os
import json
import logging
import argparse
import copy
import warnings
import time
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool


from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


MP_DOWNLOAD_CONFIG = {
    "fields": ["structure", "material_id", "formula_pretty", "energy_above_hull"],
    "representation": {
        "symprec": 0.01,
        "angle_tolerance": 5.0,
    },
    "filters": {
        "max_atoms": 500,
        "min_volume": None,
        "max_volume": 100000.0,
        "require_spacegroup_stability": True,
        "symprecs": [0.01, 0.05, 0.1],
        "angle_tolerances": [0.1, 1.0, 2.5, 5.0],
        "check_reduced_cell_consistency": True,
        "rtol": 1e-3,
    },
}


CONFIG = {


    "representation": {
        "symprec": 0.01,
        "angle_tolerance": 5.0,
    },


    "n_workers": max(1, (os.cpu_count() or 2) - 1),


    "wavelength": "CuKa",
    "theta_range": (4, 90),
    "signal_length": 8500,


    "intensity_threshold_pct": 5.0,
    "top_n_peaks": 20,
    "max_peaks": 20,


    "output_dir": "source_dataset",

    "sources": {
        "mp": {
            "enabled": True,
            "workers": 8,
            "batch_size": 128,
            "resume": True,
            "checkpoint_every": 512,
            "api_retries": 6,
            "api_retry_sleep_sec": 60,
            "max_records": None,
        },
        "cod": {
            "enabled": True,
            "cod_root": "",
            "max_files": None,
            "workers": 8,
            "batch_size": 64,
        },
    },

    "cod_filters": {
        "max_elements": 8,
        "max_sites": 200,
        "max_volume": 10000.0,
        "min_volume": 1.0,
        "require_ordered": True,
        "require_full_occupancy": True,
        "remove_oxidation_states": True,
        "skip_theoretical": True,
        "skip_duplicate_flag": True,
        "skip_error_flag": True,
        "allowed_atomic_numbers_max": 94,
        "exclude_radioactive": True,
    },

    "dedup": {
        "enabled": True,
        "prefer_source": "COD",
        "structure_matcher": True,
        "formula_sg_volume_prefilter": True,
        "volume_per_atom_tol": 0.05,
        "ltol": 0.2,
        "stol": 0.3,
        "angle_tol": 5,
    },

    "split": {
        "enabled": True,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "stratify_by": ["source", "crystal_system"],
        "random_seed": 42,
    },
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def configure_warning_filters() -> None:
    """Keep noisy pymatgen CIF cleanup warnings out of full-build progress."""
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"pymatgen\.io\.cif",
    )


    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*No structure parsed for section.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*Occupancy .* exceeded tolerance.*",
    )

    warnings.filterwarnings(
        "ignore",
        message=r".* parsed as .*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Issues encountered while parsing CIF:.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Incorrect stoichiometry:.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Site labels are not unique.*",
        category=UserWarning,
    )


configure_warning_filters()


POINTGROUP_TO_LAUE = {

    "1": "-1", "-1": "-1",

    "2": "2/m", "m": "2/m", "2/m": "2/m",

    "222": "mmm", "mm2": "mmm", "mmm": "mmm",

    "4": "4/m", "-4": "4/m", "4/m": "4/m",
    "422": "4/mmm", "4mm": "4/mmm", "-42m": "4/mmm",
    "-4m2": "4/mmm", "4/mmm": "4/mmm",

    "3": "-3", "-3": "-3",
    "32": "-3m", "3m": "-3m", "-3m": "-3m",
    "321": "-3m", "312": "-3m", "3m1": "-3m", "31m": "-3m",
    "-3m1": "-3m", "-31m": "-3m",

    "6": "6/m", "-6": "6/m", "6/m": "6/m",
    "622": "6/mmm", "6mm": "6/mmm", "-6m2": "6/mmm",
    "-62m": "6/mmm", "6/mmm": "6/mmm",

    "23": "m-3", "m-3": "m-3",
    "432": "m-3m", "-43m": "m-3m", "m-3m": "m-3m",
}

CRYSTAL_SYSTEM_BRAVAIS = {
    "triclinic": {"P": "aP"},
    "monoclinic": {"P": "mP", "C": "mS", "A": "mS", "B": "mS", "I": "mS"},
    "orthorhombic": {"P": "oP", "C": "oS", "A": "oS", "B": "oS",
                     "F": "oF", "I": "oI"},
    "tetragonal": {"P": "tP", "I": "tI"},
    "trigonal": {"P": "hP", "R": "hR"},
    "hexagonal": {"P": "hP"},
    "cubic": {"P": "cP", "F": "cF", "I": "cI"},
}

RADIOACTIVE_Z = {43, 61, *range(84, 119)}


def get_laue_class(structure) -> str:
    """从 pymatgen Structure 获取 Laue 群。"""
    try:
        analyzer = SpacegroupAnalyzer(structure)
        pg = analyzer.get_point_group_symbol()
        return POINTGROUP_TO_LAUE.get(pg, "unknown")
    except Exception:
        return "unknown"


def get_bravais_lattice(crystal_system: str, sg_symbol: str) -> str:
    """从晶系和空间群符号推断 Bravais 格子类型。"""
    if not sg_symbol:
        return "unknown"
    first_letter = sg_symbol[0]
    cs_map = CRYSTAL_SYSTEM_BRAVAIS.get(crystal_system, {})
    return cs_map.get(first_letter, "unknown")


def filter_peaks_hybrid(pattern, threshold_pct: float, top_n: int) -> list:
    """
    混合模式筛选峰：(强度>=阈值%) ∪ (Top-N强度峰)
    返回按 2θ 升序排列的 [(two_theta, intensity, d_spacing), ...]
    """
    max_intensity = max(pattern.y)
    all_peaks = list(zip(pattern.x, pattern.y, pattern.d_hkls))
    threshold_idx = {
        i for i, (_, intensity, _) in enumerate(all_peaks)
        if (intensity / max_intensity * 100) >= threshold_pct
    }
    top_n_idx = {
        i for i, _ in sorted(
            enumerate(all_peaks), key=lambda x: x[1][1], reverse=True
        )[:top_n]
    }
    selected = threshold_idx | top_n_idx
    return [all_peaks[i] for i in sorted(selected)]


def compute_base_peaks(structure, cfg: dict) -> list:
    """
    计算给定晶体结构的 XRD 衍射花样并筛选峰位。
    结果与随机种子无关，可被所有增强实例共享。
    """
    xrd_calc = XRDCalculator(wavelength=cfg["wavelength"])
    pattern = xrd_calc.get_pattern(
        structure, scaled=True, two_theta_range=cfg["theta_range"]
    )
    selected_peaks = filter_peaks_hybrid(
        pattern, cfg["intensity_threshold_pct"], cfg["top_n_peaks"]
    )
    max_peaks = cfg["max_peaks"]
    if len(selected_peaks) > max_peaks:
        selected_peaks = sorted(selected_peaks, key=lambda x: x[1], reverse=True)[:max_peaks]
        selected_peaks = sorted(selected_peaks, key=lambda x: x[0])
    return selected_peaks


def generate_structure_labels(structure, base_mid: str, formula: str,
                               crystal_system: str, sg_symbol: str,
                               laue_class: str, bravais: str,
                               sg_number: int = -1,
                               source: str = "MP") -> dict:
    """Return fixed crystallographic labels independent of signal augmentation.

The record contains formula, crystal system, space group, Laue class,
Bravais lattice, lattice parameters, and cell volume."""
    lattice = structure.lattice
    return {
        "material_id": base_mid,
        "source": source,
        "formula": formula,
        "crystal_system": crystal_system,
        "space_group": sg_symbol,
        "space_group_number": sg_number,
        "laue_class": laue_class,
        "bravais_lattice": bravais,
        "a": round(lattice.a, 4),
        "b": round(lattice.b, 4),
        "c": round(lattice.c, 4),
        "alpha": round(lattice.alpha, 3),
        "beta": round(lattice.beta, 3),
        "gamma": round(lattice.gamma, 3),
        "volume": round(lattice.volume, 4),
    }


def init_dataset_dir(output_dir: str) -> dict:
    """Create directories for per-material diffraction peaks, labels, and structures."""
    paths = {
        "root": Path(output_dir),
        "base_peaks": Path(output_dir) / "base_peaks",
        "labels": Path(output_dir) / "labels",
        "structures": Path(output_dir) / "structures",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    logger.info(f"数据集目录已初始化 (v6 模式): {output_dir}")
    return paths


def _dataset_paths(output_dir) -> dict:
    root = Path(output_dir)
    return {
        "root": root,
        "base_peaks": root / "base_peaks",
        "labels": root / "labels",
        "structures": root / "structures",
    }


def _ensure_dataset_paths(output_dir) -> dict:
    paths = _dataset_paths(output_dir)
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _rel_path(path, root) -> str:
    path = Path(path)
    root = Path(root)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_dataset_path(root, value) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return Path(root) / path


def _metadata_path_for_source(output_dir, source: str) -> Path:
    return Path(output_dir) / f"metadata_{str(source).lower()}.csv"


def _artifacts_exist(output_dir, material_id: str) -> bool:
    root = Path(output_dir)
    material_id = str(material_id).lower()
    return (
        (root / "base_peaks" / f"{material_id}.npz").exists()
        and (root / "labels" / f"{material_id}.json").exists()
        and (root / "structures" / f"{material_id}.cif").exists()
    )


def _read_existing_metadata(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except Exception as exc:
            logger.warning(f"Could not read existing metadata {path}: {exc}")
    return pd.DataFrame()


def _merge_metadata_rows(rows: list, new_rows: list) -> list:
    if not new_rows:
        return rows
    if not rows:
        return list(new_rows)
    df = pd.concat([pd.DataFrame(rows), pd.DataFrame(new_rows)], ignore_index=True)
    if "material_id" in df.columns:
        df = df.drop_duplicates("material_id", keep="last")
    return df.to_dict("records")


def _flush_rows_to_csv(rows: list, path: Path, log_label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if "material_id" in df.columns:
        df = df.drop_duplicates("material_id", keep="last")
    df.to_csv(path, index=False)
    logger.info(f"{log_label} checkpoint saved: {path} ({len(df)} rows)")


def reconstruct_record_from_artifacts(
    material_id: str,
    source: str,
    output_dir,
    extra_metadata: dict = None,
):
    root = Path(output_dir)
    material_id = str(material_id).lower()
    labels_path = root / "labels" / f"{material_id}.json"
    base_peaks_path = root / "base_peaks" / f"{material_id}.npz"
    structure_path = root / "structures" / f"{material_id}.cif"
    if not (labels_path.exists() and base_peaks_path.exists() and structure_path.exists()):
        return None

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    with np.load(base_peaks_path) as peaks:
        n_base_peaks = int(len(peaks["mus"])) if "mus" in peaks.files else 0

    structure = _load_structure_for_dedup(root, structure_path)
    summary = _composition_summary(structure)
    volume = float(labels.get("volume", structure.volume))
    n_sites = max(1, int(summary["n_sites"]))

    row = {
        "source": str(source).upper(),
        "material_id": material_id,
        "base_material_id": material_id,
        "formula": labels.get("formula", summary["formula"]),
        "reduced_formula": summary["reduced_formula"],
        "anonymous_formula": summary["anonymous_formula"],
        "crystal_system": labels.get("crystal_system", "unknown"),
        "space_group": labels.get("space_group", "unknown"),
        "space_group_number": labels.get("space_group_number", -1),
        "laue_class": labels.get("laue_class", "unknown"),
        "bravais_lattice": labels.get("bravais_lattice", "unknown"),
        "a": labels.get("a"),
        "b": labels.get("b"),
        "c": labels.get("c"),
        "alpha": labels.get("alpha"),
        "beta": labels.get("beta"),
        "gamma": labels.get("gamma"),
        "volume": labels.get("volume", round(volume, 4)),
        "n_elements": summary["n_elements"],
        "n_sites": n_sites,
        "volume_per_atom": round(volume / n_sites, 6),
        "n_base_peaks": n_base_peaks,
        "base_peaks_path": _rel_path(base_peaks_path, root),
        "labels_path": _rel_path(labels_path, root),
        "structure_path": _rel_path(structure_path, root),
    }
    for key, value in (extra_metadata or {}).items():
        if key not in row:
            row[key] = value
    return row


def reconcile_metadata_with_artifacts(
    metadata_df: pd.DataFrame,
    output_dir,
    source: str,
) -> tuple:
    root = Path(output_dir)
    source = str(source).upper()
    prefix = source.lower() + "-"

    base_ids = {p.stem for p in (root / "base_peaks").glob(f"{prefix}*.npz")}
    label_ids = {p.stem for p in (root / "labels").glob(f"{prefix}*.json")}
    structure_ids = {p.stem for p in (root / "structures").glob(f"{prefix}*.cif")}
    complete_ids = base_ids & label_ids & structure_ids

    if metadata_df.empty or "material_id" not in metadata_df.columns:
        metadata_ids = set()
    else:
        metadata_ids = set(metadata_df["material_id"].astype(str).str.lower())

    rows_to_add = []
    for material_id in sorted(complete_ids - metadata_ids):
        extra = {}
        if source == "COD":
            extra["cod_id"] = material_id.removeprefix("cod-")
        try:
            row = reconstruct_record_from_artifacts(
                material_id=material_id,
                source=source,
                output_dir=root,
                extra_metadata=extra,
            )
        except Exception:
            row = None
        if row is not None:
            rows_to_add.append(row)

    if rows_to_add:
        metadata_df = pd.concat(
            [metadata_df, pd.DataFrame(rows_to_add)],
            ignore_index=True,
        )
        metadata_df = metadata_df.drop_duplicates("material_id", keep="last")

    stats = {
        "source": source,
        "metadata_rows_before": int(len(metadata_ids)),
        "complete_artifacts": int(len(complete_ids)),
        "reconstructed_rows": int(len(rows_to_add)),
        "base_without_label_or_structure": sorted(base_ids - label_ids - structure_ids),
        "base_missing_label": sorted(base_ids - label_ids),
        "base_missing_structure": sorted(base_ids - structure_ids),
    }
    if rows_to_add or stats["base_missing_label"] or stats["base_missing_structure"]:
        logger.info(
            f"{source} artifact reconciliation: +{len(rows_to_add)} metadata rows, "
            f"{len(stats['base_missing_label'])} missing labels, "
            f"{len(stats['base_missing_structure'])} missing structures"
        )
    return metadata_df, stats


def _safe_json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Counter):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    if isinstance(value, defaultdict):
        return {str(k): _safe_json_value(v) for k, v in dict(value).items()}
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json_value(v) for v in value]
    return value


def _chunked(items, batch_size: int):
    batch = []
    batch_size = max(1, int(batch_size))
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _task_progress_units(task) -> int:
    if isinstance(task, dict):
        try:
            return int(task.get("progress_units", 1))
        except Exception:
            return 1
    if isinstance(task, (list, tuple)):
        return len(task)
    return 1


def _failed_pool_result(task, reason="process_pool_broken") -> dict:
    return {
        "success": False,
        "reason": reason,
        "row": None,
        "progress_units": _task_progress_units(task),
    }


def iter_bounded_process_pool(worker_fn, tasks, total, workers, desc,
                              exception_reason="process_failed",
                              max_pending=None):
    """
    Run process-pool tasks with a bounded number of in-flight futures.

    Limit pending futures to keep memory usage bounded while results are consumed.
    """
    tasks = iter(tasks)
    if total <= 0:
        return

    if workers <= 1:
        with tqdm(total=total, desc=desc) as pbar:
            for task in tasks:
                try:
                    result = worker_fn(task)
                except Exception:
                    result = {
                        "success": False,
                        "reason": exception_reason,
                        "row": None,
                        "progress_units": _task_progress_units(task),
                    }
                pbar.update(int(result.get("progress_units", _task_progress_units(task))))
                yield result
        return

    if max_pending is None:
        max_pending = max(workers * 4, workers)

    with tqdm(total=total, desc=desc) as pbar:
        while True:
            pending = {}
            failed_submit_task = None
            pool_broken = False

            with ProcessPoolExecutor(max_workers=workers) as executor:
                def submit_one():
                    nonlocal failed_submit_task
                    try:
                        task = next(tasks)
                    except StopIteration:
                        return False
                    try:
                        pending[executor.submit(worker_fn, task)] = task
                    except BrokenProcessPool:
                        failed_submit_task = task
                        raise
                    return True

                try:
                    for _ in range(min(max_pending, total)):
                        if not submit_one():
                            break
                except BrokenProcessPool:
                    pool_broken = True

                while pending and not pool_broken:
                    for future in as_completed(pending):
                        task = pending.pop(future)
                        try:
                            result = future.result()
                        except BrokenProcessPool:
                            result = _failed_pool_result(task)
                            pool_broken = True
                        except Exception:
                            result = {
                                "success": False,
                                "reason": exception_reason,
                                "row": None,
                                "progress_units": _task_progress_units(task),
                            }
                        pbar.update(int(result.get("progress_units", _task_progress_units(task))))
                        yield result

                        if pool_broken:
                            break

                        try:
                            submit_one()
                        except BrokenProcessPool:
                            pool_broken = True
                        break

                if pool_broken:
                    if failed_submit_task is not None:
                        result = _failed_pool_result(failed_submit_task)
                        pbar.update(int(result["progress_units"]))
                        yield result
                    for task in list(pending.values()):
                        result = _failed_pool_result(task)
                        pbar.update(int(result["progress_units"]))
                        yield result

            if not pending and not pool_broken:
                break

            logger.warning(
                f"{desc}: process pool broke because a child process terminated. "
                "The affected batch(es) were marked failed and the pool will restart."
            )


def save_build_stats(output_dir, stats: dict) -> None:
    stats_path = Path(output_dir) / "build_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(_safe_json_value(stats), f, ensure_ascii=False, indent=2)


def configure_build_logging(output_dir) -> None:
    log_path = Path(output_dir) / "build.log"
    root_logger = logging.getLogger()
    existing = {
        getattr(h, "baseFilename", None)
        for h in root_logger.handlers
        if isinstance(h, logging.FileHandler)
    }
    if str(log_path) in existing:
        return
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root_logger.addHandler(handler)


def _copy_structure(structure):
    try:
        return structure.copy()
    except Exception:
        return copy.deepcopy(structure)


def strip_oxidation_states_safe(structure):
    cleaned = _copy_structure(structure)
    try:
        cleaned.remove_oxidation_states()
        return cleaned, True
    except Exception:
        return structure, False


def save_standardized_structure(structure, structure_path) -> None:
    structure_path = Path(structure_path)
    structure_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        configure_warning_filters()
        CifWriter(structure).write_file(str(structure_path))


def _composition_summary(structure) -> dict:
    comp = structure.composition
    try:
        formula = comp.reduced_formula
    except Exception:
        formula = str(comp)
    try:
        anonymous_formula = comp.anonymized_formula
    except Exception:
        anonymous_formula = ""
    return {
        "formula": formula,
        "reduced_formula": formula,
        "anonymous_formula": anonymous_formula,
        "n_elements": int(len(comp.elements)),
        "n_sites": int(len(structure)),
    }


def _as_float_d_spacing(value) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, dict):
        for key in ("d_hkl", "d_spacing", "d"):
            if key in value:
                return float(value[key])
    if isinstance(value, (list, tuple)) and value:
        first = value[0]
        if isinstance(first, dict):
            for key in ("d_hkl", "d_spacing", "d"):
                if key in first:
                    return float(first[key])
        return float(first)
    return float("nan")


def process_structure_to_record(
    structure,
    material_id: str,
    source: str,
    output_dir,
    cfg: dict,
    extra_metadata: dict = None,
):
    """Process an MP or COD structure into a material record.

Symmetry labels are read from the input structure. Base peaks, lattice
labels, and the saved CIF use the Niggli-reduced structure."""
    extra_metadata = extra_metadata or {}
    source = str(source).upper()
    material_id = str(material_id).strip().lower()
    if source == "MP" and not material_id.startswith("mp-"):
        material_id = f"mp-{material_id}"
    if source == "COD" and not material_id.startswith("cod-"):
        material_id = f"cod-{material_id}"

    paths = _ensure_dataset_paths(output_dir)
    root = paths["root"]

    try:
        rep_cfg = cfg.get("representation", {})
        analyzer = SpacegroupAnalyzer(
            structure,
            symprec=float(rep_cfg.get("symprec", 0.01)),
            angle_tolerance=float(rep_cfg.get("angle_tolerance", 5.0)),
        )
        sg_symbol = analyzer.get_space_group_symbol()
        sg_number = int(analyzer.get_space_group_number())
        crystal_system = analyzer.get_crystal_system()
        laue_class = get_laue_class(structure)
        bravais = get_bravais_lattice(crystal_system, sg_symbol)
    except Exception as exc:
        raise RuntimeError(f"symmetry_failed: {exc}") from exc

    try:
        structure_niggli = structure.get_reduced_structure(reduction_algo="niggli")
    except Exception:
        structure_niggli = structure

    summary = _composition_summary(structure)
    formula = extra_metadata.get("formula") or summary["formula"]

    try:
        base_peaks = compute_base_peaks(structure_niggli, cfg)
    except Exception as exc:
        raise RuntimeError(f"xrd_failed: {exc}") from exc
    if not base_peaks:
        raise RuntimeError("xrd_failed: no_base_peaks")

    mus = np.array([float(p[0]) for p in base_peaks], dtype=np.float32)
    intensities = np.array([float(p[1]) for p in base_peaks], dtype=np.float32)
    d_hkls = np.array([_as_float_d_spacing(p[2]) for p in base_peaks], dtype=np.float32)

    base_peaks_path = paths["base_peaks"] / f"{material_id}.npz"
    labels_path = paths["labels"] / f"{material_id}.json"
    structure_path = paths["structures"] / f"{material_id}.cif"

    np.savez_compressed(
        str(base_peaks_path),
        mus=mus,
        intensities=intensities,
        d_hkls=d_hkls,
    )
    save_standardized_structure(structure_niggli, structure_path)

    labels = generate_structure_labels(
        structure_niggli,
        material_id,
        formula,
        crystal_system,
        sg_symbol,
        laue_class,
        bravais,
        sg_number=sg_number,
        source=source,
    )
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    lattice = structure_niggli.lattice
    volume_per_atom = float(lattice.volume) / max(1, len(structure_niggli))
    row = {
        "source": source,
        "material_id": material_id,
        "base_material_id": material_id,
        "formula": formula,
        "reduced_formula": summary["reduced_formula"],
        "anonymous_formula": summary["anonymous_formula"],
        "crystal_system": crystal_system,
        "space_group": sg_symbol,
        "space_group_number": sg_number,
        "laue_class": labels["laue_class"],
        "bravais_lattice": labels["bravais_lattice"],
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
        "base_peaks_path": _rel_path(base_peaks_path, root),
        "labels_path": _rel_path(labels_path, root),
        "structure_path": _rel_path(structure_path, root),
    }

    for key, value in extra_metadata.items():
        if key not in row:
            row[key] = value
    return row


def iter_cod_cif_files(cod_root, max_files=None):
    cod_root = Path(cod_root)
    if max_files is None:
        yield from sorted(cod_root.rglob("*.cif"))
        return

    files = []
    limit = int(max_files)
    for path in cod_root.rglob("*.cif"):
        if len(files) >= limit:
            break
        files.append(path)
    yield from sorted(files)


def cod_id_from_path(path) -> str:
    return Path(path).stem


def load_cod_structure(cif_path):
    with warnings.catch_warnings():
        configure_warning_filters()
        parser = CifParser(str(cif_path), occupancy_tolerance=1.0)
        if hasattr(parser, "parse_structures"):
            structures = parser.parse_structures(primitive=False)
        else:
            structures = parser.get_structures(primitive=False)
    if not structures:
        raise ValueError("empty_structure")
    return structures[0]


def _cod_tag_lines(cif_path):
    try:
        with open(cif_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                text = line.strip().lower()
                if not text.startswith("_"):
                    continue
                tag = text.split(maxsplit=1)[0]
                if tag.startswith(("_cod", "_audit", "_chemical", "_publ")):
                    yield text
    except OSError:
        return


def _cod_has_flag(cif_path, words) -> bool:
    for line in _cod_tag_lines(cif_path):
        if any(word in line for word in words):
            return True
    return False


def _site_has_full_occupancy(site) -> bool:
    try:
        total_occu = sum(float(occu) for occu in site.species.values())
    except Exception:
        return False
    return abs(total_occu - 1.0) <= 1e-6 and all(
        abs(float(occu) - 1.0) <= 1e-6 for occu in site.species.values()
    )


def filter_cod_structure(structure, cif_path, cfg) -> tuple:
    filters = cfg.get("cod_filters", {})
    if structure is None or len(structure) == 0:
        return False, "empty_structure"

    if filters.get("skip_theoretical", True) and _cod_has_flag(cif_path, ["theoretical"]):
        return False, "theoretical_flag"
    if filters.get("skip_duplicate_flag", True) and _cod_has_flag(cif_path, ["duplicate"]):
        return False, "duplicate_flag"
    if filters.get("skip_error_flag", True) and _cod_has_flag(cif_path, ["error"]):
        return False, "error_flag"

    if filters.get("require_ordered", True) and not getattr(structure, "is_ordered", False):
        return False, "disordered"
    if filters.get("require_full_occupancy", True):
        if any(not _site_has_full_occupancy(site) for site in structure):
            return False, "partial_occupancy"

    max_sites = filters.get("max_sites")
    if max_sites is not None and len(structure) > int(max_sites):
        return False, "too_many_sites"

    volume = float(structure.volume)
    min_volume = filters.get("min_volume")
    max_volume = filters.get("max_volume")
    if min_volume is not None and volume < float(min_volume):
        return False, "bad_volume"
    if max_volume is not None and volume > float(max_volume):
        return False, "bad_volume"

    elements = list(structure.composition.elements)
    max_elements = filters.get("max_elements")
    if max_elements is not None and len(elements) > int(max_elements):
        return False, "too_many_elements"

    max_z = filters.get("allowed_atomic_numbers_max")
    for el in elements:
        z = int(getattr(el, "Z", 999))
        if max_z is not None and z > int(max_z):
            return False, "bad_elements"
        if filters.get("exclude_radioactive", True) and z in RADIOACTIVE_Z:
            return False, "radioactive_element"

    return True, "ok"


def _process_cod_file(task: dict) -> dict:
    configure_warning_filters()
    cif_path = Path(task["cif_path"])
    output_dir = task["output_dir"]
    cfg = task["cfg"]
    cod_id = cod_id_from_path(cif_path)
    material_id = f"cod-{cod_id}"

    try:
        structure = load_cod_structure(cif_path)
    except Exception:
        return {"success": False, "reason": "parse_failed", "row": None}

    ok, reason = filter_cod_structure(structure, cif_path, cfg)
    if not ok:
        return {"success": False, "reason": reason, "row": None}

    if cfg.get("cod_filters", {}).get("remove_oxidation_states", True):
        structure, removed = strip_oxidation_states_safe(structure)
        if not removed:
            return {"success": False, "reason": "oxidation_remove_failed", "row": None}

    try:
        row = process_structure_to_record(
            structure=structure,
            material_id=material_id,
            source="COD",
            output_dir=output_dir,
            cfg=cfg,
            extra_metadata={
                "cod_id": cod_id,
                "cif_path": str(cif_path),
            },
        )
        return {"success": True, "reason": "success", "row": row}
    except Exception as exc:
        text = str(exc)
        if text.startswith("symmetry_failed"):
            return {"success": False, "reason": "symmetry_failed", "row": None}
        if text.startswith("xrd_failed"):
            return {"success": False, "reason": "xrd_failed", "row": None}
        return {"success": False, "reason": "process_failed", "row": None}


def _process_cod_batch(task: dict) -> dict:
    configure_warning_filters()
    output_dir = task["output_dir"]
    cfg = task["cfg"]
    results = []
    for cif_path in task.get("cif_paths", []):
        try:
            res = _process_cod_file({
                "cif_path": cif_path,
                "output_dir": output_dir,
                "cfg": cfg,
            })
        except Exception:
            res = {"success": False, "reason": "process_failed", "row": None}
        results.append(res)
    return {
        "success": True,
        "reason": "batch",
        "results": results,
        "progress_units": len(task.get("cif_paths", [])),
    }


def process_cod_dataset(cod_root, output_dir, cfg: dict):
    cod_root = Path(cod_root)
    stats = Counter()
    rows = []

    if not cod_root.exists():
        raise FileNotFoundError(f"COD root does not exist: {cod_root}")

    max_files = cfg.get("sources", {}).get("cod", {}).get("max_files")
    cod_source_cfg = cfg.get("sources", {}).get("cod", {})
    workers = int(cod_source_cfg.get("workers") or cfg.get("n_workers", 1))
    batch_size = int(cod_source_cfg.get("batch_size", 64) or 1)
    files = list(iter_cod_cif_files(cod_root, max_files=max_files))
    stats["n_cif_total"] = len(files)
    logger.info(
        f"COD CIF files queued: {len(files)} from {cod_root} "
        f"(workers={workers}, batch_size={batch_size})"
    )

    def task_iter():
        for batch in _chunked(files, batch_size):
            cif_paths = [str(path) for path in batch]
            yield {
                "cif_paths": cif_paths,
                "output_dir": str(output_dir),
                "cfg": cfg,
                "progress_units": len(cif_paths),
            }

    if files:
        for batch_res in iter_bounded_process_pool(
            _process_cod_batch,
            task_iter(),
            total=len(files),
            workers=workers,
            desc="COD CIF",
            exception_reason="process_failed",
        ):
            sub_results = batch_res.get("results")
            if not sub_results:
                stats[f"n_{batch_res['reason']}"] += int(batch_res.get("progress_units", 1))
                continue
            for res in sub_results:
                stats[f"n_{res['reason']}"] += 1
                if res["success"]:
                    rows.append(res["row"])

    df = pd.DataFrame(rows)

    metadata_path = Path(output_dir) / "metadata_cod.csv"
    df.to_csv(metadata_path, index=False)
    logger.info(f"COD metadata saved: {metadata_path} ({len(df)} rows)")
    return df, dict(stats)


def get_mp_download_config() -> dict:
    """Return the immutable dataset MP curation policy."""
    return copy.deepcopy(MP_DOWNLOAD_CONFIG)


def merge_mp_download_config(cfg: dict, download_cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    rep_cfg = download_cfg.get("representation")
    if isinstance(rep_cfg, dict):
        cfg.setdefault("representation", {})
        if rep_cfg.get("symprec") is not None:
            cfg["representation"]["symprec"] = float(rep_cfg["symprec"])
        if rep_cfg.get("angle_tolerance") is not None:
            cfg["representation"]["angle_tolerance"] = float(rep_cfg["angle_tolerance"])
    cfg["mp_structure_filters"] = download_cfg.get("filters", {}) or {}
    return cfg


def _first_mapping(*values):
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _range_or_value(value):
    if isinstance(value, dict):
        low = value.get("min", value.get("gte", value.get("lower")))
        high = value.get("max", value.get("lte", value.get("upper")))
        if low is not None or high is not None:
            return (low, high)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return tuple(value)
    return value


def _as_optional_float(value):
    if value is None or value == "":
        return None
    return float(value)


def _range_from_min_max(min_value, max_value, default_min=0):
    low = _as_optional_float(min_value)
    high = _as_optional_float(max_value)
    if low is None and high is None:
        return None
    if low is None:
        low = default_min
    return (low, high)


def _download_filter_cfg(download_cfg: dict) -> dict:
    filters = download_cfg.get("filters", {})
    return filters if isinstance(filters, dict) else {}


def _mp_search_kwargs_from_config(download_cfg: dict) -> dict:
    mp_cfg = download_cfg.get("mp", {}) if isinstance(download_cfg.get("mp"), dict) else {}
    search_cfg = _first_mapping(
        download_cfg.get("summary_search"),
        download_cfg.get("search"),
        download_cfg.get("criteria"),
        mp_cfg.get("summary_search"),
        mp_cfg.get("search"),
        mp_cfg.get("criteria"),
        mp_cfg.get("filters"),
    )
    search_cfg = dict(search_cfg)

    fields = (
        download_cfg.get("fields")
        or mp_cfg.get("fields")
        or search_cfg.pop("fields", None)
        or ["structure", "material_id", "formula_pretty", "energy_above_hull"]
    )
    if "structure" not in fields:
        fields = list(fields) + ["structure"]

    aliases = {
        "energy_above_hull_max": "energy_above_hull",
        "e_above_hull": "energy_above_hull",
        "max_elements": "num_elements",
        "num_elements_max": "num_elements",
        "max_atoms": "num_sites",
        "max_sites": "num_sites",
        "num_sites_max": "num_sites",
        "min_volume": "volume",
        "max_volume": "volume",
        "space_group_number": "spacegroup_number",
    }
    allowed = {
        "band_gap", "chemsys", "crystal_system", "density", "deprecated",
        "elements", "energy_above_hull", "exclude_elements", "formula",
        "formation_energy", "has_props", "is_gap_direct", "is_metal",
        "is_stable", "magnetic_ordering", "material_ids", "num_elements",
        "num_sites", "spacegroup_number", "theoretical", "total_energy",
        "uncorrected_energy", "volume", "fields", "chunk_size", "num_chunks",
        "all_fields",
    }

    kwargs = {"fields": fields}
    for raw_key, value in search_cfg.items():
        key = aliases.get(raw_key, raw_key)
        if key not in allowed:
            logger.info(f"Ignoring unsupported MP config key for summary.search: {raw_key}")
            continue
        if raw_key in {"energy_above_hull_max", "max_elements", "num_elements_max",
                       "max_atoms", "max_sites", "num_sites_max"}:
            value = (0, value)
        if raw_key == "min_volume":
            value = (value, None)
        if raw_key == "max_volume":
            value = (0, value)
        kwargs[key] = _range_or_value(value)

    filters = _download_filter_cfg(download_cfg)
    if "num_sites" not in kwargs and filters.get("max_atoms") is not None:
        kwargs["num_sites"] = (1, int(filters["max_atoms"]))
    if "volume" not in kwargs:
        volume_range = _range_from_min_max(
            filters.get("min_volume"),
            filters.get("max_volume"),
            default_min=0,
        )
        if volume_range is not None:
            kwargs["volume"] = volume_range
    return kwargs


def _spacegroup_is_stable(structure, filters: dict) -> bool:
    symprecs = filters.get("symprecs") or [0.01]
    angle_tolerances = filters.get("angle_tolerances") or [5.0]
    numbers = set()
    for symprec in symprecs:
        for angle_tol in angle_tolerances:
            try:
                analyzer = SpacegroupAnalyzer(
                    structure,
                    symprec=float(symprec),
                    angle_tolerance=float(angle_tol),
                )
                numbers.add(int(analyzer.get_space_group_number()))
            except Exception:
                return False
    return len(numbers) == 1


def _lattice_signature(structure):
    lattice = structure.lattice
    lengths = sorted([float(lattice.a), float(lattice.b), float(lattice.c)])
    angles = sorted([float(lattice.alpha), float(lattice.beta), float(lattice.gamma)])
    return np.array(lengths + angles + [float(lattice.volume)], dtype=float)


def _reduced_cell_is_consistent(structure, filters: dict) -> bool:
    symprecs = filters.get("symprecs") or [0.01]
    angle_tolerances = filters.get("angle_tolerances") or [5.0]
    rtol = float(filters.get("rtol", 1e-3))
    reference = None
    for symprec in symprecs:
        for angle_tol in angle_tolerances:
            try:
                analyzer = SpacegroupAnalyzer(
                    structure,
                    symprec=float(symprec),
                    angle_tolerance=float(angle_tol),
                )
                conventional = analyzer.get_conventional_standard_structure()
                reduced = conventional.get_reduced_structure(reduction_algo="niggli")
                signature = _lattice_signature(reduced)
            except Exception:
                return False
            if reference is None:
                reference = signature
                continue
            if not np.allclose(signature, reference, rtol=rtol, atol=rtol):
                return False
    return True


def structure_passes_mp_filters(structure, download_cfg: dict) -> tuple:
    filters = _download_filter_cfg(download_cfg)
    if not filters:
        return True, "ok"

    max_atoms = filters.get("max_atoms")
    if max_atoms is not None and len(structure) > int(max_atoms):
        return False, "max_atoms"

    min_volume = _as_optional_float(filters.get("min_volume"))
    max_volume = _as_optional_float(filters.get("max_volume"))
    volume = float(structure.volume)
    if min_volume is not None and volume < min_volume:
        return False, "min_volume"
    if max_volume is not None and volume > max_volume:
        return False, "max_volume"

    if filters.get("require_spacegroup_stability", False):
        if not _spacegroup_is_stable(structure, filters):
            return False, "spacegroup_stability"

    if filters.get("check_reduced_cell_consistency", False):
        if not _reduced_cell_is_consistent(structure, filters):
            return False, "reduced_cell_consistency"

    return True, "ok"


def _process_mp_doc_task(task: dict) -> dict:
    configure_warning_filters()
    structure = task.get("structure")
    material_id = task.get("material_id")
    if structure is None:
        return {"success": False, "reason": "missing_structure", "row": None}
    if not material_id:
        return {"success": False, "reason": "missing_material_id", "row": None}

    ok, reason = structure_passes_mp_filters(
        structure,
        task.get("download_cfg", {}),
    )
    if not ok:
        return {"success": False, "reason": f"filtered_{reason}", "row": None}

    try:
        row = process_structure_to_record(
            structure=structure,
            material_id=material_id,
            source="MP",
            output_dir=task["output_dir"],
            cfg=task["cfg"],
            extra_metadata={
                "formula": task.get("formula"),
                "energy_above_hull": task.get("energy_above_hull"),
            },
        )
        return {"success": True, "reason": "success", "row": row}
    except Exception as exc:
        text = str(exc)
        if text.startswith("symmetry_failed"):
            return {"success": False, "reason": "symmetry_failed", "row": None}
        if text.startswith("xrd_failed"):
            return {"success": False, "reason": "xrd_failed", "row": None}
        return {"success": False, "reason": "process_failed", "row": None}


def _process_mp_doc_batch(task: dict) -> dict:
    configure_warning_filters()
    cfg = task["cfg"]
    download_cfg = task.get("download_cfg", {})
    output_dir = task["output_dir"]
    results = []
    for item in task.get("items", []):
        try:
            res = _process_mp_doc_task({
                "structure": item.get("structure"),
                "material_id": item.get("material_id"),
                "formula": item.get("formula"),
                "energy_above_hull": item.get("energy_above_hull"),
                "download_cfg": download_cfg,
                "output_dir": output_dir,
                "cfg": cfg,
            })
        except Exception:
            res = {"success": False, "reason": "process_failed", "row": None}
        results.append(res)
    return {
        "success": True,
        "reason": "batch",
        "results": results,
        "progress_units": len(task.get("items", [])),
    }


def _mp_summary_search_with_retries(api_key: str, kwargs: dict, mp_cfg: dict):
    from mp_api.client import MPRester
    attempts = max(1, int(mp_cfg.get("api_retries", 6) or 1))
    sleep_sec = max(0.0, float(mp_cfg.get("api_retry_sleep_sec", 60) or 0))

    for attempt in range(1, attempts + 1):
        try:
            with MPRester(api_key) as mpr:
                return mpr.materials.summary.search(**kwargs)
        except Exception as exc:
            if attempt >= attempts:
                break
            wait = sleep_sec * attempt
            logger.warning(
                "MP API request failed "
                f"({attempt}/{attempts}, {type(exc).__name__}). "
                f"Retrying in {wait:.0f}s..."
            )
            if wait > 0:
                time.sleep(wait)

    raise RuntimeError(
        "MP API did not return a valid response after "
        f"{attempts} attempt(s). Existing metadata/checkpoints were kept; "
        "rerun later to continue."
    ) from None


def process_mp_dataset(cfg: dict, output_dir):
    stats = Counter()
    rows = []
    mp_cfg = cfg.get("sources", {}).get("mp", {})
    download_cfg = get_mp_download_config()
    if mp_cfg.get("fast_filters", False):
        download_cfg = copy.deepcopy(download_cfg)
        filters = download_cfg.setdefault("filters", {})
        filters["require_spacegroup_stability"] = False
        filters["check_reduced_cell_consistency"] = False
        logger.warning("MP fast_filters enabled: skipping heavy stability checks.")
    cfg = merge_mp_download_config(cfg, download_cfg)
    kwargs = _mp_search_kwargs_from_config(download_cfg)
    max_records = mp_cfg.get("max_records")
    if max_records is not None:
        kwargs.setdefault("chunk_size", int(max_records))
        kwargs.setdefault("num_chunks", 1)
    api_key = os.environ.get("MP_API_KEY", "").strip()
    if not api_key:
        raise ValueError("MP is enabled but the MP_API_KEY environment variable is empty.")

    logger.info(f"MP summary.search kwargs from embedded dataset policy: {kwargs}")
    docs = _mp_summary_search_with_retries(api_key, kwargs, mp_cfg)
    stats["n_mp_downloaded"] = len(docs)
    logger.info(f"MP records downloaded: {len(docs)}")

    if max_records is not None:
        docs = docs[:int(max_records)]
        logger.info(f"MP debug max_records applied: {len(docs)}")

    metadata_path = _metadata_path_for_source(output_dir, "mp")
    resume_enabled = bool(mp_cfg.get("resume", True))
    checkpoint_every = max(1, int(mp_cfg.get("checkpoint_every", 512) or 512))
    processed_ids = set()
    items_to_process = []

    if resume_enabled:
        existing_df = _read_existing_metadata(metadata_path)
        if not existing_df.empty and "material_id" in existing_df.columns:
            for _, row in existing_df.iterrows():
                material_id = str(row.get("material_id", "")).lower()
                if material_id and _artifacts_exist(output_dir, material_id):
                    rows.append(row.to_dict())
                    processed_ids.add(material_id)
                elif material_id:
                    stats["n_mp_resume_metadata_missing_artifacts"] += 1
            if processed_ids:
                logger.info(f"MP resume: loaded {len(processed_ids)} rows from metadata_mp.csv")

    for doc in docs:
        material_id = str(getattr(doc, "material_id", "")).lower()
        item = {
            "structure": getattr(doc, "structure", None),
            "material_id": material_id,
            "formula": getattr(doc, "formula_pretty", None),
            "energy_above_hull": getattr(doc, "energy_above_hull", None),
        }
        if resume_enabled and material_id:
            if material_id in processed_ids:
                stats["n_mp_skipped_existing"] += 1
                continue
            if _artifacts_exist(output_dir, material_id):
                try:
                    row = reconstruct_record_from_artifacts(
                        material_id=material_id,
                        source="MP",
                        output_dir=output_dir,
                        extra_metadata={
                            "formula": item["formula"],
                            "energy_above_hull": item["energy_above_hull"],
                        },
                    )
                except Exception:
                    row = None
                if row is not None:
                    rows.append(row)
                    processed_ids.add(material_id)
                    stats["n_mp_resumed_from_artifacts"] += 1
                    continue
        items_to_process.append(item)

    if resume_enabled and rows:
        _flush_rows_to_csv(rows, metadata_path, "MP metadata")

    workers = int(mp_cfg.get("workers") or cfg.get("n_workers", 1))
    batch_size = int(mp_cfg.get("batch_size", 16) or 1)
    logger.info(
        f"MP processing workers={workers}, batch_size={batch_size}, "
        f"remaining={len(items_to_process)}, resumed={len(rows)}"
    )

    def task_iter():
        for batch in _chunked(items_to_process, batch_size):
            yield {
                "items": batch,
                "download_cfg": download_cfg,
                "output_dir": str(output_dir),
                "cfg": cfg,
                "progress_units": len(batch),
            }

    last_checkpoint_count = len(rows)
    try:
        for batch_res in iter_bounded_process_pool(
            _process_mp_doc_batch,
            task_iter(),
            total=len(items_to_process),
            workers=workers,
            desc="MP structures",
            exception_reason="process_failed",
        ):
            sub_results = batch_res.get("results")
            if not sub_results:
                stats["n_mp_process_failed"] += int(batch_res.get("progress_units", 1))
                continue
            for res in sub_results:
                if res["success"]:
                    rows.append(res["row"])
                    stats["n_mp_success"] += 1
                else:
                    reason = res["reason"]
                    if reason.startswith("filtered_"):
                        stats[f"n_mp_{reason}"] += 1
                        stats["n_mp_filtered_by_policy"] += 1
                    elif reason == "process_failed":
                        stats["n_mp_process_failed"] += 1
                    else:
                        stats[f"n_mp_{reason}"] += 1
                        stats["n_mp_process_failed"] += 1

            if len(rows) - last_checkpoint_count >= checkpoint_every:
                _flush_rows_to_csv(rows, metadata_path, "MP metadata")
                last_checkpoint_count = len(rows)
    finally:
        if rows:
            _flush_rows_to_csv(rows, metadata_path, "MP metadata")

    df = pd.DataFrame(rows)
    if "material_id" in df.columns:
        df = df.drop_duplicates("material_id", keep="last")
    df.to_csv(metadata_path, index=False)
    logger.info(f"MP metadata saved: {metadata_path} ({len(df)} rows)")
    return df, dict(stats)


def _load_structure_for_dedup(root, structure_path):
    path = _resolve_dataset_path(root, structure_path)
    return load_cod_structure(path)


def _dedup_key(row, tol: float):
    formula = row.get("reduced_formula") or row.get("formula") or row.get("anonymous_formula") or ""
    sg = row.get("space_group_number", -1)
    try:
        sg = int(sg)
    except Exception:
        sg = -1
    try:
        vbin = int(round(float(row.get("volume_per_atom")) / tol))
    except Exception:
        vbin = -1
    return f"{formula}|{sg}|{vbin}"


def deduplicate_metadata(metadata_df: pd.DataFrame, output_dir, cfg: dict):
    root = Path(output_dir)
    dedup_cfg = cfg.get("dedup", {})
    stats = {
        "n_cod_before": int((metadata_df.get("source", pd.Series(dtype=str)).astype(str).str.upper() == "COD").sum()),
        "n_mp_before": int((metadata_df.get("source", pd.Series(dtype=str)).astype(str).str.upper() == "MP").sum()),
        "n_all_before": int(len(metadata_df)),
        "n_cod_internal_duplicates": 0,
        "n_mp_internal_duplicates": 0,
        "n_mp_removed_due_to_cod_duplicate": 0,
    }

    if metadata_df.empty:
        metadata_df.to_csv(root / "metadata_dedup.csv", index=False)
        metadata_df.to_csv(root / "metadata.csv", index=False)
        stats["n_all_after"] = 0
        return metadata_df, stats

    if not dedup_cfg.get("enabled", True):
        metadata_df.to_csv(root / "metadata_dedup.csv", index=False)
        metadata_df.to_csv(root / "metadata.csv", index=False)
        stats["n_all_after"] = int(len(metadata_df))
        return metadata_df, stats

    work = metadata_df.copy()
    prefer = str(dedup_cfg.get("prefer_source", "COD")).upper()
    work["_input_order"] = np.arange(len(work))
    work["_source_priority"] = work["source"].astype(str).str.upper().apply(
        lambda s: 0 if s == prefer else 1
    )
    if "energy_above_hull" in work.columns:
        work["_energy"] = pd.to_numeric(work["energy_above_hull"], errors="coerce").fillna(1e9)
    else:
        work["_energy"] = 1e9
    work = work.sort_values(["_source_priority", "_energy", "_input_order"]).reset_index(drop=True)

    tol = float(dedup_cfg.get("volume_per_atom_tol", 0.05))
    if dedup_cfg.get("formula_sg_volume_prefilter", True):
        work["_dedup_key"] = work.apply(lambda row: _dedup_key(row, tol), axis=1)
    else:
        work["_dedup_key"] = "all"

    keep_indices = []
    duplicate_records = []
    if dedup_cfg.get("structure_matcher", True):
        matcher = StructureMatcher(
            ltol=float(dedup_cfg.get("ltol", 0.2)),
            stol=float(dedup_cfg.get("stol", 0.3)),
            angle_tol=float(dedup_cfg.get("angle_tol", 5)),
            primitive_cell=True,
            scale=True,
            attempt_supercell=False,
        )
        for _, group in tqdm(work.groupby("_dedup_key", sort=False), desc="Dedup"):
            representatives = []
            for idx, row in group.iterrows():
                try:
                    structure = _load_structure_for_dedup(root, row["structure_path"])
                except Exception:
                    keep_indices.append(idx)
                    continue
                matched = None
                for rep_idx, rep_source, rep_structure in representatives:
                    try:
                        if matcher.fit(structure, rep_structure):
                            matched = (rep_idx, rep_source)
                            break
                    except Exception:
                        continue
                if matched is None:
                    keep_indices.append(idx)
                    representatives.append((idx, str(row["source"]).upper(), structure))
                else:
                    duplicate_records.append({
                        "source": str(row["source"]).upper(),
                        "matched_source": matched[1],
                    })
    else:
        kept = work.drop_duplicates("_dedup_key", keep="first")
        keep_indices = list(kept.index)
        dropped = work.loc[~work.index.isin(keep_indices)]
        duplicate_records = [
            {"source": str(row["source"]).upper(), "matched_source": "unknown"}
            for _, row in dropped.iterrows()
        ]

    for rec in duplicate_records:
        source = rec["source"]
        matched = rec["matched_source"]
        if source == "COD" and matched == "COD":
            stats["n_cod_internal_duplicates"] += 1
        elif source == "MP" and matched == "MP":
            stats["n_mp_internal_duplicates"] += 1
        elif source == "MP" and matched == "COD":
            stats["n_mp_removed_due_to_cod_duplicate"] += 1

    dedup_df = work.loc[sorted(set(keep_indices))].copy()
    drop_cols = [c for c in dedup_df.columns if c.startswith("_")]
    dedup_df = dedup_df.drop(columns=drop_cols)
    dedup_df = dedup_df.sort_values(["source", "material_id"]).reset_index(drop=True)
    stats["n_all_after"] = int(len(dedup_df))

    dedup_df.to_csv(root / "metadata_dedup.csv", index=False)
    dedup_df.to_csv(root / "metadata.csv", index=False)
    logger.info(f"Dedup complete: {len(metadata_df)} -> {len(dedup_df)}")
    return dedup_df, stats


def _split_counts(df, column):
    if df.empty or column not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[column].value_counts().to_dict().items()}


def split_dataset_source_aware(metadata_df: pd.DataFrame, output_dir, cfg: dict):
    split_cfg = cfg.get("split", {})
    root = Path(output_dir)
    if metadata_df.empty:
        for name in ("train.csv", "val.csv", "test.csv", "test_open.csv"):
            metadata_df.to_csv(root / name, index=False)
        return {"train_size": 0, "val_size": 0, "test_size": 0}

    if not split_cfg.get("enabled", True):
        metadata_df.to_csv(root / "train.csv", index=False)
        return {"train_size": int(len(metadata_df)), "val_size": 0, "test_size": 0}

    seed = int(split_cfg.get("random_seed", 42))
    rng = np.random.default_rng(seed)
    train_ratio = float(split_cfg.get("train_ratio", 0.8))
    val_ratio = float(split_cfg.get("val_ratio", 0.1))
    test_ratio = float(split_cfg.get("test_ratio", 0.1))
    stratify_by = list(split_cfg.get("stratify_by", ["source", "crystal_system"]))

    materials = metadata_df.drop_duplicates("base_material_id").copy()
    for col in stratify_by:
        if col not in materials.columns:
            materials[col] = "unknown"

    train_ids, val_ids, test_ids = set(), set(), set()
    for _, group in materials.groupby(stratify_by, dropna=False, sort=False):
        ids = group["base_material_id"].astype(str).tolist()
        rng.shuffle(ids)
        n = len(ids)
        if n == 1:
            train_ids.add(ids[0])
            continue
        if n == 2:
            train_ids.add(ids[0])
            test_ids.add(ids[1])
            continue

        n_val = max(1, int(round(n * val_ratio)))
        n_test = max(1, int(round(n * test_ratio)))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train = 1
            overflow = n_train + n_val + n_test - n
            if n_test > 1:
                n_test -= overflow
            else:
                n_val = max(0, n_val - overflow)

        train_ids.update(ids[:n_train])
        val_ids.update(ids[n_train:n_train + n_val])
        test_ids.update(ids[n_train + n_val:n_train + n_val + n_test])

    all_ids = set(materials["base_material_id"].astype(str))
    assigned = train_ids | val_ids | test_ids
    train_ids.update(all_ids - assigned)

    key_series = metadata_df["base_material_id"].astype(str)
    train_df = metadata_df[key_series.isin(train_ids)]
    val_df = metadata_df[key_series.isin(val_ids)]
    test_df = metadata_df[key_series.isin(test_ids)]

    train_df.to_csv(root / "train.csv", index=False)
    val_df.to_csv(root / "val.csv", index=False)
    test_df.to_csv(root / "test.csv", index=False)
    test_df.to_csv(root / "test_open.csv", index=False)

    stats = {
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "source_distribution": {
            "train": _split_counts(train_df, "source"),
            "val": _split_counts(val_df, "source"),
            "test": _split_counts(test_df, "source"),
        },
        "crystal_system_distribution": {
            "train": _split_counts(train_df, "crystal_system"),
            "val": _split_counts(val_df, "crystal_system"),
            "test": _split_counts(test_df, "crystal_system"),
        },
        "space_group_distribution_summary": {
            "train_n_space_groups": int(train_df["space_group_number"].nunique()) if "space_group_number" in train_df else 0,
            "val_n_space_groups": int(val_df["space_group_number"].nunique()) if "space_group_number" in val_df else 0,
            "test_n_space_groups": int(test_df["space_group_number"].nunique()) if "space_group_number" in test_df else 0,
        },
    }
    logger.info(
        f"Split complete: train={len(train_df)} val={len(val_df)} test={len(test_df)}"
    )
    return stats


def run_mixed_pipeline(cfg: dict = CONFIG) -> None:
    cfg = copy.deepcopy(cfg)
    output_dir = Path(cfg["output_dir"])
    init_dataset_dir(str(output_dir))
    configure_build_logging(output_dir)

    cfg = merge_mp_download_config(cfg, get_mp_download_config())

    all_stats = {}
    mp_df = pd.DataFrame()
    cod_df = pd.DataFrame()

    if cfg.get("sources", {}).get("cod", {}).get("enabled", False):
        cod_root = str(cfg["sources"]["cod"].get("cod_root") or "").strip()
        if not cod_root:
            raise ValueError("COD is enabled but cod_root is empty.")
        cod_df, cod_stats = process_cod_dataset(cod_root, output_dir, cfg)
        all_stats["cod"] = cod_stats
    else:
        cod_meta_path = _metadata_path_for_source(output_dir, "cod")
        cod_df = _read_existing_metadata(cod_meta_path)
        all_stats["cod"] = {
            "disabled": True,
            "loaded_existing_rows": int(len(cod_df)),
            "metadata_path": str(cod_meta_path),
        }
        if not cod_df.empty:
            logger.info(f"Loaded existing COD metadata: {cod_meta_path} ({len(cod_df)} rows)")
    cod_df, cod_reconcile_stats = reconcile_metadata_with_artifacts(cod_df, output_dir, "COD")
    if not cod_df.empty:
        cod_df.to_csv(_metadata_path_for_source(output_dir, "cod"), index=False)
    all_stats["cod_artifact_reconcile"] = cod_reconcile_stats

    if cfg.get("sources", {}).get("mp", {}).get("enabled", False):
        mp_df, mp_stats = process_mp_dataset(cfg, output_dir)
        all_stats["mp"] = mp_stats
    else:
        mp_meta_path = _metadata_path_for_source(output_dir, "mp")
        mp_df = _read_existing_metadata(mp_meta_path)
        all_stats["mp"] = {
            "disabled": True,
            "loaded_existing_rows": int(len(mp_df)),
            "metadata_path": str(mp_meta_path),
        }
        if not mp_df.empty:
            logger.info(f"Loaded existing MP metadata: {mp_meta_path} ({len(mp_df)} rows)")
    mp_df, mp_reconcile_stats = reconcile_metadata_with_artifacts(mp_df, output_dir, "MP")
    if not mp_df.empty:
        mp_df.to_csv(_metadata_path_for_source(output_dir, "mp"), index=False)
    all_stats["mp_artifact_reconcile"] = mp_reconcile_stats

    metadata_all = pd.concat([cod_df, mp_df], ignore_index=True)
    metadata_all.to_csv(output_dir / "metadata_all.csv", index=False)
    if metadata_all.empty:
        logger.warning("No MP/COD records were built. Empty split files will be written.")

    metadata_dedup, dedup_stats = deduplicate_metadata(metadata_all, output_dir, cfg)
    all_stats["dedup"] = dedup_stats

    split_stats = split_dataset_source_aware(metadata_dedup, output_dir, cfg)
    all_stats["split"] = split_stats
    save_build_stats(output_dir, all_stats)
    logger.info(f"Build stats saved: {output_dir / 'build_stats.json'}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Build MP and COD structure artifacts."
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--include_mp", dest="include_mp", action="store_true", default=None)
    parser.add_argument("--no_mp", dest="include_mp", action="store_false")
    parser.add_argument("--include_cod", dest="include_cod", action="store_true", default=None)
    parser.add_argument("--no_cod", dest="include_cod", action="store_false")
    parser.add_argument("--cod_root", type=str, default=None)
    parser.add_argument("--cod_max_files", type=int, default=None)
    parser.add_argument("--cod_batch_size", type=int, default=None)
    parser.add_argument("--mp_max_records", type=int, default=None)
    parser.add_argument("--mp_batch_size", type=int, default=None)
    parser.add_argument("--mp_checkpoint_every", type=int, default=None)
    parser.add_argument("--mp_api_retries", type=int, default=None)
    parser.add_argument("--mp_api_retry_sleep_sec", type=float, default=None)
    parser.add_argument("--resume", dest="resume", action="store_true", default=None)
    parser.add_argument("--no_resume", "--no-resume", dest="resume", action="store_false")
    parser.add_argument("--fast_mp_filters", action="store_true", default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--dedup", dest="dedup", action="store_true", default=None)
    parser.add_argument("--no_dedup", "--no-dedup", dest="dedup", action="store_false")
    parser.add_argument("--split", dest="split", action="store_true", default=None)
    parser.add_argument("--no_split", "--no-split", dest="split", action="store_false")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def apply_cli_overrides(cfg: dict, args) -> dict:
    cfg = copy.deepcopy(cfg)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.include_mp is not None:
        cfg["sources"]["mp"]["enabled"] = bool(args.include_mp)
    if args.include_cod is not None:
        cfg["sources"]["cod"]["enabled"] = bool(args.include_cod)
    if args.cod_root:
        cfg["sources"]["cod"]["cod_root"] = args.cod_root
    if args.cod_max_files is not None:
        cfg["sources"]["cod"]["max_files"] = args.cod_max_files
    if args.cod_batch_size is not None:
        cfg["sources"]["cod"]["batch_size"] = args.cod_batch_size
    if args.mp_max_records is not None:
        cfg["sources"]["mp"]["max_records"] = args.mp_max_records
    if args.mp_batch_size is not None:
        cfg["sources"]["mp"]["batch_size"] = args.mp_batch_size
    if args.mp_checkpoint_every is not None:
        cfg["sources"]["mp"]["checkpoint_every"] = args.mp_checkpoint_every
    if args.mp_api_retries is not None:
        cfg["sources"]["mp"]["api_retries"] = args.mp_api_retries
    if args.mp_api_retry_sleep_sec is not None:
        cfg["sources"]["mp"]["api_retry_sleep_sec"] = args.mp_api_retry_sleep_sec
    if args.resume is not None:
        cfg["sources"]["mp"]["resume"] = bool(args.resume)
    if args.num_workers is not None:
        cfg["n_workers"] = args.num_workers
        cfg["sources"]["cod"]["workers"] = args.num_workers
        cfg["sources"]["mp"]["workers"] = args.num_workers
    if args.fast_mp_filters is not None:
        cfg["sources"]["mp"]["fast_filters"] = bool(args.fast_mp_filters)
    if args.dedup is not None:
        cfg["dedup"]["enabled"] = bool(args.dedup)
    if args.split is not None:
        cfg["split"]["enabled"] = bool(args.split)
    if args.seed is not None:
        cfg["split"]["random_seed"] = args.seed
    return cfg


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = copy.deepcopy(CONFIG)
    cfg = apply_cli_overrides(cfg, args)
    run_mixed_pipeline(cfg)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
