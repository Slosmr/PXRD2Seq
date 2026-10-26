"""Build the labeled RRUFF base set and unlabeled adaptation set.

The labeled set requires cell metadata, a matching DIF record, resolvable
space group, symmetry consistency, XY-DIF cell agreement, and a non-calculated
experimental profile. The adapt set follows a separate permissive path: any
parseable XY pattern with at least 15 degrees of coverage is retained unless
its RRUFF id is already present in the labeled set.

Both datasets use 8500-point signals by default. Labeled samples also receive
Niggli-reduced lattice labels and JSON reports.
"""

import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks, peak_widths

from rruff_parser import (
    parse_xy_file, parse_dif_file,
    sg_symbol_to_info, get_laue_from_sg_number, get_bravais,
    is_cu_ka,
    resample_pattern,
    niggli_reduce_cell,
    extract_rruff_id_from_name,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def check_symmetry_consistency(cell: dict, rtol: float = 1e-3) -> bool:
    """
    验证 cell 参数与晶系声明一致。对应论文：
        "verified that the Niggli reduced cell was consistent with crystal
         system symmetry within a relative tolerance of 10^-3"
    """
    cs = cell.get("crystal_system", "").lower()
    a, b, c = cell["a"], cell["b"], cell["c"]
    al, be, ga = cell["alpha"], cell["beta"], cell["gamma"]

    def _close(x, y, rel=rtol):
        return abs(x - y) <= rel * max(abs(x), abs(y), 1e-9)

    def _angle_close(x, y, atol=0.5):
        return abs(x - y) <= atol

    if cs == "cubic":
        return (_close(a, b) and _close(b, c)
                and _angle_close(al, 90) and _angle_close(be, 90)
                and _angle_close(ga, 90))
    if cs == "tetragonal":
        return (_close(a, b)
                and _angle_close(al, 90) and _angle_close(be, 90)
                and _angle_close(ga, 90))
    if cs == "orthorhombic":
        return (_angle_close(al, 90) and _angle_close(be, 90)
                and _angle_close(ga, 90))
    if cs == "hexagonal":
        return (_close(a, b)
                and _angle_close(al, 90) and _angle_close(be, 90)
                and _angle_close(ga, 120))
    if cs == "trigonal":
        hp = (_close(a, b)
              and _angle_close(al, 90) and _angle_close(be, 90)
              and _angle_close(ga, 120))
        hr = (_close(a, b) and _close(b, c)
              and _angle_close(al, be) and _angle_close(be, ga))
        return hp or hr
    if cs == "monoclinic":
        b_unique = _angle_close(al, 90) and _angle_close(ga, 90)
        c_unique = _angle_close(al, 90) and _angle_close(be, 90)
        a_unique = _angle_close(be, 90) and _angle_close(ga, 90)
        return b_unique or c_unique or a_unique
    if cs == "triclinic":
        return True
    return False


def check_xy_dif_match(xy_cell: dict, dif_cell: tuple,
                       length_rtol: float = 0.01,
                       angle_atol: float = 0.5) -> bool:
    """验证 XY header 的晶胞与 DIF 的晶胞数值匹配。"""
    da, db, dc, dal, dbe, dga = dif_cell
    for xy_len, dif_len in [(xy_cell["a"], da), (xy_cell["b"], db), (xy_cell["c"], dc)]:
        if abs(xy_len - dif_len) > length_rtol * max(abs(xy_len), abs(dif_len), 1e-9):
            return False
    for xy_ang, dif_ang in [(xy_cell["alpha"], dal),
                             (xy_cell["beta"], dbe),
                             (xy_cell["gamma"], dga)]:
        if abs(xy_ang - dif_ang) > angle_atol:
            return False
    return True


def is_calculated_profile(xy_info: dict) -> bool:
    desc = xy_info.get("description", "").lower()
    return "calculated" in desc


CONFIG = {
    "theta_range": (4.0, 90.0),
    "signal_length": 8500,
    "max_peaks": 20,
    "peak_dim": 14,
    "wavelength_ang": 1.5406,
    "peak_prominence_pct": 3.0,
    "peak_min_distance_deg": 0.09,
    "min_2theta_span": 15.0,
}


def detect_peaks_from_signal(y: np.ndarray,
                             x: np.ndarray,
                             cfg: dict) -> np.ndarray:
    """Extract a (max_peaks, 8) feature matrix from an intensity array.

The minimum peak separation is configured in degrees and converted to
sample indices using the spacing of x."""
    max_peaks = cfg["max_peaks"]
    theta_lo, theta_hi = cfg["theta_range"]
    wavelength = cfg["wavelength_ang"]

    features = np.zeros((max_peaks, 8), dtype=np.float32)
    if y.max() <= 0:
        return features

    y_norm = y / y.max() * 100.0
    dx = x[1] - x[0]
    min_distance_pix = max(1, int(round(cfg["peak_min_distance_deg"] / dx)))
    peaks, _ = find_peaks(
        y_norm,
        prominence=cfg["peak_prominence_pct"],
        distance=min_distance_pix,
    )
    if len(peaks) == 0:
        return features

    widths_pix, _, _, _ = peak_widths(y_norm, peaks, rel_height=0.5)
    fwhms_deg = widths_pix * dx

    peak_2theta = x[peaks]
    peak_intensity = y[peaks]

    if len(peaks) > max_peaks:
        top_idx = np.argsort(peak_intensity)[::-1][:max_peaks]
        top_idx = np.sort(top_idx)
        peak_2theta = peak_2theta[top_idx]
        peak_intensity = peak_intensity[top_idx]
        fwhms_deg = fwhms_deg[top_idx]

    n_valid = len(peak_2theta)
    for i in range(n_valid):
        t = peak_2theta[i]
        inten = peak_intensity[i]
        fwhm = fwhms_deg[i]
        d = wavelength / (2 * np.sin(np.radians(t / 2)) + 1e-10)
        area = inten * fwhm
        delta_left = t - (peak_2theta[i - 1] if i > 0 else theta_lo)
        delta_right = (peak_2theta[i + 1] if i < n_valid - 1 else theta_hi) - t
        features[i] = [t, inten, fwhm, area, d, delta_left, delta_right, 1.0]
    return features


def expand_peak_matrix_with_observed_mask(peak_matrix: np.ndarray,
                                          peak_dim: int = 14) -> np.ndarray:
    """Expand the peak matrix to the configured feature width, including observation masks."""
    if peak_matrix.shape[1] == peak_dim:
        result = peak_matrix.copy().astype(np.float32)
        valid_rows = result[:, 7] == 1.0
        result[:, 8:14] = (result[:, 8:14] > 0.5).astype(np.float32)
        result[~valid_rows, 8:14] = 0.0
        return result
    if peak_matrix.shape[1] != 8:
        raise ValueError(f"peak_matrix expected 8 or {peak_dim} columns, got {peak_matrix.shape}")
    result = np.zeros((peak_matrix.shape[0], peak_dim), dtype=np.float32)
    result[:, :8] = peak_matrix.astype(np.float32)
    valid_rows = result[:, 7] == 1.0
    result[valid_rows, 8:14] = 1.0
    return result


def load_heldout_rruff_ids(path: Path | None) -> tuple[set[str], set[str]]:
    """读取已有 RRUFF test CSV，返回 (full_ids, base_ids)。"""
    if path is None or not Path(path).is_file():
        return set(), set()
    df = pd.read_csv(path)
    full_ids, base_ids = set(), set()
    candidate_cols = [
        c for c in ("rruff_id", "rruff_id_full", "material_id", "base_material_id")
        if c in df.columns
    ]
    for col in candidate_cols:
        for raw in df[col].dropna().astype(str):
            full, base = extract_rruff_id_from_name(raw)
            if full:
                full_ids.add(full)
            if base:
                base_ids.add(base)
    return full_ids, base_ids


def build_report(name: str, formula: str,
                 cell: dict,
                 sg_symbol: str = None,
                 sg_number: int = -1,
                 laue_class: str = "unknown",
                 bravais: str = "unknown",
                 peak_matrix: np.ndarray = None) -> dict:
    """产出与 generate_report() 结构一致的 JSON。"""
    top3 = []
    if peak_matrix is not None:
        valid = peak_matrix[peak_matrix[:, 7] == 1.0]
        if len(valid) > 0:
            top_idx = np.argsort(valid[:, 1])[::-1][:3]
            top3 = [
                {"two_theta": float(valid[i, 0]),
                 "d_spacing": float(valid[i, 4])}
                for i in top_idx
            ]
    while len(top3) < 3:
        top3.append({"two_theta": 0.0, "d_spacing": 0.0})

    n_valid_peaks = int((peak_matrix[:, 7] == 1.0).sum())\
        if peak_matrix is not None else 0

    ctx = {
        "formula": formula or name,
        "crystal_system": cell.get("crystal_system", "unknown"),
        "space_group": sg_symbol if sg_symbol else "unknown",
        "laue_class": laue_class,
        "bravais_lattice": bravais,
        "a": round(cell["a"], 4),
        "b": round(cell["b"], 4),
        "c": round(cell["c"], 4),
        "alpha": round(cell["alpha"], 3),
        "beta": round(cell["beta"], 3),
        "gamma": round(cell["gamma"], 3),
        "volume": round(cell.get("volume", cell["a"] * cell["b"] * cell["c"]), 4),
        "n_valid_peaks": n_valid_peaks,
        "top3_peaks": top3,
    }

    text = (
        f"Phase: {ctx['formula']}. "
        f"Crystal system: {ctx['crystal_system']}. "
        f"Space group: {ctx['space_group']}. "
        f"Laue class: {ctx['laue_class']}. "
        f"Bravais lattice: {ctx['bravais_lattice']}. "
        f"a={ctx['a']}Å, b={ctx['b']}Å, c={ctx['c']}Å; "
        f"α={ctx['alpha']}°, β={ctx['beta']}°, γ={ctx['gamma']}°; "
        f"V={ctx['volume']}Å³. "
        f"Peaks detected: {ctx['n_valid_peaks']}."
    )

    kv_parts = [
        f"crystal_system: {ctx['crystal_system']}",
        f"laue_class: {ctx['laue_class']}",
        f"bravais_lattice: {ctx['bravais_lattice']}",
        f"space_group: {ctx['space_group']}",
        f"a: {ctx['a']}", f"b: {ctx['b']}", f"c: {ctx['c']}",
        f"alpha: {ctx['alpha']}", f"beta: {ctx['beta']}", f"gamma: {ctx['gamma']}",
        f"volume: {ctx['volume']}",
        f"n_peaks: {ctx['n_valid_peaks']}",
    ]
    for i, p in enumerate(top3):
        kv_parts.append(f"peak{i+1}_2theta: {p['two_theta']:.2f}")
        kv_parts.append(f"peak{i+1}_d: {p['d_spacing']:.3f}")
    kv_text = " | ".join(kv_parts)

    return {
        **ctx,
        "material_id": name,
        "space_group_number": sg_number,
        "text": text,
        "kv_text": kv_text,
    }


def build_adapt_dataset(xy_files: list,
                        output_dir: Path,
                        cfg: dict,
                        heldout_full_ids: set,
                        heldout_base_ids: set) -> list:
    """
    构建 rruff_adapt 数据集：XY_Processed 全集 - rruff_labeled。

    宽松路径：
      - 不要求 cell / DIF / 对称性 / XY-DIF 匹配
      - 计算谱保留
      - 唯一过滤：parse 成功 + 2θ 覆盖 ≥ min_2theta_span
      - 排除：rruff_labeled 中的 full_id 或 base_id

    落盘：signals/<mid>.npy 和 peaks/<mid>.npy（不写 reports/，因为无 cell/SG 标签）。
    若该 material_id 已被严格管线写过，本函数会覆盖写入相同字节内容（确定性）。

    Returns
    -------
    rows_adapt : list[dict]
        每行只包含信号路径 + 基础元信息。
    """
    rows_adapt = []
    n_parse_failed = 0
    n_excluded_heldout = 0
    n_signal_too_short = 0
    n_dup_id = 0
    seen_ids = set()

    x_grid = np.linspace(*cfg["theta_range"], cfg["signal_length"])

    for p in tqdm(xy_files, desc="构建 adapt 集"):
        xy = parse_xy_file(p)
        if xy is None:
            n_parse_failed += 1
            continue


        if (xy["rruff_id_full"] in heldout_full_ids
                or xy["rruff_id_base"] in heldout_base_ids):
            n_excluded_heldout += 1
            continue


        t_lo, t_hi = xy["two_theta"].min(), xy["two_theta"].max()
        overlap_lo = max(t_lo, cfg["theta_range"][0])
        overlap_hi = min(t_hi, cfg["theta_range"][1])
        if overlap_hi - overlap_lo < cfg["min_2theta_span"]:
            n_signal_too_short += 1
            continue

        material_id = f"RRUFF_{xy['rruff_id_full']}"
        if material_id in seen_ids:
            n_dup_id += 1
            continue
        seen_ids.add(material_id)


        y_resampled = resample_pattern(
            xy["two_theta"], xy["intensity"],
            target_range=cfg["theta_range"],
            target_length=cfg["signal_length"],
        ).astype(np.float32)


        peak_matrix = detect_peaks_from_signal(y_resampled, x_grid, cfg)
        peak_matrix = expand_peak_matrix_with_observed_mask(
            peak_matrix, peak_dim=cfg.get("peak_dim", 14)
        )

        signal_path = output_dir / "signals" / f"{material_id}.npy"
        peak_path = output_dir / "peaks" / f"{material_id}.npy"
        np.save(signal_path, y_resampled)
        np.save(peak_path, peak_matrix)

        rows_adapt.append({
            "material_id": material_id,
            "rruff_id": xy["rruff_id_full"],
            "rruff_base_id": xy["rruff_id_base"],
            "name": xy["name"],
            "formula": xy["formula"],
            "signal_path": f"signals/{material_id}.npy",
            "peak_path": f"peaks/{material_id}.npy",
        })

    logger.info(
        f"adapt 集构建完成："
        f"parse 失败 {n_parse_failed}, "
        f"sg held-out 排除 {n_excluded_heldout}, "
        f"信号过短 {n_signal_too_short}, "
        f"重复 ID {n_dup_id}, "
        f"入选 {len(rows_adapt)}"
    )
    return rows_adapt


def process_rruff(xy_dir: Path, dif_dir: Path, output_dir: Path,
                  cfg: dict = CONFIG,
                  strict_cu_ka: bool = True,
                  require_dif: bool = True,
                  check_symmetry: bool = True,
                  check_xy_dif: bool = True,
                  exclude_calculated: bool = True,
                  labeled_csv_name: str = "rruff_labeled.csv",
                  adapt_csv_name: str = "rruff_adapt.csv") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    输出 rruff_labeled.csv + rruff_adapt.csv。
    严格管线（对称性/XY-DIF/SG 解析）→ sg
    宽松独立路径（仅信号）→ adapt = XY 全集 - sg
    """
    output_dir = Path(output_dir)
    (output_dir / "signals").mkdir(parents=True, exist_ok=True)
    (output_dir / "peaks").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)

    logger.info("启用的检查（仅作用于严格管线 / sg 集）：")
    logger.info(f"  check_symmetry     = {check_symmetry}")
    logger.info(f"  check_xy_dif       = {check_xy_dif}")
    logger.info(f"  exclude_calculated = {exclude_calculated}")
    logger.info(f"  require_dif        = {require_dif}")
    logger.info(f"  strict_cu_ka       = {strict_cu_ka}")


    logger.info(f"扫描 DIF 目录：{dif_dir}")
    dif_files = sorted(set(Path(dif_dir).rglob("*.txt"))
                        | set(Path(dif_dir).rglob("*.dif")))
    logger.info(f"发现 {len(dif_files)} 个 DIF 候选文件")

    dif_index = {}
    n_dif_ok, n_dif_non_cuka, n_dif_bad = 0, 0, 0
    for p in tqdm(dif_files, desc="解析 DIF"):
        info = parse_dif_file(p)
        if info is None:
            n_dif_bad += 1
            continue
        if info["rruff_id_base"] is None:
            n_dif_bad += 1
            continue

        if strict_cu_ka and info["wavelength"] is not None and not is_cu_ka(info["wavelength"]):
            n_dif_non_cuka += 1
            continue
        dif_index.setdefault(info["rruff_id_base"], info)
        n_dif_ok += 1
    logger.info(
        f"DIF 解析：成功 {n_dif_ok}（独立 base_id {len(dif_index)} 个），"
        f"非 Cu Kα 跳过 {n_dif_non_cuka}，解析失败 {n_dif_bad}"
    )


    logger.info(f"扫描 XY 目录：{xy_dir}")
    xy_files = sorted(Path(xy_dir).rglob("*.txt"))
    logger.info(f"发现 {len(xy_files)} 个 XY 文件")

    rows_sg = []
    stats = {k: 0 for k in [
        "xy_parsed",
        "xy_skip_no_cell",
        "xy_skip_2theta_range",
        "xy_skip_symmetry_inconsistent",
        "xy_skip_xy_dif_mismatch",
        "xy_skip_calculated",
        "xy_skip_no_dif_required",
        "xy_no_dif",
        "xy_with_dif",
        "xy_with_sg_resolved",
        "n_calculated_kept",
        "success_sg",
    ]}

    for p in tqdm(xy_files, desc="处理 XY (严格管线)"):
        xy = parse_xy_file(p)
        if xy is None:
            continue
        stats["xy_parsed"] += 1


        if xy["cell"] is None:
            stats["xy_skip_no_cell"] += 1
            continue


        is_calc = is_calculated_profile(xy)
        if exclude_calculated and is_calc:
            stats["xy_skip_calculated"] += 1
            continue
        if is_calc:
            stats["n_calculated_kept"] += 1


        if check_symmetry and not check_symmetry_consistency(xy["cell"]):
            stats["xy_skip_symmetry_inconsistent"] += 1
            continue


        t_lo, t_hi = xy["two_theta"].min(), xy["two_theta"].max()
        overlap_lo = max(t_lo, cfg["theta_range"][0])
        overlap_hi = min(t_hi, cfg["theta_range"][1])
        if overlap_hi - overlap_lo < cfg["min_2theta_span"]:
            stats["xy_skip_2theta_range"] += 1
            continue


        dif = dif_index.get(xy["rruff_id_base"])
        if dif is None:
            stats["xy_no_dif"] += 1
            if require_dif:
                stats["xy_skip_no_dif_required"] += 1
                continue
        else:

            if check_xy_dif and not check_xy_dif_match(xy["cell"], dif["cell"]):
                stats["xy_skip_xy_dif_mismatch"] += 1
                continue
            stats["xy_with_dif"] += 1


        sg_number = -1
        sg_symbol = None
        crystal_system_from_sg = None
        if dif is not None:
            sg_number, crystal_system_from_sg = sg_symbol_to_info(dif["space_group"])
            sg_number = sg_number if sg_number is not None else -1
            sg_symbol = dif["space_group"]
            if sg_number > 0:
                stats["xy_with_sg_resolved"] += 1


        if sg_number <= 0:
            continue


        _CS_ALIAS = {"rhombohedral": "trigonal"}
        raw_cs = xy["cell"]["crystal_system"].strip().lower()
        crystal_system = _CS_ALIAS.get(raw_cs, raw_cs)
        if crystal_system_from_sg and crystal_system_from_sg != crystal_system:
            crystal_system = crystal_system_from_sg


        laue_class = get_laue_from_sg_number(sg_number)
        bravais = get_bravais(crystal_system, sg_symbol) if sg_symbol else "unknown"


        cell_niggli = niggli_reduce_cell(xy["cell"])


        y_resampled = resample_pattern(
            xy["two_theta"], xy["intensity"],
            target_range=cfg["theta_range"],
            target_length=cfg["signal_length"],
        ).astype(np.float32)
        x_grid = np.linspace(*cfg["theta_range"], cfg["signal_length"])
        peak_matrix = detect_peaks_from_signal(y_resampled, x_grid, cfg)
        peak_matrix = expand_peak_matrix_with_observed_mask(
            peak_matrix, peak_dim=cfg.get("peak_dim", 14)
        )
        n_valid_peaks = int((peak_matrix[:, 7] == 1.0).sum())


        material_id = f"RRUFF_{xy['rruff_id_full']}"
        report = build_report(
            name=material_id,
            formula=xy["formula"] or xy["name"],
            cell=cell_niggli,
            sg_symbol=sg_symbol,
            sg_number=sg_number,
            laue_class=laue_class,
            bravais=bravais,
            peak_matrix=peak_matrix,
        )
        signal_path = output_dir / "signals" / f"{material_id}.npy"
        peak_path = output_dir / "peaks" / f"{material_id}.npy"
        report_path = output_dir / "reports" / f"{material_id}.json"
        np.save(signal_path, y_resampled)
        np.save(peak_path, peak_matrix)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        rows_sg.append({
            "material_id": material_id,
            "base_material_id": material_id,
            "rruff_id": xy["rruff_id_full"],
            "rruff_base_id": xy["rruff_id_base"],
            "name": xy["name"],
            "formula": xy["formula"],
            "is_calculated": is_calc,
            "crystal_system": crystal_system,
            "space_group": sg_symbol or "",
            "space_group_number": sg_number,
            "laue_class": laue_class,
            "bravais_lattice": bravais,
            "a": cell_niggli["a"], "b": cell_niggli["b"], "c": cell_niggli["c"],
            "alpha": cell_niggli["alpha"],
            "beta": cell_niggli["beta"],
            "gamma": cell_niggli["gamma"],
            "volume": cell_niggli["volume"],
            "n_valid_peaks": n_valid_peaks,
            "signal_path": f"signals/{material_id}.npy",
            "peak_path": f"peaks/{material_id}.npy",
            "report_path": f"reports/{material_id}.json",
        })
        stats["success_sg"] += 1


    df_sg = pd.DataFrame(rows_sg)
    sg_csv_path = output_dir / labeled_csv_name
    df_sg.to_csv(sg_csv_path, index=False)
    logger.info(f"  {labeled_csv_name} {len(df_sg):>5d} rows -> {sg_csv_path}")


    logger.info("=" * 60)
    logger.info("严格管线统计：")
    for k, v in stats.items():
        logger.info(f"  {k:35s}: {v}")
    if not df_sg.empty:
        logger.info(f"\nsg 集晶系分布：\n"
                    f"{df_sg['crystal_system'].value_counts().to_string()}")


    logger.info("=" * 60)
    logger.info("开始构建 adapt 集（XY 全集 - labeled base set）...")
    heldout_full_ids, heldout_base_ids = load_heldout_rruff_ids(sg_csv_path)
    logger.info(
        f"sg held-out: full_id={len(heldout_full_ids)}, "
        f"base_id={len(heldout_base_ids)}"
    )

    rows_adapt = build_adapt_dataset(
        xy_files=xy_files,
        output_dir=output_dir,
        cfg=cfg,
        heldout_full_ids=heldout_full_ids,
        heldout_base_ids=heldout_base_ids,
    )
    df_adapt = pd.DataFrame(rows_adapt)
    adapt_csv_path = output_dir / adapt_csv_name
    df_adapt.to_csv(adapt_csv_path, index=False)

    logger.info("=" * 60)
    logger.info("最终输出：")
    logger.info(f"  {labeled_csv_name:<18s} {len(df_sg):>5d} rows")
    logger.info(f"  {adapt_csv_name:<18s} {len(df_adapt):>5d} 条")

    return df_sg, df_adapt, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xy_dir", default='XY_Processed')
    parser.add_argument("--dif_dir", default='DIF')
    parser.add_argument("--output_dir", default='RRUFF_dataset')
    parser.add_argument("--labeled_csv_name", default="rruff_labeled.csv",
                        help="Output filename for the labeled base dataset")
    parser.add_argument("--adapt_csv_name", default="rruff_adapt.csv",
                        help="输出的无标签域适配 CSV 名称")

    parser.add_argument("--allow_non_cu_ka", action="store_true",
                        help="启用则对 DIF 不做 Cu Kα 波长过滤（默认：严格过滤）")
    parser.add_argument("--no_require_dif", action="store_true",
                        help="启用则保留没有 DIF 配对的样本（默认：必须有 DIF）")
    parser.add_argument("--no_exclude_calculated", action="store_true",
                        help="启用则严格管线保留计算谱（默认：丢弃计算谱）；"
                             "注意此开关不影响 adapt 集，adapt 集始终保留计算谱")
    parser.add_argument("--no_check_symmetry", action="store_true",
                        help="禁用对称性一致性检查（论文默认启用）")
    parser.add_argument("--no_check_xy_dif", action="store_true",
                        help="禁用 XY-DIF 晶胞数值交叉验证（论文默认启用）")
    args = parser.parse_args()

    process_rruff(
        xy_dir=Path(args.xy_dir),
        dif_dir=Path(args.dif_dir),
        output_dir=Path(args.output_dir),
        strict_cu_ka=not args.allow_non_cu_ka,
        require_dif=not args.no_require_dif,
        check_symmetry=not args.no_check_symmetry,
        check_xy_dif=not args.no_check_xy_dif,
        exclude_calculated=not args.no_exclude_calculated,
        labeled_csv_name=args.labeled_csv_name,
        adapt_csv_name=args.adapt_csv_name,
    )
