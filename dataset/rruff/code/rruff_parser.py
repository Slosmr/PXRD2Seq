"""Parse RRUFF experimental XY patterns and DIF crystallographic metadata.

XY headers use ##KEY=VALUE records ending at ##END=. Numeric data may be
comma-separated or whitespace-separated. DIF records use text prefixes
such as CELL PARAMETERS, SPACE GROUP, and X-RAY WAVELENGTH.
RRUFF identifiers retain sample suffixes (for example, R050336-1)."""

import re
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


CU_KA_TARGETS = [1.5406, 1.5418, 1.541838, 1.540562]
CU_KA_TOLERANCE = 0.005


_RRUFF_ID_FULL = re.compile(r"(R\d{5,7}(?:-\d+)?)")
_RRUFF_ID_BASE = re.compile(r"R\d{5,7}")


def extract_rruff_id_from_name(filename: str) -> tuple:
    """从文件名提取 (full_id, base_id)，如 ('R050336-1', 'R050336')。"""
    m_full = _RRUFF_ID_FULL.search(filename)
    if not m_full:
        return None, None
    full = m_full.group(1)
    m_base = _RRUFF_ID_BASE.match(full)
    return full, (m_base.group(0) if m_base else full)


def is_cu_ka(wavelength: Optional[float],
             tol: float = CU_KA_TOLERANCE) -> bool:
    """判断波长是否为 Cu Kα。"""
    if wavelength is None:
        return False
    return any(abs(wavelength - w) <= tol for w in CU_KA_TARGETS)


_CELL_FIELD_RE = re.compile(
    r"a\s*:\s*([-+]?\d*\.?\d+).*?"
    r"b\s*:\s*([-+]?\d*\.?\d+).*?"
    r"c\s*:\s*([-+]?\d*\.?\d+).*?"
    r"alpha\s*:\s*([-+]?\d*\.?\d+).*?"
    r"beta\s*:\s*([-+]?\d*\.?\d+).*?"
    r"gamma\s*:\s*([-+]?\d*\.?\d+).*?"
    r"volume\s*:\s*([-+]?\d*\.?\d+).*?"
    r"crystal\s*system\s*:\s*(\w+)",
    re.IGNORECASE | re.DOTALL,
)


def _parse_xy_header(text: str) -> dict:
    """解析 XY 文件头 ##KEY=VALUE 形式的元数据。"""
    meta = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("##"):
            continue
        body = line[2:]
        if "=" in body:
            key, _, val = body.partition("=")
            meta[key.strip().upper()] = val.strip()
        elif ":" in body:
            key, _, val = body.partition(":")
            meta[key.strip().upper()] = val.strip()
    return meta


def _parse_xy_cell(cell_str: str) -> Optional[dict]:
    """
    解析 ##CELL PARAMETERS= 字段值，如:
    'a: 11.5379 b: 11.5379 c: 11.5379 alpha: 90 beta: 90 gamma: 90 volume: 1535.98 crystal system: cubic'
    """
    m = _CELL_FIELD_RE.search(cell_str)
    if not m:
        return None
    try:
        return {
            "a": float(m.group(1)),
            "b": float(m.group(2)),
            "c": float(m.group(3)),
            "alpha": float(m.group(4)),
            "beta": float(m.group(5)),
            "gamma": float(m.group(6)),
            "volume": float(m.group(7)),
            "crystal_system": m.group(8).lower(),
        }
    except ValueError:
        return None


def _parse_xy_wavelength(meta: dict) -> Optional[float]:
    """预留波长解析（RRUFF XY 目前 header 无波长字段）。"""
    for key in ("WAVELENGTH", "X-RAY WAVELENGTH"):
        if key in meta:
            m = re.search(r"[-+]?\d*\.?\d+", meta[key])
            if m:
                try:
                    return float(m.group())
                except ValueError:
                    pass
    return None


def parse_xy_file(path: Path) -> Optional[dict]:
    """
    解析 RRUFF XY_Processed/*.txt，兼容两种数据行格式：
      - 逗号分隔:  `5.000, 0.000`
      - 空格分隔 + `X Y` 列名行
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"读取失败 {path.name}: {e}")
        return None

    meta = _parse_xy_header(text)
    full_id, base_id = extract_rruff_id_from_name(path.name)
    if base_id is None:
        base_id = meta.get("RRUFFID", "").strip() or None
        full_id = base_id

    cell_info = None
    if "CELL PARAMETERS" in meta:
        cell_info = _parse_xy_cell(meta["CELL PARAMETERS"])

    wavelength = _parse_xy_wavelength(meta)


    two_theta, intensity = [], []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("##") or s.startswith("#"):
            continue
        if not re.search(r"\d", s):
            continue
        parts = re.split(r"[,\s]+", s)
        if len(parts) < 2:
            continue
        try:
            t = float(parts[0])
            i = float(parts[1])
        except ValueError:
            continue
        two_theta.append(t)
        intensity.append(i)

    if len(two_theta) < 10:
        logger.debug(f"数据点过少 {path.name}: {len(two_theta)}")
        return None

    return {
        "rruff_id_full": full_id,
        "rruff_id_base": base_id,
        "name": meta.get("NAMES", ""),
        "formula": meta.get("IDEAL CHEMISTRY", ""),
        "wavelength": wavelength,
        "cell": cell_info,
        "description": meta.get("DIFFRACTION SAMPLE DESCRIPTION", ""),
        "status": meta.get("STATUS", ""),
        "two_theta": np.array(two_theta, dtype=np.float64),
        "intensity": np.array(intensity, dtype=np.float64),
    }


_DIF_CELL_RE = re.compile(
    r"CELL\s+PARAMETERS\s*:?\s*"
    r"([-+]?\d*\.?\d+)\s+"
    r"([-+]?\d*\.?\d+)\s+"
    r"([-+]?\d*\.?\d+)\s+"
    r"([-+]?\d*\.?\d+)\s+"
    r"([-+]?\d*\.?\d+)\s+"
    r"([-+]?\d*\.?\d+)",
    re.IGNORECASE,
)
_DIF_SG_RE = re.compile(r"SPACE\s+GROUP\s*:?\s*(\S+)", re.IGNORECASE)
_DIF_WL_RE = re.compile(
    r"X[- ]?RAY\s+WAVELENGTH\s*:?\s*([-+]?\d*\.?\d+)", re.IGNORECASE
)


def parse_dif_file(path: Path) -> Optional[dict]:
    """解析 DIF 文件（通过 regex 全文搜索关键字段）。"""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"读取失败 {path.name}: {e}")
        return None

    full_id, base_id = extract_rruff_id_from_name(path.name)

    m_cell = _DIF_CELL_RE.search(text)
    m_sg = _DIF_SG_RE.search(text)
    m_wl = _DIF_WL_RE.search(text)

    if not m_cell or not m_sg:
        logger.debug(f"DIF 缺关键字段 {path.name}")
        return None

    try:
        cell = tuple(float(m_cell.group(i)) for i in range(1, 7))
    except ValueError:
        return None

    sg_symbol = m_sg.group(1).strip()
    wavelength = float(m_wl.group(1)) if m_wl else None

    name = ""
    for line in text.splitlines():
        s = line.strip()
        if s:
            name = s
            break

    return {
        "rruff_id_full": full_id,
        "rruff_id_base": base_id,
        "name": name,
        "wavelength": wavelength,
        "cell": cell,
        "space_group": sg_symbol,
    }


def _normalize_sg_symbol(s: str) -> str:
    s = s.strip().replace(" ", "")
    s = s.replace("–", "-").replace("—", "-")
    return s


def sg_symbol_to_info(sg_symbol: str):
    """
    多策略把空间群符号转 (ita_number, crystal_system)。
    应对 P3_221 / P2_1/c 等非标准记号。
    """
    try:
        from pymatgen.symmetry.groups import SpaceGroup
    except ImportError:
        logger.error("pymatgen 未安装")
        return None, None

    original = sg_symbol.strip()
    candidates = [
        original,
        _normalize_sg_symbol(original),
        _normalize_sg_symbol(original).replace("_", ""),
    ]
    norm = _normalize_sg_symbol(original)
    if "_" in norm:
        candidates.append(re.sub(r"_(\d)", r"\1", norm))

    for sym in candidates:
        try:
            sg = SpaceGroup(sym)
            return sg.int_number, sg.crystal_system
        except (ValueError, KeyError):
            continue

    try:
        n = int(original)
        if 1 <= n <= 230:
            sg = SpaceGroup.from_int_number(n)
            return n, sg.crystal_system
    except (ValueError, KeyError):
        pass
    return None, None


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


def get_laue_from_sg_number(sg_number: int) -> str:
    try:
        from pymatgen.symmetry.groups import SpaceGroup
        sg = SpaceGroup.from_int_number(sg_number)
        return POINTGROUP_TO_LAUE.get(sg.point_group, "unknown")
    except Exception:
        return "unknown"


def get_bravais(crystal_system: str, sg_symbol: str) -> str:
    if not sg_symbol:
        return "unknown"
    return CRYSTAL_SYSTEM_BRAVAIS.get(crystal_system, {}).get(sg_symbol[0], "unknown")


def resample_pattern(two_theta: np.ndarray,
                     intensity: np.ndarray,
                     target_range: tuple = (4.0, 90.0),
                     target_length: int = 8500) -> np.ndarray:
    """Resample onto an evenly spaced grid, using zeros outside the measured range."""
    x_target = np.linspace(target_range[0], target_range[1], target_length)
    y_target = np.zeros_like(x_target)
    src_lo, src_hi = float(two_theta.min()), float(two_theta.max())
    valid = (x_target >= src_lo) & (x_target <= src_hi)
    if valid.any():
        y_target[valid] = np.interp(x_target[valid], two_theta, intensity)
    return y_target


def niggli_reduce_cell(cell: dict) -> dict:
    """Return Niggli-reduced cell parameters while preserving the crystal-system label.

DIF and XY lattice metadata may use conventional settings, including the
hexagonal setting for trigonal crystals. If reduction is unavailable or
fails, return the original cell."""
    try:
        from pymatgen.core import Lattice
    except ImportError:
        logger.warning("pymatgen 未安装，跳过 Niggli 约化")
        return cell

    try:
        lat = Lattice.from_parameters(
            a=cell["a"], b=cell["b"], c=cell["c"],
            alpha=cell["alpha"], beta=cell["beta"], gamma=cell["gamma"],
        )

        lat_n = lat.get_niggli_reduced_lattice()

        return {
            "a": float(lat_n.a),
            "b": float(lat_n.b),
            "c": float(lat_n.c),
            "alpha": float(lat_n.alpha),
            "beta": float(lat_n.beta),
            "gamma": float(lat_n.gamma),
            "volume": float(lat_n.volume),
            "crystal_system": cell.get("crystal_system", ""),
        }
    except Exception as e:
        logger.debug(f"Niggli 约化失败 ({e})，保留原 cell")
        return cell
