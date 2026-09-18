from __future__ import annotations

import re


# True metals protect carbon-containing inorganic compounds from being
# classified as organic. Metalloids are handled separately below.
METAL_SYMBOLS = frozenset(
    {
        "Li", "Na", "K", "Rb", "Cs", "Fr",
        "Be", "Mg", "Ca", "Sr", "Ba", "Ra",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Rf", "Db", "Sg", "Bh", "Hs",
        "Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi", "Po",
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
        "Ho", "Er", "Tm", "Yb", "Lu",
        "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
        "Es", "Fm", "Md", "No", "Lr",
    }
)

METALLOIDS = frozenset({"B", "Si", "Ge", "As", "Se", "Te"})
ORGANIC_HETEROATOMS = frozenset({"N", "O", "S", "P", "F", "Cl", "Br", "I"})
FORMULA_TOKEN_WITH_COUNT_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def parse_element_counts(formula: str) -> dict[str, float]:
    """Extract approximate element counts from a formula string."""
    if not isinstance(formula, str) or not formula.strip():
        return {}
    counts: dict[str, float] = {}
    for symbol, number in FORMULA_TOKEN_WITH_COUNT_RE.findall(formula):
        if not symbol:
            continue
        amount = float(number) if number else 1.0
        counts[symbol] = counts.get(symbol, 0.0) + amount
    return counts


def looks_organic_v2(
    formula: str,
    c_frac_threshold: float = 0.3,
) -> tuple[bool, str]:
    """Conservative COD organic/molecular-crystal heuristic used by the dataset pipeline."""
    counts = parse_element_counts(formula)
    symbols = set(counts)

    if symbols & METAL_SYMBOLS:
        return False, "not_organic_metal"

    # Keep inorganic covalent materials such as SiC and B4C unless hydrogen
    # is present, which is a stronger signal for molecular chemistry.
    if (symbols & METALLOIDS) and "H" not in symbols:
        return False, "not_organic_metalloid_no_H"

    if "C" not in symbols:
        return False, "not_organic_no_C"

    if "H" in symbols and symbols & ORGANIC_HETEROATOMS:
        return True, "organic_v1_CH_hetero"

    total = sum(counts.values())
    carbon_fraction = counts.get("C", 0.0) / total if total > 0 else 0.0
    if carbon_fraction >= c_frac_threshold:
        return True, "organic_v2_C_dominant"

    return False, "not_organic_other"
