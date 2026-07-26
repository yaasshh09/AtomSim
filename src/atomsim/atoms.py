"""Elements, subshells, and Aufbau configurations for screened atoms (Phase 6).

Pure data and combinatorics — no physics engine. A Configuration is an ordered
tuple of ((n, l), occupancy) in Madelung filling order. The screened potential
depends only on (Z, N); the configuration decides which computed orbitals are
occupied and thus the summed energy. See docs/superpowers/specs/
2026-07-18-phase6-screened-atoms-design.md.
"""

from dataclasses import dataclass

SUBSHELL_LABELS = "spdfgh"

Subshell = tuple[int, int]                       # (n, l)
Configuration = tuple[tuple[Subshell, int], ...]  # ordered, non-zero shells


def subshell_capacity(l: int) -> int:
    return 2 * (2 * l + 1)


def _madelung_order() -> list[Subshell]:
    """(n, l) shells sorted by (n + l, n) — the Madelung/Aufbau rule."""
    shells = [(n, l) for n in range(1, 8) for l in range(n)]
    shells.sort(key=lambda nl: (nl[0] + nl[1], nl[0]))
    return shells


_MADELUNG = _madelung_order()


def aufbau_configuration(n_electrons: int) -> Configuration:
    if n_electrons < 1:
        raise ValueError(f"n_electrons must be >= 1, got {n_electrons}")
    remaining = n_electrons
    out: list[tuple[Subshell, int]] = []
    for n, l in _MADELUNG:
        if remaining <= 0:
            break
        fill = min(subshell_capacity(l), remaining)
        out.append(((n, l), fill))
        remaining -= fill
    if remaining > 0:
        raise ValueError(f"{n_electrons} electrons exceeds supported shells")
    return tuple(out)


def format_config(config: Configuration) -> str:
    return " ".join(f"{n}{SUBSHELL_LABELS[l]}{occ}" for (n, l), occ in config)


def parse_config(text: str) -> Configuration:
    out: list[tuple[Subshell, int]] = []
    for tok in text.split():
        n = int(tok[0])
        l = SUBSHELL_LABELS.index(tok[1])
        occ = int(tok[2:])
        out.append(((n, l), occ))
    return tuple(out)


def total_electrons(config: Configuration) -> int:
    return sum(occ for _, occ in config)


def is_ground(config: Configuration) -> bool:
    return config == aufbau_configuration(total_electrons(config))


def validate_config(config: Configuration) -> None:
    for (n, l), occ in config:
        if n <= l:
            raise ValueError(f"n must be > l for a real subshell, got n={n}, l={l}")
        if occ < 0:
            raise ValueError(f"occupancy must be >= 0, got {occ}")
        if occ > subshell_capacity(l):
            raise ValueError(
                f"occupancy {occ} exceeds capacity {subshell_capacity(l)} for l={l}"
            )


@dataclass(frozen=True)
class Element:
    z: int
    symbol: str
    name: str
    #: Standard atomic weight, in unified atomic mass units. This is the mass
    #: of the whole neutral atom for the natural terrestrial isotope mixture,
    #: which is what a Doppler width needs: the emitter that recoils is the
    #: atom, not the electron. A mixture is not one mass, and the spread shows
    #: up in a real spectrum as an isotope shift that this does not model.
    mass_u: float


#: Standard atomic weights, IUPAC/CIAAW 2021 table (abridged to the digits that
#: matter here: a Doppler width goes as 1/sqrt(m), so the fourth digit moves it
#: by parts in ten thousand, far below the model's own error).
ELEMENTS: tuple[Element, ...] = (
    Element(1, "H", "Hydrogen", 1.008), Element(2, "He", "Helium", 4.002602),
    Element(3, "Li", "Lithium", 6.94), Element(4, "Be", "Beryllium", 9.012183),
    Element(5, "B", "Boron", 10.81), Element(6, "C", "Carbon", 12.011),
    Element(7, "N", "Nitrogen", 14.007), Element(8, "O", "Oxygen", 15.999),
    Element(9, "F", "Fluorine", 18.998403), Element(10, "Ne", "Neon", 20.1797),
    Element(11, "Na", "Sodium", 22.989769), Element(12, "Mg", "Magnesium", 24.305),
    Element(13, "Al", "Aluminium", 26.981538), Element(14, "Si", "Silicon", 28.085),
    Element(15, "P", "Phosphorus", 30.973762), Element(16, "S", "Sulfur", 32.06),
    Element(17, "Cl", "Chlorine", 35.45), Element(18, "Ar", "Argon", 39.95),
)

_BY_SYMBOL = {e.symbol: e for e in ELEMENTS}
_BY_Z = {e.z: e for e in ELEMENTS}

# Elements with no published neutral GSZ screening parameters. Szydlik & Green,
# Phys. Rev. A 9, 1885 (1974), Table I tabulates neutral atoms He..P and Ar, but
# skips neutral S and Cl (their 3s^2 3p^4 / 3p^5 blocks list only Ar^2+ / Ar^+).
# Rather than invent parameters, we omit these atoms from the preset library — the
# prime directive forbids quietly shipping physics we cannot source.
NO_GSZ_PARAMETERS: frozenset[int] = frozenset({16, 17})  # S, Cl

# Named screened-atom presets are neutral He..P and Ar (H stays hydrogenic/analytic;
# S and Cl are excluded, see NO_GSZ_PARAMETERS).
ATOM_KEYS: tuple[str, ...] = tuple(
    e.symbol.lower() for e in ELEMENTS if e.z >= 2 and e.z not in NO_GSZ_PARAMETERS
)


def element_by_symbol(sym: str) -> Element:
    return _BY_SYMBOL[sym]


def element_by_z(z: int) -> Element:
    return _BY_Z[z]


def is_atom_key(key: str) -> bool:
    return key in ATOM_KEYS


def atom_for_key(key: str) -> Element:
    if not is_atom_key(key):
        raise KeyError(f"unknown atom key {key!r}; known: {ATOM_KEYS}")
    return _BY_SYMBOL[key.capitalize()]
