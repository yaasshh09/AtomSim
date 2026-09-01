"""My exotic-but-real hydrogen-like system presets (spec 5.5).

Each preset hands me a nuclear charge Z and the exact reduced-mass ratio
mu/m_e, as a Quantity whose provenance cites the CODATA mass ratios I built it
from. My m_over_M (orbiting mass / nuclear mass) feeds the fine-structure
recoil error scale: I stay honest about positronium by quantifying that error,
never by serving a silent wrong number.
"""

from dataclasses import dataclass

from scipy import constants as _sc

from atomsim.provenance import Fidelity, Provenance, Quantity


@dataclass(frozen=True)
class System:
    key: str
    name: str
    Z: int
    mu_ratio: Quantity  # mu / m_e, unit "m_e"
    m_over_M: float     # orbiting mass / nuclear mass (recoil scale)
    description: str
    # The rms charge radius of the nucleus, in bohr, my canonical unit.
    # None is how I say honestly absent: a point lepton (positronium's
    # positron) or a generic Z with no nucleus I can name, never a silent zero.
    nuclear_radius: Quantity | None = None


def _mass_ratio(constant_name: str) -> tuple[float, float]:
    value, _unit, unc = _sc.physical_constants[constant_name]
    return value, unc


_A0_M = _sc.physical_constants["Bohr radius"][0]


def _radius_quantity(
    r_m: float, unc_m: float, nucleus: str, source: str
) -> Quantity:
    """I wrap a nuclear rms charge radius as a Quantity of mine, in bohr."""
    return Quantity(
        value=r_m / _A0_M,
        unit="bohr",
        label=f"nuclear rms charge radius ({nucleus})",
        provenance=Provenance(
            fidelity=Fidelity.EXACT,
            method=f"the measured rms charge radius of the {nucleus}, {source}",
            assumptions=(
                "this is a reference measurement, not something I predicted",
                "my Coulomb engine still treats the nucleus as a point charge",
            ),
            error_estimate=unc_m / _A0_M,
        ),
    )


def _codata_radius(constant_name: str, nucleus: str) -> Quantity:
    r_m, _unit, unc_m = _sc.physical_constants[constant_name]
    return _radius_quantity(
        r_m, unc_m, nucleus, f"CODATA (scipy.constants: {constant_name})"
    )


# The triton is absent from scipy's CODATA table, so I take the standard
# compilation value instead.
_TRITON_RADIUS = _radius_quantity(
    1.7591e-15, 0.0363e-15, "triton",
    "Angeli & Marinova (2013), At. Data Nucl. Data Tables 99, 69",
)


def _codata_system(
    key: str, name: str, Z: int, nucleus_constant: str, description: str,
    orbiter_constant: str | None = None,
    nuclear_radius: Quantity | None = None,
) -> System:
    """I build a preset from CODATA mass ratios, orbited by an electron unless
    you name something else.
    """
    R_nuc, u_nuc = _mass_ratio(nucleus_constant)  # M / m_e
    if orbiter_constant is None:
        r_orb, u_orb, orb_name = 1.0, 0.0, "electron"
    else:
        r_orb, u_orb = _mass_ratio(orbiter_constant)
        orb_name = orbiter_constant.split("-")[0]
    mu = r_orb * R_nuc / (r_orb + R_nuc)
    rel_unc = (u_nuc / R_nuc if R_nuc else 0.0) + (u_orb / r_orb if r_orb else 0.0)
    return System(
        key=key,
        name=name,
        Z=Z,
        mu_ratio=Quantity(
            value=mu,
            unit="m_e",
            label=f"mu/m_e ({name})",
            provenance=Provenance(
                fidelity=Fidelity.EXACT,
                method=(
                    "I take mu/m_e = m_orb M / (m_orb + M) from CODATA mass "
                    f"ratios (scipy.constants: {nucleus_constant}"
                    + (f", {orbiter_constant})" if orbiter_constant else ")")
                ),
                assumptions=(f"I orbit a {orb_name} here",),
                error_estimate=mu * rel_unc,
            ),
        ),
        m_over_M=r_orb / R_nuc,
        description=description,
        nuclear_radius=nuclear_radius,
    )


_POSITRONIUM = System(
    key="ps",
    name="Positronium",
    Z=1,
    mu_ratio=Quantity(
        value=0.5,
        unit="m_e",
        label="mu/m_e (Positronium)",
        provenance=Provenance(
            fidelity=Fidelity.EXACT,
            method="mu = m_e/2 exactly, because the electron and positron weigh the same",
            assumptions=("I orbit an electron here, and my 'nucleus' is a positron",),
            error_estimate=0.0,
        ),
    ),
    m_over_M=1.0,
    description="Electron bound to a positron; recoil is O(1), fine structure "
    "unreliable at alpha^2 (error estimate says so).",
)

_SYSTEMS: tuple[System, ...] = (
    _codata_system("h", "Hydrogen", 1, "proton-electron mass ratio",
                   "Ordinary hydrogen: electron + proton.",
                   nuclear_radius=_codata_radius("proton rms charge radius", "proton")),
    _codata_system("d", "Deuterium", 1, "deuteron-electron mass ratio",
                   "Heavy hydrogen: electron + deuteron.",
                   nuclear_radius=_codata_radius("deuteron rms charge radius", "deuteron")),
    _codata_system("t", "Tritium", 1, "triton-electron mass ratio",
                   "Radioactive hydrogen isotope: electron + triton.",
                   nuclear_radius=_TRITON_RADIUS),
    _codata_system("mu-h", "Muonic hydrogen", 1, "proton-electron mass ratio",
                   "Muon orbiting a proton: ~186x smaller, ~186x deeper.",
                   orbiter_constant="muon-electron mass ratio",
                   nuclear_radius=_codata_radius("proton rms charge radius", "proton")),
    _POSITRONIUM,
    _codata_system("he+", "Helium ion He+", 2, "alpha particle-electron mass ratio",
                   "One-electron helium: Z=2 scaling on real helium-4.",
                   nuclear_radius=_codata_radius(
                       "alpha particle rms charge radius", "alpha particle")),
)


def list_systems() -> tuple[System, ...]:
    return _SYSTEMS


def get_system(key: str) -> System:
    for s in _SYSTEMS:
        if s.key == key:
            return s
    raise KeyError(f"unknown system {key!r}; available: {[s.key for s in _SYSTEMS]}")


def emitter_mass(system: System) -> Quantity:
    """The mass of my whole radiating atom (orbiter plus nucleus), in kg.

    A Doppler width is set by the mass of the thing that recoils, which is the
    atom and not the electron. Getting that wrong is a factor of 1836, so I
    derive it here once rather than at each call site.

    I need no new data: the preset already implies the mass. With
    `x = m_orb / M_nuc` (my stored `m_over_M`) and my stored reduced mass
    `mu = m_orb M / (m_orb + M) = m_orb / (1 + x)`,

        m_orb = mu (1 + x)
        M_nuc = m_orb / x = mu (1 + x) / x
        M_atom = m_orb + M_nuc = mu (1 + x)^2 / x

    which gives me 1837.15 m_e for hydrogen (proton plus electron), exactly
    2 m_e for positronium, and 2042.8 m_e for muonic hydrogen.

    `m_over_M = 0` is the infinite-nucleus idealization in my generic Z preset.
    My honest answer there is an infinite mass, and so a line with no Doppler
    width at all, because a nucleus that cannot recoil cannot shift its own
    photon. That is exact *for my model* and wrong about every real ion, so I
    say it out loud instead of serving you a sharp line.
    """
    x = system.m_over_M
    mu = system.mu_ratio.value
    if x <= 0.0:
        return Quantity(
            value=float("inf"),
            unit="kg",
            label=f"M_atom ({system.key})",
            provenance=Provenance(
                fidelity=Fidelity.APPROXIMATION,
                method="I use an infinite nuclear mass, so my nucleus cannot recoil",
                assumptions=(
                    "this preset of mine carries no nuclear mass (m_over_M = 0), "
                    "so my atom is infinitely heavy and every thermal velocity I "
                    "get is zero; a real ion of this Z has a finite mass and a "
                    "finite Doppler width",
                ),
                refinement="give me a nuclear mass and I will give you a real thermal width",
            ),
        )
    ratio = mu * (1.0 + x) ** 2 / x  # M_atom / m_e
    return Quantity(
        value=ratio * _sc.m_e,
        unit="kg",
        label=f"M_atom ({system.key})",
        provenance=Provenance(
            fidelity=system.mu_ratio.provenance.fidelity,
            method=(
                "I invert M_atom = mu (1 + x)^2 / x with x = m_orb/M_nuc, out of "
                f"[{system.mu_ratio.provenance.method}]"
            ),
            assumptions=system.mu_ratio.provenance.assumptions
            + ("I take the mass of the bound system as the sum of its parts: the "
               "binding energy is ~1e-8 of the rest mass, so I drop it",),
            error_estimate=(
                None if system.mu_ratio.provenance.error_estimate is None
                else system.mu_ratio.provenance.error_estimate
                * (1.0 + x) ** 2 / x * _sc.m_e
            ),
        ),
    )


def element_emitter_mass(element) -> Quantity:
    """The mass of a neutral atom of `element`, in kg, which I need for a
    Doppler width.

    `element` is an atoms.Element. I leave it untyped here to keep that module
    free of physics imports, which is its stated boundary. Unlike my
    hydrogen-like presets, a screened atom's mass is not implied by anything I
    already carry, so I take it from the standard atomic weight table.

    That weight is for the natural isotope mixture, which is not one mass. The
    spread is real and shows up in a spectrum as an isotope shift. What I give
    you is the width the mean mass would produce.
    """
    return Quantity(
        value=element.mass_u * _sc.physical_constants["atomic mass constant"][0],
        unit="kg",
        label=f"M_atom ({element.symbol})",
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method=(
                f"I take the standard atomic weight {element.mass_u:g} u "
                "(IUPAC/CIAAW) and multiply by the atomic mass constant"
            ),
            assumptions=(
                "I take the natural terrestrial isotope mixture as a single mean "
                "mass: the real mixture broadens and shifts lines by an isotope "
                "effect I do not model",
                "I assume a neutral atom: an ion has lost an electron's worth of "
                "mass, which is parts in 1e4 and below anything I resolve here",
            ),
        ),
    )


def hydrogen_like(Z: int, mu_ratio: float = 1.0) -> System:
    """A generic one-electron ion of charge Z. I take the nuclear mass as
    infinite unless you say otherwise.
    """
    if Z < 1:
        raise ValueError(f"Z must be >= 1, got {Z}")
    assumptions = (
        ("I used an infinite nuclear mass (mu_ratio = 1)",) if mu_ratio == 1.0
        else (f"you gave me mu_ratio = {mu_ratio:g}",)
    )
    return System(
        key=f"z{Z}",
        name=f"Hydrogen-like Z={Z}",
        Z=Z,
        mu_ratio=Quantity(
            value=mu_ratio,
            unit="m_e",
            label=f"mu/m_e (Z={Z})",
            provenance=Provenance(
                fidelity=Fidelity.EXACT,
                method="the reduced-mass ratio you specified",
                assumptions=assumptions,
            ),
        ),
        m_over_M=0.0,
        description=f"Generic one-electron ion, Z={Z}.",
    )
