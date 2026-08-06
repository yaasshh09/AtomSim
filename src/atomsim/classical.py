"""Classical ghost: what classical electrodynamics predicts for a one-electron atom.

A classical orbiting electron is an accelerating charge; by the Larmor formula it
radiates, loses energy, and spirals into the nucleus in picoseconds. That is the
COUNTERFACTUAL truth of classical rules, computed exactly. The circular Bohr orbits
are a semi-classical APPROXIMATION (right energy scale, wrong picture). Quantum
mechanics forbids the collapse, which is the whole lesson.
"""

import math
from dataclasses import dataclass

from atomsim.constants import FundamentalConstants
from atomsim.provenance import Fidelity, Provenance, Quantity
from atomsim.systems import System, get_system

_APPROX = "Bohr model r_n = n^2 a0 / Z (semi-classical circular orbit)"
_LARMOR = "Larmor radiative collapse, classical E&M, exact under classical rules"
_ASSUME = ("classical electrodynamics (deliberately non-quantum)",)


@dataclass(frozen=True)
class BohrOrbit:
    n: int
    radius_bohr: Quantity   # for drawing in the cloud's Bohr units
    radius_pm: Quantity     # for display


@dataclass(frozen=True)
class ClassicalGhost:
    n: int
    system_key: str
    z: int
    orbits: tuple[BohrOrbit, ...]
    r0_bohr: Quantity
    collapse_time_s: Quantity
    orbital_period_s: Quantity
    orbit_count: Quantity


def _bohr_orbit(n: int, a0_sys_m: float, a0_m: float, z: int) -> BohrOrbit:
    r_m = n * n * a0_sys_m / z
    prov = Provenance(fidelity=Fidelity.APPROXIMATION, method=_APPROX, assumptions=_ASSUME)
    return BohrOrbit(
        n=n,
        radius_bohr=Quantity(r_m / a0_m, "bohr", f"Bohr orbit n={n}", prov),
        radius_pm=Quantity(r_m * 1e12, "pm", f"Bohr orbit n={n}", prov),
    )


def classical_ghost(n: int, system: str | System = "h") -> ClassicalGhost:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    # Accept a bare key (registered systems) or an already-resolved System, so the
    # server can hand us generic hydrogen-like ions (z{N}) that get_system doesn't know.
    sys = system if isinstance(system, System) else get_system(system)
    z = sys.Z
    c = FundamentalConstants.codata()
    m = sys.mu_ratio.value * c.m_e                 # orbiting reduced mass (kg)
    a0_m = c.bohr_radius                           # real-electron Bohr radius (m)
    a0_sys_m = a0_m / sys.mu_ratio.value           # reduced-mass Bohr radius (m)
    k = 1.0 / (4.0 * math.pi * c.eps0)

    orbits = tuple(_bohr_orbit(i, a0_sys_m, a0_m, z) for i in range(1, n + 1))
    r0_m = n * n * a0_sys_m / z

    # Larmor closed form: t = r0^3 m^2 c^3 / (4 Z k^2 e^4)
    t_collapse = (r0_m**3 * m**2 * c.c**3) / (4.0 * z * k**2 * c.e**4)
    # initial circular speed and angular velocity
    v0 = math.sqrt(z * k * c.e**2 / (m * r0_m))
    omega0 = v0 / r0_m
    period = 2.0 * math.pi / omega0
    n_orbits = omega0 * t_collapse / math.pi

    cf = Provenance(fidelity=Fidelity.COUNTERFACTUAL, method=_LARMOR, assumptions=_ASSUME)
    return ClassicalGhost(
        n=n,
        system_key=sys.key,
        z=z,
        orbits=orbits,
        r0_bohr=Quantity(r0_m / a0_m, "bohr", "classical start radius", cf),
        collapse_time_s=Quantity(t_collapse, "s", "classical collapse time", cf),
        orbital_period_s=Quantity(period, "s", "initial orbital period", cf),
        orbit_count=Quantity(n_orbits, "", "orbits before collapse", cf),
    )
