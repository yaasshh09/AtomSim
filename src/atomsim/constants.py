"""Fundamental constants (CODATA, via scipy) and the counterfactual-universe hook.

Engine-internal computations use Hartree atomic units; this module supplies the
SI anchors and display conversions. A FundamentalConstants instance with altered
fields IS a counterfactual universe (What-If Lab, later phase).
"""

import math
from dataclasses import dataclass

from scipy import constants as _sc

# Real-universe display anchor ONLY: counterfactual universes must derive
# their own conversion via FundamentalConstants.hartree_energy, never this.
HARTREE_EV: float = _sc.physical_constants["Hartree energy in eV"][0]

# Real-universe display anchor ONLY (same caveat as HARTREE_EV): counterfactual
# universes must derive alpha via FundamentalConstants.alpha, never this.
ALPHA: float = _sc.fine_structure

# Real-universe display anchor ONLY (same caveat): pm-per-bohr for readouts.
BOHR_RADIUS_PM: float = _sc.physical_constants["Bohr radius"][0] * 1e12

# Real-universe display anchor ONLY (same caveat): fm-per-bohr for nuclear radii.
BOHR_RADIUS_FM: float = _sc.physical_constants["Bohr radius"][0] * 1e15

# Real-universe display anchor ONLY (same caveat): atomic unit of magnetic field
# (hbar / (e a0^2)), in tesla, for Tesla<->a.u. conversion at the server boundary.
B0_TESLA: float = _sc.physical_constants["atomic unit of mag. flux density"][0]

# Real-universe display anchor ONLY (same caveat): atomic unit of electric field
# (E_h / (e a0)), in volts per metre, for MV/m<->a.u. conversion at the boundary.
E0_V_PER_M: float = _sc.physical_constants["atomic unit of electric field"][0]


@dataclass(frozen=True)
class FundamentalConstants:
    hbar: float  # J s
    e: float     # C
    m_e: float   # kg
    eps0: float  # F/m
    c: float     # m/s

    @classmethod
    def codata(cls) -> "FundamentalConstants":
        return cls(
            hbar=_sc.hbar,
            e=_sc.elementary_charge,
            m_e=_sc.electron_mass,
            eps0=_sc.epsilon_0,
            c=_sc.speed_of_light,
        )

    @property
    def alpha(self) -> float:
        """Fine-structure constant e^2 / (4 pi eps0 hbar c), dimensionless."""
        return self.e**2 / (4 * math.pi * self.eps0 * self.hbar * self.c)

    @property
    def bohr_radius(self) -> float:
        """a0 = 4 pi eps0 hbar^2 / (m_e e^2), in metres."""
        return 4 * math.pi * self.eps0 * self.hbar**2 / (self.m_e * self.e**2)

    @property
    def hartree_energy(self) -> float:
        """E_h = hbar^2 / (m_e a0^2), in joules."""
        return self.hbar**2 / (self.m_e * self.bohr_radius**2)
