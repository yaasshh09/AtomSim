"""Helium beyond Hartree-Fock: Hylleraas's correlated variational method.

Hartree-Fock gives each electron its own orbital and lets it see the other
only as an averaged cloud. The real electrons also keep apart instant by
instant, and the energy that buys is the correlation energy,

    E_corr = E_exact - E_HF,

-0.042 hartree for helium: about 1% of the total, and 1.1 eV of the ionization
energy, which is why Hartree-Fock's helium ionizes at 23.45 eV against the
measured 24.59.

Hylleraas (1929) computed it by writing the wavefunction in the coordinates

    s = r1 + r2,   t = r1 - r2,   u = r12,

so that the electron-electron distance is a variable the trial function can
depend on directly. The basis here is e^(-zeta s) s^i t^(2j) u^m, symmetric in
the two electrons because t only appears squared. Every matrix element is a
sum of integrals of s^a t^b u^c e^(-2 zeta s) over 0 <= u <= s, -u <= t <= u,
which close to factorials, so nothing here is quadrature. The kinetic term is
(1/2) sum_i |grad_i psi|^2 rewritten in (s, t, u) with the volume element
u (s^2 - t^2):

    u (s^2 - t^2) (psi_s^2 + psi_t^2 + psi_u^2)
      + 2 s (u^2 - t^2) psi_s psi_u + 2 t (s^2 - u^2) psi_t psi_u,

and the potential is (-4 Z s u + s^2 - t^2) psi^2, which is -Z (1/r1 + 1/r2)
+ 1/r12 under the same weight.

The energy is variational: every basis gives an upper bound on the exact
non-relativistic energy, and the bound falls as the basis grows. The method
solves the exact non-relativistic, clamped-nucleus Hamiltonian, so the result
is NUMERICAL, with the step between the last two basis sizes as its error
estimate. That estimate is a convergence spread and not a bound; the
reference value it is tested against is a cited constant, not a query.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import constants as _sc
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar

from atomsim.atoms import aufbau_configuration
from atomsim.hf_atom import solve_hartree_fock
from atomsim.provenance import Fidelity, Provenance, Quantity

#: The exact non-relativistic, infinite-nuclear-mass helium ground state, in
#: hartree, to 13 digits.
HELIUM_EXACT = -2.9037243770341
HELIUM_EXACT_CITATION = (
    "H. Nakashima and H. Nakatsuji, J. Chem. Phys. 127, 224104 (2007): "
    "-2.903724377034119598... hartree, non-relativistic, infinite nuclear mass"
)

Monomial = tuple[float, int, int, int]  # (coefficient, s power, t power, u power)

_METHOD = (
    "Hylleraas variational method: basis e^(-zeta s) s^i t^(2j) u^m with "
    "i + 2j + m <= omega, matrix elements in closed form, generalized "
    "eigenproblem, zeta optimized"
)
_ASSUMPTIONS = (
    "the exact non-relativistic Hamiltonian with a clamped (infinitely heavy) nucleus",
    "variational: an upper bound on that Hamiltonian's ground state",
)


def _integral(a: int, b: int, c: int, zeta: float) -> float:
    """int s^a t^b u^c e^(-2 zeta s) over 0 <= u <= s, -u <= t <= u, s >= 0."""
    if b % 2:
        return 0.0
    n = a + b + c + 2
    return (
        2.0 / ((b + 1) * (b + c + 2))
        * math.exp(math.lgamma(n + 1) - (n + 1) * math.log(2.0 * zeta))
    )


def _derivatives(i: int, j: int, m: int, zeta: float):
    """phi, phi_s, phi_t, phi_u as monomial lists (the e^(-zeta s) is implicit)."""
    b = 2 * j
    phi = [(1.0, i, b, m)]
    d_s = [(-zeta, i, b, m)] + ([(float(i), i - 1, b, m)] if i else [])
    d_t = [(float(b), i, b - 1, m)] if b else []
    d_u = [(float(m), i, b, m - 1)] if m else []
    return phi, d_s, d_t, d_u


def _times(p: list[Monomial], q: list[Monomial]) -> list[Monomial]:
    return [
        (c1 * c2, a1 + a2, b1 + b2, e1 + e2)
        for c1, a1, b1, e1 in p
        for c2, a2, b2, e2 in q
    ]


def _value(poly: list[Monomial], zeta: float) -> float:
    return sum(c * _integral(a, b, e, zeta) for c, a, b, e in poly if c)


# The weights, as monomials in (s, t, u).
_W_KIN = [(1.0, 2, 0, 1), (-1.0, 0, 2, 1)]  # u (s^2 - t^2)
_W_SU = [(1.0, 1, 0, 2), (-1.0, 1, 2, 0)]  # s (u^2 - t^2)
_W_TU = [(1.0, 2, 1, 0), (-1.0, 0, 1, 2)]  # t (s^2 - u^2)


def _matrices(basis, zeta: float, z: float):
    """S, T and V; T + V is the Hamiltonian, and T and V apart give the virial."""
    parts = [_derivatives(i, j, m, zeta) for i, j, m in basis]
    w_pot = [(-4.0 * z, 1, 0, 1), (1.0, 2, 0, 0), (-1.0, 0, 2, 0)]
    n = len(basis)
    S, T, V = (np.zeros((n, n)) for _ in range(3))
    for p in range(n):
        f, fs, ft, fu = parts[p]
        for q in range(p, n):
            g, gs, gt, gu = parts[q]
            S[p, q] = _value(_times(_W_KIN, _times(f, g)), zeta)
            # The bilinear form of the kinetic functional: the cross terms
            # 2 s (u^2 - t^2) psi_s psi_u become s (u^2 - t^2) (f_s g_u + f_u g_s).
            T[p, q] = (
                _value(_times(_W_KIN, _times(fs, gs) + _times(ft, gt) + _times(fu, gu)), zeta)
                + _value(_times(_W_SU, _times(fs, gu) + _times(fu, gs)), zeta)
                + _value(_times(_W_TU, _times(ft, gu) + _times(fu, gt)), zeta)
            )
            V[p, q] = _value(_times(w_pot, _times(f, g)), zeta)
    lower = np.tril_indices(n, -1)
    for M in (S, T, V):
        M[lower] = M.T[lower]
    return S, T, V


def basis_of_order(omega: int) -> tuple[tuple[int, int, int], ...]:
    """Every (i, j, m) with i + 2j + m <= omega."""
    return tuple(
        (total - 2 * j - m, j, m)
        for total in range(omega + 1)
        for j in range(total // 2 + 1)
        for m in range(total - 2 * j + 1)
    )


def _ground(basis, zeta: float, z: float):
    S, T, V = _matrices(basis, zeta, z)
    w, c = eigh(T + V, S)
    return float(w[0]), c[:, 0], T, V


@dataclass(frozen=True)
class HylleraasLevel:
    omega: int
    terms: int
    zeta: float
    energy: float  # hartree
    virial: float  # -<V>/<T>, 2 at the optimal zeta


def solve_basis(basis, z: float = 2.0, omega: int = -1) -> HylleraasLevel:
    """Optimize zeta for one basis and return the ground-state energy."""
    best = minimize_scalar(
        lambda zeta: _ground(basis, zeta, z)[0], bounds=(0.8, 4.0), method="bounded",
        options={"xatol": 1e-9},
    )
    e, c, T, V = _ground(basis, best.x, z)
    return HylleraasLevel(
        omega=omega, terms=len(basis), zeta=float(best.x), energy=e,
        virial=float(-(c @ V @ c) / (c @ T @ c)),
    )


@lru_cache(maxsize=4)
def hylleraas_series(omega_max: int = 5, z: float = 2.0) -> tuple[HylleraasLevel, ...]:
    """The ground-state energy for basis orders 0..omega_max."""
    return tuple(solve_basis(basis_of_order(w), z, w) for w in range(omega_max + 1))


def correlated_energy(omega_max: int = 5) -> Quantity:
    """Helium's ground state from the largest basis, with the last step as its spread."""
    series = hylleraas_series(omega_max)
    last, before = series[-1], series[-2]
    return Quantity(last.energy, "hartree", "E_Hylleraas", Provenance(
        fidelity=Fidelity.NUMERICAL,
        method=f"{_METHOD}; omega = {omega_max}, {last.terms} terms",
        assumptions=_ASSUMPTIONS,
        error_estimate=abs(before.energy - last.energy),
        refinement="a larger basis; the bound falls toward the exact value from above",
    ))


#: The basis order the report uses: 50 terms, 7e-7 hartree above the exact value.
REPORT_OMEGA = 6


@dataclass(frozen=True)
class HeliumReport:
    series: tuple[HylleraasLevel, ...]
    correlated: Quantity  # hartree, NUMERICAL
    hartree_fock: Quantity  # hartree, this package's own HF solve
    correlation: Quantity  # correlated - hartree_fock, hartree
    ie_hartree_fock: Quantity  # -2 - E_HF, hartree (Delta-SCF; He+ is exact)
    ie_correlated: Quantity  # -2 - E, hartree, clamped nucleus
    ie_reduced_mass: Quantity  # the same, scaled to helium-4's reduced mass


def helium_report() -> HeliumReport:
    """Correlated helium beside Hartree-Fock: the correlation energy and the IE."""
    series = hylleraas_series(REPORT_OMEGA)
    correlated = correlated_energy(REPORT_OMEGA)
    hf = solve_hartree_fock(2, 2, aufbau_configuration(2)).total_energy
    err_c = correlated.provenance.error_estimate or 0.0
    err_hf = hf.provenance.error_estimate or 0.0
    numerical = Provenance(
        fidelity=Fidelity.NUMERICAL,
        method="the Hylleraas energy minus this package's Hartree-Fock energy",
        assumptions=(
            "both solve the same non-relativistic, clamped-nucleus Hamiltonian, so "
            "the difference is a property of the model and not of the measurement",
        ),
        error_estimate=err_c + err_hf,
    )
    ie_prov = Provenance(
        fidelity=Fidelity.NUMERICAL,
        method="E(He+) - E(He), with E(He+) = -Z^2/2 = -2 hartree exactly",
        assumptions=_ASSUMPTIONS[:1],
        error_estimate=err_c,
    )
    mass_ratio = _sc.physical_constants["alpha particle-electron mass ratio"][0]
    mu = 1.0 / (1.0 + 1.0 / mass_ratio)
    return HeliumReport(
        series=series,
        correlated=correlated,
        hartree_fock=hf,
        correlation=Quantity(correlated.value - hf.value, "hartree", "E_corr", numerical),
        ie_hartree_fock=Quantity(-2.0 - hf.value, "hartree", "IE_HF", Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method="Delta-SCF Hartree-Fock: E(He+) - E_HF(He)",
            assumptions=("no correlation",) + _ASSUMPTIONS[:1],
            error_estimate=err_hf,
        )),
        ie_correlated=Quantity(-2.0 - correlated.value, "hartree", "IE", ie_prov),
        ie_reduced_mass=Quantity((-2.0 - correlated.value) * mu, "hartree", "IE_mu", Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method=(
                "the clamped-nucleus IE scaled by mu / m_e for helium-4, CODATA "
                "alpha-particle mass"
            ),
            assumptions=(
                "mass polarization (the p1 . p2 / M term) left out",
                "non-relativistic, no QED",
            ),
            error_estimate=err_c * mu,
        )),
    )
