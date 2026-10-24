"""The spontaneous decay cascade of an excited hydrogen-like level.

An atom left in (n, l) does not drop straight to the ground state. Each level
empties at its total rate Gamma = sum A, splitting between its electric-dipole
channels in the branching ratios b = A / Gamma, and every level it lands in does
the same. Because the gross-structure energy depends on n alone, every E1 decay
lowers n, so the levels form a directed acyclic graph and the cascade is solved
in one pass from the top:

    P(start) = 1,   P(lower) = sum_upper P(upper) b(upper -> lower)

P is the probability that the cascade passes through a level, P(upper) b is the
expected number of photons on that line per excited atom, and the mean time to
settle is sum P tau over the levels that decay. The populations in time solve
dN/dt = M N with M built from the same rates, by matrix exponential.

Two levels have no E1 channel at all and end the cascade: 1s, and 2s. A real 2s
atom decays by two-photon emission at 8.229 Z^6 s^-1 (about 1e8 times slower
than 2p), which is not modelled here; it is a terminal level, disclosed as such,
and the fraction trapped there is one of the results. Everything is NUMERICAL,
inheriting the Gauss-Laguerre rates of `analytic.transitions`.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from atomsim.analytic.hydrogen import energy, validate_quantum_numbers
from atomsim.analytic.transitions import einstein_A
from atomsim.constants import HARTREE_EV
from atomsim.provenance import Field, Fidelity, Provenance, Quantity
from atomsim.spectra import _EV_NM, orbital_label

_ASSUMPTIONS = (
    "one-electron hydrogenic rates in the electric-dipole approximation",
    "gross structure only: no fine-structure splitting of the rates",
    "an isolated atom: no collisions, no absorption, no stimulated emission",
    "1s and 2s have no E1 channel and end the cascade; the 2s two-photon decay "
    "(8.229 Z^6 s^-1) and positronium annihilation are not modelled",
)


@dataclass(frozen=True)
class CascadeLevel:
    n: int
    l: int
    visit_probability: Quantity  # chance the cascade passes through this level
    lifetime: Quantity           # seconds; infinite for a terminal level
    terminal: bool


@dataclass(frozen=True)
class CascadeBranch:
    upper: tuple[int, int]
    lower: tuple[int, int]
    einstein_A: Quantity
    branching_ratio: Quantity
    photons: Quantity            # expected photons on this line per excited atom
    wavelength: Quantity         # nm, vacuum


@dataclass(frozen=True)
class Cascade:
    start: tuple[int, int]
    levels: list[CascadeLevel]   # ordered from the start downward in energy
    branches: list[CascadeBranch]
    mean_settle_time: Quantity
    populations: list[Field]     # one per level, on a shared log-spaced time grid


def _rate_table(n: int, l: int, Z: int, mu_ratio: float):
    """Every reachable level and its E1 decay rates, found top-down."""
    rates: dict[tuple[int, int], dict[tuple[int, int], Quantity]] = {}
    todo = [(n, l)]
    while todo:
        up = todo.pop()
        if up in rates:
            continue
        rates[up] = {}
        nu, lu = up
        for n2 in range(1, nu):
            for l2 in (lu - 1, lu + 1):
                if 0 <= l2 < n2:
                    rates[up][(n2, l2)] = einstein_A(nu, lu, n2, l2, Z=Z, mu_ratio=mu_ratio)
                    todo.append((n2, l2))
    # Energy order is n order; within one n, l order keeps the output stable.
    order = sorted(rates, key=lambda k: (-k[0], -k[1]))
    return order, rates


def _time_grid(lifetimes: list[float], points: int) -> np.ndarray:
    finite = [t for t in lifetimes if np.isfinite(t)]
    if not finite:
        return np.array([0.0])
    return np.geomspace(1e-3 * min(finite), 30.0 * max(finite), points)


def decay_cascade(
    n: int, l: int, Z: int = 1, mu_ratio: float = 1.0, points: int = 160
) -> Cascade:
    """The full radiative cascade from (n, l) down to the levels that cannot decay."""
    validate_quantum_numbers(n, l)
    order, rates = _rate_table(n, l, Z, mu_ratio)
    index = {k: i for i, k in enumerate(order)}

    gamma = {k: sum(a.value for a in rates[k].values()) for k in order}
    # Each rate carries its roundoff; a branching ratio A / sum A then has at most
    # twice the worst relative error, and a visit probability is a sum of
    # products of at most n - 1 of them.
    eps = max(
        ((a.provenance.error_estimate or 0.0) / a.value
         for k in order for a in rates[k].values() if a.value > 0.0),
        default=0.0,
    )
    rel_b = 2.0 * eps
    rel_p = 2.0 * max(n - 1, 1) * eps

    visit = {k: 0.0 for k in order}
    visit[(n, l)] = 1.0
    branches: list[CascadeBranch] = []
    for up in order:                      # parents always precede children
        for low, a in rates[up].items():
            b = a.value / gamma[up]
            flux = visit[up] * b
            visit[low] += flux
            de_ev = (energy(up[0], Z=Z, mu_ratio=mu_ratio).value
                     - energy(low[0], Z=Z, mu_ratio=mu_ratio).value) * HARTREE_EV
            label = f"{orbital_label(*up)}->{orbital_label(*low)}"
            branches.append(CascadeBranch(
                upper=up,
                lower=low,
                einstein_A=a,
                branching_ratio=Quantity(b, "dimensionless", f"b {label}", Provenance(
                    fidelity=Fidelity.NUMERICAL,
                    method="b = A / sum A over the upper level's E1 channels",
                    assumptions=_ASSUMPTIONS,
                    error_estimate=rel_b * b,
                )),
                photons=Quantity(flux, "photons per excited atom", f"N {label}", Provenance(
                    fidelity=Fidelity.NUMERICAL,
                    method="P(upper) * b, with P propagated down the cascade",
                    assumptions=_ASSUMPTIONS,
                    error_estimate=(rel_p + rel_b) * flux,
                )),
                wavelength=Quantity(_EV_NM / de_ev, "nm (vacuum)", f"lambda {label}", Provenance(
                    fidelity=Fidelity.EXACT,
                    method="hc / (E_upper - E_lower), Bohr levels with reduced mass",
                    assumptions=("gross structure: no fine-structure or Lamb shift",),
                )),
            ))

    lifetimes = {k: (1.0 / gamma[k] if gamma[k] > 0.0 else np.inf) for k in order}
    levels = [
        CascadeLevel(
            n=k[0],
            l=k[1],
            visit_probability=Quantity(visit[k], "dimensionless", f"P {orbital_label(*k)}",
                                       Provenance(
                fidelity=Fidelity.NUMERICAL,
                method="P(lower) = sum P(upper) b(upper -> lower), top down",
                assumptions=_ASSUMPTIONS,
                error_estimate=rel_p * visit[k],
            )),
            lifetime=Quantity(lifetimes[k], "s", f"tau {orbital_label(*k)}", Provenance(
                fidelity=Fidelity.NUMERICAL,
                method="tau = 1 / sum A over E1 channels; infinite where there are none",
                assumptions=_ASSUMPTIONS,
                error_estimate=rel_b * lifetimes[k] if gamma[k] > 0.0 else 0.0,
            )),
            terminal=gamma[k] == 0.0,
        )
        for k in order
    ]

    settle = sum(visit[k] * lifetimes[k] for k in order if gamma[k] > 0.0)
    mean_settle = Quantity(settle, "s", "mean time to settle", Provenance(
        fidelity=Fidelity.NUMERICAL,
        method="sum P(level) tau(level) over the levels that decay (mean first-passage time)",
        assumptions=_ASSUMPTIONS,
        error_estimate=(rel_p + rel_b) * settle,
    ))

    # dN_i/dt = -Gamma_i N_i + sum_j A(j -> i) N_j, solved as N(t) = exp(M t) N(0).
    size = len(order)
    M = np.zeros((size, size))
    for up in order:
        M[index[up], index[up]] = -gamma[up]
        for low, a in rates[up].items():
            M[index[low], index[up]] += a.value
    t = _time_grid([lifetimes[k] for k in order], points)
    n0 = np.zeros(size)
    n0[index[(n, l)]] = 1.0
    N = np.array([expm(M * ti) @ n0 for ti in t]).T
    # The populations must sum to one; the drift measures the exponential's roundoff.
    drift = float(np.max(np.abs(N.sum(axis=0) - 1.0)))
    pop_prov = Provenance(
        fidelity=Fidelity.NUMERICAL,
        method="N(t) = expm(M t) N(0), M from the E1 rates (scipy.linalg.expm)",
        assumptions=_ASSUMPTIONS,
        error_estimate=drift,
        refinement="the error is the worst departure of sum N from 1 across the grid",
    )
    populations = [
        Field(N[index[k]], t, "dimensionless", "s", f"N_{orbital_label(*k)}(t)", pop_prov)
        for k in order
    ]
    return Cascade(
        start=(n, l),
        levels=levels,
        branches=branches,
        mean_settle_time=mean_settle,
        populations=populations,
    )
