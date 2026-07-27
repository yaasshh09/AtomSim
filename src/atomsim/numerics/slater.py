"""Slater radial integrals for the electron-electron interaction.

The pair potential

    U_k[a,b](r) = integral_0^inf ( r_<^k / r_>^(k+1) ) P_a(s) P_b(s) ds
                = r^-(k+1) integral_0^r  s^k     P_a P_b ds
                + r^k      integral_r^inf s^-(k+1) P_a P_b ds

is everything two-electron in Hartree-Fock. Both halves are cumulative
trapezoid integrals, so a whole pair potential costs O(N) and no ODE solve.

Two properties make this self-checking and are asserted in tests: U_0 tends to
1/r beyond the charge for a normalized density, and U_k is symmetric under
exchange of its two orbital arguments.

Hartree atomic units. P = r R(r) throughout, normalized as integral P^2 dr = 1.

Returns plain arrays and floats, not Quantity or Field. This is pure
quadrature, not physics: the caller in hf_atom.py wraps these results with
the provenance they belong to. That is a decision, not an oversight, and it
matches the exemption analytic/wigner.py already documents for 3j and 6j
symbols.
"""

import numpy as np

__all__ = ["pair_potential", "slater_f", "slater_g"]


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative integral of y dx from x[0], same length as x, starting at 0."""
    increments = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    out = np.empty_like(y)
    out[0] = 0.0
    np.cumsum(increments, out=out[1:])
    return out


def pair_potential(
    p_a: np.ndarray, p_b: np.ndarray, r: np.ndarray, k: int
) -> np.ndarray:
    """U_k[a,b](r), the multipole-k potential of the pair density P_a P_b."""
    if k < 0:
        raise ValueError(f"multipole order k must be >= 0, got {k}")
    if not (p_a.shape == p_b.shape == r.shape):
        raise ValueError("orbitals and grid must have the same shape")
    if r[0] <= 0.0:
        raise ValueError(
            f"radial grid must start strictly above zero, got r[0]={r[0]!r}; "
            "r = 0 makes the outer integrand 0 * inf = nan and np.cumsum then "
            "propagates that nan across the whole grid"
        )

    density = p_a * p_b
    inner = _cumulative_trapezoid(density * r**k, r)
    outer_total = _cumulative_trapezoid(density * r ** (-(k + 1)), r)
    outer = outer_total[-1] - outer_total
    return inner / r ** (k + 1) + outer * r**k


def slater_f(p_a: np.ndarray, p_b: np.ndarray, r: np.ndarray, k: int) -> float:
    """F^k(ab) = integral P_a^2(r) U_k[b,b](r) dr."""
    return float(np.trapezoid(p_a**2 * pair_potential(p_b, p_b, r, k), r))


def slater_g(p_a: np.ndarray, p_b: np.ndarray, r: np.ndarray, k: int) -> float:
    """G^k(ab) = integral P_a(r) P_b(r) U_k[a,b](r) dr."""
    return float(np.trapezoid(p_a * p_b * pair_potential(p_a, p_b, r, k), r))
