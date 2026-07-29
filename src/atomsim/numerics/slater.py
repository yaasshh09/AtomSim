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

from dataclasses import dataclass

import numpy as np

__all__ = [
    "MultipoleGeometry",
    "multipole_geometry",
    "pair_potential",
    "slater_f",
    "slater_g",
]


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative integral of y dx from x[0], same length as x, starting at 0."""
    increments = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    out = np.empty_like(y)
    out[0] = 0.0
    np.cumsum(increments, out=out[1:])
    return out


@dataclass(frozen=True)
class MultipoleGeometry:
    """Everything in U_k[a,b] that depends on the grid and k but not the pair.

    Exists for one reason: inside an SCF, `pair_potential` is called from the
    LOBPCG matvec, hundreds of times per channel, against an unchanged grid and
    an unchanged k while only the pair changes. Each of those calls was
    rebuilding r**k, r**-(k+1) and diff(r) from scratch. On argon that was
    25043 calls doing four array powers apiece, and it was the single largest
    cost in the whole solve.

    Splitting it out is a pure hoist: `potential` does the same arithmetic on
    the same values in the same order, so it returns bit-identical results.
    Build one per (grid, k) and reuse it.
    """

    r: np.ndarray
    half_dr: np.ndarray  # 0.5 * diff(r), the trapezoid weights
    r_k: np.ndarray  # r**k
    r_k1: np.ndarray  # r**(k+1), for the inner term's divisor
    r_inv_k1: np.ndarray  # r**-(k+1), for the outer term's integrand

    def potential(self, p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
        """U_k[a,b](r) for this grid and this k."""
        density = p_a * p_b
        inner = self._cumulative(density * self.r_k)
        outer_total = self._cumulative(density * self.r_inv_k1)
        # Divides by r**(k+1) rather than multiplying by r**-(k+1). Both terms
        # are needed anyway, and the division is what the unhoisted code did:
        # a * (1/b) rounds twice where a / b rounds once, so keeping the divide
        # makes this hoist bit-identical instead of merely close. Measured, the
        # swap moved argon's total energy in the eleventh decimal.
        return inner / self.r_k1 + (outer_total[-1] - outer_total) * self.r_k

    def _cumulative(self, y: np.ndarray) -> np.ndarray:
        out = np.empty_like(y)
        out[0] = 0.0
        np.cumsum((y[1:] + y[:-1]) * self.half_dr, out=out[1:])
        return out


def multipole_geometry(r: np.ndarray, k: int) -> MultipoleGeometry:
    """Build the reusable grid factors for multipole order k.

    Carries the validation, so anything built through here has already been
    checked and `MultipoleGeometry.potential` can stay a hot inner loop.
    """
    if k < 0:
        raise ValueError(f"multipole order k must be >= 0, got {k}")
    if r[0] <= 0.0:
        raise ValueError(
            f"radial grid must start strictly above zero, got r[0]={r[0]!r}; "
            "r = 0 makes the outer integrand 0 * inf = nan and np.cumsum then "
            "propagates that nan across the whole grid"
        )
    return MultipoleGeometry(
        r=r,
        half_dr=0.5 * np.diff(r),
        r_k=r**k,
        r_k1=r ** (k + 1),
        r_inv_k1=r ** (-(k + 1)),
    )


def pair_potential(
    p_a: np.ndarray, p_b: np.ndarray, r: np.ndarray, k: int
) -> np.ndarray:
    """U_k[a,b](r), the multipole-k potential of the pair density P_a P_b.

    Rebuilds the grid factors every call. Fine for one-off use; inside a loop
    over pairs or over LOBPCG iterations, build a `multipole_geometry` once and
    call its `potential` instead.
    """
    if not (p_a.shape == p_b.shape == r.shape):
        raise ValueError("orbitals and grid must have the same shape")
    return multipole_geometry(r, k).potential(p_a, p_b)


def slater_f(p_a: np.ndarray, p_b: np.ndarray, r: np.ndarray, k: int) -> float:
    """F^k(ab) = integral P_a^2(r) U_k[b,b](r) dr."""
    return float(np.trapezoid(p_a**2 * pair_potential(p_b, p_b, r, k), r))


def slater_g(p_a: np.ndarray, p_b: np.ndarray, r: np.ndarray, k: int) -> float:
    """G^k(ab) = integral P_a(r) P_b(r) U_k[a,b](r) dr."""
    return float(np.trapezoid(p_a * p_b * pair_potential(p_a, p_b, r, k), r))
