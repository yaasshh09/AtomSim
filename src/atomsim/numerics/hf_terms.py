"""My angular coefficients and Fock-operator terms for average-of-configuration
HF.

I derived them by varying the average-of-configuration energy functional with
respect to P_a. The functional is

    E = sum_a q_a I(a)
      + sum_a  (q_a (q_a - 1) / 2) [ F0(aa)
            - sum_{k>0} ((2 l_a + 1)/(4 l_a + 1)) tj(l_a,k,l_a)^2 Fk(aa) ]
      + sum_{a<b} q_a q_b [ F0(ab) - (1/2) sum_k tj(l_a,k,l_b)^2 Gk(ab) ]

writing tj(l1,k,l2) for wigner_3j(l1, k, l2, 0, 0, 0). Varying and dividing by
2 q_a gives me the Fock equation

    h P_a + (q_a - 1) U0[a,a] P_a + sum_{b != a} q_b U0[b,b] P_a
          - (q_a - 1) sum_{k>0} ((2 l_a + 1)/(4 l_a + 1)) tj(l_a,k,l_a)^2 Uk[a,a] P_a
          - sum_{b != a} (q_b / 2) sum_k tj(l_a,k,l_b)^2 Uk[a,b] P_b
          = eps_a P_a

Four checks pin this, all of them tests in tests/test_hf_terms.py and
tests/test_hartree_fock.py: hydrogen sees no self-interaction at all (the
(q_a - 1) factor), helium sees exactly one unit of U_0, my averaged
coefficients reduce to the independently derived closed-shell ones when q is
full, and beryllium gives the textbook 4J - 2K.

This is the part of me that fails silently when it is wrong: bad coefficients
give me converged, smooth, believable orbitals with the wrong energy. Do not
adjust a coefficient to make a benchmark match without re-deriving it.

My sign convention: `direct_potential` and `exchange_apply` both return the
magnitude of their term as written above, so my caller assembles the Fock
operator as h + direct - exchange. I return exchange positive and let the
caller subtract it; I do not pre-negate it here.

I return plain arrays and floats, not Quantity or Field, for the same reason I
do in numerics/slater.py: these are the pieces of an operator, not a physical
result. hf_atom.py attaches the provenance when it reports an energy.

I work in Hartree atomic units. P = r R(r), normalized so integral P^2 dr = 1.
"""

from dataclasses import dataclass

import numpy as np

from atomsim.analytic.wigner import wigner_3j
from atomsim.numerics.slater import MultipoleGeometry, multipole_geometry

__all__ = [
    "ExchangeOperator",
    "Subshell",
    "direct_potential",
    "exchange_apply",
    "exchange_coefficient",
    "exchange_operator",
    "same_shell_coefficient",
]


@dataclass(frozen=True)
class Subshell:
    """One (n, l) subshell with its occupancy and its current radial function."""

    n: int
    l: int
    q: int
    p: np.ndarray  # P_nl on my solver grid


def same_shell_coefficient(l_a: int, k: int, q_a: int) -> float:
    """The coefficient I put on U_k[a,a] P_a in the Fock equation, for k > 0."""
    if k <= 0:
        raise ValueError(f"same-shell exchange needs k > 0, got {k}")
    tj = wigner_3j(l_a, k, l_a, 0, 0, 0)
    return (q_a - 1) * ((2 * l_a + 1) / (4 * l_a + 1)) * tj * tj


def exchange_coefficient(l_a: int, k: int, l_b: int, q_b: int) -> float:
    """The coefficient I put on U_k[a,b] P_b in the Fock equation, for b != a."""
    tj = wigner_3j(l_a, k, l_b, 0, 0, 0)
    return 0.5 * q_b * tj * tj


def direct_potential(
    subshells: tuple[Subshell, ...], a_index: int, r: np.ndarray
) -> np.ndarray:
    """The local Hartree potential I let subshell a see.

    (q_a - 1) U0[a,a] + sum_{b != a} q_b U0[b,b]. The (q_a - 1) is what makes a
    one-electron atom see nothing at all.
    """
    a = subshells[a_index]
    # Every term here is k = 0, so I build the grid factors once rather than
    # once per subshell.
    geometry = multipole_geometry(r, 0)
    v = (a.q - 1) * geometry.potential(a.p, a.p)
    for i, b in enumerate(subshells):
        if i != a_index:
            v = v + b.q * geometry.potential(b.p, b.p)
    return v


@dataclass(frozen=True)
class ExchangeOperator:
    """My exchange operator for one subshell, with psi factored out.

    Everything about exchange except the trial function is fixed once the
    orbitals are: which partners contribute, at which multipole order, with
    which angular coefficient, and the grid factors each order needs. Only the
    pair potential depends on psi.

    That split matters because of my caller. LOBPCG applies this to its search
    directions hundreds of times per channel, and the previous shape of my code
    rebuilt the Wigner coefficients and the r**k arrays on every one of those
    applications. Building once and applying many times is the same arithmetic
    in the same order, so my result is bit-identical.

    Each term is (coefficient, partner P, geometry for its k), and I sum over
    terms of coefficient * U_k[partner, psi] * partner.
    """

    terms: tuple[tuple[float, np.ndarray, MultipoleGeometry], ...]

    def apply(self, psi: np.ndarray) -> np.ndarray:
        out = np.zeros_like(psi)
        for coefficient, partner, geometry in self.terms:
            out += coefficient * geometry.potential(partner, psi) * partner
        return out


def exchange_operator(
    subshells: tuple[Subshell, ...], a_index: int, r: np.ndarray
) -> ExchangeOperator:
    """I build the exchange operator for subshell a on this grid.

    I drop terms with a vanishing angular coefficient here rather than skipping
    them per application, so once the operator exists a selection rule costs me
    nothing at all.
    """
    a = subshells[a_index]
    geometries: dict[int, MultipoleGeometry] = {}

    def geometry(k: int) -> MultipoleGeometry:
        if k not in geometries:
            geometries[k] = multipole_geometry(r, k)
        return geometries[k]

    terms: list[tuple[float, np.ndarray, MultipoleGeometry]] = []

    # I leave k = 0 out deliberately: the (q_a - 1) factor in direct_potential
    # already carries the same-shell k = 0 exchange. Including it here is the
    # most likely way for me to break helium. Odd k vanish by parity, hence the
    # step of 2.
    for k in range(2, 2 * a.l + 1, 2):
        c = same_shell_coefficient(a.l, k, a.q)
        if c:
            terms.append((c, a.p, geometry(k)))

    for i, b in enumerate(subshells):
        if i == a_index:
            continue
        for k in range(abs(a.l - b.l), a.l + b.l + 1):
            c = exchange_coefficient(a.l, k, b.l, b.q)
            if c:
                terms.append((c, b.p, geometry(k)))

    return ExchangeOperator(terms=tuple(terms))


def exchange_apply(
    subshells: tuple[Subshell, ...], a_index: int, psi: np.ndarray, r: np.ndarray
) -> np.ndarray:
    """I apply the non-local exchange operator for subshell a to a trial psi.

    psi is any function in the l_a channel, not only an occupied orbital:
    LOBPCG applies this to its search directions, so I rebuild the pair
    potentials from psi on every call rather than caching them from the last
    SCF step.

    I build the operator here and throw it away. If you are applying it more
    than once against the same orbitals, go through `exchange_operator`
    instead.
    """
    return exchange_operator(subshells, a_index, r).apply(psi)
