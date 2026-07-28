"""Angular coefficients and Fock-operator terms for average-of-configuration HF.

Derived by varying the average-of-configuration energy functional with respect
to P_a. The functional is

    E = sum_a q_a I(a)
      + sum_a  (q_a (q_a - 1) / 2) [ F0(aa)
            - sum_{k>0} ((2 l_a + 1)/(4 l_a + 1)) tj(l_a,k,l_a)^2 Fk(aa) ]
      + sum_{a<b} q_a q_b [ F0(ab) - (1/2) sum_k tj(l_a,k,l_b)^2 Gk(ab) ]

writing tj(l1,k,l2) for wigner_3j(l1, k, l2, 0, 0, 0). Varying and dividing by
2 q_a gives the Fock equation

    h P_a + (q_a - 1) U0[a,a] P_a + sum_{b != a} q_b U0[b,b] P_a
          - (q_a - 1) sum_{k>0} ((2 l_a + 1)/(4 l_a + 1)) tj(l_a,k,l_a)^2 Uk[a,a] P_a
          - sum_{b != a} (q_b / 2) sum_k tj(l_a,k,l_b)^2 Uk[a,b] P_b
          = eps_a P_a

Four checks pin this, all of them tests in tests/test_hf_terms.py and
tests/test_hartree_fock.py: hydrogen has no self-interaction at all (the
(q_a - 1) factor), helium sees exactly one unit of U_0, the averaged
coefficients reduce to the independently derived closed-shell ones when q is
full, and beryllium gives the textbook 4J - 2K.

This is the part of the phase that fails silently if it is wrong: bad
coefficients produce converged, smooth, believable orbitals with the wrong
energy. Do not adjust a coefficient to make a benchmark match without
re-deriving it.

Sign convention: `direct_potential` and `exchange_apply` both return the
magnitude of their term as written above, so the caller assembles the Fock
operator as h + direct - exchange. Exchange is returned positive and
subtracted by the caller; it is not pre-negated here.

Returns plain arrays and floats, not Quantity or Field, for the same reason
numerics/slater.py does: these are the pieces of an operator, not a physical
result. hf_atom.py attaches the provenance when it reports an energy.

Hartree atomic units. P = r R(r), normalized as integral P^2 dr = 1.
"""

from dataclasses import dataclass

import numpy as np

from atomsim.analytic.wigner import wigner_3j
from atomsim.numerics.slater import pair_potential

__all__ = [
    "Subshell",
    "direct_potential",
    "exchange_apply",
    "exchange_coefficient",
    "same_shell_coefficient",
]


@dataclass(frozen=True)
class Subshell:
    """One (n, l) subshell with its occupancy and current radial function."""

    n: int
    l: int
    q: int
    p: np.ndarray  # P_nl on the solver grid


def same_shell_coefficient(l_a: int, k: int, q_a: int) -> float:
    """Coefficient of U_k[a,a] P_a in the Fock equation, for k > 0."""
    if k <= 0:
        raise ValueError(f"same-shell exchange needs k > 0, got {k}")
    tj = wigner_3j(l_a, k, l_a, 0, 0, 0)
    return (q_a - 1) * ((2 * l_a + 1) / (4 * l_a + 1)) * tj * tj


def exchange_coefficient(l_a: int, k: int, l_b: int, q_b: int) -> float:
    """Coefficient of U_k[a,b] P_b in the Fock equation, for b != a."""
    tj = wigner_3j(l_a, k, l_b, 0, 0, 0)
    return 0.5 * q_b * tj * tj


def direct_potential(
    subshells: tuple[Subshell, ...], a_index: int, r: np.ndarray
) -> np.ndarray:
    """The local Hartree potential seen by subshell a.

    (q_a - 1) U0[a,a] + sum_{b != a} q_b U0[b,b]. The (q_a - 1) is what makes a
    one-electron atom see nothing at all.
    """
    a = subshells[a_index]
    v = (a.q - 1) * pair_potential(a.p, a.p, r, 0)
    for i, b in enumerate(subshells):
        if i != a_index:
            v = v + b.q * pair_potential(b.p, b.p, r, 0)
    return v


def exchange_apply(
    subshells: tuple[Subshell, ...], a_index: int, psi: np.ndarray, r: np.ndarray
) -> np.ndarray:
    """Apply the non-local exchange operator for subshell a to a trial psi.

    psi is any function in the l_a channel, not only an occupied orbital:
    LOBPCG applies this to its search directions, so the pair potentials are
    rebuilt from psi on every call rather than cached from the last SCF step.
    """
    a = subshells[a_index]
    out = np.zeros_like(psi)

    # k = 0 is deliberately absent: the (q_a - 1) factor in direct_potential
    # already carries the same-shell k = 0 exchange. Including it here is the
    # most likely way to break helium. Odd k vanish by parity, hence step 2.
    for k in range(2, 2 * a.l + 1, 2):
        c = same_shell_coefficient(a.l, k, a.q)
        if c:
            out = out + c * pair_potential(a.p, psi, r, k) * a.p

    for i, b in enumerate(subshells):
        if i == a_index:
            continue
        for k in range(abs(a.l - b.l), a.l + b.l + 1):
            c = exchange_coefficient(a.l, k, b.l, b.q)
            if c:
                out = out + c * pair_potential(b.p, psi, r, k) * b.p

    return out
