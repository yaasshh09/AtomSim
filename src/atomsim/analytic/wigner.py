"""My Wigner 6j symbols, by the Racah formula.

This is the angular-momentum recoupling algebra I need to split a
gross-structure transition rate across its fine-structure components. See
docs/specs/2026-07-25-phase15-fine-structure-line-strengths-design.md.

**On my provenance rule.** Everywhere else in this package I return a
`Quantity` or a `Field` carrying a `Fidelity`. I make a 6j symbol deliberately
different: it is a dimensionless algebraic constant determined entirely by six
angular-momentum quantum numbers, in the same category as a Clebsch-Gordan
coefficient or pi, not a modelled physical value with a fidelity tier. So I
return plain `float` here, and the physical `Quantity` I build from them in
`transitions.py` carries the provenance for the rate. That is a decision of
mine, not an oversight.

You may give me integer or half-integer arguments. I carry them internally as
doubled integers, so I decide triangle conditions by integer arithmetic and
never by a floating-point comparison. My values are exact up to float64 roundoff
in the alternating factorial sum, and I sum over a short range for the small
momenta this project uses (l <= 10, spin 1/2), so cancellation is not a
practical issue for me.
"""

import math
from functools import lru_cache

__all__ = ["triangular", "wigner_3j", "wigner_6j"]

# Both symbols are pure functions of six small numbers, so when I memoize them I
# am being exact rather than approximate: my cache returns the identical float
# the Racah sum would have produced, not a rounded or interpolated one.
#
# It is worth doing because of where these get called from. My Hartree-Fock
# exchange operator asks for its angular coefficients inside the LOBPCG matvec,
# which runs hundreds of times per channel against an unchanged (l_a, k, l_b),
# and each miss costs me a Racah sum of factorials. When I profiled argon I saw
# 23907 calls to wigner_3j driving 389484 calls to math.factorial.
#
# The distinct arguments in an atomic calculation number in the dozens, since I
# build them from l values of 0 to 3 and small multipole orders, so the physics
# bounds my cache in practice rather than maxsize doing it.
_CACHE_SIZE = 4096


def _doubled(j: float, name: str) -> int:
    """2j as an exact integer. I reject anything that is not a (half-)integer."""
    two_j = round(2 * j)
    if abs(2 * j - two_j) > 1e-9:
        raise ValueError(f"{name} must be integer or half-integer, got {j}")
    if two_j < 0:
        raise ValueError(f"{name} must be non-negative, got {j}")
    return two_j


def _triangular_doubled(a2: int, b2: int, c2: int) -> bool:
    """The triangle condition I apply on doubled momenta: |a-b| <= c <= a+b, and a+b+c integral."""
    if (a2 + b2 + c2) % 2 != 0:
        return False
    return abs(a2 - b2) <= c2 <= a2 + b2


def triangular(a: float, b: float, c: float) -> bool:
    """Whether (a, b, c) form a valid angular-momentum triad."""
    return _triangular_doubled(_doubled(a, "a"), _doubled(b, "b"), _doubled(c, "c"))


def _delta(a2: int, b2: int, c2: int) -> float:
    """Racah's Delta(abc) = sqrt( (a+b-c)! (a-b+c)! (-a+b+c)! / (a+b+c+1)! )."""
    return math.sqrt(
        math.factorial((a2 + b2 - c2) // 2)
        * math.factorial((a2 - b2 + c2) // 2)
        * math.factorial((-a2 + b2 + c2) // 2)
        / math.factorial((a2 + b2 + c2) // 2 + 1)
    )


@lru_cache(maxsize=_CACHE_SIZE)
def wigner_6j(
    j1: float, j2: float, j3: float, j4: float, j5: float, j6: float
) -> float:
    """The 6j symbol {j1 j2 j3; j4 j5 j6}.

    I return exactly 0.0 when any of the four triads fails its triangle
    condition, which is what makes the selection rules structural rather than
    something my caller has to special-case.
    """
    a = [_doubled(j, n) for j, n in
         ((j1, "j1"), (j2, "j2"), (j3, "j3"), (j4, "j4"), (j5, "j5"), (j6, "j6"))]
    j1_, j2_, j3_, j4_, j5_, j6_ = a

    # The four triads the symbol couples; any failure means I return a zero.
    triads = ((j1_, j2_, j3_), (j1_, j5_, j6_), (j4_, j2_, j6_), (j4_, j5_, j3_))
    if not all(_triangular_doubled(*t) for t in triads):
        return 0.0

    prefactor = 1.0
    for t in triads:
        prefactor *= _delta(*t)

    # My Racah sum: t runs over the range where every factorial argument is >= 0.
    lower = max(sum(t) for t in triads)
    upper = min(
        j1_ + j2_ + j4_ + j5_,
        j2_ + j3_ + j5_ + j6_,
        j1_ + j3_ + j4_ + j6_,
    )
    total = 0.0
    for t2 in range(lower, upper + 1, 2):   # doubled, so step by 2
        t = t2 // 2
        denom = (
            math.factorial(t - sum(triads[0]) // 2)
            * math.factorial(t - sum(triads[1]) // 2)
            * math.factorial(t - sum(triads[2]) // 2)
            * math.factorial(t - sum(triads[3]) // 2)
            * math.factorial((j1_ + j2_ + j4_ + j5_) // 2 - t)
            * math.factorial((j2_ + j3_ + j5_ + j6_) // 2 - t)
            * math.factorial((j1_ + j3_ + j4_ + j6_) // 2 - t)
        )
        total += (-1.0) ** t * math.factorial(t + 1) / denom

    return prefactor * total


def _doubled_m(m: float, name: str) -> int:
    """2m as an exact integer. Unlike in _doubled, I let m be negative here."""
    two_m = round(2 * m)
    if abs(2 * m - two_m) > 1e-9:
        raise ValueError(f"{name} must be integer or half-integer, got {m}")
    return two_m


@lru_cache(maxsize=_CACHE_SIZE)
def wigner_3j(
    j1: float, j2: float, j3: float, m1: float, m2: float, m3: float
) -> float:
    """The 3j symbol (j1 j2 j3; m1 m2 m3), by the Racah formula.

    I return exactly 0.0 when the projections do not sum to zero, when any
    |m_i| > j_i, or when the triangle condition fails, so the selection rules
    are structural rather than something my caller has to special-case.

    My Hartree-Fock angular coefficients need only the m1=m2=m3=0 case, where
    the symbol also vanishes unless j1+j2+j3 is even. I do not special-case that
    parity rule: it falls out of the general formula.
    """
    j1_, j2_, j3_ = (_doubled(j, n) for j, n in ((j1, "j1"), (j2, "j2"), (j3, "j3")))
    m1_, m2_, m3_ = (_doubled_m(m, n) for m, n in ((m1, "m1"), (m2, "m2"), (m3, "m3")))

    if m1_ + m2_ + m3_ != 0:
        return 0.0
    for j, m in ((j1_, m1_), (j2_, m2_), (j3_, m3_)):
        if abs(m) > j or (j - m) % 2 != 0:  # I require m to share j's half-integrality
            return 0.0
    if not _triangular_doubled(j1_, j2_, j3_):
        return 0.0

    prefactor = _delta(j1_, j2_, j3_)
    for j, m in ((j1_, m1_), (j2_, m2_), (j3_, m3_)):
        prefactor *= math.sqrt(
            math.factorial((j + m) // 2) * math.factorial((j - m) // 2)
        )

    # My Racah sum: t runs where every factorial argument stays non-negative.
    # My three lower bounds come from t >= 0, (j3-j2+m1)/2 + t >= 0 and
    # (j3-j1-m2)/2 + t >= 0; my three upper bounds from the remaining three
    # factorials.
    lower = max(0, -((j3_ - j2_ + m1_) // 2), -((j3_ - j1_ - m2_) // 2))
    upper = min(
        (j1_ + j2_ - j3_) // 2,
        (j1_ - m1_) // 2,
        (j2_ + m2_) // 2,
    )
    total = 0.0
    for t in range(lower, upper + 1):
        denom = (
            math.factorial(t)
            * math.factorial((j3_ - j2_ + m1_) // 2 + t)
            * math.factorial((j3_ - j1_ - m2_) // 2 + t)
            * math.factorial((j1_ + j2_ - j3_) // 2 - t)
            * math.factorial((j1_ - m1_) // 2 - t)
            * math.factorial((j2_ + m2_) // 2 - t)
        )
        total += (-1.0) ** t / denom

    sign = (-1.0) ** ((j1_ - j2_ - m3_) // 2)
    return sign * prefactor * total
