"""Hartree-Fock past argon, and the 3d/4s ordering it can now speak to.

Two kinds of claim, kept apart on purpose. The first is accuracy: potassium,
calcium and scandium have to land on their published Hartree-Fock limits inside
the solver's own error bar, or nothing said about their orderings means
anything. The second is the ordering puzzle itself, which every textbook states
and few show: from scandium on, a 3d orbital is deeper than the 4s (lower
epsilon), yet the atom puts its electrons in 4s anyway, because what decides a
configuration is its total energy, not one orbital's. Each assertion below is a
line the Levels view prints.

Configuration averages throughout: the solver minimizes the average energy of a
configuration, not a term, so these compare configurations the way the
textbook argument does.
"""

from functools import lru_cache

import pytest

from atomsim.atoms import parse_config
from atomsim.hf_atom import solve_hartree_fock

AR = "1s2 2s2 2p6 3s2 3p6"


@lru_cache(maxsize=None)
def _solve(z: int, n: int, tail: str):
    return solve_hartree_fock(z, n, parse_config(f"{AR} {tail}"))


def _eps(res, n: int, l: int) -> float:
    return next(o.energy.value for o in res.orbitals if (o.n, o.l) == (n, l))


# Numerical Hartree-Fock limits, Bunge, Barrientos and Bunge, At. Data Nucl.
# Data Tables 53, 113 (1993), for the configuration each ground term lives in.
@pytest.mark.parametrize(
    "z,tail,limit",
    [(19, "4s1", -599.164787), (20, "4s2", -676.758186), (21, "3d1 4s2", -759.735718)],
)
def test_past_argon_lands_on_the_hartree_fock_limit(z, tail, limit):
    res = _solve(z, z, tail)
    e = res.total_energy
    assert abs(e.value - limit) <= max(e.provenance.error_estimate, 3e-3)


def test_potassium_puts_its_last_electron_in_4s():
    s, d = _solve(19, 19, "4s1"), _solve(19, 19, "3d1")
    gap_ev = (d.total_energy.value - s.total_energy.value) * 27.211386
    # NIST puts K 3d at 2.67 eV above 4s; Hartree-Fock, missing correlation, lands near it.
    assert 2.0 < gap_ev < 3.0
    assert _eps(s, 4, 0) < _eps(d, 3, 2)  # and in potassium the 4s is also the deeper orbital


def test_calcium_fills_4s2_before_any_3d():
    energies = {t: _solve(20, 20, t).total_energy.value for t in ("4s2", "3d1 4s1", "3d2")}
    assert min(energies, key=energies.get) == "4s2"


def test_scandium_fills_4s_although_3d_is_the_deeper_orbital():
    """The puzzle, in one atom: epsilon says 3d, the total energy says 4s^2."""
    ground = _solve(21, 21, "3d1 4s2")
    assert _eps(ground, 3, 2) < _eps(ground, 4, 0)  # 3d is below 4s...
    moved = _solve(21, 21, "3d2 4s1")
    assert ground.total_energy.value < moved.total_energy.value  # ...yet 4s^2 wins


def test_ionizing_scandium_takes_a_4s_electron_not_the_3d():
    energies = {t: _solve(21, 20, t).total_energy.value for t in ("3d1 4s1", "3d2", "4s2")}
    assert min(energies, key=energies.get) == "3d1 4s1"  # Sc+ ground: 3d 4s, as NIST has it
