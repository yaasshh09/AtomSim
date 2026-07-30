"""Regression guards on what a Hartree-Fock solve costs.

These are guards, not benchmarks. A benchmark asks "how fast is this"; a guard
asks "did someone silently undo the reason it is fast", and the two want
different thresholds. Every number here is set from a measurement with enough
headroom to survive a slow CI runner, and every one of them is a trap this
module has actually fallen into:

  - argon once ran on 72000 uniform points and took about an hour. The
    exponential mesh cut it to ~2800 points and a few seconds. A grid change
    that looks harmless can put that hour back.
  - the SCF mixing parameter was 0.4 before it was tuned to 0.65, which is a
    2.7x wall-time difference for no change in any digit of the answer, so
    nothing about the physics would go red if it drifted back.
  - solve_hartree_fock is memoized, and several test files lean on that to run
    in seconds rather than minutes.

Wall clock is the least informative of the three, because it measures the
machine as much as the code. The iteration and point counts do not, so they
carry most of the weight here and the clock is the backstop.
"""

import time

import pytest

from atomsim.atoms import aufbau_configuration
from atomsim.hf_atom import hf_mesh, solve_hartree_fock

# Measured on the development machine (Windows, Python 3.12), cold: He 0.18s,
# Be 1.07s, Ne 1.63s, Mg 3.40s, Ar 4.66s, Cl 7.18s. Argon is not the slowest
# atom - the third-row open shells cost more - so the ceiling is set from
# chlorine and applied to both.
_WALL_CLOCK_CEILING = 60.0

# Measured coarse-mesh SCF counts across Z = 1..18 span 2 (H) to 17 (Li, B, S).
# At the old mixing parameter of 0.4 the same solves took 34 (Ar) to 40 (Ne),
# so a ceiling of 25 separates the tuned case from the untuned one with about
# 1.5x headroom on the tuned side. Iteration counts are integers produced by a
# deterministic loop, so unlike the clock they do not need slack for the host.
_COARSE_SCF_CEILING = 25

# Fine-mesh counts span 2 to 8 over the same range.
_FINE_SCF_CEILING = 15

# Argon's fine mesh is ~2800 points, its coarse one ~1400. The uniform grid
# this replaced needed 72000. Anything between those is a mesh regression
# whether or not the answer is still right.
_ARGON_POINT_CEILING = 6000


def _solve_cold(z: int):
    """Solve without touching the memo table, in either direction.

    Calling solve_hartree_fock directly would measure a cache hit whenever some
    earlier test in the session solved the same atom, and this suite is run
    under pytest-randomly, so whether that happened is not knowable from here -
    a timing guard that silently measures a dict lookup passes forever.

    Clearing the cache would fix that and create a worse problem: the HF test
    files share it, and wiping it mid-session makes every later test re-solve
    from scratch. Reaching past the decorator costs nothing and takes nothing
    away from anyone.
    """
    assert hasattr(solve_hartree_fock, "__wrapped__"), (
        "solve_hartree_fock is no longer memoized; test_repeated_solves_hit_"
        "the_cache should have caught this first"
    )
    return solve_hartree_fock.__wrapped__(z, z, aufbau_configuration(z))


@pytest.mark.parametrize("symbol,z", [("Ar", 18), ("Cl", 17)])
def test_a_cold_solve_stays_within_the_budget(symbol, z):
    """Argon is the reference atom; chlorine is the slowest one measured."""
    start = time.perf_counter()
    _solve_cold(z)
    elapsed = time.perf_counter() - start
    assert elapsed < _WALL_CLOCK_CEILING, (
        f"{symbol} took {elapsed:.1f}s against a ceiling of "
        f"{_WALL_CLOCK_CEILING:.0f}s; profile before raising this"
    )


def test_repeated_solves_hit_the_cache():
    """A second solve of the same atom must not redo the work.

    Deliberately does not clear the cache first: if an earlier test already
    solved neon then the first call here is itself a hit, and the claim being
    made - that a repeat is free - holds either way.
    """
    solve_hartree_fock(10, 10, aufbau_configuration(10))
    start = time.perf_counter()
    solve_hartree_fock(10, 10, aufbau_configuration(10))
    assert time.perf_counter() - start < 0.1


def test_the_configuration_argument_does_not_defeat_the_cache():
    """Two independently built configurations for the same atom must collide.

    aufbau_configuration returns a fresh object each call. If it ever returned
    something that hashes by identity - a list, or a dataclass without eq - the
    memo would miss every time and nothing would fail except the clock.
    """
    a = aufbau_configuration(10)
    b = aufbau_configuration(10)
    assert a is not b
    assert hash(a) == hash(b)
    before = solve_hartree_fock.cache_info()
    solve_hartree_fock(10, 10, a)
    solve_hartree_fock(10, 10, b)
    after = solve_hartree_fock.cache_info()
    assert after.hits - before.hits >= 1


@pytest.mark.parametrize("symbol,z", [("Li", 3), ("Ne", 10), ("S", 16), ("Ar", 18)])
def test_scf_iteration_counts_stay_bounded(symbol, z):
    """The mixing parameter, made falsifiable.

    The two counts are asserted separately on purpose. The fine solve is warm
    started from the coarse one, so it converges in a handful whatever the
    mixing does - it moved only 6 to 8 on argon when the mixing was regressed
    to the value that cost 2.7x wall time. The coarse solve starts from a
    central field and is where the cost actually lives; it went 13 to 34 on the
    same change. Guarding only the fine count would guard the number that does
    not move.
    """
    result = _solve_cold(z)
    assert result.coarse_iterations < _COARSE_SCF_CEILING, (
        f"{symbol} took {result.coarse_iterations} coarse SCF iterations; "
        f"check the mixing parameter in numerics.hartree_fock.scf"
    )
    assert result.iterations < _FINE_SCF_CEILING, (
        f"{symbol} took {result.iterations} fine SCF iterations"
    )


def test_the_coarse_solve_is_the_expensive_one():
    """Pins the premise the test above is built on.

    If a change ever made the fine solve dominate - dropping the warm start,
    say - then the coarse ceiling would stop being the informative guard and
    this fails to say so, rather than leaving a stale comment claiming
    otherwise.
    """
    result = _solve_cold(18)
    assert result.coarse_iterations > result.iterations


@pytest.mark.parametrize("refinement,ceiling", [(1, _ARGON_POINT_CEILING // 2),
                                                (2, _ARGON_POINT_CEILING)])
def test_argon_runs_on_a_small_mesh(refinement, ceiling):
    """The exponential mesh, made falsifiable.

    Every cost in this module is roughly linear in the point count times the
    iteration count, so this is the other half of the same guard - and it is
    the half that catches a grid regression even on a machine fast enough to
    hide it in the clock.
    """
    config = aufbau_configuration(18)
    n_top = max(n for (n, _), _ in config)
    mesh = hf_mesh(18, 18, n_top, refinement=refinement)
    assert len(mesh.r) < ceiling, (
        f"argon's refinement-{refinement} mesh has {len(mesh.r)} points; "
        f"a uniform grid of comparable accuracy needed 72000 and about an hour"
    )
