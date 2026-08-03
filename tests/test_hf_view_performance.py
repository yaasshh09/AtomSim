"""What the picture views actually cost, recorded rather than assumed.

Budgets here are deliberately loose. The number worth guarding is the shape of
the cost - one solve shared by four views, and an isosurface that is grid work
rather than solve work - and a tight budget on a shared CI runner guards the
runner's mood instead.
"""

import time

import pytest

from atomsim.atoms import aufbau_configuration, parse_config
from atomsim.hf_atom import solve_hartree_fock
from atomsim.isosurface import hf_isosurface
from atomsim.plane import hf_plane_grid
from atomsim.sampling import sample_hf_density


def test_four_views_share_one_solve():
    """The solve is the expensive part, so it must be paid once.

    Measured through the cache counters rather than a stopwatch: a wall-clock
    assertion on "the second one was faster" passes on a machine where both
    were slow for unrelated reasons.
    """
    solve_hartree_fock.cache_clear()
    config = aufbau_configuration(10)

    sample_hf_density(10, 10, 2, 1, 0, 2_000, config=config)
    hf_plane_grid(10, 10, 2, 1, 0, resolution=32, config=config)
    hf_isosurface(10, 10, 2, 1, 0, resolution=48, config=config)

    info = solve_hartree_fock.cache_info()
    assert info.misses == 1, f"expected one solve, got {info.misses}"
    assert info.hits > 0


def test_default_config_shares_the_explicit_one():
    """config=None must resolve to the aufbau tuple BEFORE the cached call.

    Resolve it after and the two spellings of the same atom become two cache
    keys, so a view that passes the configuration explicitly and one that
    leaves it None each pay their own solve. Nothing would look wrong; the
    application would just be twice as slow as it reads.
    """
    solve_hartree_fock.cache_clear()
    sample_hf_density(10, 10, 2, 1, 0, 1_000)
    sample_hf_density(10, 10, 2, 1, 0, 1_000, config=aufbau_configuration(10))
    assert solve_hartree_fock.cache_info().misses == 1


def test_the_counterfactual_key_space_fits_the_cache():
    """One atom, both switches, two configurations: does anything evict?

    Eight slots against (atom, configuration, exchange, pauli). This records
    what a user actually reaches by flipping switches on the atom they are
    looking at, and fails if that sequence starts re-solving. If it ever does,
    raise maxsize with this test as the reason rather than as a precaution.
    """
    solve_hartree_fock.cache_clear()
    ground = aufbau_configuration(10)
    excited = parse_config("1s2 2s2 2p5 3s1")
    collapsed = aufbau_configuration(10, pauli=False)

    keys = [
        (ground, True, True),
        (ground, False, True),
        (collapsed, False, False),
        (excited, True, True),
        (excited, False, True),
    ]
    for config, exchange, pauli in keys:
        solve_hartree_fock(10, 10, config, exchange, pauli)
    first_misses = solve_hartree_fock.cache_info().misses

    # Walk the same set again. Every one should now be a hit.
    for config, exchange, pauli in keys:
        solve_hartree_fock(10, 10, config, exchange, pauli)
    assert solve_hartree_fock.cache_info().misses == first_misses


@pytest.mark.parametrize("resolution", [96])
def test_isosurface_budget(resolution):
    """The expensive path, timed once with the solve already paid.

    96^3 plus the box fit plus the halved grid for the error bar, each point
    through evaluate_hf_state's interpolation. The solve is cached and the
    interpolation is vectorised, so this should sit near the screened path.
    """
    solve_hartree_fock(10, 10, aufbau_configuration(10))  # warm
    t0 = time.monotonic()
    surf = hf_isosurface(10, 10, 2, 1, 0, resolution=resolution)
    elapsed = time.monotonic() - t0
    assert surf.vertices.shape[0] > 0
    assert elapsed < 60.0, f"96^3 Hartree-Fock isosurface took {elapsed:.1f}s"
    print(f"\nHF isosurface {resolution}^3, warm solve: {elapsed:.2f}s")
