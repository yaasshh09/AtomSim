"""The decay cascade: conservation, the published branching ratio, and two
independent solutions (the top-down graph pass and the matrix exponential)
agreeing with each other and with the closed-form Bateman chain."""

import math

import numpy as np
import pytest

from atomsim.analytic.transitions import lifetime
from atomsim.cascade import decay_cascade
from atomsim.provenance import Fidelity


def _level(c, n, l):
    return next(lv for lv in c.levels if (lv.n, lv.l) == (n, l))


def test_3p_branches_to_2s_at_the_published_ratio():
    # A(3p->1s) = 1.6725e8, A(3p->2s) = 2.2449e7 s^-1 (Wiese & Fuhr 2009).
    c = decay_cascade(3, 1)
    b = {br.lower: br.branching_ratio.value for br in c.branches}
    assert b[(2, 0)] == pytest.approx(2.2449e7 / (1.6725e8 + 2.2449e7), rel=1e-3)
    assert _level(c, 2, 0).terminal and _level(c, 1, 0).terminal


@pytest.mark.parametrize("n", range(2, 9))
def test_every_cascade_ends_somewhere(n):
    for l in range(n):
        c = decay_cascade(n, l, points=8)
        ends = sum(lv.visit_probability.value for lv in c.levels if lv.terminal)
        assert ends == pytest.approx(1.0, abs=1e-13)
        # Only 1s and 2s lack an E1 channel.
        assert {(lv.n, lv.l) for lv in c.levels if lv.terminal} <= {(1, 0), (2, 0)}


def test_photon_flux_into_a_level_is_its_visit_probability():
    c = decay_cascade(6, 2, points=8)
    for lv in c.levels:
        if (lv.n, lv.l) == c.start:
            continue
        inflow = sum(br.photons.value for br in c.branches if br.lower == (lv.n, lv.l))
        assert inflow == pytest.approx(lv.visit_probability.value, rel=1e-13)


def test_matrix_exponential_settles_where_the_graph_says():
    c = decay_cascade(7, 3)
    for lv, pop in zip(c.levels, c.populations):
        final = pop.values[-1]
        expected = lv.visit_probability.value if lv.terminal else 0.0
        assert final == pytest.approx(expected, abs=1e-9)
    total = np.sum([p.values for p in c.populations], axis=0)
    assert np.max(np.abs(total - 1.0)) < 1e-12


def test_3s_chain_matches_bateman():
    # 3s -> 2p -> 1s is a single chain: N_2p = g3/(g2-g3) (e^-g3 t - e^-g2 t).
    c = decay_cascade(3, 0)
    g3 = 1.0 / _level(c, 3, 0).lifetime.value
    g2 = 1.0 / _level(c, 2, 1).lifetime.value
    pop = dict(zip([(lv.n, lv.l) for lv in c.levels], c.populations))
    t = pop[(2, 1)].grid
    bateman = g3 / (g2 - g3) * (np.exp(-g3 * t) - np.exp(-g2 * t))
    assert np.max(np.abs(pop[(2, 1)].values - bateman)) < 1e-12
    assert np.allclose(pop[(3, 0)].values, np.exp(-g3 * t), atol=1e-12)
    assert c.mean_settle_time.value == pytest.approx(1 / g3 + 1 / g2, rel=1e-13)


def test_lifetimes_agree_with_the_transitions_module():
    c = decay_cascade(5, 1, points=8)
    for lv in c.levels:
        if not lv.terminal:
            assert lv.lifetime.value == pytest.approx(lifetime(lv.n, lv.l).value, rel=1e-13)
        else:
            assert math.isinf(lv.lifetime.value)


def test_z_scaling_moves_the_clock_not_the_branches():
    h = decay_cascade(4, 1, points=8)
    he = decay_cascade(4, 1, Z=2, points=8)
    for a, b in zip(h.branches, he.branches):
        assert b.branching_ratio.value == pytest.approx(a.branching_ratio.value, rel=1e-12)
    assert he.mean_settle_time.value == pytest.approx(h.mean_settle_time.value / 16, rel=1e-12)


def test_ground_state_has_nothing_to_do():
    c = decay_cascade(1, 0)
    assert len(c.levels) == 1 and c.levels[0].terminal
    assert c.branches == [] and c.mean_settle_time.value == 0.0


def test_provenance_discloses_the_two_photon_gap():
    c = decay_cascade(3, 1, points=8)
    prov = c.levels[0].visit_probability.provenance
    assert prov.fidelity is Fidelity.NUMERICAL
    assert any("two-photon" in a for a in prov.assumptions)
