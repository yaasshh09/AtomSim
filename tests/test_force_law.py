import math

import numpy as np
import pytest
from scipy.optimize import brentq

from atomsim.analytic.hydrogen import energy as hydrogen_energy
from atomsim.analytic.oscillator import oscillator_energy
from atomsim.numerics.force_law import force_law_levels
from atomsim.provenance import Fidelity
from atomsim.systems import get_system


def test_powerlaw_p1_matches_exact_hydrogen():
    res = force_law_levels("powerlaw", {"p": 1.0}, l=0, system="h", n_states=3)
    assert res.preset_key == "powerlaw"
    assert res.bound_count == 3 and res.requested_count == 3
    mu = get_system("h").mu_ratio.value  # compare like-for-like reduced mass
    for k, level in enumerate(res.counterfactual):
        n = k + 1  # l = 0
        exact = hydrogen_energy(n, Z=1, mu_ratio=mu).value
        assert level.energy.provenance.fidelity is Fidelity.NUMERICAL
        assert math.isclose(level.energy.value, exact, rel_tol=2e-4)


def test_powerlaw_reference_is_exact_hydrogen_ladder():
    res = force_law_levels("powerlaw", {"p": 1.2}, l=1, system="h", n_states=3)
    assert res.reference.kind == "levels"
    assert [item.label for item in res.reference.items] == ["n=2", "n=3", "n=4"]
    assert all(i.energy.provenance.fidelity is Fidelity.EXACT for i in res.reference.items)


def test_powerlaw_degeneracy_breaks_off_p1():
    s = force_law_levels("powerlaw", {"p": 1.2}, l=0, system="h", n_states=2)
    p = force_law_levels("powerlaw", {"p": 1.2}, l=1, system="h", n_states=1)
    e_2s = s.counterfactual[1].energy.value  # (l=0, k=1) -> n=2
    e_2p = p.counterfactual[0].energy.value  # (l=1, k=0) -> n=2
    assert abs(e_2s - e_2p) > 1e-4
    assert e_2s < e_2p  # p > 1 (harder): s below p (alkali ordering)


def test_powerlaw_out_of_range_p_raises():
    with pytest.raises(ValueError, match="p"):
        force_law_levels("powerlaw", {"p": 1.9}, l=0)


def test_potential_curve_is_field_in_hartree():
    res = force_law_levels("powerlaw", {"p": 1.0}, l=0, system="h", n_states=2)
    curve = res.potential_curve
    assert curve.values.shape == curve.grid.shape
    assert curve.unit == "hartree" and curve.grid_unit == "bohr"
    assert curve.provenance.fidelity is Fidelity.EXACT
    assert np.all(curve.grid > 0)


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="unknown preset"):
        force_law_levels("nope", {}, l=0)


def test_yukawa_approaches_hydrogen_as_screening_weakens():
    # Yukawa -> Coulomb as lambda grows. lambda=20 bohr (the ParamSpec max) still
    # leaves a real ~10% screening shift on the 1s: the leading correction is
    # dE ~= Z/lambda = 0.05 hartree vs |E_1s| = 0.5. Reaching a 5e-3 absolute match
    # would need lambda ~ 400 bohr (an absurd UI range), so the honest validation of
    # the Coulomb limit is the *approach*: every Yukawa 1s is less bound than exact
    # Coulomb, and larger lambda binds monotonically closer to it.
    mu = get_system("h").mu_ratio.value
    exact = hydrogen_energy(1, Z=1, mu_ratio=mu).value
    e = {
        lam: force_law_levels("yukawa", {"lambda": lam}, l=0, system="h", n_states=1)
        .counterfactual[0]
        .energy.value
        for lam in (2.0, 6.0, 20.0)
    }
    # all bound (E<0) yet above exact Coulomb, and monotone approach from above:
    assert exact < e[20.0] < e[6.0] < e[2.0] < 0.0
    # residual screening at the widest lambda is the expected ~10%, not a blow-up:
    assert abs(e[20.0] - exact) < 0.12 * abs(exact)


def test_yukawa_screening_removes_bound_states():
    loose = force_law_levels("yukawa", {"lambda": 20.0}, l=0, system="h", n_states=4)
    tight = force_law_levels("yukawa", {"lambda": 1.0}, l=0, system="h", n_states=4)
    assert tight.bound_count < loose.bound_count
    assert all(level.energy.value < 0 for level in tight.counterfactual)


def test_yukawa_reference_is_full_hydrogen_ladder_even_under_shortfall():
    res = force_law_levels("yukawa", {"lambda": 1.0}, l=0, system="h", n_states=4)
    assert len(res.reference.items) == 4  # full ideal ladder
    assert res.bound_count <= 4


def test_coulombcore_c0_recovers_hydrogen():
    res = force_law_levels("coulombcore", {"core": 0.0}, l=0, system="h", n_states=2)
    mu = get_system("h").mu_ratio.value  # compare like-for-like reduced mass
    for k, level in enumerate(res.counterfactual):
        exact = hydrogen_energy(k + 1, Z=1, mu_ratio=mu).value
        assert math.isclose(level.energy.value, exact, rel_tol=2e-4)


def test_coulombcore_repulsive_core_raises_penetrating_s_above_p():
    s = force_law_levels("coulombcore", {"core": 0.5}, l=0, system="h", n_states=2)
    p = force_law_levels("coulombcore", {"core": 0.5}, l=1, system="h", n_states=1)
    e_2s = s.counterfactual[1].energy.value  # (l=0, k=1) -> n=2
    e_2p = p.counterfactual[0].energy.value  # (l=1, k=0) -> n=2
    assert abs(e_2s - e_2p) > 1e-4
    assert e_2s > e_2p  # +c/r^2 repulsion hits the penetrating low-l state harder


def test_harmonic_matches_exact_qho():
    for omega in (0.3, 0.6):
        for l in (0, 1):
            res = force_law_levels("harmonic", {"omega": omega}, l=l, system="h", n_states=3)
            assert res.bound_count == 3  # confining: all bound
            for k, level in enumerate(res.counterfactual):
                exact = oscillator_energy(k, l, omega).value
                assert math.isclose(level.energy.value, exact, rel_tol=2e-3)


def test_harmonic_reference_is_qho_levels():
    res = force_law_levels("harmonic", {"omega": 0.5}, l=0, system="h", n_states=2)
    assert res.reference.kind == "levels"
    assert all(i.energy.provenance.fidelity is Fidelity.EXACT for i in res.reference.items)
    assert math.isclose(res.reference.items[0].energy.value, 0.5 * 1.5, rel_tol=1e-9)


def _well_ground_state(v0: float, a: float, mu: float) -> float:
    # s-wave spherical well: k1 cot(k1 a) = -k2, E in (-v0, 0), independent of the FD solver
    def f(E):
        k1 = math.sqrt(2 * mu * (E + v0))
        k2 = math.sqrt(-2 * mu * E)
        return k1 / math.tan(k1 * a) + k2
    lo, hi = -v0 + 1e-9, -1e-9
    return brentq(f, lo, hi)


def test_finitewell_ground_state_matches_transcendental():
    v0, a = 2.0, 3.0
    res = force_law_levels("finitewell", {"v0": v0, "a": a}, l=0, system="h", n_states=3)
    assert res.bound_count >= 1
    mu = get_system("h").mu_ratio.value  # compare like-for-like reduced mass
    ref = _well_ground_state(v0, a, mu=mu)
    assert math.isclose(res.counterfactual[0].energy.value, ref, rel_tol=5e-3)
    for level in res.counterfactual:
        assert -v0 < level.energy.value < 0


def test_finitewell_markers_reference():
    res = force_law_levels("finitewell", {"v0": 2.0, "a": 3.0}, l=0, system="h", n_states=2)
    assert res.reference.kind == "markers"
    labels = {i.label for i in res.reference.items}
    assert labels == {"well floor", "continuum threshold"}


def test_finitewell_too_shallow_has_no_bound_states():
    # sqrt(2*mu*v0)*a < pi/2  =>  no s-wave bound state
    res = force_law_levels("finitewell", {"v0": 0.1, "a": 0.5}, l=0, system="h", n_states=3)
    assert res.bound_count == 0
