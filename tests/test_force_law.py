import pytest

from atomsim.analytic.hydrogen import energy as hydrogen_energy
from atomsim.numerics.force_law import P_MAX, P_MIN, force_law_levels


def test_identity_case_matches_exact_hydrogen():
    # At p=1 the numerical levels reproduce the exact Bohr formula.
    for l in (0, 1):
        res = force_law_levels(p=1.0, l=l, system="h", n_states=2)
        assert len(res.counterfactual) == len(res.reference) == 2
        for lvl, ref in zip(res.counterfactual, res.reference):
            exact = hydrogen_energy(ref.n, Z=1, mu_ratio=1.0).value
            assert lvl.energy.value == pytest.approx(exact, rel=2e-3)


def test_provenance_tiers_are_distinct():
    res = force_law_levels(p=1.2, l=0, system="h", n_states=2)
    assert all(x.energy.provenance.fidelity.value == "numerical" for x in res.counterfactual)
    assert all(x.energy.provenance.fidelity.value == "exact" for x in res.reference)
    # numerical levels carry a grid-halving error estimate
    assert all(x.energy.provenance.error_estimate is not None for x in res.counterfactual)
    # the counterfactual is self-describing: its provenance names the bent potential,
    # not just the generic solver method (prime directive — honest in isolation)
    assert all(
        "counterfactual power-law potential V=-Z/r^1.2" in x.energy.provenance.method
        for x in res.counterfactual
    )
    # the EXACT reference is the genuine Coulomb closed form — never tagged counterfactual
    assert all("counterfactual" not in r.energy.provenance.method for r in res.reference)


def test_reference_gated_to_n_ge_l_plus_1():
    res = force_law_levels(p=1.0, l=1, system="h", n_states=3)
    assert [r.n for r in res.reference] == [2, 3, 4]
    assert [c.radial_index for c in res.counterfactual] == [0, 1, 2]


def _two_s_minus_two_p(p: float) -> float:
    # 2s = (l=0, radial index 1); 2p = (l=1, radial index 0); both n=2.
    e_2s = force_law_levels(p=p, l=0, system="h", n_states=2).counterfactual[1].energy.value
    e_2p = force_law_levels(p=p, l=1, system="h", n_states=1).counterfactual[0].energy.value
    return e_2s - e_2p


def test_degeneracy_intact_at_coulomb():
    assert _two_s_minus_two_p(1.0) == pytest.approx(0.0, abs=3e-3)


def test_degeneracy_breaks_with_correct_ordering():
    # p<1 (softer, DeltaV>0): s above p  -> E_2s > E_2p  -> positive
    # p>1 (harder, DeltaV<0): s below p  -> E_2s < E_2p  -> negative
    soft = _two_s_minus_two_p(0.8)
    hard = _two_s_minus_two_p(1.2)
    assert soft > 3e-3
    assert hard < -3e-3


def test_p_out_of_range_rejected():
    with pytest.raises(ValueError):
        force_law_levels(p=P_MAX + 0.1, l=0)
    with pytest.raises(ValueError):
        force_law_levels(p=P_MIN - 0.1, l=0)


def test_negative_l_rejected():
    with pytest.raises(ValueError):
        force_law_levels(p=1.0, l=-1)
