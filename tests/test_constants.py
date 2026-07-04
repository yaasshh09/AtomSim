from scipy.constants import physical_constants

from atomsim.constants import HARTREE_EV, FundamentalConstants


def test_hartree_ev_matches_codata():
    assert abs(HARTREE_EV - 27.2113862460) < 1e-8


def test_derived_alpha_matches_published_value():
    c = FundamentalConstants.codata()
    published = physical_constants["fine-structure constant"][0]
    assert abs(c.alpha - published) / published < 1e-9


def test_derived_bohr_radius_matches_published_value():
    c = FundamentalConstants.codata()
    published = physical_constants["Bohr radius"][0]
    assert abs(c.bohr_radius - published) / published < 1e-9


def test_derived_hartree_matches_published_value():
    c = FundamentalConstants.codata()
    published = physical_constants["Hartree energy"][0]
    assert abs(c.hartree_energy - published) / published < 1e-9


def test_counterfactual_universe_rescales():
    # doubling e quadruples alpha (e^2) and shrinks the atom (a0 ~ 1/e^2)
    real = FundamentalConstants.codata()
    weird = FundamentalConstants(
        hbar=real.hbar, e=2 * real.e, m_e=real.m_e, eps0=real.eps0, c=real.c
    )
    assert abs(weird.alpha / real.alpha - 4.0) < 1e-12
    assert abs(weird.bohr_radius / real.bohr_radius - 0.25) < 1e-12
