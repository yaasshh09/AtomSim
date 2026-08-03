"""Hartree-Fock reaching the picture views.

The load-bearing test here is the first one. Every other check in this file
would pass on a build that silently drew the aufbau configuration under any
label the user picked, which is exactly the failure this phase exists to
prevent, so the configuration is asserted to reach the orbital before anything
else is asserted about the orbital.
"""

import numpy as np
import pytest

from atomsim.atoms import aufbau_configuration, parse_config
from atomsim.hf_atom import evaluate_hf_state, hf_radial
from atomsim.provenance import Fidelity


def test_explicit_configuration_reaches_the_orbital():
    """A non-aufbau configuration must change the orbital it produces.

    Sodium's 3s sits outside a closed neon core. Promote it to 3p and the 3s is
    gone; ask instead for the 2p, which BOTH configurations occupy, and the
    orbital still has to differ, because the Fock operator for the 2p is built
    from the other occupied orbitals and one of them moved.
    """
    ground = aufbau_configuration(11)
    excited = parse_config("1s2 2s2 2p6 3p1")

    r_ground, _ = hf_radial(11, 11, 2, 1, points=200, config=ground)
    r_excited, _ = hf_radial(11, 11, 2, 1, points=200, config=excited)

    assert not np.allclose(r_ground.values, r_excited.values, atol=1e-9)


def test_exchange_off_reaches_the_orbital_and_the_badge():
    """The Hartree 2p is a different curve, and says so in its own tier."""
    config = aufbau_configuration(10)
    r_hf, _ = hf_radial(10, 10, 2, 1, points=200, config=config)
    r_hartree, _ = hf_radial(
        10, 10, 2, 1, points=200, config=config, exchange=False
    )

    assert not np.allclose(r_hf.values, r_hartree.values, atol=1e-9)
    assert r_hf.provenance.fidelity is Fidelity.APPROXIMATION
    assert r_hartree.provenance.fidelity is Fidelity.COUNTERFACTUAL


def test_evaluate_hf_state_inherits_the_counterfactual_tier():
    """The 3-D evaluator must not staple APPROXIMATION onto a Hartree orbital.

    It used to: the tier was a literal in the Provenance constructor rather
    than something read off the solve, so every picture came back
    APPROXIMATION whatever the flags said.
    """
    pos = np.array([[0.0, 0.0, 1.0], [0.5, 0.0, 0.5]])
    real = evaluate_hf_state(10, 10, 2, 1, 0, pos)
    hartree = evaluate_hf_state(10, 10, 2, 1, 0, pos, exchange=False)

    assert real.provenance.fidelity is Fidelity.APPROXIMATION
    assert hartree.provenance.fidelity is Fidelity.COUNTERFACTUAL


def test_pauli_off_refuses_every_subshell_but_the_one_that_exists():
    """With the cap lifted the configuration is 1s^N and nothing else is there.

    The refusal has to name the reason. A bare "not occupied" would read as a
    contingent fact about this atom rather than as the consequence of the
    switch the caller just flipped.
    """
    collapsed = aufbau_configuration(10, pauli=False)
    with pytest.raises(ValueError, match="occupancy cap"):
        hf_radial(
            10, 10, 2, 1, points=200,
            config=collapsed, exchange=False, pauli=False,
        )
    # And the one that does exist still comes back.
    r, _ = hf_radial(
        10, 10, 1, 0, points=200, config=collapsed, exchange=False, pauli=False
    )
    assert r.provenance.fidelity is Fidelity.COUNTERFACTUAL


def test_orbital_carries_the_not_an_observable_claim():
    """Every Hartree-Fock picture routes through hf_radial, so the claim does."""
    r, p = hf_radial(10, 10, 2, 1, points=200)
    joined = " ".join(r.provenance.assumptions)
    assert "not an observable" in joined
    assert "spherical" in joined
    assert joined == " ".join(p.provenance.assumptions)


def test_hf_sampling_reduces_to_hydrogen():
    """At Z=1, N=1 the Fock operator IS the bare Coulomb Hamiltonian.

    There is no other electron, so no direct term, no exchange term, and
    nothing for self-consistency to do. The sampler therefore has to reproduce
    the closed-form 1s radial CDF, 1 - e^(-2r)(1 + 2r + 2r^2), and a KS test is
    the check the analytic sampler already gets held to.

    A ground truth this tier rarely has, which is why it is spent here.
    """
    from scipy import stats

    from atomsim.sampling import sample_hf_density

    cloud = sample_hf_density(1, 1, 1, 0, 0, 20_000, seed=7)
    r = np.linalg.norm(cloud.positions.astype(np.float64), axis=1)

    def cdf(x):
        return 1.0 - np.exp(-2.0 * x) * (1.0 + 2.0 * x + 2.0 * x * x)

    assert stats.kstest(r, cdf).pvalue > 0.01


def test_hf_cloud_carries_the_solve_and_the_claim():
    from atomsim.sampling import sample_hf_density

    cloud = sample_hf_density(10, 10, 2, 1, 0, 2_000, seed=1)
    joined = " ".join(cloud.provenance.assumptions)
    assert cloud.provenance.fidelity is Fidelity.APPROXIMATION
    assert "not an observable" in joined
    assert "correlation" in joined  # the solve's own disclosure survived


def test_hf_cloud_goes_counterfactual_with_exchange_off():
    from atomsim.sampling import sample_hf_density

    cloud = sample_hf_density(10, 10, 2, 1, 0, 2_000, seed=1, exchange=False)
    assert cloud.provenance.fidelity is Fidelity.COUNTERFACTUAL
