import math

import numpy as np
import pytest

from atomsim.atoms import aufbau_configuration, parse_config
from atomsim.provenance import Fidelity
from atomsim.sampling import sample_screened_density
from atomsim.screened_atom import (
    evaluate_screened_state,
    screened_radial,
    solve_screened_atom,
    valence_ionization_energy,
)

# NIST first ionization energies (eV), for the tier-defining tolerance test.
# Source: NIST ASD ionization energies (public). Retrieved 2026-07-18.
_NIST_IE_EV = {"he": 24.587, "li": 5.392, "na": 5.139}
HARTREE_EV = 27.211386245988


def test_n1_recovers_hydrogenic():
    # One electron in the screened field == bare Coulomb == -Z^2 / 2n^2.
    res = solve_screened_atom(z=3, n_electrons=1, config=parse_config("1s1"))
    e1s = res.orbitals[0].energy.value
    assert math.isclose(e1s, -(3**2) / 2.0, rel_tol=2e-4)
    assert res.orbitals[0].energy.provenance.fidelity is Fidelity.APPROXIMATION


def test_orbital_energy_carries_numerical_subscale():
    res = solve_screened_atom(z=11, n_electrons=11, config=aufbau_configuration(11))
    assert res.orbitals[0].energy.provenance.error_estimate is not None


def test_total_energy_is_occupancy_weighted_sum():
    res = solve_screened_atom(z=6, n_electrons=6, config=aufbau_configuration(6))
    expect = sum(o.occupancy * o.energy.value for o in res.orbitals)
    assert math.isclose(res.total_energy.value, expect, rel_tol=1e-12)


def test_s_below_p_for_same_n():
    # Screening lifts the Coulomb degeneracy: 2s below 2p in carbon.
    res = solve_screened_atom(z=6, n_electrons=6, config=aufbau_configuration(6))
    e = {(o.n, o.l): o.energy.value for o in res.orbitals}
    assert e[(2, 0)] < e[(2, 1)]


@pytest.mark.parametrize("key,z,n", [("he", 2, 2), ("li", 3, 3), ("na", 11, 11)])
def test_valence_ionization_matches_nist(key, z, n):
    res = solve_screened_atom(z=z, n_electrons=n, config=aufbau_configuration(n))
    ie_ev = valence_ionization_energy(res).value * HARTREE_EV
    ref = _NIST_IE_EV[key]
    assert abs(ie_ev - ref) / ref < 0.12  # GSZ valence energies: ~10% class


def test_screened_radial_shapes():
    r_field, p_field = screened_radial(z=11, n_electrons=11, n=3, l=0, points=300)
    assert r_field.values.shape == r_field.grid.shape == (300,)
    assert p_field.unit == "bohr^-1" and r_field.provenance.fidelity is Fidelity.APPROXIMATION


def test_a_shape_field_carries_no_energy_error_bar():
    """An error estimate is in the unit of the thing it describes, or it lies.

    Both fields here are shapes: R in bohr^-3/2 and r^2 R^2 in bohr^-1. The
    eigenvalue's error estimate is in hartree. Attaching it to either one is
    not a conservative bar on the shape, it is a number in the wrong dimension
    read as an uncertainty on R, which is exactly the class of quiet lie the
    provenance system exists to make impossible.

    The energy's own bar is a separate assertion below, because dropping it
    from the fields must not be achieved by dropping it everywhere.
    """
    r_field, p_field = screened_radial(z=11, n_electrons=11, n=3, l=0, points=300)
    assert r_field.unit == "bohr^-3/2" and p_field.unit == "bohr^-1"
    assert r_field.provenance.error_estimate is None
    assert p_field.provenance.error_estimate is None

    # The two surfaces downstream of that field, which is how a hartree bar
    # would creep back in without anyone editing the line that sets it: psi
    # values are bohr^-3/2 and a sampled cloud is positions in bohr.
    pos = np.array([[1.0, 0.0, 0.0]])
    psi = evaluate_screened_state(11, 11, 3, 0, 0, pos)
    assert psi.provenance.error_estimate is None
    cloud = sample_screened_density(11, 11, 3, 0, 0, 50, seed=1)
    assert cloud.provenance.error_estimate is None

    # ...and the energy, which IS in hartree, still carries one. Dropping the
    # bar from the shapes must not be achieved by dropping it everywhere.
    res = solve_screened_atom(z=11, n_electrons=11, config=aufbau_configuration(11))
    energy = res.orbitals[0].energy
    assert energy.unit == "hartree"
    assert energy.provenance.error_estimate is not None


def test_evaluate_screened_state_is_real_on_y0_plane():
    # Central-field orbital is real on y=0 (e^{i m phi} = +/-1 there).
    x = np.linspace(0.1, 20.0, 50)
    pos = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
    psi = evaluate_screened_state(11, 11, 3, 1, 0, pos, basis="complex")
    assert psi.values.shape == (50,)
    assert np.max(np.abs(psi.values.imag)) < 1e-9
    assert psi.provenance.fidelity is Fidelity.APPROXIMATION


def test_evaluate_screened_state_factorizes_R_times_Y():
    from atomsim.analytic.angular import spherical_harmonic

    rng = np.random.default_rng(0)
    pos = rng.normal(size=(200, 3)) * 3.0
    psi = evaluate_screened_state(11, 11, 3, 0, 0, pos, basis="complex")
    r = np.linalg.norm(pos, axis=1)
    theta = np.arccos(np.clip(pos[:, 2] / np.where(r > 0, r, 1.0), -1.0, 1.0))
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    r_field, _ = screened_radial(11, 11, 3, 0, points=4096)
    R = np.interp(r, r_field.grid, r_field.values, right=0.0)
    Y = spherical_harmonic(0, 0, theta, phi, basis="complex").values
    assert np.allclose(psi.values, R * Y, atol=1e-8)


def test_evaluate_screened_state_node_count_along_ray():
    # Na 3s has n-l-1 = 2 radial nodes.
    z = np.linspace(0.05, 40.0, 4000)
    pos = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    psi = evaluate_screened_state(11, 11, 3, 0, 0, pos, basis="complex").values.real
    nz = psi[np.abs(psi) > 1e-6]
    sign_changes = int(np.sum(np.diff(np.sign(nz)) != 0))
    assert sign_changes == 2
