"""Bosonic electrons as a table: the real atoms' ionization energy has the
periodic sawtooth, and the collapsed atoms' has none."""

import warnings

import numpy as np
import pytest

from atomsim.constants import HARTREE_EV
from atomsim.hf_atom import hf_valence_ionization_energy, pauli_collapse

ZS = range(2, 13)


@pytest.fixture(scope="module")
def ies():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = {}
        for z in ZS:
            c = pauli_collapse(z)
            out[z] = (
                hf_valence_ionization_energy(c.real).value * HARTREE_EV,
                hf_valence_ionization_energy(c.collapsed).value * HARTREE_EV,
            )
    return out


def test_helium_is_the_same_atom_either_way(ies):
    """1s2 is the ground configuration with or without the cap, and for two
    electrons in one orbital exchange cancels self-interaction exactly."""
    real, collapsed = ies[2]
    assert collapsed == pytest.approx(real, abs=1e-6)


def test_real_atoms_drop_at_every_alkali(ies):
    for alkali in (3, 11):
        assert ies[alkali][0] < 0.3 * ies[alkali - 1][0]


def test_collapsed_atoms_have_no_periodicity(ies):
    collapsed = np.array([ies[z][1] for z in ZS])
    assert np.all(np.diff(collapsed) > 0)
