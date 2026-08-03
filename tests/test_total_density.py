"""The total radial density: the one shape in this application that is real.

Every other picture here is an orbital, and the app says in as many words that
an orbital is not an observable. This is the counterweight, so the checks are
about the two things that make it an observable rather than a drawing: that it
integrates to the electron count, and that its peaks are the shells.
"""

import numpy as np
import pytest
from scipy.signal import find_peaks

from atomsim.atoms import aufbau_configuration
from atomsim.hf_atom import hf_total_radial_density
from atomsim.provenance import Fidelity


def test_density_integrates_to_the_electron_count():
    """integral D(r) dr = N, which is what makes this a density and not a curve.

    Exact on the solver mesh by construction (every P is normalized there), so
    what is left is the resampling onto the display grid, and the provenance
    reports exactly that residual in electrons.
    """
    d = hf_total_radial_density(10, 10)
    total = np.trapezoid(d.values, d.grid)
    # Loose in absolute terms and tight in the terms that matter: the curve
    # must account for all ten electrons to well under one part in a thousand,
    # and the error bar must be the actual miss rather than a decoration.
    assert total == pytest.approx(10.0, rel=1e-3)
    assert d.provenance.error_estimate == pytest.approx(abs(total - 10.0), abs=1e-9)


def test_the_closure_residual_is_quadrature_and_not_a_defect():
    """Refine the grid and the miss must fall as h^2, or it is not the grid.

    Worth a test of its own because a fixed tolerance cannot tell the two
    apart. A 0.02% shortfall at 400 points looks the same whether it is the
    trapezoid rule doing its job or an orbital being dropped from the sum, and
    only the convergence rate separates them. Doubling the points must quarter
    the error; a dropped orbital would sit flat instead.

    This is the check that found the original bug. Built on a uniform grid
    first, this function lost 0.35 of neon's electrons and did NOT converge
    away, because a uniform grid steps over the 1s however fine it gets.
    """
    errors = []
    for points in (400, 800, 1600):
        d = hf_total_radial_density(10, 10, points=points)
        errors.append(abs(float(np.trapezoid(d.values, d.grid)) - 10.0))

    for coarse, fine in zip(errors, errors[1:], strict=False):
        # Second order, with room for the interpolation floor underneath it.
        assert 3.0 < coarse / fine < 6.0, f"{coarse:.3e} -> {fine:.3e} is not h^2"


def test_argon_has_three_shells_and_neon_has_two():
    """K, L, M. This is the periodic table showing up in a plot.

    The count is the assertion, not the positions: peak positions move with Z
    and with the model, but the NUMBER of peaks is the shell structure itself,
    and a solve that lost it would be broken in a way no energy check catches.
    """
    for z, expected in ((10, 2), (18, 3)):
        d = hf_total_radial_density(z, z)
        # Prominence threshold rejects the shoulder ripples a numerical P can
        # carry in its tail; the shells are the dominant features by far.
        peaks, _ = find_peaks(d.values, prominence=0.05 * d.values.max())
        assert len(peaks) == expected, f"Z={z} gave {len(peaks)} shells"


def test_the_collapsed_atom_has_one_shell():
    """No exclusion principle, no shells. Phase 24's lesson as a shape.

    Every electron is in the 1s, so there is one peak and nothing else, and
    that is the whole reason chemistry needs the principle.
    """
    collapsed = aufbau_configuration(18, pauli=False)
    d = hf_total_radial_density(
        18, 18, config=collapsed, exchange=False, pauli=False
    )
    peaks, _ = find_peaks(d.values, prominence=0.05 * d.values.max())
    assert len(peaks) == 1
    assert d.provenance.fidelity is Fidelity.COUNTERFACTUAL


def test_the_density_is_the_observable_and_says_so():
    d = hf_total_radial_density(10, 10)
    joined = " ".join(d.provenance.assumptions)
    assert "observable" in joined
    assert d.unit == "electrons/bohr"
    assert d.provenance.fidelity is Fidelity.APPROXIMATION


def test_a_non_aufbau_configuration_changes_the_density():
    """The configuration trap again, on the new quantity."""
    from atomsim.atoms import parse_config

    ground = hf_total_radial_density(10, 10)
    excited = hf_total_radial_density(10, 10, config=parse_config("1s2 2s2 2p5 3s1"))
    assert not np.allclose(ground.values, excited.values)
