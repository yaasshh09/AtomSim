"""The plotted grid has to resolve the orbital it is plotting.

A solve box is sized so the eigenvalue is not squeezed by its own wall, and
for an inner orbital that box is enormous compared with the orbital. Both
many-electron models then resampled their output uniformly across the whole
box, which threw the solve away at the last step: argon's 1s is solved to 160
bohr on 48000 points, is finished by 0.4, and arrived at the browser as a
400-point grid with ONE sample on the orbital and 399 zeros. The app drew a
straight line where a 1s should be, under a correct axis and a correct badge.

These tests are about the picture, not the energy, and the split matters. The
solve box is deliberately untouched, because argon's 1s energy is worth about
2 hartree to that box; what changed is which samples get drawn. The check that
this is a window and not a truncation is the normalization: P(r) integrates to
1 over the drawn range, so nothing real was cut off.
"""

import numpy as np
import pytest

from atomsim.hf_atom import hf_radial
from atomsim.numerics.mesh import display_window
from atomsim.screened_atom import screened_radial

# Enough of the 400 output points on the orbital that a plot of it is a curve
# rather than a polyline through a handful of samples. Before the window,
# argon's 1s got one.
MIN_POINTS_ON_ORBITAL = 100


def _points_on_orbital(p_field) -> int:
    """Output samples where P(r) is above a thousandth of its peak."""
    v = p_field.values
    return int(np.count_nonzero(v > v.max() * 1e-3))


class TestDisplayWindow:
    def test_windows_to_where_the_function_lives(self):
        r = np.linspace(0.01, 100.0, 10000)
        # A tight peak near the origin on a box a thousand times wider.
        density = r**2 * np.exp(-2 * r / 0.05) ** 1
        assert display_window(r, density) < 5.0

    def test_keeps_the_whole_box_when_the_function_uses_it(self):
        r = np.linspace(0.01, 10.0, 1000)
        density = np.ones_like(r)
        assert display_window(r, density) == pytest.approx(r[-1])

    def test_never_cuts_inside_the_peak(self):
        r = np.linspace(0.01, 100.0, 10000)
        density = np.exp(-((r - 30.0) ** 2) / 2.0)
        assert display_window(r, density) > 30.0

    def test_never_exceeds_the_box(self):
        r = np.linspace(0.01, 10.0, 1000)
        density = np.exp(-((r - 9.99) ** 2) / 0.01)
        assert display_window(r, density) <= r[-1]

    def test_degenerate_inputs_fall_back_to_the_box(self):
        r = np.linspace(0.01, 10.0, 100)
        assert display_window(r, np.zeros_like(r)) == pytest.approx(r[-1])
        assert display_window(r, np.full_like(r, np.nan)) == pytest.approx(r[-1])
        assert display_window(np.array([]), np.array([])) == 0.0

    def test_keeps_more_tail_than_a_display_trim_would(self):
        # The floor here decides what data exists; the frontend's 1e-3 decides
        # what gets drawn from it. This one must be the looser of the two, or
        # the client would be trimming a curve that was already trimmed.
        r = np.linspace(0.01, 50.0, 20000)
        density = np.exp(-r)
        at_1em3 = r[np.nonzero(density > density.max() * 1e-3)[0][-1]]
        assert display_window(r, density) > at_1em3


@pytest.mark.parametrize(
    ("z", "n_electrons", "n", "l"),
    [
        (18, 18, 1, 0),  # argon's 1s: the case that was one sample wide
        (18, 18, 3, 1),  # argon's valence, at the other end of the same box
        (10, 10, 1, 0),
        (2, 2, 1, 0),
    ],
)
class TestScreenedOutputGrid:
    def test_resolves_the_orbital(self, z, n_electrons, n, l):
        _, p = screened_radial(z, n_electrons, n, l)
        assert _points_on_orbital(p) >= MIN_POINTS_ON_ORBITAL

    def test_the_window_cuts_no_probability(self, z, n_electrons, n, l):
        # The physics check. A window that lost real amplitude would show up
        # here as a norm below 1, and a grid too coarse to integrate would show
        # up as a norm that misses in either direction.
        _, p = screened_radial(z, n_electrons, n, l)
        assert float(np.trapezoid(p.values, p.grid)) == pytest.approx(1.0, abs=2e-3)

    def test_the_peak_is_inside_the_drawn_range(self, z, n_electrons, n, l):
        _, p = screened_radial(z, n_electrons, n, l)
        assert p.grid[int(p.values.argmax())] < p.grid[-1]

    def test_provenance_states_both_radii(self, z, n_electrons, n, l):
        # A window is honest only while it says it is one. The method names the
        # radius drawn and the radius solved, so the two can never be confused.
        _, p = screened_radial(z, n_electrons, n, l)
        assert "windowed from a solve box" in p.provenance.method


@pytest.mark.parametrize(
    ("z", "n_electrons", "n", "l"),
    [(18, 18, 1, 0), (18, 18, 3, 1), (10, 10, 1, 0)],
)
class TestHartreeFockOutputGrid:
    def test_resolves_the_orbital(self, z, n_electrons, n, l):
        _, p = hf_radial(z, n_electrons, n, l)
        assert _points_on_orbital(p) >= MIN_POINTS_ON_ORBITAL

    def test_the_window_cuts_no_probability(self, z, n_electrons, n, l):
        _, p = hf_radial(z, n_electrons, n, l)
        assert float(np.trapezoid(p.values, p.grid)) == pytest.approx(1.0, abs=2e-3)

    def test_provenance_states_both_radii(self, z, n_electrons, n, l):
        _, p = hf_radial(z, n_electrons, n, l)
        assert "windowed from a mesh reaching" in p.provenance.method


def test_the_inner_orbital_is_drawn_far_tighter_than_the_valence_one():
    """The window is per orbital, which is the whole point of having one.

    One box for the atom would have to be the valence box, and that is the box
    that loses the 1s. These two come out of the same solve and the same
    `points`, and their drawn ranges must differ by more than an order of
    magnitude, because the orbitals do.
    """
    _, core = screened_radial(18, 18, 1, 0)
    _, valence = screened_radial(18, 18, 3, 1)
    assert valence.grid[-1] > 10 * core.grid[-1]
