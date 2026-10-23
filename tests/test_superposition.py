import numpy as np
import pytest

from atomsim.analytic.wavefunction import evaluate_state
from atomsim.provenance import Fidelity
from atomsim.superposition import (
    Component,
    evolve_current,
    evolve_density,
    normalized_coefficients,
    superposition_grid,
)

# 1s + 2p0, equal populations: the textbook sloshing dipole.
SLOSH = [Component(1, 0, 0, 1.0), Component(2, 1, 0, 1.0)]
BEAT = 0.5 - 0.125  # E_2 - E_1 for hydrogen, hartree


def _z_centroid(sg, rho):
    z = sg.axis[:, None]
    return float((rho * z).sum() / rho.sum())


def test_matches_direct_complex_evaluation_at_any_time():
    """The plane grids are real, and the sum over them is the full complex Psi.

    This is the claim the whole design leans on: on y=0 the complex-basis
    psi_nlm has no imaginary part, so the browser's real-grid phase sum loses
    nothing. Checked with m = +/-1, where e^{i m phi} is not trivially 1.
    """
    comps = [Component(2, 1, 1, 0.3, 0.4), Component(3, 2, -1, 0.7, -1.1)]
    sg = superposition_grid(comps, resolution=21)
    t = 7.3
    i, j = 4, 15
    pos = np.array([[sg.axis[j], 0.0, sg.axis[i]]])
    psi = sum(
        c * evaluate_state(k.n, k.l, k.m, pos).values[0] * np.exp(-1j * e.value * t)
        for c, k, e in zip(sg.coefficients, comps, sg.energies)
    )
    assert evolve_density(sg, t)[i, j] == pytest.approx(abs(psi) ** 2, rel=1e-10)


def test_populations_are_normalized_and_phases_kept():
    c = normalized_coefficients([Component(1, 0, 0, 2.0, 0.0), Component(2, 1, 0, 6.0, 1.0)])
    assert np.sum(np.abs(c) ** 2) == pytest.approx(1.0)
    assert abs(c[0]) ** 2 == pytest.approx(0.25)
    assert np.angle(c[1]) == pytest.approx(1.0)


def test_density_returns_after_one_bohr_period():
    sg = superposition_grid(SLOSH, resolution=33)
    period = 2 * np.pi / BEAT
    assert sg.beat_frequencies() == pytest.approx([BEAT])
    rho0 = evolve_density(sg, 0.0)
    np.testing.assert_allclose(evolve_density(sg, period), rho0, rtol=1e-9, atol=1e-14)
    # and it really moved in between
    assert np.abs(evolve_density(sg, period / 2) - rho0).max() > 0.1 * rho0.max()


def test_the_dipole_sloshes_along_z():
    """psi_2p0 > 0 for z > 0, so at t = 0 the in-phase sum piles up at +z and
    half a period later at -z."""
    sg = superposition_grid(SLOSH, resolution=65)
    period = 2 * np.pi / BEAT
    assert _z_centroid(sg, evolve_density(sg, 0.0)) > 0.1
    assert _z_centroid(sg, evolve_density(sg, period / 2)) < -0.1
    assert abs(_z_centroid(sg, evolve_density(sg, period / 4))) < 1e-9


def test_time_average_kills_the_interference():
    """Averaged over a period, cross terms vanish: <rho> = sum_k |c_k|^2 psi_k^2."""
    sg = superposition_grid(SLOSH, resolution=33)
    period = 2 * np.pi / BEAT
    ts = np.arange(64) * period / 64  # exact for a single cosine
    mean = np.mean([evolve_density(sg, t) for t in ts], axis=0)
    incoherent = np.tensordot(np.abs(sg.coefficients) ** 2, sg.psi**2, axes=1)
    np.testing.assert_allclose(mean, incoherent, rtol=1e-9, atol=1e-14)


def test_degenerate_states_are_stationary():
    sg = superposition_grid([Component(2, 0, 0, 1.0), Component(2, 1, 0, 1.0)], resolution=17)
    assert sg.beat_frequencies() == []
    np.testing.assert_allclose(evolve_density(sg, 123.4), evolve_density(sg, 0.0))


def test_zero_weight_component_does_not_beat():
    sg = superposition_grid([Component(1, 0, 0, 1.0), Component(2, 1, 0, 0.0)], resolution=9)
    assert sg.beat_frequencies() == []


def test_reduced_mass_speeds_the_clock():
    """Every E_n scales with mu, so every beat does too: muonic hydrogen at
    mu/m_e ~ 186 beats 186 times faster in a frame 186 times smaller."""
    mu = 185.84
    h = superposition_grid(SLOSH, resolution=9)
    muh = superposition_grid(SLOSH, mu_ratio=mu, resolution=9)
    assert muh.beat_frequencies()[0] == pytest.approx(mu * h.beat_frequencies()[0])
    assert muh.axis[-1] == pytest.approx(h.axis[-1] / mu)


def test_frame_fits_the_largest_n():
    sg = superposition_grid([Component(1, 0, 0, 1.0), Component(3, 1, 0, 1.0)], resolution=9)
    assert sg.axis[-1] == pytest.approx(2.5 * 9)


def test_provenance_is_exact_and_names_what_it_leaves_out():
    sg = superposition_grid(SLOSH, resolution=9)
    assert sg.provenance.fidelity is Fidelity.EXACT
    text = " ".join(sg.provenance.assumptions)
    assert "spontaneous emission" in text
    assert "degenerate" in text


@pytest.mark.parametrize(
    "comps",
    [
        [],
        [Component(1, 0, 0, 1.0)] * 2,
        [Component(n, 0, 0, 1.0) for n in range(1, 6)],
        [Component(1, 0, 0, 0.0)],
        [Component(1, 0, 0, -1.0), Component(2, 0, 0, 2.0)],
        [Component(1, 1, 0, 1.0)],
        [Component(2, 1, 2, 1.0)],
        [Component(1, 0, 0, 1.0, float("nan"))],
    ],
)
def test_bad_mixes_are_refused(comps):
    with pytest.raises(ValueError):
        superposition_grid(comps, resolution=9)


def _exact_inplane_j(comps, Z, mu, t, x, z, h=1e-5):
    """Im(Psi* grad Psi)/mu at (x, 0, z), from the complex psi_k off the grid."""
    sg = superposition_grid(comps, Z=Z, mu_ratio=mu, resolution=2)
    e = np.array([q.value for q in sg.energies])
    amp = sg.coefficients * np.exp(-1j * e * t)
    pos = np.array([[x, 0, z], [x + h, 0, z], [x - h, 0, z], [x, 0, z + h], [x, 0, z - h]])
    psi = sum(
        a * evaluate_state(c.n, c.l, c.m, pos, Z=Z, mu_ratio=mu).values
        for a, c in zip(amp, comps)
    )
    gx = (psi[1] - psi[2]) / (2 * h)
    gz = (psi[3] - psi[4]) / (2 * h)
    return np.imag(np.conj(psi[0]) * gx) / mu, np.imag(np.conj(psi[0]) * gz) / mu


def test_current_converges_at_second_order_to_im_psi_star_grad_psi():
    comps = [Component(1, 0, 0, 1.0), Component(2, 1, 0, 1.0, 0.3), Component(3, 2, 1, 0.5)]
    pts = [(1.5, 2.0), (-2.5, 1.0), (3.0, -2.0), (0.5, -3.5)]
    errs = []
    for res in (33, 65):  # steps 0.5 and 0.25 bohr; both grids hold every point
        sg = superposition_grid(comps, resolution=res, half_extent=8.0)
        jx, jz = evolve_current(sg, 3.0)
        err = 0.0
        for x, z in pts:
            i, j = int(np.argmin(abs(sg.axis - z))), int(np.argmin(abs(sg.axis - x)))
            ex, ez = _exact_inplane_j(comps, 1, 1.0, 3.0, x, z)
            err = max(err, abs(jx[i, j] - ex), abs(jz[i, j] - ez))
        errs.append(err)
    assert errs[1] < 1e-3
    assert 3.0 < errs[0] / errs[1] < 5.0  # second order: halve h, quarter the error


def test_a_single_stationary_state_has_no_inplane_current():
    sg = superposition_grid([Component(3, 2, 1, 1.0)], resolution=33)
    jx, jz = evolve_current(sg, 7.0)
    assert not jx.any() and not jz.any()


def test_current_obeys_continuity_at_reduced_mass():
    # Axisymmetric (all m = 0), so on y=0 at x > 0: div j = d(jx)/dx + jx/x + d(jz)/dz.
    # mu = 0.5 is the check on the 1/mu: the clock runs at mu too, and a
    # mismatch between the two would break d(rho)/dt + div j = 0 by a factor of 2.
    comps = [Component(1, 0, 0, 1.0), Component(2, 1, 0, 1.0)]
    sg = superposition_grid(comps, mu_ratio=0.5, resolution=401, half_extent=10.0)
    t, dt = 2.0, 1e-4
    drho = (evolve_density(sg, t + dt) - evolve_density(sg, t - dt)) / (2 * dt)
    jx, jz = evolve_current(sg, t)
    h = sg.axis[1] - sg.axis[0]
    div = np.gradient(jx, h, axis=1) + jx / np.where(sg.axis == 0, np.inf, sg.axis) \
        + np.gradient(jz, h, axis=0)
    for x, z in [(2.0, 1.0), (3.0, -2.0), (1.0, 3.0), (4.0, 0.5)]:
        i, j = int(np.argmin(abs(sg.axis - z))), int(np.argmin(abs(sg.axis - x)))
        assert div[i, j] == pytest.approx(-drho[i, j], rel=0.02)
