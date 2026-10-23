import numpy as np
import pytest

from atomsim.analytic.hydrogen import _radial_eval, energy
from atomsim.expectations import hydrogen_expectations, hydrogen_virial, virial_check
from atomsim.numerics.radial_solver import solve_radial
from atomsim.provenance import Fidelity

STATES = [(1, 0, 1, 1.0), (2, 1, 1, 1.0), (3, 2, 2, 0.99986), (5, 1, 1, 0.5), (6, 5, 3, 1.0)]


@pytest.mark.parametrize("n,l,Z,mu", STATES)
def test_kramers_moments_match_direct_quadrature(n, l, Z, mu):
    a = 1 / (Z * mu)
    r = np.linspace(0.0, 40 * n * n * a, 400001)
    rr = _radial_eval(n, l, r, Z * mu) ** 2  # R^2, finite at r = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        moment = {k: rr * r ** (2 + k) for k in (1, 2, -1)}
    moment[-2] = rr  # r^2 R^2 / r^2
    got = {q.label: q for q in hydrogen_expectations(n, l, Z, mu)}
    for label, power in [("<r>", 1), ("<r^2>", 2), ("<1/r>", -1), ("<1/r^2>", -2)]:
        assert got[label].value == pytest.approx(np.trapezoid(moment[power], r), rel=1e-6)
        assert got[label].provenance.fidelity is Fidelity.EXACT
    if l > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            inv3 = np.where(r > 0, rr / r, 0.0)  # r^2 R^2 / r^3; R ~ r^l, so it vanishes at 0
        assert got["<1/r^3>"].value == pytest.approx(np.trapezoid(inv3, r), rel=1e-5)
    else:
        assert "<1/r^3>" not in got  # diverges for s states
    spread = np.sqrt(got["<r^2>"].value - got["<r>"].value ** 2)
    assert got["Delta r"].value == pytest.approx(spread)
    e = energy(n, Z=Z, mu_ratio=mu).value
    assert got["<T>"].value == pytest.approx(-e) and got["<V>"].value == pytest.approx(2 * e)


@pytest.mark.parametrize("n,l,Z,mu", STATES)
def test_closed_form_wavefunctions_pass_the_virial_check(n, l, Z, mu):
    v = hydrogen_virial(n, l, Z, mu)
    assert abs(v.residual.value) < 1e-5
    # <T> comes from the shape of R, never from E, and still lands on -E
    assert v.kinetic.value == pytest.approx(-energy(n, Z=Z, mu_ratio=mu).value, rel=1e-5)
    assert v.potential.value == pytest.approx(2 * energy(n, Z=Z, mu_ratio=mu).value, rel=1e-5)


def test_a_numerical_oscillator_obeys_2T_equals_2V():
    pot = lambda r: 0.5 * 0.3**2 * r**2  # noqa: E731
    sol = solve_radial(pot, l=1, r_max=40.0, n_points=8000, n_states=3)
    for k in range(3):
        v = virial_check(sol.r, sol.u[k], pot, 1, 1.0, source="the solver")
        assert abs(v.residual.value) < 1e-4
        assert v.r_dv_dr.value == pytest.approx(2 * v.potential.value, rel=1e-6)  # k = 2


def test_the_check_catches_a_state_the_box_is_squeezing():
    coulomb = lambda r: -1.0 / r  # noqa: E731
    roomy = solve_radial(coulomb, l=0, r_max=80.0, n_points=16000, n_states=3)
    tight = solve_radial(coulomb, l=0, r_max=12.0, n_points=16000, n_states=3)
    # 3s reaches out to ~25 bohr: a 12 bohr box pushes on it, an 80 bohr box does not
    ok = virial_check(roomy.r, roomy.u[2], coulomb, 0, 1.0, source="roomy")
    bad = virial_check(tight.r, tight.u[2], coulomb, 0, 1.0, source="tight")
    assert abs(ok.residual.value) < 1e-3
    assert abs(bad.residual.value) > 0.1
