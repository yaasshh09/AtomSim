import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import spherical_jn

from atomsim.analytic.hydrogen import momentum_wavefunction, radial_wavefunction
from atomsim.provenance import Fidelity

STATES = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 1), (5, 3), (6, 0)]


def _bessel_transform(n, l, p, Z, mu):
    """sqrt(2/pi) int j_l(p r) R_nl(r) r^2 dr, by brute quadrature of the r-space form."""
    def f(r):
        return r * r * spherical_jn(l, p * r) * radial_wavefunction(
            n, l, np.array([r]), Z=Z, mu_ratio=mu
        ).values[0]
    return math.sqrt(2.0 / math.pi) * quad(f, 0.0, 60.0 * n * n / (Z * mu), limit=4000)[0]


@pytest.mark.parametrize("n,l", STATES)
@pytest.mark.parametrize("Z,mu", [(1, 1.0), (2, 0.5)])
def test_closed_form_is_the_fourier_transform_of_r_space(n, l, Z, mu):
    p = np.array([0.07, 0.31, 0.9]) * Z * mu
    closed = momentum_wavefunction(n, l, p, Z=Z, mu_ratio=mu).values
    numeric = np.array([_bessel_transform(n, l, pi, Z, mu) for pi in p])
    assert closed == pytest.approx(numeric, abs=1e-7 * np.max(np.abs(closed)) + 1e-12)


@pytest.mark.parametrize("n,l", STATES)
def test_normalized_and_p_squared_is_the_virial_value(n, l):
    # <p^2> = 2 mu <T> = (Z mu / n)^2 exactly; Z=3, mu=0.7 so a scaling slip shows.
    Z, mu = 3, 0.7

    def f(p):
        return momentum_wavefunction(n, l, np.array([p]), Z=Z, mu_ratio=mu).values[0]

    norm = quad(lambda p: (f(p) * p) ** 2, 0, np.inf, limit=400)[0]
    p2 = quad(lambda p: (f(p) * p * p) ** 2, 0, np.inf, limit=400)[0]
    assert norm == pytest.approx(1.0, abs=1e-9)
    assert p2 == pytest.approx((Z * mu / n) ** 2, rel=1e-8)


def test_node_count_and_provenance():
    p = np.linspace(1e-4, 10.0, 20001)
    for n, l in STATES:
        field = momentum_wavefunction(n, l, p)
        assert np.count_nonzero(np.diff(np.sign(field.values)) != 0) == n - l - 1
    assert field.provenance.fidelity is Fidelity.EXACT
    assert field.grid_unit == "hbar/bohr"
