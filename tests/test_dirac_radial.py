"""Dirac radial functions: normalized, solving the radial Dirac equations, with
the 1s closed forms and the Schrodinger limit as alpha -> 0."""

import math

import numpy as np
import pytest
from scipy.integrate import quad

from atomsim.analytic.dirac import dirac_energy, dirac_kappa, dirac_radial
from atomsim.analytic.hydrogen import radial_wavefunction
from atomsim.constants import ALPHA

STATES = [(1, 0, 0.5), (2, 0, 0.5), (2, 1, 0.5), (2, 1, 1.5), (3, 2, 1.5), (4, 3, 3.5)]


def test_kappa():
    assert dirac_kappa(0, 0.5) == -1
    assert dirac_kappa(1, 0.5) == 1
    assert dirac_kappa(1, 1.5) == -2
    with pytest.raises(ValueError):
        dirac_kappa(0, 1.5)


@pytest.mark.parametrize("n,l,j", STATES)
@pytest.mark.parametrize("Z", [1, 50, 92])
def test_normalized(n, l, j, Z):
    def density(r):
        g, f = dirac_radial(n, l, j, np.array([r]), Z=Z)
        return float((g.values[0] ** 2 + f.values[0] ** 2) * r * r)

    assert quad(density, 0, np.inf, limit=400)[0] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("n,l,j", STATES)
@pytest.mark.parametrize("Z", [1, 80])
def test_solves_the_radial_dirac_equations(n, l, j, Z):
    # dG/dr + (k/r) G = (E - V + c^2) F / c and dF/dr - (k/r) F = -(E - V - c^2) G / c,
    # with G = r g, F = r f and E the total energy. Central differences on two
    # grids: the residual must fall by ~4 when h halves, so it is the stencil's.
    c = 1.0 / ALPHA
    k = dirac_kappa(l, j)
    e = dirac_energy(n, j, Z=Z).value + c * c

    def residual(points):
        # Away from r = 0, where j = 1/2 diverges as r^(gamma-1) and swamps the stencil.
        r = np.linspace(0.1 * n / Z, 8.0 * n * n / Z, points)
        g, f = dirac_radial(n, l, j, r, Z=Z)
        G, F = r * g.values, r * f.values
        h = r[1] - r[0]
        dG, dF = np.gradient(G, h), np.gradient(F, h)
        V = -Z / r
        r1 = dG + k / r * G - (e - V + c * c) / c * F
        r2 = dF - k / r * F + (e - V - c * c) / c * G
        scale = np.max(np.abs(dG))
        return max(np.max(np.abs(r1[2:-2])), np.max(np.abs(r2[2:-2]))) / scale

    coarse, fine = residual(4001), residual(8001)
    assert fine < 1e-4
    assert coarse / fine > 3.0


@pytest.mark.parametrize("Z", [1, 60, 92])
def test_1s_closed_forms(Z):
    # g_1s = C r^(gamma-1) e^(-Z r), and f / g = -sqrt((1-gamma)/(1+gamma)).
    gam = math.sqrt(1 - (Z * ALPHA) ** 2)
    r = np.array([0.1, 0.5, 1.3]) / Z
    g, f = dirac_radial(1, 0, 0.5, r, Z=Z)
    shape = r ** (gam - 1) * np.exp(-Z * r)
    assert np.allclose(g.values / shape, g.values[0] / shape[0], rtol=1e-12)
    assert np.allclose(f.values / g.values, -math.sqrt((1 - gam) / (1 + gam)), rtol=1e-12)


@pytest.mark.parametrize("n,l,j", STATES + [(3, 1, 0.5), (4, 2, 1.5)])
def test_schrodinger_limit(n, l, j):
    r = np.linspace(0.01, 6.0 * n * n, 600)
    g, f = dirac_radial(n, l, j, r, alpha=1e-6)
    R = radial_wavefunction(n, l, r).values
    assert np.max(np.abs(g.values - R)) < 1e-5 * np.max(np.abs(R))
    assert np.max(np.abs(f.values)) < 1e-5 * np.max(np.abs(R))


def test_refusals_and_tier():
    with pytest.raises(ValueError):
        dirac_radial(2, 2, 1.5, np.array([1.0]))
    g, _ = dirac_radial(1, 0, 0.5, np.array([1.0]))
    assert g.provenance.fidelity.value == "exact"
    g, _ = dirac_radial(1, 0, 0.5, np.array([1.0]), alpha=0.01)
    assert g.provenance.fidelity.value == "counterfactual"
