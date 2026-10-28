"""Hydrogen in other numbers of dimensions: the same -Z/r pull, D = 2, 3 or 4.

Separating the Schrodinger equation in D dimensions leaves a radial equation
in which l(l + 1) is replaced by L(L + 1), with L = l + (D - 3)/2, and the
Coulomb problem then solves exactly as in three dimensions with n shifted:

    E_n = -Z^2 mu / (2 (n + (D - 3)/2)^2),   n = n_r + l + 1 = 1, 2, ...

In two dimensions the ground state is four times deeper than hydrogen's and
the levels are spaced by half-integers; in four they are shallower. The
degeneracy of a level counts the angular states of every l below n: 2n - 1 in
two dimensions, n^2 in three, n(n + 1)(2n + 1)/6 in four.

This is COUNTERFACTUAL twice over, and the provenance says which way. Space
has three dimensions, and in D dimensions Gauss's law would not give a 1/r
potential at all: it gives ln r in two dimensions, which binds every state and
lets none escape, and 1/r^2 in four, which is too strong at short range for
any ground state to exist. Keeping -Z/r is the choice that isolates what the
dimension alone does to the ladder.

The closed form is checked against a numerical solve of the D-dimensional
radial equation, which shares nothing with it but the potential.
"""

from math import comb

import numpy as np
from scipy.linalg import eigh_tridiagonal, solve_banded

from atomsim.numerics.mesh import exponential_mesh, mesh_for_atom_at_step
from atomsim.provenance import Fidelity, Provenance, Quantity

DIMENSIONS = (2, 3, 4)

_ASSUMPTIONS = (
    "the potential is kept as -Z/r in every dimension; a D-dimensional Gauss's law "
    "would give ln r in two dimensions and 1/r^2 in four",
    "non-relativistic, infinite nuclear mass unless mu is given",
)


def _validate(dim: int, n: int) -> None:
    if dim not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {DIMENSIONS}, got {dim}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")


def energy_d(n: int, dim: int, Z: int = 1, mu_ratio: float = 1.0) -> Quantity:
    """E_n of hydrogen in `dim` dimensions with the 1/r potential, hartree."""
    _validate(dim, n)
    value = -(Z**2) * mu_ratio / (2.0 * (n + (dim - 3) / 2.0) ** 2)
    fidelity = Fidelity.EXACT if dim == 3 else Fidelity.COUNTERFACTUAL
    return Quantity(value, "hartree", f"E_{n} (D = {dim})", Provenance(
        fidelity=fidelity,
        method="closed form: E_n = -Z^2 mu / (2 (n + (D - 3)/2)^2)",
        assumptions=_ASSUMPTIONS if dim != 3 else _ASSUMPTIONS[1:],
    ))


def degeneracy(n: int, dim: int) -> int:
    """How many states share level n: the angular states of every l from 0 to n - 1.

    The degree-l harmonics on the (D - 1)-sphere number
    C(l + D - 1, D - 1) - C(l + D - 3, D - 1), the second term absent below l = 2.
    """
    _validate(dim, n)
    total = 0
    for l in range(n):
        total += comb(l + dim - 1, dim - 1) - (comb(l + dim - 3, dim - 1) if l >= 2 else 0)
    return total


def _bands(mesh, big_l: float, Z: int):
    """The radial Hamiltonian on an exponential mesh, for a centrifugal L that may be
    half-integer. Built by hand rather than with hamiltonian_bands, whose inner-wall
    ghost term assumes S ~ r^(l + 3/2) for an integer l; here S ~ r^(L + 3/2), and
    with a wall coupling near 1/(delta^2 r_min^2) the wrong exponent moves the 2-D
    ground state by most of a hartree."""
    r = mesh.r
    d = mesh.kinetic_diag + big_l * (big_l + 1) / (2.0 * r**2) - Z / r
    d[0] += mesh.inner_wall_coupling * mesh.inner_ghost_ratio ** (big_l + 1.5)
    return d, mesh.kinetic_offdiag


def numerical_energy(n: int, l: int, dim: int, Z: int = 1, step: float = 0.004) -> float:
    """The same level from a numerical solve of the D-dimensional radial equation.

    With R = u / r^((D - 1)/2) the radial equation is a one-dimensional one with
    centrifugal term L(L + 1) / (2 r^2), L = l + (D - 3)/2, which can be attractive:
    D = 2, l = 0 gives -1/(8 r^2). On the exponential mesh the change of variable
    adds exactly +1/(8 r^2), so that channel is regular there. Shares nothing with
    energy_d but the potential.

    The two-dimensional s channel needs care. With (L + 1/2) = 0 its irregular
    solution is ln r rather than a power, so the error from the inner wall falls
    only as fast as the wall moves in: 1.6e-2 hartree with the wall at 1e-3 bohr,
    1.6e-4 at 1e-5. The wall therefore sits at 1e-9 / Z, where the matrix norm is
    far too large for the tridiagonal eigensolver's own precision, so its answer on
    the usual mesh only seeds a shifted inverse iteration on the fine one.
    """
    _validate(dim, n)
    if not 0 <= l < n:
        raise ValueError(f"l must be in [0, n), got {l}")
    big_l = l + (dim - 3) / 2.0
    r_max = 60.0 * n * n
    k = n - l - 1
    coarse = mesh_for_atom_at_step(Z, r_max, step)
    d, o = _bands(coarse, big_l, Z)
    e = float(eigh_tridiagonal(d, o, select="i", select_range=(k, k))[0][0])
    r_min = 1e-9 / Z
    fine = exponential_mesh(r_min, r_max, int(np.log(r_max / r_min) / step) + 1)
    d, o = _bands(fine, big_l, Z)
    ab = np.zeros((3, d.size))
    ab[0, 1:] = o
    ab[2, :-1] = o
    s = np.ones(d.size)
    for _ in range(6):
        ab[1] = d - e
        s = solve_banded((1, 1), ab, s)
        s /= np.linalg.norm(s)
        hs = d * s
        hs[:-1] += o * s[1:]
        hs[1:] += o * s[:-1]
        e = float(s @ hs)
    return e
