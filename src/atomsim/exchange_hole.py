"""The exchange hole: where the other electrons are, given where one of them is.

A density rho(r) says where an electron is likely to be. It cannot say how the
electrons avoid each other, and they do: two electrons of the same spin can
never be at the same place, because the wavefunction is antisymmetric. Fix one
electron of spin sigma at r1 and ask where the others are, and the answer is
the conditional density

    rho(r2 | r1) = rho(r2) + h_x(r1, r2),
    h_x(r1, r2) = -|gamma_sigma(r1, r2)|^2 / rho_sigma(r1),

with gamma_sigma the one-spin density matrix of the Hartree-Fock determinant.
Two facts about h_x are exact for any single determinant:

* at r2 = r1 it equals -rho_sigma(r1): the same-spin density vanishes on top
  of the electron, which is the exclusion principle as a picture. This holds
  by construction of the formula, so it is drawn, not tested;
* it integrates to exactly -1: the reference electron is taken out of the
  crowd, and only out of its own spin. That one depends on the orbitals being
  orthonormal under the solver's own quadrature, so it is computed and
  reported as a check.

For a closed-shell atom the addition theorem collapses the sum over m:

    gamma_sigma(r1, r2) = sum_nl (2l + 1) / (4 pi) P_nl(r1) P_nl(r2) / (r1 r2) P_l(cos theta12).

Only closed-shell ground states are offered. An open shell in this solver is an
average over its terms, a spherical density that no single determinant has,
and its "exchange hole" would be the hole of a state that does not exist.

What this does not show is the hole around opposite-spin electrons, the
Coulomb hole. Hartree-Fock has none: it puts opposite-spin electrons down
independently, which is exactly the correlation it leaves out. The provenance
says so.

The profile is drawn along the line through the nucleus and the reference
electron, where cos theta12 is +1 on the electron's side and -1 beyond the
nucleus. Every point on it is a solver mesh node, so nothing is interpolated;
the reference electron itself is snapped to the nearest node and reported
where it was snapped to.
"""

from dataclasses import dataclass

import numpy as np

from atomsim.atoms import aufbau_configuration, subshell_capacity
from atomsim.hf_atom import hf_mesh, solve_hartree_fock
from atomsim.provenance import Fidelity, Provenance, Quantity

#: Neutral closed-shell ground states the Hartree-Fock solver reaches.
CLOSED_SHELL_Z = tuple(
    z for z in range(2, 37)
    if all(q == subshell_capacity(l) for (_, l), q in aufbau_configuration(z))
)

PROFILE_REACH = 6.0  # bohr either side of the nucleus

_ASSUMPTIONS = (
    "the Hartree-Fock determinant of a closed-shell ground state",
    "same-spin (exchange) hole only: Hartree-Fock has no Coulomb hole between "
    "opposite spins, which is the correlation it leaves out",
    "drawn on the line through the nucleus and the reference electron",
)


@dataclass(frozen=True)
class ExchangeHole:
    z: int
    a: float  # bohr, the reference electron's distance from the nucleus, snapped
    s: np.ndarray  # signed position along the line, bohr; + is the electron's side
    density: np.ndarray  # rho(s), bohr^-3, all electrons
    conditional: np.ndarray  # rho(s | a), bohr^-3, the other N - 1
    sum_rule: Quantity  # integral of h_x over all space; -1 exactly in theory
    provenance: Provenance


def exchange_hole(z: int, a: float) -> ExchangeHole:
    """The exchange hole around an electron a bohr from the nucleus of atom Z."""
    if z not in CLOSED_SHELL_Z:
        raise ValueError(
            f"the exchange hole is drawn for closed-shell atoms, Z in {CLOSED_SHELL_Z}, "
            f"got {z}; an open shell here is an average over terms, not a determinant"
        )
    if not 0.0 < a <= PROFILE_REACH:
        raise ValueError(f"a must be in (0, {PROFILE_REACH}] bohr, got {a}")
    config = aufbau_configuration(z)
    result = solve_hartree_fock(z, z, config)
    n_top = max(n for (n, _), _ in config)
    mesh = hf_mesh(z, z, n_top, refinement=2)
    r = mesh.r
    if not np.array_equal(r, result.orbitals[0].P.grid):
        raise RuntimeError("the rebuilt mesh is not the one the solve ran on")

    i = int(np.argmin(np.abs(r - a)))
    a_node = float(r[i])
    shells = [(o.l, o.P.values) for o in result.orbitals]

    # c_l(r) = (2l+1)/(4 pi) sum_n P_nl(a) P_nl(r) / (a r): gamma's l-th Legendre part.
    ls = sorted({l for l, _ in shells})
    c = {l: np.zeros_like(r) for l in ls}
    for l, p in shells:
        c[l] += (2 * l + 1) / (4 * np.pi) * p[i] * p / (a_node * r)
    rho_sigma = sum((2 * l + 1) / (4 * np.pi) * (p / r) ** 2 for l, p in shells)
    rho_sigma_a = float(rho_sigma[i])

    # Sum rule by Legendre orthogonality, integral dOmega P_l P_l' = 4 pi / (2l+1)
    # delta, then the mesh's own radial quadrature. This does not assume the
    # orbitals are orthonormal; that is what it checks.
    hole_integral = -sum(
        4 * np.pi / (2 * l + 1) * mesh.integrate(c[l] ** 2 * r**2) for l in ls
    ) / rho_sigma_a

    keep = r <= PROFILE_REACH
    rr = r[keep]
    gamma_same = sum(c[l][keep] for l in ls)  # cos theta12 = +1
    gamma_opp = sum((-1) ** l * c[l][keep] for l in ls)  # cos theta12 = -1
    density = 2.0 * rho_sigma[keep]

    return ExchangeHole(
        z=z,
        a=a_node,
        s=np.concatenate([-rr[::-1], rr]),
        density=np.concatenate([density[::-1], density]),
        conditional=np.concatenate([
            (density - gamma_opp**2 / rho_sigma_a)[::-1],
            density - gamma_same**2 / rho_sigma_a,
        ]),
        sum_rule=Quantity(float(hole_integral), "dimensionless", "integral h_x", Provenance(
            fidelity=Fidelity.NUMERICAL,
            method="the exchange hole's integral, by the solver mesh's own quadrature",
            assumptions=("a deviation from -1 measures the mesh and the SCF, not the atom",),
        )),
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method=(
                "rho(r2|r1) = rho(r2) - |gamma_sigma(r1,r2)|^2 / rho_sigma(r1) from the "
                "converged Hartree-Fock orbitals, at the solver's mesh nodes"
            ),
            assumptions=_ASSUMPTIONS + (
                f"the reference electron snapped to the mesh node at {a_node:.4g} bohr",
            ),
            refinement="a correlated wavefunction would add the opposite-spin Coulomb hole",
        ),
    )
