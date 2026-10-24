"""Where psi_nlm is exactly zero, in closed form.

    psi_nlm = R_nl(r) Y(theta, phi),  R_nl ~ L_{n-l-1}^{2l+1}(2 Z mu' r / n)

Radial nodes are the n - l - 1 roots of that Laguerre polynomial: spheres.

The angular part depends on the basis, and the usual "l angular nodes" is only
true of one of them. Theta-dependence is P_l^|m|(cos theta), which vanishes at
the l - |m| roots of the |m|-th derivative of the Legendre polynomial (nodal
cones; theta = pi/2 is the xy plane) and, when m != 0, on the z axis itself,
through the (1 - x^2)^{|m|/2} factor. Then:

* complex Y_lm: |e^{i m phi}| = 1 never vanishes, so |psi|^2 has the cones and
  the z axis and nothing more. A complex m != 0 orbital has l - |m| nodal
  surfaces, not l: its |m| "missing" nodes became a nodal line.
* real S_lm: cos(|m| phi) or sin(|m| phi) adds |m| nodal planes through the z
  axis, which brings the count to l.

All of it EXACT: these are the zero sets of the closed-form functions.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.polynomial import legendre
from scipy.special import roots_genlaguerre

from atomsim.analytic.angular import validate_angular
from atomsim.analytic.hydrogen import validate_quantum_numbers
from atomsim.provenance import Fidelity, Provenance, Quantity


@dataclass(frozen=True)
class NodeSet:
    radial: tuple[Quantity, ...]   # sphere radii, bohr, ascending
    cones: tuple[Quantity, ...]    # polar angles from +z, rad, ascending, in (0, pi)
    planes: tuple[Quantity, ...]   # azimuths of nodal planes through z, rad, in [0, pi)
    z_axis: bool                   # psi vanishes on the whole z axis (m != 0)
    basis: str
    provenance: Provenance


def nodes(
    n: int, l: int, m: int, Z: int = 1, mu_ratio: float = 1.0, basis: str = "complex"
) -> NodeSet:
    validate_quantum_numbers(n, l)
    validate_angular(l, m)
    if basis not in ("complex", "real"):
        raise ValueError(f"basis must be 'complex' or 'real', got {basis!r}")
    am = abs(m)
    prov = Provenance(
        fidelity=Fidelity.EXACT,
        method=(
            "roots of L_{n-l-1}^{2l+1} (radial), of d^|m| P_l / dx^|m| (cones), and "
            "of cos/sin(|m| phi) (planes, real basis); float64 root finding"
        ),
        assumptions=("hydrogen-like closed-form psi_nlm",),
    )

    def q(value, unit, label):
        return Quantity(value=float(value), unit=unit, label=label, provenance=prov)

    radial: tuple[Quantity, ...] = ()
    if n - l - 1 > 0:
        rho = np.sort(roots_genlaguerre(n - l - 1, 2 * l + 1)[0])
        radial = tuple(
            q(x * n / (2 * Z * mu_ratio), "bohr", f"radial node {k + 1}")
            for k, x in enumerate(rho)
        )

    cones: tuple[Quantity, ...] = ()
    if l - am > 0:
        coef = legendre.legder(np.eye(l + 1)[l], am)  # d^|m| P_l / dx^|m|
        xs = np.sort(np.real(legendre.legroots(coef)))
        cones = tuple(
            q(math.acos(min(1.0, max(-1.0, x))), "rad", f"nodal cone {k + 1}")
            for k, x in enumerate(xs[::-1])  # ascending theta
        )

    planes: tuple[Quantity, ...] = ()
    if basis == "real" and am > 0:
        # m > 0 is the cos(|m| phi) combination, m < 0 the sin(|m| phi) one.
        offset = math.pi / (2 * am) if m > 0 else 0.0
        planes = tuple(
            q(offset + k * math.pi / am, "rad", f"nodal plane {k + 1}") for k in range(am)
        )

    return NodeSet(
        radial=radial, cones=cones, planes=planes, z_axis=am > 0, basis=basis,
        provenance=prov,
    )
