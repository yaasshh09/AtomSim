"""Radial meshes, and the discretization that goes with them (NUMERICAL).

A uniform grid has to resolve two scales at once. The 1s core contracts as
1/Z, so the step must shrink as 1/Z; the valence still reaches tens of bohr,
so the box cannot shrink with it. Argon needs h = 5.6e-4 over 40 bohr, which
is 72000 points, and 99% of them sit in vacuum describing a function that is
zero. Measured, that costs 88 seconds per SCF step and about an hour per atom.

An exponential mesh r = r_min e^(i delta) spends its points where the physics
is: constant RELATIVE resolution, so the innermost interval is as fine
compared to the 1s as the outermost is compared to the valence tail. Argon
needs a few thousand points instead of seventy-two thousand.

## The discretization

Each mesh type carries the kinetic operator its own change of variables
produces, rather than sharing one generic assembly. The obvious alternative -
the finite-element weak form, which would cover any mesh at once - was tried
and does not survive the origin: lumping the mass matrix costs O(delta^2)
RELATIVE, the kinetic term it divides is O(1/delta^2), and the product is a
finite spurious multiple of 1/r^2. That is 4e10 hartree of fictitious
repulsion at r = 1e-6, and it does not shrink when the mesh refines. A term
proportional to 1/r^2 has to be exactly right or not present at all.

The exponential mesh's own transformation, with x = ln r and P = sqrt(r) Q,
gives the exact operator:

    diag      V(r) + (2l+1)^2 / 8r^2 + 1 / delta^2 r^2
    offdiag   -1 / 2 delta^2 r_i r_i+1

where the (2l+1)^2/8 is the usual l(l+1)/2 plus the 1/8 the substitution
generates. The uniform mesh keeps the 3-point stencil it has always used, so
the exponential mesh is a new capability rather than a rewrite of results
already validated.

Working variable: the eigenproblem is solved for S = sqrt(delta J) P, not for
P. That substitution is what makes the plain Euclidean dot product on S equal
the physical integral P^2 dr, so an eigensolver that knows nothing about the
mesh still measures the right norm.

## The inner wall, and why this mesh has an accuracy floor

An exponential mesh cannot reach the origin, so P = 0 is imposed at r_min
instead of at 0. Left as a hard wall that is expensive: a hard sphere of
radius a shifts an s-state by 2 pi a |psi(0)|^2, which for hydrogen is exactly
2a, so r_min = 1e-2 costs 3.5% of the ground state and the error falls only
LINEARLY in r_min. Measured: 3.5e-2, 3.9e-3, 3.8e-4 at r_min = 1e-2, 1e-3,
1e-4.

The fix is to stop pretending P vanishes there. P ~ r^(l+1) near the origin,
so the ghost node one step below r_min is not zero, it is
S_-1 = S_0 exp(-(l+3/2) delta), and folding that into the first diagonal entry
turns the linear error quadratic. It buys two orders at every r_min.

That leaves a genuine floor, and it is worth stating rather than discovering
later. Two errors move in opposite directions:

    inner-wall truncation    ~ 2 (Z r_min)^2, relative
    eigensolver conditioning ~ eps / (delta^2 r_min^2 |E|), relative

the second because the symmetric operator's largest entry is 1/delta^2 r_min^2
while its lowest eigenvalue is order Z^2, and a symmetric eigensolve is
accurate to eps times the norm. The product is minimized at Z r_min ~ 1.1e-3
with a floor near 2.4e-6 relative, INDEPENDENT of Z and of the point count.
Measured at the optimum: 2.2e-6 for hydrogen and 1.8e-6 for Z = 18.

So `for_atom` places r_min at 1e-3/Z, and 2e-6 relative is what this mesh
delivers - about fifty times better than the 1e-4 the Hartree-Fock benchmarks
need, and reached with forty times fewer points than the uniform grid. It is
also why nothing here should be Richardson-extrapolated: past the optimum the
residual is conditioning noise, not a smooth power of delta, and extrapolating
noise sharpens nothing.

Hartree atomic units. Plain arrays in, plain arrays out: this is quadrature
and discretization, not physics, so provenance belongs to the caller. Same
exemption slater.py documents.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["RadialMesh", "exponential_mesh", "mesh_for_atom", "uniform_mesh"]

# Z r_min at the crossover between wall truncation and conditioning noise.
_OPTIMAL_SCALED_INNER_RADIUS = 1.0e-3


@dataclass(frozen=True)
class RadialMesh:
    """A radial grid together with the coordinate its quadrature is uniform in.

    `jacobian` is dr/dx at each node. `kinetic_diag` and `kinetic_offdiag` are
    the l-independent part of -1/2 d2/dr2 in the S representation, which is
    where the mesh's own change of variables lives; the centrifugal term is
    added by hamiltonian_bands because it depends on l.

    `inner_wall_coupling` is what the first row would couple to a node below
    r[0] with, and `inner_ghost_ratio` is that node's radius over r[0]. A mesh
    whose wall sits exactly on the origin sets the ratio to zero, which makes
    the correction vanish identically rather than by cancellation.
    """

    r: np.ndarray
    jacobian: np.ndarray
    step: float
    kinetic_diag: np.ndarray
    kinetic_offdiag: np.ndarray
    inner_wall_coupling: float
    inner_ghost_ratio: float

    def __post_init__(self) -> None:
        if self.r.ndim != 1 or self.r.size < 3:
            raise ValueError(f"mesh must be 1-D with at least 3 points, got {self.r.shape}")
        if self.r[0] <= 0.0:
            raise ValueError(
                f"mesh must start strictly above zero, got r[0]={float(self.r[0])!r}; "
                "r = 0 makes the centrifugal term and the pair-potential outer "
                "integrand infinite"
            )
        if not np.all(np.diff(self.r) > 0.0):
            raise ValueError("mesh must be strictly increasing")
        if self.jacobian.shape != self.r.shape:
            raise ValueError("jacobian must have one entry per node")
        if self.kinetic_diag.shape != self.r.shape:
            raise ValueError("kinetic diagonal must have one entry per node")
        if self.kinetic_offdiag.shape != (self.r.size - 1,):
            raise ValueError(
                f"kinetic off-diagonal must have one entry per interior "
                f"interval, so {self.r.size - 1}, got {self.kinetic_offdiag.shape}"
            )
        if self.step <= 0.0:
            raise ValueError(f"mesh step must be positive, got {self.step!r}")
        if not 0.0 <= self.inner_ghost_ratio < 1.0:
            raise ValueError(
                f"inner ghost node must sit below r[0], got a ratio of "
                f"{self.inner_ghost_ratio!r}"
            )

    @property
    def points(self) -> int:
        return int(self.r.size)

    @property
    def outer_wall(self) -> float:
        """Where the outer Dirichlet condition sits, one step past r[-1].

        Not the same as r[-1]: a uniform mesh built for a box of 40 bohr has
        its last node just inside 40 and its wall exactly on it.
        """
        return float(self.r[-1] + self.step * self.jacobian[-1])

    def integrate(self, f: np.ndarray) -> float:
        """integral f(r) dr, as integral f J dx by the trapezoid rule in x.

        On a uniform mesh J = 1 and x = r, so this is exactly
        np.trapezoid(f, r) - the quadrature every existing result was computed
        with.
        """
        return float(np.trapezoid(np.asarray(f, dtype=float) * self.jacobian) * self.step)

    def cumulative(self, f: np.ndarray) -> np.ndarray:
        """integral from the inner wall to r of f ds, same length as r."""
        y = np.asarray(f, dtype=float) * self.jacobian
        increments = 0.5 * (y[..., 1:] + y[..., :-1]) * self.step
        out = np.empty_like(y)
        out[..., 0] = 0.0
        np.cumsum(increments, axis=-1, out=out[..., 1:])
        return out

    def to_s(self, p: np.ndarray) -> np.ndarray:
        """P -> S = sqrt(delta J) P, the variable the eigenproblem is in."""
        return np.asarray(p, dtype=float) * np.sqrt(self.step * self.jacobian)

    def to_p(self, s: np.ndarray) -> np.ndarray:
        """S -> P, the physical radial function r R(r)."""
        return np.asarray(s, dtype=float) / np.sqrt(self.step * self.jacobian)

    def normalized(self, p: np.ndarray) -> np.ndarray:
        """P scaled so that integral P^2 dr = 1 in this mesh's own quadrature."""
        return np.asarray(p, dtype=float) / np.sqrt(self.integrate(np.asarray(p) ** 2))

    def hamiltonian_bands(
        self, v_local: np.ndarray, l: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Diagonal and off-diagonal of the local Hamiltonian in S."""
        if l < 0:
            raise ValueError(f"orbital quantum number l must be >= 0, got {l}")
        v = np.asarray(v_local, dtype=float)
        if v.shape != self.r.shape:
            raise ValueError("potential must be sampled on this mesh")

        diag = self.kinetic_diag + l * (l + 1) * 0.5 / self.r**2 + v
        # P ~ r^(l+1) near the origin, so S ~ r^(l+3/2). Folding the ghost node
        # in is what makes the wall error quadratic in r_min instead of linear.
        diag[0] += self.inner_wall_coupling * self.inner_ghost_ratio ** (l + 1.5)
        return diag, self.kinetic_offdiag


def uniform_mesh(r_max: float, points: int) -> RadialMesh:
    """r = h, 2h, ... with h = r_max / (points + 1).

    The convention radial_solver.solve_radial has always used: r[0] == h, so
    the Dirichlet wall lands exactly on r = 0 rather than one step inside it.
    """
    if r_max <= 0.0:
        raise ValueError(f"box radius must be positive, got {r_max!r}")
    if points < 3:
        raise ValueError(f"mesh needs at least 3 points, got {points}")
    h = r_max / (points + 1)
    return RadialMesh(
        r=h * np.arange(1, points + 1),
        jacobian=np.ones(points),
        step=h,
        kinetic_diag=np.full(points, 1.0 / h**2),
        kinetic_offdiag=np.full(points - 1, -0.5 / h**2),
        inner_wall_coupling=-0.5 / h**2,
        inner_ghost_ratio=0.0,  # the wall IS the origin; P there is exactly 0
    )


def exponential_mesh(r_min: float, r_max: float, points: int) -> RadialMesh:
    """r = r_min e^(i delta), geometrically spaced from r_min to r_max."""
    if r_min <= 0.0:
        raise ValueError(
            f"inner radius must be strictly above zero, got {r_min!r}; an "
            "exponential mesh cannot reach the origin, it approaches it"
        )
    if r_max <= r_min:
        raise ValueError(f"box radius {r_max!r} must exceed inner radius {r_min!r}")
    if points < 3:
        raise ValueError(f"mesh needs at least 3 points, got {points}")

    delta = float(np.log(r_max / r_min) / (points - 1))
    r = r_min * np.exp(delta * np.arange(points))
    ghost = float(np.exp(-delta))

    # The 1/8 is not the centrifugal term. It is what P = sqrt(r) Q generates
    # on its own, and together with the l(l+1)/2 that hamiltonian_bands adds it
    # makes the (2l+1)^2/8 of the exponential-mesh literature.
    return RadialMesh(
        r=r,
        jacobian=r.copy(),
        step=delta,
        kinetic_diag=(1.0 / delta**2 + 0.125) / r**2,
        kinetic_offdiag=-0.5 / (delta**2 * r[:-1] * r[1:]),
        inner_wall_coupling=-0.5 / (delta**2 * ghost * r[0] ** 2),
        inner_ghost_ratio=ghost,
    )


def mesh_for_atom(z: int, r_max: float, points: int) -> RadialMesh:
    """The exponential mesh at the inner radius that minimizes total error.

    r_min = 1e-3 / Z is where wall truncation and eigensolver conditioning
    cross; see the module docstring for the measured balance. Both scales in
    the problem are set by Z, which is why one constant covers every element.
    """
    if z < 1:
        raise ValueError(f"Z must be >= 1, got {z}")
    return exponential_mesh(_OPTIMAL_SCALED_INNER_RADIUS / z, r_max, points)
