"""My radial meshes, and the discretization that goes with them (NUMERICAL).

A uniform grid makes me resolve two scales at once. The 1s core contracts as
1/Z, so my step has to shrink as 1/Z; the valence still reaches tens of bohr,
so my box cannot shrink with it. Argon needs h = 5.6e-4 over 40 bohr, which is
72000 points, and 99% of them sit in vacuum describing a function that is zero.
I measured it: that costs 88 seconds per SCF step and about an hour per atom.

An exponential mesh r = r_min e^(i delta) spends its points where the physics
is: constant RELATIVE resolution, so my innermost interval is as fine compared
to the 1s as my outermost is compared to the valence tail. Argon then needs a
few thousand points instead of seventy-two thousand.

## My discretization

I give each mesh type the kinetic operator its own change of variables
produces, rather than sharing one generic assembly. I tried the obvious
alternative, the finite-element weak form, which would have covered any mesh at
once, and it does not survive the origin: lumping the mass matrix costs
O(delta^2) RELATIVE, the kinetic term it divides is O(1/delta^2), and the
product is a finite spurious multiple of 1/r^2. That is 4e10 hartree of
fictitious repulsion at r = 1e-6, and it does not shrink when I refine the
mesh. A term proportional to 1/r^2 has to be exactly right or not present at
all.

The exponential mesh's own transformation, with x = ln r and P = sqrt(r) Q,
gives me the exact operator:

    diag      V(r) + (2l+1)^2 / 8r^2 + 1 / delta^2 r^2
    offdiag   -1 / 2 delta^2 r_i r_i+1

where the (2l+1)^2/8 is the usual l(l+1)/2 plus the 1/8 the substitution
generates. I leave the uniform mesh on the 3-point stencil it has always used,
so my exponential mesh is a new capability rather than a rewrite of results I
have already validated.

My working variable: I solve the eigenproblem for S = sqrt(delta J) P, not for
P. That substitution is what makes the plain Euclidean dot product on S equal
the physical integral P^2 dr, so an eigensolver that knows nothing about my
mesh still measures the right norm.

## The inner wall, and why this mesh has an accuracy floor of mine

An exponential mesh cannot reach the origin, so I impose P = 0 at r_min
instead of at 0. Left as a hard wall that is expensive: a hard sphere of radius
a shifts an s-state by 2 pi a |psi(0)|^2, which for hydrogen is exactly 2a, so
r_min = 1e-2 costs me 3.5% of the ground state and the error falls only
LINEARLY in r_min. I measured 3.5e-2, 3.9e-3 and 3.8e-4 at r_min = 1e-2, 1e-3,
1e-4.

My fix is to stop pretending P vanishes there. P ~ r^(l+1) near the origin, so
the ghost node one step below r_min is not zero, it is
S_-1 = S_0 exp(-(l+3/2) delta), and folding that into my first diagonal entry
turns the linear error quadratic. It buys me two orders at every r_min.

That leaves a genuine floor, and I would rather state it than have you discover
it later. Two errors of mine move in opposite directions:

    inner-wall truncation    ~ 2 (Z r_min)^2, relative
    eigensolver conditioning ~ eps / (delta^2 r_min^2 |E|), relative

the second because my symmetric operator's largest entry is 1/delta^2 r_min^2
while its lowest eigenvalue is order Z^2, and a symmetric eigensolve is
accurate to eps times the norm. The product is minimized at Z r_min ~ 1.1e-3
with a floor near 2.4e-6 relative, INDEPENDENT of Z and of the point count. At
the optimum I measured 2.2e-6 for hydrogen and 1.8e-6 for Z = 18.

So `for_atom` places r_min at 1e-3/Z, and 2e-6 relative is what I deliver from
this mesh: about fifty times better than the 1e-4 my Hartree-Fock benchmarks
need, and reached with forty times fewer points than the uniform grid. It is
also why I should never Richardson-extrapolate anything here: past the optimum
my residual is conditioning noise, not a smooth power of delta, and
extrapolating noise sharpens nothing.

I work in Hartree atomic units. Plain arrays in, plain arrays out: this is
quadrature and discretization rather than physics, so provenance belongs to my
caller. Same exemption I document in slater.py.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "RadialMesh",
    "display_window",
    "exponential_mesh",
    "mesh_for_atom",
    "mesh_for_atom_at_step",
    "uniform_mesh",
]

# The Z r_min where my wall truncation crosses my conditioning noise.
_OPTIMAL_SCALED_INNER_RADIUS = 1.0e-3


@dataclass(frozen=True)
class RadialMesh:
    """A radial grid together with the coordinate its quadrature is uniform in.

    `jacobian` is dr/dx at each node. `kinetic_diag` and `kinetic_offdiag` are
    the l-independent part of -1/2 d2/dr2 in my S representation, which is
    where the mesh's own change of variables lives; I add the centrifugal term
    in hamiltonian_bands, because it depends on l.

    `inner_wall_coupling` is what my first row would couple to a node below
    r[0] with, and `inner_ghost_ratio` is that node's radius over r[0]. For a
    mesh whose wall sits exactly on the origin I set the ratio to zero, which
    makes the correction vanish identically rather than by cancellation.
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
                f"I need the mesh to start strictly above zero, got "
                f"r[0]={float(self.r[0])!r}; at r = 0 my centrifugal term and my "
                "pair-potential outer integrand are both infinite"
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
        """Where I put the outer Dirichlet condition, one step past r[-1].

        It is not the same as r[-1]: a uniform mesh I build for a box of 40
        bohr has its last node just inside 40 and its wall exactly on it.
        """
        return float(self.r[-1] + self.step * self.jacobian[-1])

    def integrate(self, f: np.ndarray) -> float:
        """integral f(r) dr, which I take as integral f J dx by the trapezoid
        rule in x.

        On a uniform mesh J = 1 and x = r, so this is exactly
        np.trapezoid(f, r), the quadrature I computed every existing result
        with.
        """
        return float(np.trapezoid(np.asarray(f, dtype=float) * self.jacobian) * self.step)

    def cumulative(self, f: np.ndarray) -> np.ndarray:
        """The integral from my inner wall out to r of f ds, same length as r."""
        y = np.asarray(f, dtype=float) * self.jacobian
        increments = 0.5 * (y[..., 1:] + y[..., :-1]) * self.step
        out = np.empty_like(y)
        out[..., 0] = 0.0
        np.cumsum(increments, axis=-1, out=out[..., 1:])
        return out

    def to_s(self, p: np.ndarray) -> np.ndarray:
        """P -> S = sqrt(delta J) P, the variable I solve the eigenproblem in."""
        return np.asarray(p, dtype=float) * np.sqrt(self.step * self.jacobian)

    def to_p(self, s: np.ndarray) -> np.ndarray:
        """S -> P, the physical radial function r R(r)."""
        return np.asarray(s, dtype=float) / np.sqrt(self.step * self.jacobian)

    def normalized(self, p: np.ndarray) -> np.ndarray:
        """P, which I scale so that integral P^2 dr = 1 in this mesh's own quadrature."""
        return np.asarray(p, dtype=float) / np.sqrt(self.integrate(np.asarray(p) ** 2))

    def hamiltonian_bands(
        self, v_local: np.ndarray, l: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """The diagonal and off-diagonal of my local Hamiltonian, in S."""
        if l < 0:
            raise ValueError(f"orbital quantum number l must be >= 0, got {l}")
        v = np.asarray(v_local, dtype=float)
        if v.shape != self.r.shape:
            raise ValueError("I need the potential sampled on this mesh")

        diag = self.kinetic_diag + l * (l + 1) * 0.5 / self.r**2 + v
        # P ~ r^(l+1) near the origin, so S ~ r^(l+3/2). Folding the ghost node
        # in is what makes my wall error quadratic in r_min instead of linear.
        diag[0] += self.inner_wall_coupling * self.inner_ghost_ratio ** (l + 1.5)
        return diag, self.kinetic_offdiag


def uniform_mesh(r_max: float, points: int) -> RadialMesh:
    """r = h, 2h, ... with h = r_max / (points + 1).

    This is the convention my radial_solver.solve_radial has always used:
    r[0] == h, so my Dirichlet wall lands exactly on r = 0 rather than one step
    inside it.
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
        inner_ghost_ratio=0.0,  # here my wall IS the origin, and P there is exactly 0
    )


def exponential_mesh(r_min: float, r_max: float, points: int) -> RadialMesh:
    """r = r_min e^(i delta), which I space geometrically from r_min to r_max."""
    if r_min <= 0.0:
        raise ValueError(
            f"I need an inner radius strictly above zero, got {r_min!r}; an "
            "exponential mesh cannot reach the origin, it only approaches it"
        )
    if r_max <= r_min:
        raise ValueError(f"box radius {r_max!r} must exceed inner radius {r_min!r}")
    if points < 3:
        raise ValueError(f"mesh needs at least 3 points, got {points}")

    delta = float(np.log(r_max / r_min) / (points - 1))
    r = r_min * np.exp(delta * np.arange(points))
    ghost = float(np.exp(-delta))

    # The 1/8 is not the centrifugal term. It is what P = sqrt(r) Q generates on
    # its own, and together with the l(l+1)/2 I add in hamiltonian_bands it
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
    """My exponential mesh at the inner radius that minimizes my total error.

    r_min = 1e-3 / Z is where my wall truncation and my eigensolver
    conditioning cross; see the module docstring for the balance I measured.
    Z sets both scales in the problem, which is why one constant covers every
    element for me.
    """
    if z < 1:
        raise ValueError(f"Z must be >= 1, got {z}")
    return exponential_mesh(_OPTIMAL_SCALED_INNER_RADIUS / z, r_max, points)


def mesh_for_atom_at_step(z: int, r_max: float, step: float) -> RadialMesh:
    """`mesh_for_atom`, sized by the step you want rather than a point count.

    I keep this so my callers do not have to know r_min. Going the other way,
    picking a point count that lands near a target step, means dividing the
    span log(r_max / r_min) by that step, which silently needs the same r_min I
    own here. A caller keeping its own copy of that constant gets no error when
    the two drift apart, just a mesh at the wrong step.

    I floor the point count, so the step I deliver is never finer than you
    asked for and overshoots by at most one point's worth. Halving `step`
    therefore always refines, which is what a refinement pair needs.
    """
    if z < 1:
        raise ValueError(f"Z must be >= 1, got {z}")
    if step <= 0.0:
        raise ValueError(f"step must be positive, got {step!r}")
    r_min = _OPTIMAL_SCALED_INNER_RADIUS / z
    if r_max <= r_min:
        raise ValueError(f"box radius {r_max!r} must exceed inner radius {r_min!r}")
    points = int(np.log(r_max / r_min) / step) + 1
    return exponential_mesh(r_min, r_max, points)


def display_window(
    r: NDArray[np.float64],
    density: NDArray[np.float64],
    floor: float = 1e-4,
    margin: float = 1.25,
) -> float:
    """The outer radius I think is worth drawing, for a curve I solved on a much
    larger box.

    I size a solve box so the eigenvalue is not squeezed by its own wall, and
    for an inner orbital that box is enormous compared with the orbital. I
    solve argon's 1s out to 160 bohr and it is finished by 0.4: resampled
    uniformly across the box for display, my 400-point output grid put exactly
    ONE sample on the orbital, and I drew a two-point straight line where a 1s
    should be. Neon's 1s got two points, argon's 3p four. My solver had 48000
    points and was right; only my picture was wrong.

    So I window the output grid and leave the solve box alone. Nothing here
    touches an energy: `r_max` in my solvers stays exactly where it was, which
    matters because argon's 1s energy is worth about 2 hartree to that box
    while the valence energies do not notice it at all. This decides which
    samples I draw, and drawing samples from a region where the function is a
    ten-thousandth of its peak is not information.

    My window is honest because my caller ships this radius as the field's own
    `grid`, so the axis is labelled with the range it actually covers. A window
    under a full-box label would be the dishonest version.

    I set `floor` deliberately well below the 1e-3 my frontend uses to trim a
    drawn curve: this one decides what data exists at all, so I keep a decade
    more tail than any display is going to want.
    """
    if r.size == 0:
        return 0.0
    peak = float(np.max(density)) if density.size else 0.0
    if not np.isfinite(peak) or peak <= 0.0:
        # I have nothing to window against, so my caller's box is as good a guess as any.
        return float(r[-1])
    above = np.nonzero(density > peak * floor)[0]
    if above.size == 0:
        return float(r[-1])
    outer = float(r[above[-1]]) * margin
    # I never go past the box, and never so tight that the peak sits on my frame.
    peak_r = float(r[int(np.argmax(density))])
    return float(min(max(outer, peak_r * 2.0), r[-1]))
