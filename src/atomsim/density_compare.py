"""The two many-electron densities on one axis, with a number on the gap.

`hf_atom` and `screened_atom` both return D(r) for the same atom, and they do
not agree. Each is an APPROXIMATION of the same observable, so the difference
between them is a statement about physics and not about convention, which is
what makes this comparison worth drawing at all and what makes the equivalent
comparison of two R(r) curves meaningless.

This is the only module that imports both models. It stays that way on purpose:
`hf_atom` mirrors the `screened_atom` API surface precisely so the two are
swappable, and that property survives exactly as long as neither imports the
other.

Neither curve is the reference. Hartree-Fock has no correlation and GSZ has
fitted parameters, so the number below is a disagreement between two
approximations and never an error in one of them. The provenance says so.
"""

import dataclasses

import numpy as np

from atomsim.provenance import Fidelity, Field

#: Increasing weakness. VISUAL_LIBERTY is deliberately absent: a density is not
#: a presentational choice, and a default that silently ranked one would hide
#: the day that stopped being true.
_WEAKNESS = {
    Fidelity.EXACT: 0,
    Fidelity.NUMERICAL: 1,
    Fidelity.APPROXIMATION: 2,
    Fidelity.COUNTERFACTUAL: 3,
}


def _weaker(a: Fidelity, b: Fidelity) -> Fidelity:
    """The weaker of two tiers, which is the strongest claim a comparison can make."""
    return a if _WEAKNESS[a] >= _WEAKNESS[b] else b


def _common_grid(a: Field, b: Field, points: int) -> np.ndarray:
    """The log grid on the intersection of the two solver boxes.

    The intersection rather than the union, because outside it one of the two
    curves would be an extrapolation past where its solver ran, and an
    extrapolated density drawn beside a computed one is exactly the quiet lie
    this project exists not to tell. What that costs is measured rather than
    assumed: see `_window_loss`, whose result goes into the error bar.
    """
    lo = max(a.grid[0], b.grid[0])
    hi = min(a.grid[-1], b.grid[-1])
    if not lo < hi:
        raise ValueError(
            f"the two solver boxes do not overlap: [{a.grid[0]:.3g}, "
            f"{a.grid[-1]:.3g}] and [{b.grid[0]:.3g}, {b.grid[-1]:.3g}]"
        )
    return np.geomspace(lo, hi, points)


def _resample(f: Field, grid: np.ndarray) -> Field:
    """`f` on `grid`, by linear interpolation, saying so in its own method string."""
    return dataclasses.replace(
        f,
        values=np.interp(grid, f.grid, f.values),
        grid=grid,
        provenance=dataclasses.replace(
            f.provenance,
            method=f.provenance.method
            + "; resampled by linear interpolation onto the common comparison grid",
        ),
    )


def _displaced_charge(grid: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Half the L1 norm, in electrons.

    Both densities integrate to N, so their signed difference integrates to
    zero and carries no information. Half the absolute difference is the whole
    story: the charge one model puts where the other does not, counted once
    rather than twice.
    """
    return 0.5 * float(np.trapezoid(np.abs(a - b), grid))


def _window_loss(f: Field, grid: np.ndarray) -> float:
    """Charge this model holds outside the common window, in electrons."""
    inside = np.trapezoid(np.interp(grid, f.grid, f.values), grid)
    return abs(float(np.trapezoid(f.values, f.grid) - inside))
