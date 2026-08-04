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
from dataclasses import dataclass

import numpy as np

from atomsim.atoms import Configuration, aufbau_configuration
from atomsim.hf_atom import hf_total_radial_density
from atomsim.provenance import Fidelity, Field, Provenance, Quantity
from atomsim.screened_atom import screened_total_radial_density

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


#: Shells by principal quantum number, in spectroscopic order.
SHELL_LABELS = ("K", "L", "M", "N", "O")


@dataclass(frozen=True)
class ShellPeak:
    """One shell, under both models, including a model that does not resolve it.

    `None` for a radius is a real answer and not missing data: it says this
    model's density has no local maximum for this shell, which is what GSZ does
    to sodium and magnesium. `depth` is the relative drop into the minimum
    before the peak, so a small number means the shell is barely separated from
    the one inside it; the innermost shell has no preceding minimum and so
    reports `None` for depth under both models.
    """

    label: str
    gsz_radius: float | None
    hf_radius: float | None
    gsz_depth: float | None
    hf_depth: float | None


@dataclass(frozen=True)
class DensityComparison:
    grid: np.ndarray
    gsz: Field
    hf: Field
    displaced_charge: Quantity
    shells: tuple[ShellPeak, ...]
    provenance: Provenance


#: Maxima below this fraction of the tallest are numerical noise, not shells.
#: Set from both ends of a measured gap that spans thirty-two orders of
#: magnitude. The faintest real shell in He..Ar is sodium's outermost
#: Hartree-Fock peak at 2.2e-2 of the tallest, and magnesium's is 5.3e-2. The
#: loudest noise is argon's Hartree-Fock tail beyond 40 bohr, which jitters at
#: about 1e-34 of the peak because the orbital amplitude out there has decayed
#: past what a float64 eigensolve can represent and starts changing sign; the
#: screened solver does the same thing past 11 bohr for neon with the
#: occupancy cap off, at 1e-60. This floor sits six orders below the faintest
#: shell and twenty-six above the loudest noise.
_NOISE_FLOOR = 1e-8


def _peaks_with_depth(
    grid: np.ndarray, values: np.ndarray
) -> list[tuple[float, float | None]]:
    """Interior maxima above the noise floor, each with the depth of the valley before it.

    The floor is as low as it can be while still doing its job, because a floor
    is also how a real shell gets dropped: sodium's outermost Hartree-Fock peak
    stands at 2 percent of the tallest one, and argon's box bug in Phase 28
    produced a spurious peak at nearly full height, so height alone sorts
    neither case correctly. What sorts them is the combination of three things:
    this floor, which only ever removes values the solve cannot represent; the
    depth of each valley, which is reported rather than thresholded on; and the
    shell count, which comes from the configuration rather than from either
    peak list.

    Minima are floored too, and by the same argument. A minimum found inside
    the noise would otherwise become the "valley" reported for a real peak
    above it, and the depth beside that peak would be a measurement of nothing.
    """
    floor = _NOISE_FLOOR * float(np.max(values))
    big = values > floor
    maxima = [
        i
        for i in range(1, len(values) - 1)
        if big[i] and values[i] > values[i - 1] and values[i] >= values[i + 1]
    ]
    minima = [
        i
        for i in range(1, len(values) - 1)
        if big[i] and values[i] < values[i - 1] and values[i] <= values[i + 1]
    ]
    out: list[tuple[float, float | None]] = []
    for i in maxima:
        before = [j for j in minima if j < i]
        if not before:
            out.append((float(grid[i]), None))
            continue
        floor = values[before[-1]]
        out.append((float(grid[i]), float((values[i] - floor) / values[i])))
    return out


def _shell_table(
    grid: np.ndarray, gsz: np.ndarray, hf: np.ndarray, config: Configuration
) -> tuple[ShellPeak, ...]:
    """Match each model's maxima to shells, inside out.

    The number of shells is the number of distinct principal quantum numbers in
    the configuration, never the length of either peak list, because that is
    the difference between "sodium has three shells and one model cannot see
    the third" and "sodium has two shells".

    Matching runs from the inside out, and the shorter list is padded at the
    OUTER end. That is not a convention, it is what physically happens: a
    valence shell that fails to separate merges into the tail, not into the
    core. Both cases in He..Ar are exactly this.

    More maxima than shells raises, because that is a density with a shell the
    atom does not have, which is what an unresolved core orbital looks like.
    It is a solver failure, and a table that quietly dropped the extra row
    would hide it.
    """
    n_shells = len({n for (n, _), _ in config})
    peaks = {"GSZ": _peaks_with_depth(grid, gsz), "HF": _peaks_with_depth(grid, hf)}
    for name, found in peaks.items():
        if len(found) > n_shells:
            radii = ", ".join(f"{r:.4g}" for r, _ in found)
            raise ValueError(
                f"the {name} density has {len(found)} maxima at r = {radii} bohr "
                f"but the configuration occupies only {n_shells} shells; that is "
                f"an unresolved orbital, not a shell"
            )
    rows = []
    for i in range(n_shells):
        g = peaks["GSZ"][i] if i < len(peaks["GSZ"]) else (None, None)
        h = peaks["HF"][i] if i < len(peaks["HF"]) else (None, None)
        rows.append(
            ShellPeak(
                label=SHELL_LABELS[i],
                gsz_radius=g[0], gsz_depth=g[1],
                hf_radius=h[0], hf_depth=h[1],
            )
        )
    return tuple(rows)


def compare_total_densities(
    z: int,
    n_electrons: int,
    *,
    config: Configuration | None = None,
    exchange: bool = True,
    pauli: bool = True,
    points: int = 800,
) -> DensityComparison:
    """Both models' D(r) on one grid, with the charge they place differently.

    `n_electrons` is separate from `z` to match the signature both density
    functions take, but must equal it: the Szydlik-Green (d, K) parameters are
    fitted to neutral atoms, and running GSZ at N != Z would compare
    Hartree-Fock against a model outside its own fit.

    With `pauli` off, both models take the same configuration, so the overlay
    answers one counterfactual question rather than two.
    """
    if n_electrons != z:
        raise ValueError(
            f"the GSZ screening parameters are fitted to neutral atoms, so this "
            f"comparison needs N = Z; got Z={z}, N={n_electrons}"
        )
    cfg = aufbau_configuration(n_electrons, pauli) if config is None else config
    hf = hf_total_radial_density(
        z, n_electrons, config=cfg, exchange=exchange, pauli=pauli, points=points
    )
    gsz = screened_total_radial_density(z, n_electrons, config=cfg, points=points)

    grid = _common_grid(gsz, hf, points)
    gsz_r = _resample(gsz, grid)
    hf_r = _resample(hf, grid)
    displaced = _displaced_charge(grid, gsz_r.values, hf_r.values)

    # Four terms, all measured: each model's own closure residual, plus the
    # charge each holds outside the window they share. None of them is assumed
    # negligible, and for argon they come to about a tenth of the number they
    # are the bar on, which is worth printing.
    bar = (
        (hf.provenance.error_estimate or 0.0)
        + (gsz.provenance.error_estimate or 0.0)
        + _window_loss(gsz, grid)
        + _window_loss(hf, grid)
    )
    fidelity = _weaker(gsz.provenance.fidelity, hf.provenance.fidelity)
    altered = [
        name
        for name, on in (("exchange", exchange), ("the occupancy cap", pauli))
        if not on
    ]
    method = (
        "half the L1 norm of D_HF - D_GSZ on the common log grid, in electrons"
    )
    if altered:
        method += f"; the Hartree-Fock side has {' and '.join(altered)} switched off"
    provenance = Provenance(
        fidelity=fidelity,
        method=method,
        assumptions=(
            "both densities resampled by linear interpolation onto a shared grid",
            "the window is the intersection of the two solver boxes; the charge "
            "outside it is measured and included in the error estimate",
            "Hartree-Fock is not truth: this is the distance between two "
            "approximations, never the error in one of them",
        ),
        error_estimate=bar,
        refinement=(
            "a correlated method (configuration interaction, coupled cluster) "
            "would give both models a reference to be measured against, rather "
            "than only against each other"
        ),
    )
    return DensityComparison(
        grid=grid,
        gsz=gsz_r,
        hf=hf_r,
        displaced_charge=Quantity(
            value=displaced,
            unit="electrons",
            label="charge the two models place differently",
            provenance=provenance,
        ),
        shells=_shell_table(grid, gsz_r.values, hf_r.values, cfg),
        provenance=provenance,
    )
