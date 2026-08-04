# Phase 29: The two densities, on one axis. Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Draw the GSZ and Hartree-Fock total radial densities on one axis in the
Radial view, with the disagreement quantified in electrons and each shell's peak
radius under both models.

**Architecture:** A new engine module `density_compare.py` is the first thing in
the codebase that imports both `hf_atom` and `screened_atom`; neither learns
about the other, which is what keeps them swappable. It resamples both densities
onto the common log grid where both solvers are defined, and returns the
displaced charge (half the L1 norm, which is exactly the number of electrons the
two models put in different places) plus a shell-by-shell peak table.
`/api/radial` gains a `compare` flag returning that structure; the Radial view
gains a toggle, a dashed overlay curve, the readout and the table.

**Tech Stack:** Python 3.12 (numpy, FastAPI, Pydantic v2, pytest), React +
TypeScript + Zustand + d3-scale + vitest.

**Spec:** `docs/superpowers/specs/2026-08-04-phase29-density-comparison-design.md`

## Global Constraints

- Every physical value crossing a module boundary is a `Quantity`, a `Field`, or
  a container carrying its own `Provenance`. A bare `float` crossing a boundary
  is a bug.
- Engine-internal math in Hartree atomic units. Any SI or display conversion
  happens at the server boundary and appends to the provenance `method`.
- Tiers are inherited from the solve, never asserted as a literal below the
  point where the solve happens.
- Line length 100. `ruff check .` clean. E741 is ignored project-wide (`l` is
  the angular momentum quantum number).
- **No em dashes anywhere**, in code, comments, captions, commit messages or
  docs.
- New physics gets a validation test (analytic ground truth, KS test, or
  grid-convergence), not a smoke test.
- Commit per logical change, straight to `main`. No AI attribution in commit
  messages, PRs, or anything pushed to GitHub.
- Leave the tree committed and push-ready at the end of every task.
- Run Python as `MKL_THREADING_LAYER=SEQUENTIAL "C:/Users/yashg/.conda/envs/atomsim/python.exe" -m pytest`
  if `conda` is not on PATH in your shell.
- Rebuild the frontend (`cd web && npm run build`) after any change under
  `web/src`, because `atomsim serve` mounts `web/dist`.

## File Structure

| File | Responsibility |
|---|---|
| `src/atomsim/density_compare.py` | **new.** Common grid, resampling, displaced charge, peak table, provenance. The only module importing both many-electron models. |
| `tests/test_density_compare.py` | **new.** Engine validation: analytic displaced charge, symmetry, window loss, the Na/Mg peak case. |
| `src/atomsim/server/schemas.py` | `ShellPeakModel`, `DensityComparisonModel`, `density_comparison` on `RadialResponse`. |
| `src/atomsim/server/app.py` | `_many_electron_target` split out of `_hf_view_target`; `compare` parameter on `/api/radial`. |
| `tests/test_server_density_compare.py` | **new.** API shape and both refusals. |
| `web/src/api/types.ts` | `ShellPeak`, `DensityComparison`, `density_comparison` on `RadialResponse`. |
| `web/src/api/client.ts` | `compare` query parameter on `getRadial`. |
| `web/src/lib/hfModel.ts` | `compareAvailable`, `resolveCompare` beside the existing `gszAvailable` / `resolveModel`. |
| `web/src/state/store.ts` | `compare` flag, `setCompare`, threading through `loadRadial`. |
| `web/src/lib/urlState.ts` | `compare=1` deep link. |
| `web/src/components/Controls.tsx` | The toggle and its refusal hint. |
| `web/src/components/RadialView.tsx` | Overlay curve, legend, readout, shell table, captions. Exports the pure formatters the tests target. |
| `web/src/components/RadialView.test.ts` | **new.** Tests those formatters. |

---

### Task 1: The comparison arithmetic

Common grid, resampling, displaced charge, and the tier rule. No shells yet, no
engine calls yet: this task is pure array work with a hand-computable answer, so
it can be checked without a solve.

**Files:**
- Create: `src/atomsim/density_compare.py`
- Test: `tests/test_density_compare.py`

**Interfaces:**
- Consumes: `atomsim.provenance` (`Field`, `Quantity`, `Provenance`, `Fidelity`).
- Produces:
  - `_weaker(a: Fidelity, b: Fidelity) -> Fidelity`
  - `_common_grid(a: Field, b: Field, points: int) -> np.ndarray`
  - `_resample(f: Field, grid: np.ndarray) -> Field`
  - `_displaced_charge(grid: np.ndarray, a: np.ndarray, b: np.ndarray) -> float`
  - `_window_loss(f: Field, grid: np.ndarray) -> float`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_density_compare.py`:

```python
"""Comparing the two many-electron densities, and putting a number on the gap.

Both models claim to approximate the same observable, so the difference between
them is a statement about physics rather than about convention. The number this
module returns is half the L1 norm, which is exactly the count of electrons the
two models place differently, and the tests below check it against a case that
can be integrated by hand before they check it against either solver.
"""

import numpy as np
import pytest

from atomsim.density_compare import (
    _common_grid,
    _displaced_charge,
    _resample,
    _weaker,
    _window_loss,
)
from atomsim.provenance import Fidelity, Field, Provenance


def _field(values, grid, fidelity=Fidelity.APPROXIMATION, error=None):
    return Field(
        values=np.asarray(values, dtype=float),
        grid=np.asarray(grid, dtype=float),
        unit="electrons/bohr",
        grid_unit="bohr",
        label="D(r)",
        provenance=Provenance(
            fidelity=fidelity, method="test fixture", error_estimate=error
        ),
    )


# --- the number, against an integral that can be done by hand ---------------


def test_displaced_charge_matches_a_hand_integral():
    """D_a = 2r and D_b = 2 - 2r on [0, 1], both integrating to one electron.

    |D_a - D_b| = |4r - 2|, whose integral over [0, 1] is 1, so half of it is
    0.5. The integrand is piecewise linear with its kink at r = 0.5, and the
    grid below puts a node there, so the trapezoid rule is exact and the
    assertion can be tight rather than approximate.
    """
    r = np.linspace(0.0, 1.0, 101)
    assert _displaced_charge(r, 2 * r, 2 - 2 * r) == pytest.approx(0.5, abs=1e-12)


def test_a_density_is_not_displaced_from_itself():
    r = np.linspace(0.0, 1.0, 101)
    assert _displaced_charge(r, 2 * r, 2 * r) == 0.0


def test_displaced_charge_is_symmetric():
    """Neither model is the reference, so the number cannot depend on the order."""
    r = np.linspace(0.0, 1.0, 101)
    a, b = 2 * r, 2 - 2 * r
    assert _displaced_charge(r, a, b) == _displaced_charge(r, b, a)


# --- the common window ------------------------------------------------------


def test_the_common_grid_is_the_intersection_of_the_two_boxes():
    """Neither model is extrapolated past where its own solver ran."""
    a = _field(np.ones(50), np.geomspace(1e-4, 60.0, 50))
    b = _field(np.ones(50), np.geomspace(1e-3, 64.0, 50))
    grid = _common_grid(a, b, 200)
    assert grid[0] == pytest.approx(1e-3)
    assert grid[-1] == pytest.approx(60.0)
    assert len(grid) == 200


def test_window_loss_is_the_charge_left_outside():
    """A flat density on [0, 2] restricted to [0, 1] loses exactly half of it."""
    f = _field(np.ones(201), np.linspace(0.0, 2.0, 201))
    assert _window_loss(f, np.linspace(0.0, 1.0, 101)) == pytest.approx(1.0, abs=1e-12)


# --- the tier rule ----------------------------------------------------------


def test_the_weaker_tier_wins():
    assert _weaker(Fidelity.APPROXIMATION, Fidelity.COUNTERFACTUAL) is Fidelity.COUNTERFACTUAL
    assert _weaker(Fidelity.COUNTERFACTUAL, Fidelity.APPROXIMATION) is Fidelity.COUNTERFACTUAL
    assert _weaker(Fidelity.APPROXIMATION, Fidelity.APPROXIMATION) is Fidelity.APPROXIMATION
    assert _weaker(Fidelity.EXACT, Fidelity.NUMERICAL) is Fidelity.NUMERICAL


def test_a_visual_liberty_has_no_place_in_this_comparison():
    """Raising beats defaulting: a density is never a presentational choice.

    If one ever becomes one, this should stop rather than quietly rank it.
    """
    with pytest.raises(KeyError):
        _weaker(Fidelity.VISUAL_LIBERTY, Fidelity.APPROXIMATION)


# --- resampling keeps the provenance, and says what it did ------------------


def test_resampling_carries_the_provenance_and_discloses_itself():
    f = _field(np.linspace(1.0, 2.0, 50), np.geomspace(1e-3, 10.0, 50))
    out = _resample(f, np.geomspace(1e-2, 5.0, 30))
    assert out.provenance.fidelity is f.provenance.fidelity
    assert "resampled" in out.provenance.method
    assert len(out.values) == 30
    assert out.unit == f.unit
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `MKL_THREADING_LAYER=SEQUENTIAL "C:/Users/yashg/.conda/envs/atomsim/python.exe" -m pytest tests/test_density_compare.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'atomsim.density_compare'`

- [ ] **Step 3: Write the module**

Create `src/atomsim/density_compare.py`:

```python
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

from atomsim.provenance import Fidelity, Field, Provenance

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `MKL_THREADING_LAYER=SEQUENTIAL "C:/Users/yashg/.conda/envs/atomsim/python.exe" -m pytest tests/test_density_compare.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Lint**

Run: `"C:/Users/yashg/.conda/envs/atomsim/python.exe" -m ruff check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/atomsim/density_compare.py tests/test_density_compare.py
git commit -m "Put a number on the gap between the two densities

Half the L1 norm on the window where both solvers actually ran, which is the
count of electrons the two models place differently."
```

---

### Task 2: Shells, including the ones a model does not resolve

The peak table, and the assembly function that calls both solvers. This is where
the measured finding lands: GSZ has no third maximum for sodium or magnesium at
all, and Hartree-Fock's third one there is a 0.3% dimple.

**Files:**
- Modify: `src/atomsim/density_compare.py`
- Modify: `tests/test_density_compare.py`

**Interfaces:**
- Consumes: everything from Task 1; `hf_atom.hf_total_radial_density`,
  `screened_atom.screened_total_radial_density`, `atoms.aufbau_configuration`,
  `atoms.Configuration`.
- Produces:
  - `SHELL_LABELS: tuple[str, ...]`
  - `ShellPeak` (frozen dataclass: `label: str`, `gsz_radius: float | None`,
    `hf_radius: float | None`, `gsz_depth: float | None`, `hf_depth: float | None`)
  - `DensityComparison` (frozen dataclass: `grid: np.ndarray`, `gsz: Field`,
    `hf: Field`, `displaced_charge: Quantity`, `shells: tuple[ShellPeak, ...]`,
    `provenance: Provenance`)
  - `_peaks_with_depth(grid, values) -> list[tuple[float, float | None]]`
  - `compare_total_densities(z, n_electrons, *, config=None, exchange=True, pauli=True, points=800) -> DensityComparison`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_density_compare.py`:

```python
from atomsim.atoms import aufbau_configuration  # noqa: E402
from atomsim.density_compare import (  # noqa: E402
    _peaks_with_depth,
    compare_total_densities,
)


# --- peaks, and how well separated they are ---------------------------------


def test_the_innermost_peak_has_no_separation_to_report():
    """Depth measures the minimum before a peak, and the first peak has none."""
    r = np.linspace(0.1, 10.0, 500)
    v = np.exp(-((r - 1.0) ** 2) / 0.02)
    peaks = _peaks_with_depth(r, v)
    assert len(peaks) == 1
    assert peaks[0][0] == pytest.approx(1.0, abs=0.05)
    assert peaks[0][1] is None


def test_depth_is_the_relative_drop_into_the_preceding_minimum():
    """Two Gaussians whose valley bottoms at half the outer peak: depth 0.5."""
    r = np.linspace(0.0, 10.0, 2001)
    v = np.exp(-((r - 2.0) ** 2) / 0.08) + np.exp(-((r - 6.0) ** 2) / 0.08)
    peaks = _peaks_with_depth(r, v)
    assert len(peaks) == 2
    assert peaks[1][1] == pytest.approx(1.0, abs=0.01)  # the valley reaches zero


# --- the two atoms where the models disagree about how many shells there are -


def test_gsz_does_not_resolve_sodiums_third_shell_and_hartree_fock_barely_does():
    """The finding this whole view exists to show, pinned in both directions.

    Sodium has three shells. Under GSZ the density falls monotonically past the
    L peak and the 3s charge rides out on the tail as a shoulder, so there is
    no third maximum and no minimum after the second. Under Hartree-Fock there
    is a third maximum, but the dip before it is 0.3 percent deep, which is a
    shell you would miss if the table did not print the depth beside it.

    Both halves are asserted. A solver change that flattens the HF dimple would
    otherwise silently drop a shell from a table nobody would think to re-check.
    """
    c = compare_total_densities(11, 11)
    assert [s.label for s in c.shells] == ["K", "L", "M"]
    k, ell, m = c.shells
    assert k.gsz_radius is not None and k.hf_radius is not None
    assert ell.gsz_radius is not None and ell.hf_radius is not None
    assert m.gsz_radius is None, "GSZ is not expected to resolve sodium's M shell"
    assert m.hf_radius == pytest.approx(3.16, rel=0.05)
    assert m.hf_depth == pytest.approx(0.003, abs=0.002)


def test_magnesium_is_the_same_case_with_a_deeper_dimple():
    c = compare_total_densities(12, 12)
    m = c.shells[2]
    assert m.label == "M"
    assert m.gsz_radius is None
    assert m.hf_radius == pytest.approx(2.43, rel=0.05)
    assert m.hf_depth == pytest.approx(0.015, abs=0.005)


@pytest.mark.parametrize("z", [13, 14, 18])
def test_three_clean_shells_under_both_models(z):
    """Aluminium up: both models resolve all three, so no cell is empty."""
    c = compare_total_densities(z, z)
    assert len(c.shells) == 3
    for s in c.shells:
        assert s.gsz_radius is not None
        assert s.hf_radius is not None


def test_the_shell_count_comes_from_the_configuration_not_from_the_peaks():
    """Otherwise the table would report that sodium has two shells."""
    c = compare_total_densities(11, 11)
    n_shells = len({n for (n, _), _ in aufbau_configuration(11)})
    assert len(c.shells) == n_shells == 3


# --- the number, on real atoms ----------------------------------------------


@pytest.mark.parametrize(
    "z,expected",
    [(2, 0.0003), (10, 0.0232), (11, 0.1218), (18, 0.0600)],
)
def test_the_measured_disagreement(z, expected):
    """The figures the captions quote, pinned so the captions stay checkable.

    Measured at 800 display points before this module existed. The tolerance is
    the reported bar, not a round number: a change that moves these past their
    own error estimate is a change in the physics, and should fail here.
    """
    c = compare_total_densities(z, z)
    assert c.displaced_charge.value == pytest.approx(
        expected, abs=max(c.displaced_charge.provenance.error_estimate, 2e-3)
    )


@pytest.mark.parametrize("z", [2, 6, 10, 11, 18])
def test_the_window_costs_less_than_the_bar_it_is_folded_into(z):
    """The intersection is honest only while what it drops is smaller than the noise."""
    c = compare_total_densities(z, z)
    loss = _window_loss(c.gsz, c.grid) + _window_loss(c.hf, c.grid)
    assert loss < c.displaced_charge.provenance.error_estimate
    assert loss < 5e-3


def test_the_models_agree_far_better_than_their_energies_do():
    """The lesson of the view, as an assertion rather than a caption.

    GSZ was fitted to reproduce Hartree-Fock potentials, so a close density is
    what the fit bought. Its valence ionization energies are 2 to 24 percent
    off NIST; its density is inside 1.5 percent of HF's for every atom here.
    """
    for z in (2, 3, 6, 10, 11, 14, 18):
        c = compare_total_densities(z, z)
        assert c.displaced_charge.value / z < 0.015


# --- provenance -------------------------------------------------------------


def test_the_comparison_is_an_approximation_and_says_neither_is_truth():
    c = compare_total_densities(10, 10)
    assert c.provenance.fidelity is Fidelity.APPROXIMATION
    assert c.displaced_charge.unit == "electrons"
    assert any("not truth" in a for a in c.provenance.assumptions)


def test_a_thrown_switch_makes_the_comparison_counterfactual():
    """Hartree-Fock without exchange is altered physics; the overlay inherits it."""
    c = compare_total_densities(10, 10, exchange=False)
    assert c.provenance.fidelity is Fidelity.COUNTERFACTUAL
    assert c.displaced_charge.provenance.fidelity is Fidelity.COUNTERFACTUAL


def test_both_models_take_the_same_configuration_with_pauli_off():
    """One counterfactual question, not two.

    With the cap off the Hartree-Fock configuration is 1s^N, and the screened
    side is handed that same configuration rather than resolving its own, so
    the overlay compares two models of the same altered atom.
    """
    c = compare_total_densities(10, 10, exchange=False, pauli=False)
    assert c.provenance.fidelity is Fidelity.COUNTERFACTUAL
    # One shell, because there is one occupied n.
    assert [s.label for s in c.shells] == ["K"]


def test_an_ion_is_refused_because_the_parameters_are_fitted_to_neutrals():
    with pytest.raises(ValueError, match="neutral"):
        compare_total_densities(10, 9)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `MKL_THREADING_LAYER=SEQUENTIAL "C:/Users/yashg/.conda/envs/atomsim/python.exe" -m pytest tests/test_density_compare.py -v`
Expected: FAIL, `ImportError: cannot import name '_peaks_with_depth'`

- [ ] **Step 3: Write the implementation**

Add to the imports at the top of `src/atomsim/density_compare.py`:

```python
from dataclasses import dataclass

from atomsim.atoms import Configuration, aufbau_configuration
from atomsim.hf_atom import hf_total_radial_density
from atomsim.provenance import Fidelity, Field, Provenance, Quantity
from atomsim.screened_atom import screened_total_radial_density
```

Append to the module:

```python
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

    The floor is on the maxima only. A deep valley is what "well separated"
    means, so flooring the minima would discard the measurement the depth
    number exists to make, and would discard it hardest in the clearest cases.
    It would also protect against nothing: the noise lives beyond every real
    peak, and the valley reported for a peak is the last minimum BEFORE it, so
    a minimum out in the tail is never the one selected.
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
        if values[i] < values[i - 1] and values[i] <= values[i + 1]
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `MKL_THREADING_LAYER=SEQUENTIAL "C:/Users/yashg/.conda/envs/atomsim/python.exe" -m pytest tests/test_density_compare.py -v`
Expected: PASS, all tests

If `test_the_measured_disagreement` fails, do NOT widen the tolerance. The
figures came from a measurement, and a disagreement means either the resampling
or the solve changed. Find which before touching the number, and if the number
is genuinely different now, update the spec's table in the same commit.

- [ ] **Step 5: Run the neighbouring density suites for regressions**

Run: `MKL_THREADING_LAYER=SEQUENTIAL "C:/Users/yashg/.conda/envs/atomsim/python.exe" -m pytest tests/test_screened_total_density.py tests/test_hf_atom.py -q`
Expected: PASS. This module only reads from those two, so a failure here means
something was changed that should not have been.

- [ ] **Step 6: Lint and commit**

```bash
"C:/Users/yashg/.conda/envs/atomsim/python.exe" -m ruff check .
git add src/atomsim/density_compare.py tests/test_density_compare.py
git commit -m "Count the shells from the configuration, not from the peaks

GSZ finds no third maximum for sodium or magnesium, and Hartree-Fock's is a
0.3 percent dimple. A table built from either peak list would report that
sodium has two shells."
```

---

### Task 3: Split the shared refusals out of the Hartree-Fock resolver

A pure refactor with no behaviour change, done on its own so the next task's
diff is only the new feature. `_hf_view_target` currently bundles two unrelated
refusals: what atom and configuration the solve is about, which the density
needs, and whether (n, l) is occupied, which only the orbital plots need.

**Files:**
- Modify: `src/atomsim/server/app.py:860-905`

**Interfaces:**
- Produces: `_many_electron_target(system: str, config: str | None, pauli: bool) -> tuple[int, int, Configuration]`,
  a closure inside `create_app` beside `_hf_view_target`.

- [ ] **Step 1: Confirm the existing suite is green before touching it**

Run: `MKL_THREADING_LAYER=SEQUENTIAL "C:/Users/yashg/.conda/envs/atomsim/python.exe" -m pytest tests/test_server_hf.py tests/test_server_hf_views.py tests/test_hf_views.py -q`
Expected: PASS. Record the count; it must be identical after the refactor.

- [ ] **Step 2: Replace `_hf_view_target` with the split pair**

In `src/atomsim/server/app.py`, replace the body of `_hf_view_target` with:

```python
    def _many_electron_target(
        system: str, config: str | None, pauli: bool
    ) -> tuple[int, int, Configuration]:
        """(Z, N, configuration) for any many-electron picture, or a refusal.

        The half of the old `_hf_view_target` that is about which atom is being
        solved rather than which orbital is being drawn. Split out because the
        total density needs the atom and does not depend on (n, l) at all, so
        refusing a density comparison over an unoccupied subshell would refuse
        a legitimate request about an orbital nobody asked for.

        Every refusal here is synchronous and carries its reason, because the
        alternative is a job that dies several seconds in with an engine
        message the client has to guess at.
        """
        if not _is_screened(system):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"this model needs an atom with a known electron count, "
                    f"and {system!r} is a one-electron system; its "
                    f"wavefunction is already exact in the other views, so "
                    f"there is nothing a self-consistent field would add"
                ),
            )
        element = atom_for_key(system)
        n_electrons = element.z
        cfg = (
            aufbau_configuration(n_electrons, pauli)
            if config is None
            else _parse_config_or_422(config, pauli)
        )
        _validate_hf_request(element.z, n_electrons, cfg, pauli)
        return element.z, n_electrons, cfg

    def _hf_view_target(req) -> tuple[int, int, Configuration]:
        """The atom, plus the occupancy check that only an orbital picture needs.

        The occupancy check needs no solve: which subshells exist is a property
        of the configuration, and the configuration is in the request.
        """
        z, n_electrons, config = _many_electron_target(
            req.system, req.config, req.pauli
        )
        if (req.n, req.l) not in [nl for nl, _ in config]:
            held = ", ".join(f"{n}{SUBSHELL_LABELS[l]}" for (n, l), _ in config)
            why = (
                "the occupancy cap is lifted, so every electron is in the 1s "
                "and no other orbital exists to draw"
                if not req.pauli
                else "Hartree-Fock builds one Fock operator per occupied "
                "subshell, so an empty one has no operator to be an "
                "eigenfunction of, and borrowing another subshell's operator "
                "would answer a different question silently"
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{req.n}{SUBSHELL_LABELS[req.l]} is not occupied in "
                    f"{atom_for_key(req.system).symbol} ({held}); {why}"
                ),
            )
        return z, n_electrons, config
```

Note the one intentional wording change: the hydrogenic refusal said
`model='hf'` and now says `this model`, because two callers reach it. If a test
asserts on that exact string, update the assertion in this commit.

- [ ] **Step 3: Run the same three suites and compare the count**

Run: `MKL_THREADING_LAYER=SEQUENTIAL "C:/Users/yashg/.conda/envs/atomsim/python.exe" -m pytest tests/test_server_hf.py tests/test_server_hf_views.py tests/test_hf_views.py -q`
Expected: PASS, the same number of tests as Step 1.

- [ ] **Step 4: Run the whole suite, because this touches a shared resolver**

Run: `MKL_THREADING_LAYER=SEQUENTIAL "C:/Users/yashg/.conda/envs/atomsim/python.exe" -m pytest -q`
Expected: PASS, 1235 tests (the count before this phase started).

- [ ] **Step 5: Lint and commit**

```bash
"C:/Users/yashg/.conda/envs/atomsim/python.exe" -m ruff check .
git add src/atomsim/server/app.py
git commit -m "Separate which atom is being solved from which orbital is drawn

A total density does not depend on (n, l), so the occupancy refusal cannot sit
in the resolver a density has to call."
```

---

### Task 4: The endpoint

**Files:**
- Modify: `src/atomsim/server/schemas.py`
- Modify: `src/atomsim/server/app.py:1224-1318`
- Test: `tests/test_server_density_compare.py`

**Interfaces:**
- Consumes: `compare_total_densities`, `DensityComparison`, `ShellPeak` from Task 2;
  `_many_electron_target`, `_gsz_element` from Task 3.
- Produces: `ShellPeakModel`, `DensityComparisonModel`,
  `RadialResponse.density_comparison: DensityComparisonModel | None`,
  and `GET /api/radial/{n}/{l}?compare=true`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_server_density_compare.py`:

```python
"""The comparison over HTTP, and the two ways it is refused."""

import pytest
from fastapi.testclient import TestClient

from atomsim.server.app import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_compare_is_off_by_default(client):
    r = client.get("/api/radial/2/1?system=ne")
    assert r.status_code == 200
    assert r.json()["density_comparison"] is None


def test_compare_returns_both_curves_on_one_grid(client):
    r = client.get("/api/radial/2/1?system=ne&compare=true")
    assert r.status_code == 200
    c = r.json()["density_comparison"]
    assert c["gsz"]["grid"] == c["hf"]["grid"]
    assert len(c["gsz"]["values"]) == len(c["hf"]["values"])
    assert c["displaced_charge"]["unit"] == "electrons"
    assert c["displaced_charge"]["provenance"]["error_estimate"] > 0


def test_the_shell_table_names_the_model_that_resolves_no_peak(client):
    r = client.get("/api/radial/3/0?system=na&compare=true")
    assert r.status_code == 200
    shells = r.json()["density_comparison"]["shells"]
    assert [s["label"] for s in shells] == ["K", "L", "M"]
    assert shells[2]["gsz_radius"] is None
    assert shells[2]["hf_radius"] is not None


def test_compare_works_from_either_model(client):
    """The overlay is symmetric, so the radio the user left it on cannot matter."""
    gsz = client.get("/api/radial/2/1?system=ne&compare=true").json()
    hf = client.get("/api/radial/2/1?system=ne&model=hf&compare=true").json()
    assert gsz["density_comparison"]["displaced_charge"]["value"] == pytest.approx(
        hf["density_comparison"]["displaced_charge"]["value"]
    )


def test_compare_does_not_need_the_drawn_orbital_to_be_occupied(client):
    """3d is empty in neon, and a density does not care.

    The orbital plots are refused on their own terms under model=hf; this asks
    for the screened orbital, which exists, plus the comparison, which is about
    the atom rather than about (n, l).
    """
    r = client.get("/api/radial/3/2?system=ne&compare=true")
    assert r.status_code == 200
    assert r.json()["density_comparison"] is not None


# --- the two refusals -------------------------------------------------------


def test_sulfur_is_refused_by_name(client):
    r = client.get("/api/radial/3/1?system=s&model=hf&compare=true")
    assert r.status_code == 400
    detail = r.json()["detail"].lower()
    assert "green" in detail


def test_a_one_electron_system_is_refused(client):
    r = client.get("/api/radial/2/1?system=h&compare=true")
    assert r.status_code == 422
    assert "one-electron" in r.json()["detail"]


def test_a_thrown_switch_reaches_the_comparison_badge(client):
    r = client.get("/api/radial/1/0?system=ne&model=hf&exchange=false&compare=true")
    assert r.status_code == 200
    c = r.json()["density_comparison"]
    assert c["provenance"]["fidelity"] == "counterfactual"
    assert "exchange" in c["provenance"]["method"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `MKL_THREADING_LAYER=SEQUENTIAL "C:/Users/yashg/.conda/envs/atomsim/python.exe" -m pytest tests/test_server_density_compare.py -v`
Expected: FAIL, `KeyError: 'density_comparison'`

- [ ] **Step 3: Add the response models**

In `src/atomsim/server/schemas.py`, after `FieldModel` (around line 159), add:

```python
class ShellPeakModel(BaseModel):
    """One shell under both models. A null radius means this model resolves none."""

    label: str
    gsz_radius: float | None
    hf_radius: float | None
    gsz_depth: float | None
    hf_depth: float | None


class DensityComparisonModel(BaseModel):
    gsz: FieldModel
    hf: FieldModel
    displaced_charge: QuantityModel
    shells: list[ShellPeakModel]
    provenance: ProvenanceModel

    @classmethod
    def from_comparison(cls, c: DensityComparison) -> "DensityComparisonModel":
        return cls(
            gsz=FieldModel.from_field(c.gsz),
            hf=FieldModel.from_field(c.hf),
            displaced_charge=QuantityModel.from_quantity(c.displaced_charge),
            shells=[
                ShellPeakModel(
                    label=s.label,
                    gsz_radius=s.gsz_radius, hf_radius=s.hf_radius,
                    gsz_depth=s.gsz_depth, hf_depth=s.hf_depth,
                )
                for s in c.shells
            ],
            provenance=ProvenanceModel.from_provenance(c.provenance),
        )
```

Add `from atomsim.density_compare import DensityComparison` to the imports, and
`density_comparison: DensityComparisonModel | None = None` to `RadialResponse`.

- [ ] **Step 4: Add the endpoint parameter**

In `src/atomsim/server/app.py`, add `compare: bool = False` to the `radial`
signature (after `pauli`), and this helper immediately before the `if model ==
"hf":` branch:

```python
        def _comparison() -> DensityComparisonModel | None:
            """Both densities, or nothing, or a refusal with the reason.

            Calls the two resolvers that already know when a model cannot
            speak, so the wording and the status codes cannot drift from the
            ones the model radio shows: 400 from `_gsz_element` for sulfur and
            chlorine, 422 from `_many_electron_target` for a one-electron
            system. Nothing new is refused here.

            The configuration comes from the Hartree-Fock side and is handed to
            both, so that with the occupancy cap off the overlay compares two
            models of the same altered atom rather than two different atoms.

            Order matters. `_many_electron_target` runs first because it is the
            one that refuses a one-electron system, and `_gsz_element` expects
            an atom key: asked about hydrogen first it would fail looking the
            element up rather than returning the 422 that says why.
            """
            if not compare:
                return None
            z, n_electrons, cfg = _many_electron_target(system, config, pauli)
            _gsz_element(system)
            return DensityComparisonModel.from_comparison(
                compare_total_densities(
                    z, n_electrons, config=cfg, exchange=exchange, pauli=pauli,
                )
            )
```

Then pass `density_comparison=_comparison()` on all three `RadialResponse(...)`
returns in that function. On the hydrogenic branch it is still called, because
that is where the 422 for a one-electron system has to come from.

Add to the imports at the top of `app.py`:

```python
from atomsim.density_compare import compare_total_densities
```

and add `DensityComparisonModel` to the existing `from atomsim.server.schemas import (...)` block.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `MKL_THREADING_LAYER=SEQUENTIAL "C:/Users/yashg/.conda/envs/atomsim/python.exe" -m pytest tests/test_server_density_compare.py -v`
Expected: PASS, 8 tests

- [ ] **Step 6: Run the full Python suite**

Run: `MKL_THREADING_LAYER=SEQUENTIAL "C:/Users/yashg/.conda/envs/atomsim/python.exe" -m pytest -q`
Expected: PASS

- [ ] **Step 7: Lint and commit**

```bash
"C:/Users/yashg/.conda/envs/atomsim/python.exe" -m ruff check .
git add src/atomsim/server/schemas.py src/atomsim/server/app.py tests/test_server_density_compare.py
git commit -m "Serve both densities from one radial request

The refusals are the two the model radio already shows, called rather than
restated, so their wording cannot drift."
```

---

### Task 5: The client, the store, and the deep link

**Files:**
- Modify: `web/src/api/types.ts`
- Modify: `web/src/api/client.ts:116-134`
- Modify: `web/src/lib/hfModel.ts`
- Modify: `web/src/state/store.ts`
- Modify: `web/src/lib/urlState.ts`
- Test: `web/src/lib/hfModel.test.ts`, `web/src/lib/urlState.test.ts`

**Interfaces:**
- Produces: `compareAvailable(systems, system) -> boolean`,
  `resolveCompare(systems, system, compare) -> boolean`, `AppState.compare`,
  `AppState.setCompare`, `UrlState.compare`.

- [ ] **Step 1: Write the failing tests**

Append to `web/src/lib/hfModel.test.ts`:

The fixture in that file is called `TABLE` and already holds argon, sulfur and
hydrogen, which is exactly the three cases needed. Add `compareAvailable` and
`resolveCompare` to its import from `./hfModel`.

```typescript
describe("compareAvailable", () => {
  it("is true for an atom both models can draw", () => {
    expect(compareAvailable(TABLE, "ar")).toBe(true);
  });

  it("is false where GSZ has no parameters, since there is nothing to compare", () => {
    expect(compareAvailable(TABLE, "s")).toBe(false);
  });

  it("is false for a one-electron system, which has no total density at all", () => {
    expect(compareAvailable(TABLE, "h")).toBe(false);
  });

  it("is false while the system table is still loading", () => {
    // The opposite default from gszAvailable, and deliberately so: greying a
    // control a moment late is cheap, but firing a request that 422s on a deep
    // link before the table lands is the bug Phase 28 found.
    expect(compareAvailable([], "ar")).toBe(false);
  });
});

describe("resolveCompare", () => {
  it("turns a deep-linked compare off where it cannot run", () => {
    expect(resolveCompare(TABLE, "s", true)).toBe(false);
  });

  it("leaves it alone where it can", () => {
    expect(resolveCompare(TABLE, "ar", true)).toBe(true);
  });
});
```

Append to `web/src/lib/urlState.test.ts`. Note the API in this file:
`serializeAppUrl(state)` returns a **string**, and `parseAppUrl(search)` returns
a `Partial<UrlState>` to be spread over `URL_DEFAULTS`. Match the `dirac` block
at the end of the file:

```typescript
describe("density comparison url state", () => {
  it("round-trips the compare toggle", () => {
    const q = serializeAppUrl({ ...URL_DEFAULTS, system: "na", compare: true });
    expect(q).toContain("compare=1");
    const back = { ...URL_DEFAULTS, ...parseAppUrl(q) };
    expect(back.compare).toBe(true);
  });

  it("omits compare when off", () => {
    expect(serializeAppUrl({ ...URL_DEFAULTS, compare: false })).not.toContain("compare");
  });
});
```

Append to `web/src/state/store.test.ts`, which resets the store in a
`beforeEach` and has a `pretendLoaded()` helper that fills every derived payload:

```typescript
describe("setCompare", () => {
  it("clears the radial payload, which is the one missing a field", () => {
    pretendLoaded();
    useAppStore.getState().setCompare(true);
    expect(useAppStore.getState().radial).toBeNull();
  });

  it("leaves the cloud, the plane and the levels alone", () => {
    // The atom did not change, so seconds of solve should not be thrown away
    // to add one curve to a different view.
    pretendLoaded();
    useAppStore.getState().setCompare(true);
    expect(useAppStore.getState().positions).not.toBeNull();
    expect(useAppStore.getState().plane).not.toBeNull();
    expect(useAppStore.getState().levels).not.toBeNull();
  });
});
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd web && npx vitest run src/lib/hfModel.test.ts src/lib/urlState.test.ts`
Expected: FAIL, `compareAvailable is not exported`

- [ ] **Step 3: Add the types**

In `web/src/api/types.ts`, add:

```typescript
export interface ShellPeak {
  label: string;
  /** null: this model's density has no local maximum for this shell. */
  gsz_radius: number | null;
  hf_radius: number | null;
  /** Relative drop into the minimum before the peak; null for the innermost. */
  gsz_depth: number | null;
  hf_depth: number | null;
}

export interface DensityComparison {
  gsz: FieldData;
  hf: FieldData;
  displaced_charge: Quantity;
  shells: ShellPeak[];
  provenance: Provenance;
}
```

and `density_comparison: DensityComparison | null;` on `RadialResponse`.

- [ ] **Step 4: Add the client parameter**

In `web/src/api/client.ts`, change `getRadial` to take `compare = false` as its
last parameter and append `if (compare) extra += "&compare=true";` after the
existing `many` block, so a request without it produces the same URL as before.

- [ ] **Step 5: Add the availability helpers**

Append to `web/src/lib/hfModel.ts`:

```typescript
/**
 * Whether both models can draw this atom's total density.
 *
 * Needs a many-electron atom (a one-electron system has no total density for
 * either model) and GSZ parameters (sulfur and chlorine have none). False while
 * the system table is still loading, which is the opposite default from
 * `gszAvailable` and deliberate: that one greys a control, and greying late is
 * harmless, while this one gates a request that would come back 422 if a deep
 * link named it before the table landed.
 */
export function compareAvailable(systems: SystemInfo[], system: string): boolean {
  const info = systems.find((s) => s.key === system);
  return info !== undefined && info.kind === "screened" && info.has_gsz;
}

/** A deep-linked `compare=1` forced off where it cannot run. */
export function resolveCompare(
  systems: SystemInfo[],
  system: string,
  compare: boolean,
): boolean {
  return compareAvailable(systems, system) && compare;
}
```

- [ ] **Step 6: Add the store field**

In `web/src/state/store.ts`: add `compare: boolean;` to `AppState` with this
comment, `setCompare: (compare: boolean) => void;` to the actions, `compare:
false,` to the initial state, and the setter:

```typescript
  /**
   * Whether the density plot draws both models at once.
   *
   * Not in INVALIDATED and not spread with it. It names an extra curve on one
   * payload rather than a different atom, so the cloud, the plane and the
   * surface are all still exactly as true as they were; throwing them away
   * would be seconds of solve spent to tell the user nothing. Only the radial
   * response goes, because only the radial response is missing a field.
   */
  setCompare: (compare) => set({ compare, radial: null }),
```

In `loadRadial`, read `compare` and pass it through, forcing it off where it
cannot run:

```typescript
  loadRadial: async () => {
    if (!(await get().ensureHF())) return;
    const { n, l, system, systems, compare } = get();
    set({
      radial: await client.getRadial(
        n, l, system, undefined, manyElectronParams(get()),
        resolveCompare(systems, system, compare),
      ),
    });
  },
```

Add `resolveCompare` to the existing `hfModel` import. In `setSystem`, reset
`compare` to false alongside `exchange` and `pauli`, for the same reason those
reset: a comparison is something the user asked for about the atom in front of
them.

- [ ] **Step 7: Add the deep link**

In `web/src/lib/urlState.ts`: add `compare: boolean;` to `UrlState`, `compare:
false,` to `URL_DEFAULTS`, `if (q.get("compare") === "1") out.compare = true;`
to the parser beside the `dirac` line (around line 276), and
`if (state.compare) q.set("compare", "1");` to the serializer (around line 378).

- [ ] **Step 8: Run the web tests and the type check**

Run: `cd web && npx vitest run && npm run build`
Expected: PASS, build clean

- [ ] **Step 9: Commit**

```bash
git add web/src/api web/src/lib web/src/state
git commit -m "Carry the comparison flag from the URL to the request

Forced off where it cannot run, because a deep link can name an atom before the
table that knows better has loaded."
```

---

### Task 6: The toggle

**Files:**
- Modify: `web/src/components/Controls.tsx:120-152`

- [ ] **Step 1: Add the checkbox**

In `Controls.tsx`, read `compare`, `setCompare` from the store and
`const canCompare = compareAvailable(systems, system);` beside the existing
`hasGsz`. Add, immediately after the `{!hasGsz && (...)}` hint block:

```tsx
          <label className="check">
            <input
              type="checkbox"
              checked={compare}
              disabled={!canCompare}
              onChange={(e) => setCompare(e.target.checked)}
            />
            Compare both models
          </label>
          <p className="panel-hint">
            {canCompare
              ? "Draws the total density under both models on one axis, with the number of electrons they place differently. The orbital plots stay on the model selected above."
              : "Needs both models, and only one of them has parameters for this element."}
          </p>
```

The toggle sits inside the existing `{isScreened && (...)}` block, so it never
appears for a hydrogen-like system. It is disabled rather than hidden for
sulfur and chlorine, matching the model radio directly above it.

- [ ] **Step 2: Build and check by hand**

Run: `cd web && npm run build`
Then from the repo root: `atomsim serve`

Check, in the browser:
1. Argon, Radial view: the toggle is enabled, and turning it on adds a second curve.
2. Sulfur: the toggle is disabled with its reason, and the model radio is too.
3. Hydrogen: no toggle at all.
4. `?system=na&view=radial&compare=1` opens with the toggle already on.

- [ ] **Step 3: Commit**

```bash
git add web/src/components/Controls.tsx
git commit -m "Offer the comparison, and say when only one model can run"
```

---

### Task 7: The overlay, the number, and the table

**Files:**
- Modify: `web/src/components/RadialView.tsx`
- Test: `web/src/components/RadialView.test.ts` (new)

**Interfaces:**
- Produces (exported from `RadialView.tsx` so the tests can reach them, matching
  the pattern in `SpectrumView.tsx`). Note there is no jsdom or testing-library
  in this project, so component tests target exported pure functions rather than
  rendered output; the spec's "legend and table render" is covered by testing
  the formatters that produce every string in them, plus the browser check in
  Step 7:
  - `displacedChargeText(q: Quantity, nElectrons: number | null): string`
  - `shellCells(s: ShellPeak): { label: string; gsz: string; hf: string; note: string | null }`

- [ ] **Step 1: Write the failing tests**

Create `web/src/components/RadialView.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import type { Quantity, ShellPeak } from "../api/types";
import { displacedChargeText, shellCells } from "./RadialView";

const q = (value: number, error: number | null): Quantity =>
  ({
    value,
    unit: "electrons",
    label: "",
    provenance: { error_estimate: error },
  }) as never;

const shell = (p: Partial<ShellPeak>): ShellPeak => ({
  label: "M",
  gsz_radius: null,
  hf_radius: null,
  gsz_depth: null,
  hf_depth: null,
  ...p,
});

describe("displacedChargeText", () => {
  it("states the number against the electron count it is a fraction of", () => {
    expect(displacedChargeText(q(0.06, 0.006), 18)).toContain("0.060");
    expect(displacedChargeText(q(0.06, 0.006), 18)).toContain("18");
  });

  it("refuses to print a figure smaller than its own error bar", () => {
    // Helium: 0.0003 displaced against a bar of about 0.0003. Printing four
    // decimals there would claim a precision the measurement does not have.
    expect(displacedChargeText(q(0.0003, 0.0004), 2)).toMatch(/agree to within/i);
  });

  it("handles an unknown electron count without inventing one", () => {
    expect(displacedChargeText(q(0.06, 0.006), null)).not.toContain("null");
  });
});

describe("shellCells", () => {
  it("says outright that a model resolves no peak, rather than leaving a blank", () => {
    const c = shellCells(shell({ hf_radius: 3.163, hf_depth: 0.003 }));
    expect(c.gsz).toMatch(/no separate peak/i);
    expect(c.hf).toContain("3.16");
  });

  it("flags a peak whose valley is too shallow to call a shell boundary", () => {
    const c = shellCells(shell({ gsz_radius: 3.1, hf_radius: 3.163, hf_depth: 0.003 }));
    expect(c.note).toMatch(/0.3%/);
  });

  it("says nothing extra about a well separated shell", () => {
    const c = shellCells(
      shell({ gsz_radius: 1.249, hf_radius: 1.241, gsz_depth: 0.4, hf_depth: 0.4 }),
    );
    expect(c.note).toBeNull();
  });
});
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd web && npx vitest run src/components/RadialView.test.ts`
Expected: FAIL, `displacedChargeText is not exported`

- [ ] **Step 3: Add the formatters**

Add to `web/src/components/RadialView.tsx`:

```tsx
/** Below which a peak's valley is too shallow to read as a shell boundary. */
const SHALLOW = 0.05;

/**
 * The disagreement in words, or an admission that it is under the noise.
 *
 * A number smaller than its own error bar is not a measurement of anything,
 * and four decimals of it would read as precision the comparison does not
 * have. Helium is that case: 0.0003 electrons displaced against a bar of the
 * same size.
 */
export function displacedChargeText(q: Quantity, nElectrons: number | null): string {
  const bar = q.provenance.error_estimate ?? 0;
  const of = nElectrons === null ? "" : ` of the atom's ${nElectrons}`;
  if (q.value <= bar) {
    return `The two models agree to within the resolution of this comparison (${q.value.toFixed(4)} electrons displaced, against a ${bar.toFixed(4)} bar).`;
  }
  return `The two models disagree about where ${q.value.toFixed(3)} ± ${bar.toFixed(3)} electrons${of} are.`;
}

/** One row of the shell table, with "no peak" as an answer rather than a gap. */
export function shellCells(s: ShellPeak): {
  label: string;
  gsz: string;
  hf: string;
  note: string | null;
} {
  const cell = (r: number | null) => (r === null ? "no separate peak" : r.toFixed(3));
  const shallow = [s.gsz_depth, s.hf_depth].filter(
    (d): d is number => d !== null && d < SHALLOW,
  );
  return {
    label: s.label,
    gsz: cell(s.gsz_radius),
    hf: cell(s.hf_radius),
    note:
      shallow.length === 0
        ? null
        : `barely separated: the dip before it is ${(Math.min(...shallow) * 100).toFixed(1)}% deep`,
  };
}
```

- [ ] **Step 4: Draw the overlay**

Give `FieldPlot` an optional second curve. Add to its props:

```tsx
  overlay?: { field: FieldData; label: string; selfLabel: string };
```

Inside, extend the y domain over both and draw the second path dashed:

```tsx
  const all = overlay ? [...field.values, ...overlay.field.values] : field.values;
  const lo = Math.min(0, ...all);
  const hi = Math.max(...all);
```

```tsx
        {overlay && (
          <path
            d={linePath(overlay.field.grid, overlay.field.values, x, y)}
            className="curve curve-overlay"
          />
        )}
```

and a legend inside the figcaption:

```tsx
        {overlay && (
          <span className="legend-inline">
            <span className="swatch-line" /> {overlay.selfLabel}
            <span className="swatch-line dashed" /> {overlay.label}
          </span>
        )}
```

The second curve is dashed rather than a second hue on purpose: this plot sits
beside canvases whose colour comes from the generated LUTs, and a hue chosen
here would be the one colour in the app that is not from the single authority in
`lib/luts.ts`. Add to the stylesheet beside the existing `.curve` rule:

```css
.curve-overlay { stroke-dasharray: 5 3; opacity: 0.85; }
.swatch-line { display: inline-block; width: 1.4em; border-top: 2px solid currentColor; }
.swatch-line.dashed { border-top-style: dashed; }
```

- [ ] **Step 5: Render the comparison block**

In the `{radial.total_density && (...)}` block, replace

```tsx
          <FieldPlot field={radial.total_density} logX />
```

with the version that overlays the other model when a comparison is present.
The primary curve stays the one the model radio selected, so the plot the reader
was already looking at does not move under them:

```tsx
          <FieldPlot
            field={radial.total_density}
            logX
            overlay={
              radial.density_comparison
                ? {
                    field:
                      model === "hf"
                        ? radial.density_comparison.gsz
                        : radial.density_comparison.hf,
                    label: model === "hf" ? "screened (GSZ)" : "Hartree-Fock",
                    selfLabel: model === "hf" ? "Hartree-Fock" : "screened (GSZ)",
                  }
                : undefined
            }
          />
```

Then add the readout and table below the existing captions:

```tsx
      {radial.density_comparison && (
        <div className="compare-block">
          <p className="hint-block">
            {displacedChargeText(
              radial.density_comparison.displaced_charge,
              radial.system.n_electrons ?? null,
            )}{" "}
            <Badge provenance={radial.density_comparison.provenance} />
          </p>
          <table className="shell-table">
            <caption>Shell peak radii [bohr]</caption>
            <thead>
              <tr><th>shell</th><th>GSZ</th><th>Hartree-Fock</th></tr>
            </thead>
            <tbody>
              {radial.density_comparison.shells.map(shellCells).map((c) => (
                <tr key={c.label}>
                  <th scope="row">{c.label}</th>
                  <td>{c.gsz}</td>
                  <td>{c.hf}</td>
                  {c.note && <td className="shell-note">{c.note}</td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
```

- [ ] **Step 6: Run the web tests and build**

Run: `cd web && npx vitest run && npm run build`
Expected: PASS, build clean

- [ ] **Step 7: Check it in the browser**

Run `atomsim serve`, then verify:
1. Argon with compare on: two curves, three shell rows, no "no separate peak" cell.
2. Sodium: the M row says "no separate peak" under GSZ and flags the 0.3% dip under HF.
3. Helium: the readout says the models agree to within the resolution.
4. The plot does not paint under the right panel, and the caption below it is not
   covered. Both are regressions of fixes landed on 2026-08-04; re-check them
   because this block adds height to the same column.

- [ ] **Step 8: Commit**

```bash
git add web/src/components/RadialView.tsx web/src/components/RadialView.test.ts web/src/index.css
git commit -m "Draw both densities on one axis, and print what they disagree about

The dashed curve is dashed rather than a second colour so that every hue in the
app still comes from the generated LUTs."
```

---

### Task 8: The captions, including one that may be wrong

**Files:**
- Modify: `web/src/components/RadialView.tsx` (the GSZ caption at the end of the
  density block)
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-08-04-phase29-density-comparison.md` (this
  file, appending the closeout section)
- Modify: `docs/superpowers/specs/2026-08-04-phase29-density-comparison-design.md`

- [ ] **Step 1: Settle the claim the spec flagged**

The GSZ branch of the caption currently says:

> Measurable, and this model is further from it than usual: GSZ was fitted to
> reproduce a potential, not a density, and every shell here sees the same one.

Task 2's `test_the_models_agree_far_better_than_their_energies_do` now measures
the opposite: under 1.5% displaced for every atom in He..Ar. Replace it with
what was measured, keeping the part that is still true (GSZ's shells all see one
potential) and dropping the part that is not:

```tsx
            <p className="hint-block">
              GSZ was fitted to reproduce a potential rather than a density, and
              every shell here sees the same one. That turns out to cost less
              than it sounds: turn the comparison on and the two models place
              under 1.5% of the electrons differently for every atom they both
              cover, because the fit was made against Hartree-Fock in the first
              place. Where the fitted model gives out is the energy, not the
              shape.
            </p>
```

If the measurement in Task 2 came out differently, write what it came out as.
The rule is that the caption states a number a test pins, not the other way
round.

- [ ] **Step 2: Update the README feature list**

The bullet beginning "The total radial density D(r) beside the orbitals" gains a
sentence naming the comparison and the sodium finding, in the same voice as the
rest of the list. No em dashes.

- [ ] **Step 3: Append a closeout section to this plan**

Add a `## What building it changed` section recording what the implementation
found that the design did not predict, including any measured number that moved.
This is the section every previous phase's plan carries and the one that is
worth reading a year from now.

- [ ] **Step 4: Mark the spec implemented**

Change the spec's header line from `Status: designed, not implemented.` to
`Status: implemented. See the plan's closeout for what building it changed.`

- [ ] **Step 5: Full verification**

Run, and paste the actual output rather than asserting success:

```bash
"C:/Users/yashg/.conda/envs/atomsim/python.exe" -m ruff check .
MKL_THREADING_LAYER=SEQUENTIAL "C:/Users/yashg/.conda/envs/atomsim/python.exe" -m pytest -q
cd web && npx vitest run && npm run build
```

Expected: ruff clean, pytest all passing with the count above the 1235 this
phase started from, vitest all passing, build clean.

- [ ] **Step 6: Commit**

```bash
git add README.md docs web/src/components/RadialView.tsx
git commit -m "Say what the two models actually disagree about, and close out the phase"
```

- [ ] **Step 7: Confirm the tree is clean**

Run: `git status --short`
Expected: no output. Nothing staged-but-uncommitted, nothing untracked that
belongs in the repo.

---

## What building it changed

**The peak finder needed a noise floor, and the design did not have one.** The
spec said "local maxima" and stopped there. Both solvers produce sign-flipping
jitter out in the tail where the orbital amplitude has decayed past what a
float64 eigensolve can represent: about 1e-34 of the peak for argon under
Hartree-Fock past 40 bohr, and 1e-60 for neon under the screened model past 11
bohr with the occupancy cap off. Every one of those wiggles is a local maximum,
and the shell table raised on all of them. The floor went to 1e-8 of the tallest
value, which is six orders below the faintest real shell in He..Ar (sodium's
outermost Hartree-Fock peak, at 2.2e-2) and twenty-six above the loudest noise.

**The floor belongs on the maxima only, and getting that wrong cost a round.**
The first fix floored both, which broke a legitimate deep valley: a minimum that
reaches near zero is not noise, it is the measurement that says two shells are
well separated. Flooring the minima discards that number hardest in exactly the
cases where it is clearest. It also protects against nothing, because the noise
lives beyond every real peak and the valley reported for a peak is the last
minimum before it.

**Helium does not take the branch it was designed for.** The spec expected the
"models agree to within the resolution" wording to fire there. Measured, helium
displaces 0.000343 electrons against a 0.000179 bar, so the number is 1.9 times
its own bar and the readout is right to state it. What the thin margin actually
needed was the other half of the same idea: `toFixed(3)` printed that resolved
measurement as "0.000 plus or minus 0.000", a zero standing in for something
that is not zero. The decimals now follow the bar, carried to two significant
figures. Both branches survive; only one of them was the one helium needed.

**The number came out the friendly way, and a caption was wrong because of it.**
The Radial view's GSZ caption said the screened model was "further from it than
usual" on the density. The measurement says the opposite: under 1.5% of the
electrons placed differently for every atom both models cover. GSZ was fitted
against Hartree-Fock potentials, so a close density is what the fit bought, and
where the fitted model gives out is the energy (2 to 24 percent off NIST on
valence ionization) rather than the shape. The caption now says what was
measured. The rule this phase kept re-learning is that the caption states a
number a test pins, never the other way round.

**Two curves that agree this well are nearly one curve.** On sodium at the y
scale the K and L peaks set, the dashed overlay sits underneath the solid one
and the eye cannot separate them. That is honest and it is the lesson, but it
means the plot is not what carries the finding: the displaced-charge readout and
the shell table are. The M row is where the disagreement is legible, and it is
legible only because "no separate peak" is printed as an answer rather than left
as an empty cell.

**A legend drawn in the caption's colour names nothing.** The swatch rules
inherited the figcaption's muted grey, so neither of them matched either curve.
They take the accent now, which is the colour the curves were already using and
introduces no hue that was not already in the app. `vertical-align: middle` on a
zero-height rule still leaves it riding the top of the line box, so the legend
row is an inline-flex with `align-items: center`.

**`compare` had to go into the RadialView effect's dependency list.** It names no
different solve, which is why it is deliberately not in the store's `INVALIDATED`
block: the cloud, the plane and the surface are all still exactly as true after
the toggle as before it, and throwing them away would be seconds of solve spent
to tell the user nothing. But it does name a field the payload is missing, and
`setCompare` drops the radial payload. Without the dependency the view cleared
and never refetched.

**The 1.5% caption was true and under-pinned, and lithium is the whole margin.**
The caption and the README both say "under 1.5% for every atom they both cover",
which is fifteen atoms (He..Ar less sulfur and chlorine). The test checked
seven. Measuring all fifteen: the claim holds, but lithium sits at 1.45%, so the
margin under the stated bound is about three percent of the bound. Aluminium is
next at 0.98% and nothing else passes 0.95%, so the comfortable gap the sampled
atoms suggested was an artifact of which ones were sampled. The loop now runs
the whole list and lithium is pinned on its own, because a bound is only as good
as its binding case and folding it in with fourteen roomier atoms reads as
though nothing in particular is holding it up. Cost: the density-comparison file
goes from 12.5s to 24.9s, which is 4% of the full suite.
