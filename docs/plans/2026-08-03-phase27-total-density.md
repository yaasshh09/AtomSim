# Phase 27: The one picture that is an observable

**Goal:** Draw the total radial electron density of a Hartree-Fock atom, D(r) = sum_a q_a P_a(r)^2, so the application shows the observable it has spent twenty-six phases explaining that its orbitals are not.

**Why now.** Phase 26 put a caption under every Hartree-Fock picture saying the orbital is not an observable and the total density is exactly spherical. That is true and it is a debt: the reader is told what the real quantity is and never shown it. Section 9 of the Phase 26 design deferred this on the grounds that it is a new quantity in a view that phase was only branching, and that it can be its own small piece. This is that piece.

**What it teaches.** Shell structure. Neon's D(r) has two peaks, argon's has three, and they are the K, L and M shells that make the periodic table have rows. Turn the exclusion principle off and the whole thing collapses to one peak, which is Phase 24's lesson in a shape rather than in two numbers.

**Architecture:** One engine function summing the converged orbitals the solve already returns, one optional field on the existing `RadialResponse`, one extra plot in `RadialView`. No new endpoints, no new job kinds.

## Global Constraints

Same as Phase 26: `Quantity`/`Field` with `Provenance` at every boundary, tiers inherited from the solve and never asserted, Hartree atomic units internally, line length 100, `ruff check .` clean, no em dashes, new physics gets a validation test, commit per logical change on `main`.

## The error bar this quantity gets, and why it is different

Every other Hartree-Fock shape in this codebase carries no error estimate, and `hf_atom.py` says why: the solve estimates its spread in hartree, and pinning a hartree number onto an amplitude in bohr^-1/2 would be wrong in dimension rather than merely loose.

D(r) is the exception, and it is worth taking. Each `P_a` is normalized so that `integral P_a^2 dr = 1` in the solver mesh's own quadrature, so `integral D dr = N` exactly, by construction, on that mesh. Resample to a uniform display grid and it no longer does. The residual `|integral D dr - N|` is therefore a real error, measured rather than modelled, and it is in electrons: the unit of the quantity it describes. That is the one shape in this project entitled to an error bar, and refusing it would be as dishonest as inventing one elsewhere.

This is the same trick as Phase 18's flux-closure check, which caught two invisible grid bugs. Expect it to be small (the display grid is 400 points over a box the solve already fits) and expect it to catch a transcription error instantly if one is made.

## Scope

Hartree-Fock only. The screened model's per-subshell radial functions are normalized the same way and the sum would work identically, so adding GSZ later is a few lines; it is left out because the claim this piece answers is a Hartree-Fock caption, and one model is enough to make the point.

---

### Task 1: hf_total_radial_density

**Files:**
- Modify: `src/atomsim/hf_atom.py` (add after `hf_mean_radius`; export in `__all__`)
- Test: `tests/test_total_density.py` (create)

**Interfaces:**
- Consumes: `solve_hartree_fock(z, n_electrons, config, exchange, pauli) -> HFResult` whose `orbitals` each carry `occupancy` and `P: Field` on the solver mesh.
- Produces: `hf_total_radial_density(z, n_electrons, *, config=None, exchange=True, pauli=True, points=400) -> Field` in `electrons/bohr` on a uniform grid.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_total_density.py`:

```python
"""The total radial density: the one shape in this application that is real.

Every other picture here is an orbital, and the app says in as many words that
an orbital is not an observable. This is the counterweight, so the checks are
about the two things that make it an observable rather than a drawing: that it
integrates to the electron count, and that its peaks are the shells.
"""

import numpy as np
import pytest
from scipy.signal import find_peaks

from atomsim.atoms import aufbau_configuration
from atomsim.hf_atom import hf_total_radial_density
from atomsim.provenance import Fidelity

def test_density_integrates_to_the_electron_count():
    """integral D(r) dr = N, which is what makes this a density and not a curve.

    Exact on the solver mesh by construction (every P is normalized there), so
    what this measures is the resampling onto the display grid. The residual is
    what the provenance reports as the error, in electrons.
    """
    d = hf_total_radial_density(10, 10)
    total = np.trapezoid(d.values, d.grid)
    assert total == pytest.approx(10.0, abs=1e-3)
    # And the reported error bar is honest about whatever residual is left.
    assert d.provenance.error_estimate == pytest.approx(abs(total - 10.0), abs=1e-9)

def test_argon_has_three_shells_and_neon_has_two():
    """K, L, M. This is the periodic table showing up in a plot.

    The count is the assertion, not the positions: peak positions move with Z
    and with the model, but the NUMBER of peaks is the shell structure itself,
    and a solve that lost it would be broken in a way no energy check catches.
    """
    for z, expected in ((10, 2), (18, 3)):
        d = hf_total_radial_density(z, z)
        # Height threshold rejects the shoulder ripples that a numerical P can
        # carry in its tail; the shells are the dominant features by far.
        peaks, _ = find_peaks(d.values, prominence=0.05 * d.values.max())
        assert len(peaks) == expected, f"Z={z} gave {len(peaks)} shells"

def test_the_collapsed_atom_has_one_shell():
    """No exclusion principle, no shells. Phase 24's lesson as a shape.

    Every electron is in the 1s, so there is one peak and nothing else, and
    that is the whole reason chemistry needs the principle.
    """
    collapsed = aufbau_configuration(18, pauli=False)
    d = hf_total_radial_density(
        18, 18, config=collapsed, exchange=False, pauli=False
    )
    peaks, _ = find_peaks(d.values, prominence=0.05 * d.values.max())
    assert len(peaks) == 1
    assert d.provenance.fidelity is Fidelity.COUNTERFACTUAL

def test_the_density_is_the_observable_and_says_so():
    d = hf_total_radial_density(10, 10)
    joined = " ".join(d.provenance.assumptions)
    assert "observable" in joined
    assert d.unit == "electrons/bohr"
    assert d.provenance.fidelity is Fidelity.APPROXIMATION

def test_a_non_aufbau_configuration_changes_the_density():
    """The configuration trap again, on the new quantity."""
    ground = hf_total_radial_density(10, 10)
    excited = hf_total_radial_density(
        10, 10, config=aufbau_configuration(10)[:-1] + (((2, 1), 5), ((3, 0), 1))
    )
    assert not np.allclose(ground.values, excited.values)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_total_density.py -q`
Expected: FAIL, `ImportError: cannot import name 'hf_total_radial_density'`.

- [ ] **Step 3: Implement**

Add to `src/atomsim/hf_atom.py`, after `hf_mean_radius`:

```python
_TOTAL_DENSITY_IS_OBSERVABLE = (
    "this one IS an observable: the total electron density, summed over every "
    "occupied subshell, is what an X-ray diffraction experiment measures. Its "
    "peaks are the shells, and the orbitals plotted above are the basis it was "
    "assembled from rather than things that can be measured one at a time"
)

def hf_total_radial_density(
    z: int,
    n_electrons: int,
    *,
    config: Configuration | None = None,
    exchange: bool = True,
    pauli: bool = True,
    points: int = 400,
) -> Field:
    """D(r) = sum_a q_a P_a(r)^2, in electrons per bohr.

    The radial distribution of the whole electron cloud, so integral D dr = N
    and the area under each peak is how many electrons that shell holds. This
    is the observable the orbital plots are not: the angular part of the total
    density is exactly uniform (Unsold on each filled subshell, and the
    average-of-configuration spreads a partly filled one equally over m), so
    all of the structure is here, in r, and none of it is a basis choice.

    Unlike every other shape this module returns, this one carries an error
    estimate, and the reason is dimensional rather than a change of policy.
    Each P_a is normalized so that integral P_a^2 dr = 1 in the solver mesh's
    own quadrature, which makes integral D dr = N exactly there. Resampling
    onto a uniform display grid breaks that, and the residual is a real error
    measured in electrons - the unit of the quantity it describes. An error bar
    the quantity's own normalization hands you is one worth reporting.
    """
    cfg = aufbau_configuration(n_electrons, pauli) if config is None else config
    result = solve_hartree_fock(z, n_electrons, cfg, exchange, pauli)

    solver_r = result.orbitals[0].P.grid
    grid = np.linspace(solver_r[0], solver_r[-1], points)
    values = np.zeros_like(grid)
    for orbital in result.orbitals:
        values += orbital.occupancy * np.interp(
            grid, orbital.P.grid, orbital.P.values
        ) ** 2

    # The closure check, kept as the error bar rather than asserted away. A
    # transcription slip in the sum above moves this immediately, which is
    # exactly what an error bar computed from the physics is for.
    residual = abs(float(np.trapezoid(values, grid)) - float(n_electrons))

    base = result.total_energy.provenance
    return Field(
        values=values,
        grid=grid,
        unit="electrons/bohr",
        grid_unit="bohr",
        label=f"D(r) = sum_a q_a P_a(r)^2 (N = {n_electrons})",
        provenance=Provenance(
            fidelity=base.fidelity,
            method=(
                f"{base.method}; total radial density summed over occupied "
                f"subshells and resampled onto {points} uniform points"
            ),
            assumptions=base.assumptions + (_TOTAL_DENSITY_IS_OBSERVABLE,),
            error_estimate=residual,
            refinement=base.refinement,
        ),
    )
```

Add `"hf_total_radial_density"` to `__all__`.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_total_density.py -q`
Expected: 5 passed.

- [ ] **Step 5: Lint and commit**

```bash
ruff check .
git add src/atomsim/hf_atom.py tests/test_total_density.py
git commit -m "Sum the occupied orbitals into the density that can be measured"
```

---

### Task 2: The radial endpoint carries it

**Files:**
- Modify: `src/atomsim/server/app.py` (`RadialResponse`, the `radial` endpoint's Hartree-Fock branch)
- Test: `tests/test_server_hf_views.py` (append)

**Interfaces:**
- Produces: `RadialResponse.total_density: FieldModel | None = None`, populated only under `model=hf`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_server_hf_views.py`:

```python
def test_radial_carries_the_total_density_under_hartree_fock(client):
    body = client.get("/api/radial/2/1?system=ne&model=hf").json()
    d = body["total_density"]
    assert d is not None
    assert d["unit"] == "electrons/bohr"
    assert "observable" in " ".join(d["provenance"]["assumptions"])
    # It integrates to the electron count, which is the point of sending it.
    total = np.trapezoid(d["values"], d["grid"])
    assert total == pytest.approx(10.0, abs=1e-2)

def test_screened_radial_has_no_total_density(client):
    """Null rather than absent: the field exists and this model does not fill it."""
    body = client.get("/api/radial/2/1?system=ne").json()
    assert body["total_density"] is None
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_server_hf_views.py -k total_density -q`
Expected: FAIL with `KeyError: 'total_density'`.

- [ ] **Step 3: Implement**

On `RadialResponse`:

```python
class RadialResponse(BaseModel):
    n: int
    l: int
    system: SystemModel
    r_wavefunction: FieldModel
    radial_probability: FieldModel
    #: D(r) for the whole cloud, summed over occupied subshells. Present only
    #: under the Hartree-Fock model, which is the one that knows the
    #: occupancies from its own solve. Null elsewhere rather than omitted, so a
    #: client reads one shape from every response.
    total_density: FieldModel | None = None
```

In the `radial` endpoint's Hartree-Fock branch, add before the `return`:

```python
            density = hf_total_radial_density(
                hf_z, hf_n, config=hf_config, exchange=exchange, pauli=pauli,
                points=points,
            )
```

and pass `total_density=FieldModel.from_field(density)` on the `RadialResponse`. Import `hf_total_radial_density` from `atomsim.hf_atom`.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_server_hf_views.py -q`
Expected: 16 passed.

- [ ] **Step 5: Lint and commit**

```bash
ruff check .
git add src/atomsim/server/app.py tests/test_server_hf_views.py
git commit -m "Send the total density beside the orbital it is not"
```

---

### Task 3: The Radial view draws it

**Files:**
- Modify: `web/src/api/types.ts` (`RadialResponse`)
- Modify: `web/src/components/RadialView.tsx`

- [ ] **Step 1: Add the type**

On the `RadialResponse` interface in `web/src/api/types.ts`:

```ts
  /**
   * Total radial electron density, present only under Hartree-Fock.
   *
   * The observable, unlike the two curves above it: its peaks are the shells
   * and the area under each is that shell's electron count.
   */
  total_density: FieldData | null;
```

- [ ] **Step 2: Draw it**

In `RadialView.tsx`, after the existing two `FieldPlot`s and before the caption:

```tsx
      {radial.total_density && (
        <>
          <FieldPlot field={radial.total_density} />
          <p className="hint-block">
            This one is measurable. Each peak is a shell and the area under it
            is how many electrons that shell holds, which is why the whole
            curve integrates to {radial.system.n_electrons ?? "N"}. The two
            plots above are one orbital out of the basis this was summed from.
          </p>
        </>
      )}
```

Keep the Phase 26 `HF_ORBITAL_CAPTION` where it is, under the orbital plots, so the contrast reads in order.

- [ ] **Step 3: Verify**

Run: `cd web && npx tsc --noEmit && npm test && npm run build`
Expected: clean, 252 passing.

Then `atomsim serve --port 8011 --no-browser` and open
`http://localhost:8011/?system=ar&n=3&l=1&m=0&view=radial&model=hf`.
Expected: three plots, the bottom one with three clear peaks. Switch to
`&nopauli=1&n=1&l=0` and it collapses to one.

- [ ] **Step 4: Commit**

```bash
git add web/src/api/types.ts web/src/components/RadialView.tsx
git commit -m "Draw the shells, and say which of the three plots is real"
```

---

### Task 4: Close out

- [ ] **Step 1:** Run `pytest -q` and `ruff check .`; both clean.
- [ ] **Step 2:** Append a short note to the Phase 26 spec's section 9 recording that the deferred radial curve landed, with a pointer to this plan.
- [ ] **Step 3:** Commit, confirm `git status --short` is empty.

---

## What building it changed

**The error bar earned its keep on the first run, and the plan's grid was wrong.**
This document specified `np.linspace` for the display grid. Built that way, the
closure residual came back at 0.35 electrons out of neon's 10, and argon
reported two shells instead of three. The cause was not the sum: a uniform
400-point grid over a 48 bohr box has 0.12 bohr spacing, and neon's 1s peaks
near 0.1 bohr and is narrower than the spacing, so the K shell fell between two
grid points entirely. The grid is now `np.geomspace`, which is the same
reasoning `numerics/mesh.py` applies to the solve itself.

This is the argument for computing an error bar out of the physics rather than
trusting a grid. Nothing else in the phase would have caught it: the curve
looked plausible, the peaks that survived were in the right places, and no
energy check touches this quantity. Only `integral D dr = N` knew.

**The tolerance was wrong too, and convergence is what settles it.** With the
log grid the residual is 1.8e-3 electrons at 400 points, over the 1e-3 the plan
asserted. Refining says why: the error falls by 4.03, 4.06 and 4.21 as the
points double, which is the trapezoid rule's h^2 and nothing else. A fixed
tolerance cannot tell quadrature from a dropped orbital, so the suite now
asserts the convergence rate rather than a magic number, and
`test_the_closure_residual_is_quadrature_and_not_a_defect` is the test that
would have caught the original bug on its own (a uniform grid's shortfall does
not converge away, however fine it gets).

**The plot needed a log axis, which the plan did not anticipate.** Argon's box
runs to 48 bohr and all three shells sit inside 2, so on the linear axis the
other plots use, the answer was one spike and 45 bohr of nothing. `FieldPlot`
gained a `logX` option, used only here. It is not a disclosed liberty: the axis
is labeled with the values it carries and nothing is clipped or rescaled. d3's
own log ticks had to be replaced with decades, since every 2x and 3x tick
overprinted into an unreadable band at this width.
