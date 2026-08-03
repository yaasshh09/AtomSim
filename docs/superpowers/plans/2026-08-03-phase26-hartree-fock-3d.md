# Phase 26: Hartree-Fock in three dimensions — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Hartree-Fock model reach the Cloud, 2-D cross-section, Radial and Surface views, carrying the exchange and Pauli counterfactual switches with it, without any view silently drawing a configuration or a model the user did not ask for.

**Architecture:** The engine already has the hard part. `hf_isosurface` (Phase 25) and `hf_radial` (Phase 21) exist and nothing calls them. Two engine functions are genuinely new (`sample_hf_density`, `hf_plane_grid`), and both take the same evaluator route their screened twins already take, so the sampling and plane machinery itself does not change. Before any of it can render, `config`, `exchange` and `pauli` have to be threaded through `_occupied_orbital` → `hf_radial` → `evaluate_hf_state` → `hf_isosurface`, because today that chain hardcodes the aufbau configuration and real-physics flags. On the wire, three existing job requests gain four fields each; no new endpoints.

**Tech Stack:** Python 3.12 (numpy, scipy, FastAPI, Pydantic v2, pytest), TypeScript/React/Three.js (Zustand, vitest, vite).

**Design spec:** `docs/superpowers/specs/2026-08-03-phase26-hartree-fock-3d-design.md`

## Global Constraints

- Every physical value crossing a module boundary is a `Quantity` or `Field` carrying a `Provenance` with a `Fidelity` tier. A bare `float` at a boundary is a bug.
- Hartree-Fock orbital pictures are `APPROXIMATION`; with `exchange=False` or `pauli=False` they are `COUNTERFACTUAL`. The tier is never hardcoded downstream of the solve — it is inherited from the solve.
- Engine-internal math is in Hartree atomic units. SI/display conversion happens at the server boundary and appends to the provenance `method`.
- `l` is the orbital angular-momentum quantum number, not a length. ruff E741 is ignored project-wide.
- Line length 100. `ruff check .` must pass.
- Never use em dashes in code, comments, docstrings, or user-facing copy.
- New physics gets a validation test (analytic ground truth, KS test, or grid convergence), not a smoke test.
- Defaults on every new request field are the already-shipped screened behaviour, so a client that has never heard of these fields cannot ask for Hartree-Fock or for a counterfactual by accident.
- Commit after every task. The tree is left committed and push-ready. No AI attribution in commit messages. Commit straight to `main`.

## Out of scope (from spec section 9, plus one found during planning)

- Total electron density as a drawable quantity, in 3-D or radially.
- Ions with no preset.
- A side-by-side screened/Hartree-Fock comparison view.
- **Sulfur and chlorine in the picture views.** The spec's section 1 is right that the HF engine solves them and GSZ cannot, but `ATOM_KEYS` excludes Z = 16 and 17 (`atoms.py:226`, they have no GSZ parameters), so the UI cannot select them and the Levels view cannot reach them either. Giving them keys is a separate change with its own consequences for every screened code path, and this phase does not make it. The wording in `Controls.tsx` must not claim otherwise.

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/atomsim/hf_atom.py` | Modify | Thread config/exchange/pauli through the orbital accessors; the orbital-is-not-an-observable claim; inherit fidelity |
| `src/atomsim/sampling.py` | Modify | `sample_hf_density` beside `sample_screened_density` |
| `src/atomsim/plane.py` | Modify | `hf_plane_grid` beside `screened_plane_grid` |
| `src/atomsim/isosurface.py` | Modify | `hf_isosurface` gains the three parameters and the inherited tier |
| `src/atomsim/server/app.py` | Modify | Four fields on three requests, one branch each, `model` on three metas, the refusals, the radial branch |
| `web/src/api/types.ts` | Modify | `ManyElectronParams`; `model` on the three metas |
| `web/src/api/client.ts` | Modify | Job params carry the four fields; `getRadial` takes them |
| `web/src/lib/hfModel.ts` | Create | Subshell availability, the short-form claim, the payload builder |
| `web/src/state/store.ts` | Modify | `ensureHF`; the four fields on every job payload |
| `web/src/components/Controls.tsx` | Modify | Delete the "Energy levels view only" sentence; disable unoccupied subshells |
| `web/src/components/RadialView.tsx` | Modify | Hartree-Fock branch and its caption |
| `tests/test_hf_views.py` | Create | Engine-level: the config trap, hydrogen reduction, helium exchange, plane/evaluator agreement |
| `tests/test_server_hf_views.py` | Create | Server-level: the branches and the four refusals |
| `web/src/lib/hfModel.test.ts` | Create | Subshell disabling and payload contents |

---

### Task 1: The configuration trap

This is task one because everything after it renders a lie without it. `_occupied_orbital` calls `solve_hartree_fock(z, n_electrons, aufbau_configuration(n_electrons))` with the configuration hardcoded and the two counterfactual flags left at their real-physics defaults. Nothing on the server reaches that path today, so nothing lies today. Wire up any view without fixing it first and a user who set `1s2 2s2 2p5 3s1` gets the aufbau orbital under a Hartree-Fock badge, and a user who switched exchange off gets the exchange-on orbital under a `COUNTERFACTUAL` badge, which is worse: the badge advertises a departure the picture does not contain.

**Files:**
- Modify: `src/atomsim/hf_atom.py` (`_occupied_orbital` at 958, `hf_radial` at 984, `evaluate_hf_state` at 1016; new module constant near 170)
- Test: `tests/test_hf_views.py` (create)

**Interfaces:**
- Consumes: `solve_hartree_fock(z, n_electrons, config, exchange, pauli) -> HFResult`, `aufbau_configuration(n_electrons, pauli) -> Configuration` (both already exist).
- Produces:
  - `_occupied_orbital(z, n_electrons, n, l, config: Configuration | None = None, exchange: bool = True, pauli: bool = True) -> HFOrbital`
  - `hf_radial(z, n_electrons, n, l, points: int = 400, *, config: Configuration | None = None, exchange: bool = True, pauli: bool = True) -> tuple[Field, Field]`
  - `evaluate_hf_state(z, n_electrons, n, l, m, positions, *, basis: str = "complex", config: Configuration | None = None, exchange: bool = True, pauli: bool = True) -> WavefunctionValues`
  - Module constant `_ORBITAL_NOT_OBSERVABLE: str`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hf_views.py`:

```python
"""Hartree-Fock reaching the picture views.

The load-bearing test here is the first one. Every other check in this file
would pass on a build that silently drew the aufbau configuration under any
label the user picked, which is exactly the failure this phase exists to
prevent, so the configuration is asserted to reach the orbital before anything
else is asserted about the orbital.
"""

import numpy as np
import pytest

from atomsim.atoms import aufbau_configuration, parse_config
from atomsim.hf_atom import evaluate_hf_state, hf_radial
from atomsim.provenance import Fidelity


def test_explicit_configuration_reaches_the_orbital():
    """A non-aufbau configuration must change the orbital it produces.

    Sodium's 3s sits outside a closed neon core. Promote it to 3p and the 3s is
    gone; ask instead for the 2p, which BOTH configurations occupy, and the
    orbital still has to differ, because the Fock operator for the 2p is built
    from the other occupied orbitals and one of them moved.
    """
    ground = aufbau_configuration(11)
    excited = parse_config("1s2 2s2 2p6 3p1")

    r_ground, _ = hf_radial(11, 11, 2, 1, points=200, config=ground)
    r_excited, _ = hf_radial(11, 11, 2, 1, points=200, config=excited)

    assert not np.allclose(r_ground.values, r_excited.values, atol=1e-9)


def test_exchange_off_reaches_the_orbital_and_the_badge():
    """The Hartree 2p is a different curve, and says so in its own tier."""
    config = aufbau_configuration(10)
    r_hf, _ = hf_radial(10, 10, 2, 1, points=200, config=config)
    r_hartree, _ = hf_radial(
        10, 10, 2, 1, points=200, config=config, exchange=False
    )

    assert not np.allclose(r_hf.values, r_hartree.values, atol=1e-9)
    assert r_hf.provenance.fidelity is Fidelity.APPROXIMATION
    assert r_hartree.provenance.fidelity is Fidelity.COUNTERFACTUAL


def test_evaluate_hf_state_inherits_the_counterfactual_tier():
    """The 3-D evaluator must not staple APPROXIMATION onto a Hartree orbital.

    It used to: the tier was a literal in the Provenance constructor rather
    than something read off the solve, so every picture came back
    APPROXIMATION whatever the flags said.
    """
    pos = np.array([[0.0, 0.0, 1.0], [0.5, 0.0, 0.5]])
    real = evaluate_hf_state(10, 10, 2, 1, 0, pos)
    hartree = evaluate_hf_state(10, 10, 2, 1, 0, pos, exchange=False)

    assert real.provenance.fidelity is Fidelity.APPROXIMATION
    assert hartree.provenance.fidelity is Fidelity.COUNTERFACTUAL


def test_pauli_off_refuses_every_subshell_but_the_one_that_exists():
    """With the cap lifted the configuration is 1s^N and nothing else is there.

    The refusal has to name the reason. A bare "not occupied" would read as a
    contingent fact about this atom rather than as the consequence of the
    switch the caller just flipped.
    """
    collapsed = aufbau_configuration(10, pauli=False)
    with pytest.raises(ValueError, match="occupancy cap"):
        hf_radial(
            10, 10, 2, 1, points=200,
            config=collapsed, exchange=False, pauli=False,
        )
    # And the one that does exist still comes back.
    r, _ = hf_radial(
        10, 10, 1, 0, points=200, config=collapsed, exchange=False, pauli=False
    )
    assert r.provenance.fidelity is Fidelity.COUNTERFACTUAL


def test_orbital_carries_the_not_an_observable_claim():
    """Every Hartree-Fock picture routes through hf_radial, so the claim does."""
    r, p = hf_radial(10, 10, 2, 1, points=200)
    joined = " ".join(r.provenance.assumptions)
    assert "not an observable" in joined
    assert "spherical" in joined
    assert joined == " ".join(p.provenance.assumptions)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_hf_views.py -v`
Expected: FAIL. `test_explicit_configuration_reaches_the_orbital` fails with `TypeError: hf_radial() got an unexpected keyword argument 'config'`; the others fail the same way.

- [ ] **Step 3: Add the claim constant**

In `src/atomsim/hf_atom.py`, after `_TOTAL_ENERGY_REFINEMENT` (around line 173), add:

```python
# Attached to every orbital this module hands out for drawing, because the
# picture invites exactly the reading it denies.
#
# The total electron density IS an observable, and for every atom this solver
# produces it is exactly spherical: the orbitals are central-field, P_nl(r)/r
# times Y_lm; a filled subshell sums over m to (2l+1)/4pi by Unsold's theorem;
# and the average-of-configuration functional spreads a partly filled subshell
# equally over m, so the m-sum is spherical there too. Carbon, oxygen and
# chlorine are all perfect balls.
#
# Drawing the observable instead would therefore produce a sphere for every
# atom in the application, honestly and uselessly. The decision is to draw the
# orbital and state the sphere in words. The counterweight belongs beside it
# and is also true: restricted Hartree-Fock on a spherically averaged
# configuration leaves the angular dependence exactly Y_lm, so the lobes are
# this model's own answer rather than hydrogen's answer reused.
_ORBITAL_NOT_OBSERVABLE = (
    "this is one orbital of a self-consistent field, and an orbital is not an "
    "observable: the total density of this atom is exactly spherical, so the "
    "shape drawn here is a basis choice rather than a photograph"
)
```

- [ ] **Step 4: Thread the parameters through `_occupied_orbital`**

Replace `_occupied_orbital` (currently lines 958-981) with:

```python
def _occupied_orbital(
    z: int,
    n_electrons: int,
    n: int,
    l: int,
    config: Configuration | None = None,
    exchange: bool = True,
    pauli: bool = True,
) -> HFOrbital:
    """The converged orbital for one subshell, or a refusal.

    Hartree-Fock cannot hand back an arbitrary channel the way a central-field
    model can. There is no single potential here: each occupied subshell has
    its own Fock operator, built from the others, so an unoccupied subshell has
    no operator to be an eigenfunction of. Asking for one is a question this
    model cannot answer, and inventing a channel by borrowing another
    subshell's operator would answer a different question silently.

    `config` defaults to the ground configuration for the rule in force, which
    is Aufbau normally and 1s^N with the cap lifted. It is a parameter and not
    a constant because the caller may have chosen a different one, and drawing
    the Aufbau orbital under a label that says otherwise is the same class of
    lie as drawing the wrong model.
    """
    if n <= l:
        raise ValueError(f"n must be > l, got n={n}, l={l}")
    cfg = aufbau_configuration(n_electrons, pauli) if config is None else config
    result = solve_hartree_fock(z, n_electrons, cfg, exchange, pauli)
    for orbital in result.orbitals:
        if (orbital.n, orbital.l) == (n, l):
            return orbital
    held = ", ".join(f"{o.n}{'spdf'[o.l]}" for o in result.orbitals)
    # Two different facts wear the same shape here, and the reader is owed the
    # one that applies. Under the exclusion principle an empty subshell is a
    # contingent fact about this atom. With the cap lifted it is the switch the
    # caller just flipped: there is one orbital and every electron is in it.
    why = (
        "the occupancy cap is lifted, so every electron is in the 1s and no "
        "other orbital exists to be an eigenfunction of anything"
        if not pauli
        else "Hartree-Fock builds one Fock operator per occupied subshell, so "
        "there is no operator for an empty one"
    )
    raise ValueError(
        f"subshell {n}{'spdf'[l]} is not occupied in Z={z}, N={n_electrons} "
        f"(which holds {held}); {why}"
    )
```

- [ ] **Step 5: Thread them through `hf_radial` and attach the claim**

Replace the `hf_radial` signature and its provenance assembly (currently 984-1004) with:

```python
def hf_radial(
    z: int,
    n_electrons: int,
    n: int,
    l: int,
    points: int = 400,
    *,
    config: Configuration | None = None,
    exchange: bool = True,
    pauli: bool = True,
) -> tuple[Field, Field]:
    """R_nl(r) and the radial density r^2 R^2, on a uniform display grid.

    Mirrors screened_atom.screened_radial, including its convention that the
    second field is the probability density and not the amplitude. Note the
    solver's own HFOrbital.P is the amplitude P = r R, a different quantity
    with a different unit; the naming follows screened_atom because these are
    what a caller plots.

    Every Hartree-Fock picture in this application routes through here - the
    radial plot directly, the cloud through its inverse CDF, the plane and the
    surface through evaluate_hf_state - so this is the one place the
    orbital-is-not-an-observable claim has to be attached to reach all four.
    """
    orbital = _occupied_orbital(
        z, n_electrons, n, l, config=config, exchange=exchange, pauli=pauli
    )
    solver_r = orbital.P.grid
    grid = np.linspace(solver_r[0], solver_r[-1], points)
    # R = P / r. The mesh never reaches r = 0, so this needs no special case,
    # which is exactly why the exponential mesh starts where it does.
    values = np.interp(grid, solver_r, orbital.P.values / solver_r)
    prov = dataclasses.replace(
        orbital.P.provenance,
        method=f"{orbital.P.provenance.method}; R_nl = P/r resampled uniformly",
        assumptions=orbital.P.provenance.assumptions + (_ORBITAL_NOT_OBSERVABLE,),
    )
```

The two `Field` constructions and the `return` below it are unchanged.

- [ ] **Step 6: Thread them through `evaluate_hf_state` and inherit the tier**

In `evaluate_hf_state`, replace the signature, the `hf_radial` call and the `Provenance` construction:

```python
def evaluate_hf_state(
    z: int,
    n_electrons: int,
    n: int,
    l: int,
    m: int,
    positions: np.ndarray,
    *,
    basis: str = "complex",
    config: Configuration | None = None,
    exchange: bool = True,
    pauli: bool = True,
) -> WavefunctionValues:
```

Then the radial call becomes:

```python
    r_field, _ = hf_radial(
        z, n_electrons, n, l, points=_HF_EVAL_POINTS,
        config=config, exchange=exchange, pauli=pauli,
    )
```

And the provenance's first field becomes:

```python
    prov = Provenance(
        # Inherited, never asserted. A Hartree orbital under an APPROXIMATION
        # badge would be a badge advertising the real atom over a picture of a
        # different universe, which is worse than no badge at all.
        fidelity=base.fidelity,
```

The rest of that constructor is unchanged. Also extend the docstring's second paragraph with:

```
    The angular factor is still the hydrogenic harmonic, and that is the
    model's own answer rather than a convenience: restricted Hartree-Fock on a
    spherically averaged configuration leaves the angular dependence exactly
    Y_lm.
```

- [ ] **Step 7: Run the tests**

Run: `pytest tests/test_hf_views.py -v`
Expected: 5 passed.

- [ ] **Step 8: Run the existing Hartree-Fock suite for regressions**

Run: `pytest tests/test_hf_atom.py tests/test_hf_pauli.py tests/test_hf_exchange.py tests/test_hf_channel.py tests/test_isosurface.py -q`
Expected: all pass. The new parameters are keyword-only with defaults matching the old hardcoded behaviour, so every existing caller is unaffected.

- [ ] **Step 9: Lint**

Run: `ruff check .`
Expected: no findings.

- [ ] **Step 10: Commit**

```bash
git add src/atomsim/hf_atom.py tests/test_hf_views.py
git commit -m "Let the chosen configuration reach the orbital that gets drawn"
```

---

### Task 2: sample_hf_density

**Files:**
- Modify: `src/atomsim/sampling.py` (add after `sample_screened_density`, which ends around line 222)
- Test: `tests/test_hf_views.py` (append)

**Interfaces:**
- Consumes: `hf_radial(z, n_electrons, n, l, points, *, config, exchange, pauli)` from Task 1; module-private `_radial_inverse_cdf_tabulated`, `_costheta_inverse_cdf`, `_phi_inverse_cdf`, `_draw_positions`, `_R_GRID_POINTS`.
- Produces: `sample_hf_density(z, n_electrons, n, l, m, count, *, seed=0, progress=None, n_chunks=10, basis="complex", config=None, exchange=True, pauli=True) -> SampleCloud`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hf_views.py`:

```python
def test_hf_sampling_reduces_to_hydrogen():
    """At Z=1, N=1 the Fock operator IS the bare Coulomb Hamiltonian.

    There is no other electron, so no direct term, no exchange term, and
    nothing for self-consistency to do. The sampler therefore has to reproduce
    the closed-form 1s radial CDF, 1 - e^(-2r)(1 + 2r + 2r^2), and a KS test is
    the check the analytic sampler already gets held to.

    A ground truth this tier rarely has, which is why it is spent here.
    """
    from scipy import stats

    from atomsim.sampling import sample_hf_density

    cloud = sample_hf_density(1, 1, 1, 0, 0, 20_000, seed=7)
    r = np.linalg.norm(cloud.positions.astype(np.float64), axis=1)

    def cdf(x):
        return 1.0 - np.exp(-2.0 * x) * (1.0 + 2.0 * x + 2.0 * x * x)

    assert stats.kstest(r, cdf).pvalue > 0.01


def test_hf_cloud_carries_the_solve_and_the_claim():
    from atomsim.sampling import sample_hf_density

    cloud = sample_hf_density(10, 10, 2, 1, 0, 2_000, seed=1)
    joined = " ".join(cloud.provenance.assumptions)
    assert cloud.provenance.fidelity is Fidelity.APPROXIMATION
    assert "not an observable" in joined
    assert "correlation" in joined  # the solve's own disclosure survived


def test_hf_cloud_goes_counterfactual_with_exchange_off():
    from atomsim.sampling import sample_hf_density

    cloud = sample_hf_density(
        10, 10, 2, 1, 0, 2_000, seed=1, exchange=False
    )
    assert cloud.provenance.fidelity is Fidelity.COUNTERFACTUAL
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_hf_views.py -k hf_sampling -v`
Expected: FAIL with `ImportError: cannot import name 'sample_hf_density'`.

- [ ] **Step 3: Implement**

Add the import at the top of `src/atomsim/sampling.py`, beside the existing `screened_radial` one:

```python
from atomsim.hf_atom import hf_radial
```

Add after `sample_screened_density`:

```python
def sample_hf_density(
    z: int,
    n_electrons: int,
    n: int,
    l: int,
    m: int,
    count: int,
    *,
    seed: int = 0,
    progress: Callable[[float], None] | None = None,
    n_chunks: int = 10,
    basis: str = "complex",
    config=None,
    exchange: bool = True,
    pauli: bool = True,
) -> SampleCloud:
    """Draw `count` positions from |psi_nlm|^2 for a Hartree-Fock orbital.

    The same shape as sample_screened_density and for the same reason: the
    radial source is a tabulated numerical R_nl and the angular part is the
    central-field Y_lm, so only the first line differs between the two models.
    Keeping them identical below that line is what makes the two comparable in
    the same camera.

    Fidelity is inherited from the solve rather than asserted here, so the two
    counterfactual switches carry through to the cloud's badge.
    """
    validate_quantum_numbers(n, l)
    if abs(m) > l:
        raise ValueError(f"|m| must be <= l, got m={m}, l={l}")
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    if basis not in ("complex", "real"):
        raise ValueError(f"basis must be 'complex' or 'real', got {basis!r}")

    r_field, _ = hf_radial(
        z, n_electrons, n, l, points=_R_GRID_POINTS,
        config=config, exchange=exchange, pauli=pauli,
    )
    r_grid, r_cdf, r_max = _radial_inverse_cdf_tabulated(r_field.grid, r_field.values)
    x_grid, x_cdf = _costheta_inverse_cdf(l, m)
    phi_sampler = _phi_inverse_cdf(m) if (basis == "real" and m != 0) else None
    positions = _draw_positions(
        count, r_grid, r_cdf, x_grid, x_cdf, phi_sampler, seed, n_chunks, progress
    )

    base = r_field.provenance
    phi_desc = (
        "phi uniform (|Y_lm|^2 is phi-independent)"
        if phi_sampler is None
        else "phi from analytic real-basis marginal (cos^2/sin^2 m phi)"
    )
    provenance = Provenance(
        fidelity=base.fidelity,
        method=(
            f"factorized inverse-CDF Monte-Carlo of |psi_nlm|^2 over a "
            f"Hartree-Fock R_nl ({basis} basis): r from P(r)=r^2 R^2 "
            f"(grid N={r_grid.size}, r_max={r_max:g} bohr), cos(theta) from "
            f"|Theta_lm|^2, {phi_desc}; {base.method}"
        ),
        assumptions=base.assumptions
        + (
            f"angular basis: {basis}",
            f"RNG PCG64 seed={seed}, count={count}",
            "positions in bohr",
        ),
        error_estimate=base.error_estimate,
        refinement=base.refinement,
    )
    return SampleCloud(
        positions=positions, n=n, l=l, m=m, Z=z, mu_ratio=1.0,
        basis=basis, provenance=provenance,
    )
```

Add `"sample_hf_density"` to `__all__` if `sampling.py` declares one; if it does not, skip this.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_hf_views.py -v`
Expected: 8 passed.

- [ ] **Step 5: Check for an import cycle**

`sampling.py` now imports `hf_atom`, which imports `numerics.*` and `analytic.*` but not `sampling`. Confirm:

Run: `python -c "import atomsim.sampling, atomsim.hf_atom; print('ok')"`
Expected: `ok`.

- [ ] **Step 6: Lint and commit**

```bash
ruff check .
git add src/atomsim/sampling.py tests/test_hf_views.py
git commit -m "Sample a Hartree-Fock orbital, checked against hydrogen's own CDF"
```

---

### Task 3: hf_plane_grid

**Files:**
- Modify: `src/atomsim/plane.py` (add after `screened_plane_grid`, which ends around line 188)
- Test: `tests/test_hf_views.py` (append)

**Interfaces:**
- Consumes: `evaluate_hf_state(..., *, basis, config, exchange, pauli)` from Task 1; module-private `_plane_values`, `default_half_extent`, `PlaneGrid`.
- Produces: `hf_plane_grid(z, n_electrons, n, l, m, quantity="density", basis="complex", resolution=512, half_extent=None, progress=None, *, config=None, exchange=True, pauli=True) -> PlaneGrid`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hf_views.py`:

```python
def test_hf_plane_agrees_with_the_evaluator_it_is_built_on():
    """The grid is not allowed to be its own authority.

    A plane routine can be wrong in two ways that look identical on screen: a
    transposed axis pair, and an off-by-one in the half-extent. Both survive
    every self-consistent check the grid can run on itself, and both die
    against psi evaluated directly at the same Cartesian points.
    """
    from atomsim.hf_atom import evaluate_hf_state
    from atomsim.plane import hf_plane_grid

    pg = hf_plane_grid(10, 10, 2, 1, 0, quantity="psi", resolution=33)
    axis = pg.axis
    # Row i is z = axis[i], column j is x = axis[j]; see the layout string the
    # server publishes for this array.
    for i in (3, 16, 29):
        for j in (5, 16, 27):
            direct = evaluate_hf_state(
                10, 10, 2, 1, 0,
                np.array([[axis[j], 0.0, axis[i]]]),
            )
            assert pg.values[i, j] == pytest.approx(
                float(np.real(direct.values[0])), rel=1e-9, abs=1e-12
            )


def test_hf_psi_is_real_on_the_y_zero_plane():
    """e^(i m phi) = +/-1 there, so a signed plot is honest and is labeled so."""
    from atomsim.hf_atom import evaluate_hf_state
    from atomsim.plane import hf_plane_grid

    pg = hf_plane_grid(10, 10, 2, 1, 1, quantity="psi", resolution=33)
    pos = np.array([[0.7, 0.0, 0.9], [-1.3, 0.0, 0.4]])
    psi = evaluate_hf_state(10, 10, 2, 1, 1, pos).values
    assert np.max(np.abs(np.imag(psi))) < 1e-12
    assert "psi is real on y=0" in " ".join(pg.provenance.assumptions)


def test_hf_plane_inherits_the_counterfactual_tier():
    from atomsim.plane import hf_plane_grid

    real = hf_plane_grid(10, 10, 2, 1, 0, resolution=17)
    hartree = hf_plane_grid(10, 10, 2, 1, 0, resolution=17, exchange=False)
    assert real.provenance.fidelity is Fidelity.APPROXIMATION
    assert hartree.provenance.fidelity is Fidelity.COUNTERFACTUAL
    assert not np.allclose(real.values, hartree.values)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_hf_views.py -k plane -v`
Expected: FAIL with `ImportError: cannot import name 'hf_plane_grid'`.

- [ ] **Step 3: Implement**

Add to the imports at the top of `src/atomsim/plane.py`:

```python
from atomsim.hf_atom import evaluate_hf_state
```

Add after `screened_plane_grid`:

```python
def hf_plane_grid(
    z: int,
    n_electrons: int,
    n: int,
    l: int,
    m: int,
    quantity: str = "density",
    basis: str = "complex",
    resolution: int = 512,
    half_extent: float | None = None,
    progress: Callable[[float], None] | None = None,
    *,
    config=None,
    exchange: bool = True,
    pauli: bool = True,
) -> PlaneGrid:
    """|psi|^2 or signed psi for a Hartree-Fock orbital on the y=0 plane.

    Structurally the screened routine with one line changed, which is the
    point: the two many-electron models have to be comparable in the same
    frame, and a plane that fitted its own extent differently would put half
    the difference between the pictures into the axes.
    """
    validate_quantum_numbers(n, l)
    validate_angular(l, m)
    if quantity not in ("density", "psi"):
        raise ValueError(f"quantity must be 'density' or 'psi', got {quantity!r}")
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}")
    z_net = max(z - n_electrons + 1, 1)  # asymptotic core charge sets display extent
    he = default_half_extent(n, z_net, 1.0) if half_extent is None else float(half_extent)
    if he <= 0.0:
        raise ValueError(f"half_extent must be positive, got {he}")

    def evaluator(pos):
        return evaluate_hf_state(
            z, n_electrons, n, l, m, pos,
            basis=basis, config=config, exchange=exchange, pauli=pauli,
        )

    values, axis, psi_assumptions = _plane_values(
        evaluator, quantity, resolution, he, progress
    )
    # The tier the solve came back with, read off the psi that was just
    # evaluated rather than assumed from the model's name.
    fidelity = evaluator(np.zeros((1, 3))).provenance.fidelity

    if quantity == "density":
        unit = "bohr^-3"
        label = f"|psi_{n},{l},{m}|^2 on the y=0 plane"
        qdesc = "|psi|^2 (probability density)"
        extra = ("plane y=0 contains the z quantization axis",)
    else:
        unit = "bohr^-3/2"
        label = f"psi_{n},{l},{m} on the y=0 plane"
        qdesc = "signed psi"
        extra = (
            "plane y=0 contains the z quantization axis",
            "psi is real on y=0 (e^{i m phi} = +/-1 there), so a signed plot is honest",
        )
    provenance = Provenance(
        fidelity=fidelity,
        method=(
            f"{qdesc} from a Hartree-Fock psi_nlm on a {resolution}x{resolution} "
            f"y=0 grid, half-extent {he:g} bohr"
        ),
        assumptions=psi_assumptions + extra,
        refinement="increase resolution, extent, or the solver mesh refinement",
    )
    return PlaneGrid(
        values=values, axis=axis, quantity=quantity, unit=unit, label=label,
        n=n, l=l, m=m, Z=z, mu_ratio=1.0, basis=basis, provenance=provenance,
    )
```

Note on the `fidelity` line: it costs one extra evaluation at the origin, which is one interpolation on an already-cached solve. That buys the tier from the same object the pixels came from rather than from a second `if not exchange` that could drift out of step with the solve's own rule.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_hf_views.py -v`
Expected: 11 passed.

- [ ] **Step 5: Lint and commit**

```bash
ruff check .
git add src/atomsim/plane.py tests/test_hf_views.py
git commit -m "Cut the y=0 plane through a Hartree-Fock orbital"
```

---

### Task 4: hf_isosurface gains the three parameters

`hf_isosurface` was built and tested in Phase 25 with no route to it, and it hardcodes `Fidelity.APPROXIMATION` and the aufbau configuration for the same reason `_occupied_orbital` did.

**Files:**
- Modify: `src/atomsim/isosurface.py:491-526`
- Test: `tests/test_hf_views.py` (append)

**Interfaces:**
- Consumes: `evaluate_hf_state(..., *, basis, config, exchange, pauli)`; `_build(...)`; `hf_mean_radius(result)` and `solve_hartree_fock(...)` from `hf_atom` for the cross-check test.
- Produces: `hf_isosurface(z, n_electrons, n, l, m, target_fraction=0.9, basis="complex", resolution=96, half_width=None, progress=None, *, config=None, exchange=True, pauli=True) -> Isosurface`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hf_views.py`:

```python
def test_hf_isosurface_reduces_to_the_closed_form_hydrogen_radius():
    """2.6612 bohr for the 1s at 90%, which Phase 25 validated closed-form.

    Z=1, N=1 makes the Fock operator the bare Coulomb Hamiltonian, so the
    numerical orbital has an exact answer to be held to and the surface built
    on it inherits that.
    """
    from atomsim.isosurface import hf_isosurface

    surf = hf_isosurface(1, 1, 1, 0, 0, target_fraction=0.9, resolution=96)
    radii = np.linalg.norm(surf.vertices, axis=1)
    assert radii.mean() == pytest.approx(2.6612, rel=5e-3)


def test_helium_hartree_and_hartree_fock_surfaces_are_bit_identical():
    """Helium's exchange energy is exactly zero, so the orbital is too.

    Exchange couples same-spin pairs only, and 1s2 holds one spin up and one
    spin down, so exchange_operator builds no terms at all. Phase 22
    established this on the energy; the surface has to inherit it to the bit,
    not merely to a tolerance, because a tolerance would hide a small real
    difference that would mean the toggle was reaching something it should not.
    """
    from atomsim.isosurface import hf_isosurface

    with_x = hf_isosurface(2, 2, 1, 0, 0, resolution=64)
    without = hf_isosurface(2, 2, 1, 0, 0, resolution=64, exchange=False)
    assert np.array_equal(with_x.vertices, without.vertices)
    # And the badge still flips, because the model the caller asked for is a
    # different model even where its answer coincides.
    assert with_x.provenance.fidelity is Fidelity.APPROXIMATION
    assert without.provenance.fidelity is Fidelity.COUNTERFACTUAL


def test_multi_shell_atom_surfaces_differ_with_exchange_off():
    """Neon has same-spin pairs, so removing exchange has to move the 2p."""
    from atomsim.isosurface import hf_isosurface

    with_x = hf_isosurface(10, 10, 2, 1, 0, resolution=64)
    without = hf_isosurface(10, 10, 2, 1, 0, resolution=64, exchange=False)
    assert not np.array_equal(with_x.vertices, without.vertices)


def test_pauli_collapse_orders_two_independent_measures_the_same_way():
    """Cross-checked rather than asserted.

    No direction is claimed for the collapsed 1s radius here, because a
    direction asserted without a derivation is a guess that a passing test then
    protects. Instead two separately computed sizes - the 90% enclosure radius
    off the triangulated surface, and hf_mean_radius off the solver mesh - must
    order the collapsed and the real atom the same way. Two code paths sharing
    no arithmetic have to agree on the sign, whatever it turns out to be.
    """
    from atomsim.atoms import aufbau_configuration
    from atomsim.hf_atom import hf_mean_radius, solve_hartree_fock
    from atomsim.isosurface import hf_isosurface

    real_cfg = aufbau_configuration(4)
    collapsed_cfg = aufbau_configuration(4, pauli=False)

    real_surf = hf_isosurface(4, 4, 1, 0, 0, resolution=64, config=real_cfg)
    collapsed_surf = hf_isosurface(
        4, 4, 1, 0, 0, resolution=64,
        config=collapsed_cfg, exchange=False, pauli=False,
    )
    surface_sign = np.sign(
        np.linalg.norm(collapsed_surf.vertices, axis=1).mean()
        - np.linalg.norm(real_surf.vertices, axis=1).mean()
    )

    real_r = hf_mean_radius(solve_hartree_fock(4, 4, real_cfg))
    collapsed_r = hf_mean_radius(
        solve_hartree_fock(4, 4, collapsed_cfg, False, False)
    )
    mean_sign = np.sign(collapsed_r.value - real_r.value)

    assert surface_sign != 0
    assert surface_sign == mean_sign
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_hf_views.py -k iso -v`
Expected: FAIL. `hf_isosurface() got an unexpected keyword argument 'exchange'` on the exchange tests; the hydrogen radius test may already pass, which is fine and expected.

- [ ] **Step 3: Implement**

Replace `hf_isosurface` in `src/atomsim/isosurface.py` (491-526) with:

```python
def hf_isosurface(
    z: int,
    n_electrons: int,
    n: int,
    l: int,
    m: int,
    target_fraction: float = 0.9,
    basis: str = "complex",
    resolution: int = 96,
    half_width: float | None = None,
    progress: Callable[[float], None] | None = None,
    *,
    config=None,
    exchange: bool = True,
    pauli: bool = True,
) -> Isosurface:
    """Same surface for a self-consistent Hartree-Fock orbital.

    `APPROXIMATION`, taking the weaker of the two tiers for the reason
    screened_isosurface gives: the model's distance from the real atom is
    larger than the grid's distance from the model. With either counterfactual
    switch thrown it is COUNTERFACTUAL instead, and that is read off the solve
    rather than decided here.
    """
    validate_quantum_numbers(n, l)
    validate_angular(l, m)

    def evaluator(pos):
        return evaluate_hf_state(
            z, n_electrons, n, l, m, pos,
            basis=basis, config=config, exchange=exchange, pauli=pauli,
        )

    fidelity = evaluator(np.zeros((1, 3))).provenance.fidelity
    counterfactual = fidelity is Fidelity.COUNTERFACTUAL
    z_net = max(z - n_electrons + 1, 1)
    return _build(
        evaluator,
        n=n, l=l, m=m, z=z, basis=basis,
        target_fraction=target_fraction,
        resolution=resolution,
        half_width=half_width,
        start_half_width=default_half_width(n, z_net, 1.0),
        fidelity=fidelity,
        method_prefix=(
            "Hartree psi_nlm" if counterfactual else "Hartree-Fock psi_nlm"
        ),
        extra_assumptions=(
            (
                "the enclosed fraction is exact for the orbital this solve "
                "produced, and that solve is a counterfactual: the fraction "
                "is not a statement about any real atom"
            )
            if counterfactual
            else (
                "the enclosed fraction is exact for the Hartree-Fock orbital, "
                "which neglects correlation"
            ),
        ),
        refinement=(
            "turn the altered rule back on"
            if counterfactual
            else "raise the grid resolution or widen the box"
        ),
        progress=progress,
    )
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_hf_views.py -v`
Expected: 15 passed. The Pauli cross-check and the neon pair each cost two solves, so this file now runs in roughly 40 to 90 seconds.

- [ ] **Step 5: Run the Phase 25 isosurface suite for regressions**

Run: `pytest tests/test_isosurface.py -q`
Expected: all pass.

- [ ] **Step 6: Lint and commit**

```bash
ruff check .
git add src/atomsim/isosurface.py tests/test_hf_views.py
git commit -m "Give the Hartree-Fock surface the configuration and the switches"
```

---

### Task 5: The job requests learn the model

**Files:**
- Modify: `src/atomsim/server/app.py` (`SampleRequest` 243, `PlaneRequest` 283, `IsoRequest` 311, the three metas, `create_sample_job` 1412, `create_plane_job` 1456, `create_iso_job` 1489, `job_meta` 1660, `create_app` state init near 680)
- Test: `tests/test_server_hf_views.py` (create)

**Interfaces:**
- Consumes: `sample_hf_density`, `hf_plane_grid`, `hf_isosurface`, `evaluate_hf_state` from Tasks 1 to 4; existing `_parse_config_or_422`, `_validate_hf_request`, `_is_screened`, `atom_for_key`, `aufbau_configuration`, `SUBSHELL_LABELS`.
- Produces:
  - `class ManyElectronRequest(BaseModel)` with `model: Literal["gsz", "hf"] = "gsz"`, `config: str | None = None`, `exchange: bool = True`, `pauli: bool = True`, and the pauli/exchange validator. `SampleRequest`, `PlaneRequest` and `IsoRequest` inherit it.
  - `_hf_view_target(req) -> tuple[int, int, Configuration]` inside `create_app`.
  - `model: str = "gsz"` on `SampleMetaModel`, `PlaneMetaModel`, `IsoMetaModel`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_server_hf_views.py`:

```python
"""Hartree-Fock over the job boundary.

Mirrors test_server_iso.py in shape. What is checked here that is not checked
in the engine tests: that the four new fields survive the trip and change the
answer, that the defaults cannot hand a client a model or a counterfactual it
did not ask for, and that the four refusals arrive with their reasons rather
than as a bare 422.
"""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from atomsim.server.app import create_app


@pytest.fixture()
def client():
    with TestClient(create_app()) as c:
        yield c


def _wait_done(client, job_id, deadline_s=120.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < deadline_s:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish")


def _run(client, path, **body):
    r = client.post(path, json=body)
    assert r.status_code == 200, r.text
    job = r.json()["id"]
    done = _wait_done(client, job)
    assert done["status"] == "done", done.get("error")
    return job


def test_sample_job_defaults_to_the_screened_model(client):
    """A client that has never heard of `model` cannot get Hartree-Fock."""
    job = _run(client, "/api/jobs/sample", n=2, l=1, m=0, count=2_000, system="ne")
    meta = client.get(f"/api/jobs/{job}/meta").json()
    assert meta["model"] == "gsz"
    assert "screened" in meta["provenance"]["method"]


def test_sample_job_under_hartree_fock(client):
    job = _run(
        client, "/api/jobs/sample",
        n=2, l=1, m=0, count=2_000, system="ne", model="hf",
    )
    meta = client.get(f"/api/jobs/{job}/meta").json()
    assert meta["model"] == "hf"
    assert "Hartree-Fock" in meta["provenance"]["method"]
    assert meta["provenance"]["fidelity"] == "APPROXIMATION"


def test_plane_job_under_hartree_fock_differs_from_the_screened_one(client):
    """Two models, two pictures. If they matched, one branch was not taken."""
    values = {}
    for model in ("gsz", "hf"):
        job = _run(
            client, "/api/jobs/plane",
            n=2, l=1, m=0, system="ne", resolution=64, model=model,
        )
        r = client.get(f"/api/jobs/{job}/data")
        values[model] = np.frombuffer(r.content, dtype=np.float32)
    assert not np.allclose(values["gsz"], values["hf"])


def test_iso_job_carries_the_counterfactual_badge(client):
    job = _run(
        client, "/api/jobs/isosurface",
        n=2, l=1, m=0, system="ne", resolution=48, model="hf", exchange=False,
    )
    meta = client.get(f"/api/jobs/{job}/meta").json()
    assert meta["model"] == "hf"
    assert meta["provenance"]["fidelity"] == "COUNTERFACTUAL"
    joined = " ".join(meta["provenance"]["assumptions"])
    assert "distinguishable" in joined


def test_explicit_config_reaches_the_picture(client):
    """The configuration trap, over the wire this time."""
    values = {}
    for config in (None, "1s2 2s2 2p5 3s1"):
        body = dict(n=2, l=1, m=0, system="ne", resolution=64, model="hf")
        if config is not None:
            body["config"] = config
        job = _run(client, "/api/jobs/plane", **body)
        r = client.get(f"/api/jobs/{job}/data")
        values[str(config)] = np.frombuffer(r.content, dtype=np.float32)
    assert not np.allclose(values["None"], values["1s2 2s2 2p5 3s1"])


def test_refuses_an_unoccupied_subshell_with_the_reason(client):
    r = client.post(
        "/api/jobs/sample",
        json=dict(n=3, l=2, m=0, count=2_000, system="ne", model="hf"),
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "not occupied" in detail
    assert "Fock operator" in detail


def test_refuses_a_non_1s_orbital_with_the_cap_lifted(client):
    r = client.post(
        "/api/jobs/sample",
        json=dict(
            n=2, l=1, m=0, count=2_000, system="ne",
            model="hf", exchange=False, pauli=False,
        ),
    )
    assert r.status_code == 422
    assert "occupancy cap" in r.json()["detail"]


def test_refuses_pauli_off_with_exchange_on(client):
    r = client.post(
        "/api/jobs/isosurface",
        json=dict(n=1, l=0, m=0, system="ne", model="hf", pauli=False),
    )
    assert r.status_code == 422
    assert "antisymmetry" in str(r.json()["detail"])


def test_refuses_a_one_electron_system(client):
    """Hartree-Fock of hydrogen is a question the other views already answer."""
    r = client.post(
        "/api/jobs/plane", json=dict(n=1, l=0, m=0, system="h", model="hf")
    )
    assert r.status_code == 422
    assert "electron count" in r.json()["detail"]


def test_counterfactual_flags_are_ignored_under_the_screened_model(client):
    """They name Hartree-Fock's rules, and GSZ has no exchange term to remove.

    Accepted and unused rather than refused: the client's model selector and
    its counterfactual switches are separate controls, and a user who leaves a
    switch set while switching models has not asked for anything incoherent.
    What must not happen is a screened picture wearing a COUNTERFACTUAL badge.
    """
    job = _run(
        client, "/api/jobs/sample",
        n=2, l=1, m=0, count=2_000, system="ne", exchange=False,
    )
    meta = client.get(f"/api/jobs/{job}/meta").json()
    assert meta["model"] == "gsz"
    assert meta["provenance"]["fidelity"] == "APPROXIMATION"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_server_hf_views.py -v`
Expected: FAIL. Pydantic rejects the unknown `model` field or ignores it; `meta["model"]` raises `KeyError`.

- [ ] **Step 3: Add the shared request base**

In `src/atomsim/server/app.py`, immediately before `class SampleRequest` (line 243), add:

```python
class ManyElectronRequest(BaseModel):
    """The four fields that pick which many-electron model draws a picture.

    Shared by the sample, plane and isosurface requests rather than repeated,
    because the pauli/exchange rule below has to be the same rule in all three
    and a copied validator is a rule waiting to drift.

    Every default is the already-shipped screened behaviour, so a client that
    has never heard of any of these fields cannot accidentally ask for
    Hartree-Fock or for a counterfactual. This mirrors HFRequest.
    """

    model: Literal["gsz", "hf"] = "gsz"
    #: Electron configuration, e.g. "1s2 2s2 2p5 3s1". None is the ground
    #: configuration for the rule in force. Ignored under model="gsz", which
    #: has no per-subshell solve for it to change.
    config: str | None = None
    exchange: bool = True
    pauli: bool = True

    @model_validator(mode="after")
    def _pauli_off_implies_exchange_off(self) -> "ManyElectronRequest":
        """Refuse the combination rather than quietly flipping a flag.

        Same rule and the same 422 as HFRequest: a Slater determinant holding
        two electrons in one spin-orbital is identically zero, so there is no
        wavefunction there for an exchange integral to act on. Correcting it
        for the client would hide that they asked for a state that does not
        exist.
        """
        if self.model == "hf" and not self.pauli and self.exchange:
            raise ValueError(
                "pauli=false requires exchange=false: exchange energy is a "
                "consequence of antisymmetry and the exclusion principle IS "
                "antisymmetry, so with the principle off there is nothing for "
                "an exchange integral to act on"
            )
        return self
```

Then change the three request classes to inherit it:

```python
class SampleRequest(ManyElectronRequest):
```
```python
class PlaneRequest(ManyElectronRequest):
```
```python
class IsoRequest(ManyElectronRequest):
```

leaving each class body exactly as it is.

- [ ] **Step 4: Add `model` to the three meta models**

On `SampleMetaModel`, `PlaneMetaModel` and `IsoMetaModel`, add beside the existing `system` field:

```python
    #: Which many-electron model drew this. Echoed from the request so the
    #: browser can name what it is looking at without parsing the provenance
    #: prose, which is a thing a view that has to parse prose eventually gets
    #: wrong.
    model: str = "gsz"
```

- [ ] **Step 5: Add the resolver and the refusals**

Inside `create_app`, next to `_resolve_config` (line 735), add:

```python
    def _hf_view_target(req) -> tuple[int, int, "Configuration"]:
        """(Z, N, configuration) for a Hartree-Fock picture, or a refusal.

        Every refusal here is synchronous and carries its reason, because the
        alternative is a job that dies several seconds in with an engine
        message the client has to guess at. The occupancy check needs no solve:
        which subshells exist is a property of the configuration, and the
        configuration is in the request.
        """
        if not _is_screened(req.system):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"model='hf' needs an atom with a known electron count, "
                    f"and {req.system!r} is a one-electron system; its "
                    f"wavefunction is already exact in the other views, so "
                    f"there is nothing a self-consistent field would add"
                ),
            )
        element = atom_for_key(req.system)
        n_electrons = element.z
        config = (
            aufbau_configuration(n_electrons, req.pauli)
            if req.config is None
            else _parse_config_or_422(req.config, req.pauli)
        )
        _validate_hf_request(element.z, n_electrons, config, req.pauli)
        if (req.n, req.l) not in [nl for nl, _ in config]:
            held = ", ".join(
                f"{n}{SUBSHELL_LABELS[l]}" for (n, l), _ in config
            )
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
                    f"{element.symbol} ({held}); {why}"
                ),
            )
        return element.z, n_electrons, config
```

`_validate_hf_request` already covers the fourth refusal (Z or outermost n outside the exercised range) with a 400 and its own reason.

- [ ] **Step 6: Record the model per job**

In `create_app`, beside `app.state.job_systems = {}` (line 680), add:

```python
    # Parallel to job_systems, and for the same reason: the meta endpoint sees
    # only the finished result object, and a SampleCloud does not know which
    # model produced it.
    app.state.job_models = {}
```

In `job_meta` (line 1664), add below the `system_key` line:

```python
        model_key = app.state.job_models.get(job_id, "gsz")
```

and pass it to the three builders: `_plane_meta(res, system_key, model_key)`, `_iso_meta(res, system_key, model_key)`, `_sample_meta(res, system_key, model_key)`. Change each builder's signature to take `model_key: str` and set `model=model_key` in the model it constructs.

- [ ] **Step 7: Branch the three job endpoints**

In `create_sample_job`, after `app.state.job_systems[job.id] = req.system` (line 1415), add:

```python
        app.state.job_models[job.id] = req.model
```

and insert the Hartree-Fock branch immediately before the existing `if _is_screened(req.system):`:

```python
        if req.model == "hf":
            hf_z, hf_n, hf_config = _hf_view_target(req)

            def work(progress):
                cloud = sample_hf_density(
                    hf_z, hf_n, req.n, req.l, req.m, req.count,
                    seed=req.seed, progress=lambda f: progress(0.9 * f),
                    basis=req.basis, config=hf_config,
                    exchange=req.exchange, pauli=req.pauli,
                )
                psi = evaluate_hf_state(
                    hf_z, hf_n, req.n, req.l, req.m,
                    cloud.positions.astype(np.float64), basis=req.basis,
                    config=hf_config, exchange=req.exchange, pauli=req.pauli,
                )
                progress(1.0)
                return SampleJobResult(cloud=cloud, psi=psi)

            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, jobs.run, job.id, work)
            return _job_model(job)
```

Note the job is created before `_hf_view_target` runs in the existing ordering; move `job = jobs.create()` to AFTER the `_hf_view_target` call so a refused request does not leave an orphan job. The two `app.state` writes move with it.

In `create_plane_job`, the same two edits, with:

```python
        if req.model == "hf":
            hf_z, hf_n, hf_config = _hf_view_target(req)

            def work(progress):
                return hf_plane_grid(
                    hf_z, hf_n, req.n, req.l, req.m,
                    quantity=req.quantity, basis=req.basis,
                    resolution=req.resolution, progress=progress,
                    config=hf_config, exchange=req.exchange, pauli=req.pauli,
                )

            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, jobs.run, job.id, work)
            return _job_model(job)
```

In `create_iso_job`, whose structure is `if / else` rather than early return, add a leading branch:

```python
        if req.model == "hf":
            hf_z, hf_n, hf_config = _hf_view_target(req)

            def work(progress):
                return hf_isosurface(
                    hf_z, hf_n, req.n, req.l, req.m,
                    target_fraction=req.fraction, basis=req.basis,
                    resolution=req.resolution, progress=progress,
                    config=hf_config, exchange=req.exchange, pauli=req.pauli,
                )

        elif _is_screened(req.system):
```

with the rest of the existing chain unchanged, and the same job-creation reordering.

Add the three engine imports at the top of `app.py`, beside the existing `screened_isosurface` / `screened_plane_grid` / `sample_screened_density` imports:

```python
from atomsim.isosurface import hf_isosurface
from atomsim.plane import hf_plane_grid
from atomsim.sampling import sample_hf_density
```

and add `evaluate_hf_state` to the existing `from atomsim.hf_atom import (...)` block.

- [ ] **Step 8: Run the tests**

Run: `pytest tests/test_server_hf_views.py -v`
Expected: 10 passed. Neon solves in about a second cold and is memoized after, so the file runs in roughly 30 to 60 seconds.

- [ ] **Step 9: Regression sweep and lint**

Run: `pytest tests/test_server.py tests/test_server_iso.py tests/test_server_hf.py tests/test_schemas.py -q`
Expected: all pass. The three requests gained fields with defaults and the metas gained one with a default, so every existing client shape is still valid.

Run: `ruff check .`
Expected: no findings.

- [ ] **Step 10: Commit**

```bash
git add src/atomsim/server/app.py tests/test_server_hf_views.py
git commit -m "Let a job ask for the Hartree-Fock picture, and refuse what it cannot draw"
```

---

### Task 6: The radial endpoint learns the branch

**Files:**
- Modify: `src/atomsim/server/app.py:1054-1088` (`radial`)
- Test: `tests/test_server_hf_views.py` (append)

**Interfaces:**
- Consumes: `hf_radial` from Task 1; `_hf_view_target` from Task 5 (which reads `req.system`, `req.config`, `req.pauli`, `req.n`, `req.l`, so the GET must present the same attribute names).
- Produces: `GET /api/radial/{n}/{l}?system=&points=&model=&config=&exchange=&pauli=` returning the existing `RadialResponse`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_server_hf_views.py`:

```python
def test_radial_under_hartree_fock(client):
    r = client.get("/api/radial/2/1?system=ne&model=hf")
    assert r.status_code == 200
    body = r.json()
    joined = " ".join(body["r_wavefunction"]["provenance"]["assumptions"])
    assert "not an observable" in joined
    assert body["r_wavefunction"]["provenance"]["fidelity"] == "APPROXIMATION"


def test_radial_hartree_fock_differs_from_screened(client):
    hf = client.get("/api/radial/2/1?system=ne&model=hf").json()
    gsz = client.get("/api/radial/2/1?system=ne").json()
    assert not np.allclose(
        hf["r_wavefunction"]["values"], gsz["r_wavefunction"]["values"]
    )


def test_radial_refuses_an_unoccupied_subshell(client):
    r = client.get("/api/radial/3/2?system=ne&model=hf")
    assert r.status_code == 422
    assert "not occupied" in r.json()["detail"]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_server_hf_views.py -k radial -v`
Expected: FAIL. `model=hf` is an unknown query parameter and is ignored, so the two responses match.

- [ ] **Step 3: Implement**

Replace the `radial` endpoint signature and add the branch after the `points` check:

```python
    @app.get("/api/radial/{n}/{l}", response_model=RadialResponse)
    def radial(
        n: int,
        l: int,
        system: str = "h",
        points: int = 400,
        model: Literal["gsz", "hf"] = "gsz",
        config: str | None = None,
        exchange: bool = True,
        pauli: bool = True,
    ) -> RadialResponse:
        _validate_state(n, l, 0)
        if not 50 <= points <= 2000:
            raise HTTPException(status_code=422, detail="points must be in [50, 2000]")
        if model == "hf":
            if not pauli and exchange:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "pauli=false requires exchange=false: exchange energy "
                        "is a consequence of antisymmetry and the exclusion "
                        "principle IS antisymmetry, so with the principle off "
                        "there is nothing for an exchange integral to act on"
                    ),
                )
            # Presented as an object because _hf_view_target reads attributes,
            # and one resolver for four views is what keeps the refusals from
            # drifting apart between the picture endpoints and this one.
            req = SimpleNamespace(
                system=system, config=config, pauli=pauli, n=n, l=l
            )
            hf_z, hf_n, hf_config = _hf_view_target(req)
            rw, p = hf_radial(
                hf_z, hf_n, n, l, points=points,
                config=hf_config, exchange=exchange, pauli=pauli,
            )
            element = atom_for_key(system)
            return RadialResponse(
                n=n, l=l,
                system=SystemModel.from_atom(
                    element, element.z,
                    f"{element.name}: self-consistent Hartree-Fock "
                    f"(APPROXIMATION; COUNTERFACTUAL with a switch thrown).",
                ),
                r_wavefunction=FieldModel.from_field(rw),
                radial_probability=FieldModel.from_field(p),
            )
        if _is_screened(system):
```

with the rest of the function unchanged. Add `from types import SimpleNamespace` to the imports at the top of `app.py`.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_server_hf_views.py -v`
Expected: 13 passed.

- [ ] **Step 5: Lint and commit**

```bash
ruff check .
git add src/atomsim/server/app.py tests/test_server_hf_views.py
git commit -m "Serve the Hartree-Fock radial curve on the endpoint that already had one"
```

---

### Task 7: The client and the store carry the four fields

**Files:**
- Modify: `web/src/api/types.ts` (`SampleMeta`, `PlaneMeta`, `IsoMeta`; add `ManyElectronParams`)
- Modify: `web/src/api/client.ts` (`SampleParams`, `PlaneParams`, `IsoParams`, `getRadial`)
- Create: `web/src/lib/hfModel.ts`
- Create: `web/src/lib/hfModel.test.ts`
- Modify: `web/src/state/store.ts` (`ensureHF`, `sample`, `loadPlane`, `loadIso`, `loadRadial`)

**Interfaces:**
- Consumes: server fields from Tasks 5 and 6; `HFLevels` and its `orbitals: HFOrbital[]` (each with `n`, `l`) from `api/types.ts`; store fields `model`, `config`, `exchange`, `pauli`, `hf`, `hfStatus`, `loadHF`.
- Produces:
  - `interface ManyElectronParams { model: "gsz" | "hf"; config: string | null; exchange: boolean; pauli: boolean }` in `api/types.ts`
  - `manyElectronParams(state) -> ManyElectronParams` in `lib/hfModel.ts`
  - `subshellAvailable(hf: HFLevels | null, model: AtomModel, n: number, l: number) -> boolean` in `lib/hfModel.ts`
  - `HF_ORBITAL_CAPTION: string` in `lib/hfModel.ts`
  - `ensureHF: () => Promise<boolean>` on the store

- [ ] **Step 1: Write the failing test**

Create `web/src/lib/hfModel.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import type { HFLevels } from "../api/types";
import { HF_ORBITAL_CAPTION, manyElectronParams, subshellAvailable } from "./hfModel";

const NEON = {
  kind: "hf",
  z: 10,
  n_electrons: 10,
  orbitals: [
    { n: 1, l: 0 },
    { n: 2, l: 0 },
    { n: 2, l: 1 },
  ],
} as unknown as HFLevels;

describe("manyElectronParams", () => {
  it("carries all four fields a job needs", () => {
    const p = manyElectronParams({
      model: "hf",
      config: "1s2 2s2 2p5 3s1",
      exchange: false,
      pauli: true,
    });
    expect(p).toEqual({
      model: "hf",
      config: "1s2 2s2 2p5 3s1",
      exchange: false,
      pauli: true,
    });
  });

  it("sends the real physics under the screened model", () => {
    // The counterfactual switches name Hartree-Fock's rules. Forwarding a
    // stale one under GSZ would put a flag on the wire that the picture does
    // not honour, and the server would echo it back into a badge.
    const p = manyElectronParams({
      model: "gsz",
      config: "1s2 2s2 2p6",
      exchange: false,
      pauli: false,
    });
    expect(p).toEqual({
      model: "gsz",
      config: "1s2 2s2 2p6",
      exchange: true,
      pauli: true,
    });
  });
});

describe("subshellAvailable", () => {
  it("allows everything under the screened model", () => {
    expect(subshellAvailable(null, "gsz", 3, 2)).toBe(true);
  });

  it("allows everything before the solve has landed", () => {
    // Not knowing is not the same as knowing it is empty, and greying a
    // control on a guess teaches the wrong thing about the atom.
    expect(subshellAvailable(null, "hf", 3, 2)).toBe(true);
  });

  it("allows an occupied subshell and refuses an empty one", () => {
    expect(subshellAvailable(NEON, "hf", 2, 1)).toBe(true);
    expect(subshellAvailable(NEON, "hf", 3, 2)).toBe(false);
    expect(subshellAvailable(NEON, "hf", 3, 0)).toBe(false);
  });
});

describe("HF_ORBITAL_CAPTION", () => {
  it("says both halves of the claim", () => {
    expect(HF_ORBITAL_CAPTION).toContain("not an observable");
    expect(HF_ORBITAL_CAPTION).toContain("spherical");
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd web && npx vitest run src/lib/hfModel.test.ts`
Expected: FAIL, cannot resolve `./hfModel`.

- [ ] **Step 3: Create `web/src/lib/hfModel.ts`**

```ts
import type { HFLevels, ManyElectronParams } from "../api/types";
import type { AtomModel } from "../state/store";

/**
 * The short form of the claim the provenance carries in full.
 *
 * It is a statement about the physics rather than about the rendering, so it
 * is not a disclosed liberty and does not flow through lib/liberties.ts. The
 * presentational disclosures on the cloud and the surface are unaffected and
 * still apply.
 */
export const HF_ORBITAL_CAPTION =
  "One orbital of a self-consistent field, and an orbital is not an " +
  "observable. This atom's total density is exactly spherical, so the shape " +
  "here is a basis choice rather than a photograph. The lobes are still this " +
  "model's own answer: restricted Hartree-Fock leaves the angular part exactly Yₗₘ.";

/** The subset of app state a job payload needs to name its many-electron model. */
export interface ModelSelection {
  model: AtomModel;
  config: string | null;
  exchange: boolean;
  pauli: boolean;
}

/**
 * The four fields every picture job sends.
 *
 * The counterfactual flags are forced back to real physics under the screened
 * model. GSZ has no exchange term to remove and no occupancy cap of its own,
 * so a false here would be a flag the server accepts, does not honour, and
 * echoes into a badge over a picture that never departed from anything.
 */
export function manyElectronParams(s: ModelSelection): ManyElectronParams {
  return {
    model: s.model,
    config: s.config,
    exchange: s.model === "hf" ? s.exchange : true,
    pauli: s.model === "hf" ? s.pauli : true,
  };
}

/**
 * Whether (n, l) can be drawn under the current model.
 *
 * Hartree-Fock builds one Fock operator per occupied subshell, so an empty one
 * has nothing to be an eigenfunction of and the server refuses it. Reading the
 * solve's own orbital list is what lets the picker grey the control instead of
 * firing a job it can already tell will come back 422.
 *
 * `true` when the solve has not landed yet: not knowing is not the same as
 * knowing the subshell is empty.
 */
export function subshellAvailable(
  hf: HFLevels | null,
  model: AtomModel,
  n: number,
  l: number,
): boolean {
  if (model !== "hf" || hf === null) return true;
  return hf.orbitals.some((o) => o.n === n && o.l === l);
}
```

- [ ] **Step 4: Add `ManyElectronParams` and the meta field**

In `web/src/api/types.ts`, add:

```ts
/**
 * Which many-electron model a picture job asks for, and under which rules.
 *
 * Every field defaults on the server to the already-shipped screened
 * behaviour, so omitting the object entirely cannot ask for Hartree-Fock or
 * for a counterfactual.
 */
export interface ManyElectronParams {
  model: "gsz" | "hf";
  /** Electron configuration, e.g. "1s2 2s2 2p5 3s1". null is the ground one. */
  config: string | null;
  exchange: boolean;
  pauli: boolean;
}
```

and add to `SampleMeta`, `PlaneMeta` and `IsoMeta`:

```ts
  /** Which many-electron model drew this: "gsz" or "hf". */
  model: string;
```

- [ ] **Step 5: Widen the client params**

In `web/src/api/client.ts`, add `Partial<ManyElectronParams>` to the three param interfaces by extending them:

```ts
export interface SampleParams extends Partial<ManyElectronParams> {
```
```ts
export interface PlaneParams extends Partial<ManyElectronParams> {
```
```ts
export interface IsoParams extends Partial<ManyElectronParams> {
```

(import `ManyElectronParams` as a type at the top). The three `post` calls already spread `params`, so no other change is needed there.

Change `getRadial`:

```ts
export function getRadial(
  n: number,
  l: number,
  system: string,
  points?: number,
  many?: ManyElectronParams,
): Promise<RadialResponse> {
  const p = points === undefined ? "" : `&points=${points}`;
  // Only the non-default half is written, so a screened request produces the
  // same URL it produced before this phase and nothing downstream has to know
  // the parameters exist.
  let extra = "";
  if (many !== undefined && many.model === "hf") {
    extra = "&model=hf";
    if (many.config !== null) extra += `&config=${encodeURIComponent(many.config)}`;
    if (!many.exchange) extra += "&exchange=false";
    if (!many.pauli) extra += "&pauli=false";
  }
  return getJson(`/api/radial/${n}/${l}?system=${key(system)}${p}${extra}`);
}
```

- [ ] **Step 6: Wire the store**

In `web/src/state/store.ts`, add to the actions interface:

```ts
  /**
   * Solve the atom before drawing it, under Hartree-Fock only.
   *
   * The solve is what says which subshells exist, so a picture fired before it
   * lands is a picture that may be about to be refused. Resolves to whether a
   * solve is available; false means the error is already on the store.
   */
  ensureHF: () => Promise<boolean>;
```

and the implementation, beside `loadHF`:

```ts
  ensureHF: async () => {
    const { model, hf } = get();
    if (model !== "hf") return true;
    if (hf !== null) return true;
    await get().loadHF();
    return get().hf !== null;
  },
```

Then at the top of `sample`, `loadIso` and `loadPlane`, before the `set({ status: "sampling" ... })` line:

```ts
    if (!(await get().ensureHF())) return;
```

and add the params to each payload. `sample`:

```ts
  sample: async () => {
    if (!(await get().ensureHF())) return;
    const { n, l, m, count, basis, system } = get();
    set({ status: "sampling", progress: 0, error: null });
    try {
      const job = await client.createSampleJob({
        n, l, m, count, basis, system, ...manyElectronParams(get()),
      });
```

`loadIso`:

```ts
      const job = await client.createIsoJob({
        n,
        l,
        m,
        system,
        basis,
        fraction: isoFraction,
        ...manyElectronParams(get()),
      });
```

`loadPlane`:

```ts
      const job = await client.createPlaneJob({
        n,
        l,
        m,
        system,
        basis,
        quantity: planeQuantity,
        ...manyElectronParams(get()),
      });
```

`loadRadial`:

```ts
  loadRadial: async () => {
    if (!(await get().ensureHF())) return;
    const { n, l, system } = get();
    set({
      radial: await client.getRadial(n, l, system, undefined, manyElectronParams(get())),
    });
  },
```

Import at the top: `import { manyElectronParams } from "../lib/hfModel";`.

Also extend the `INVALIDATED` block's comment and contents: `radial` is derived from the model now as well, so `setModel` must clear it. Check whether `radial` is already listed in `INVALIDATED`; if it is not, add it, since `setModel` spreads `INVALIDATED` and a stale screened curve under a Hartree-Fock badge is exactly the class of bug that block exists to prevent.

- [ ] **Step 7: Run the frontend tests**

Run: `cd web && npx vitest run src/lib/hfModel.test.ts src/state/store.test.ts src/api/client.test.ts`
Expected: all pass. If `store.test.ts` mocks the client, `ensureHF` may need the mock to expose `createHFJob`; add it to the mock rather than weakening the store.

- [ ] **Step 8: Typecheck and commit**

```bash
cd web && npx tsc --noEmit
git add web/src/api/types.ts web/src/api/client.ts web/src/lib/hfModel.ts web/src/lib/hfModel.test.ts web/src/state/store.ts
git commit -m "Send the model, the configuration and the switches with every picture job"
```

---

### Task 8: The views

**Files:**
- Modify: `web/src/components/Controls.tsx` (the sentence at 127; the (n, l) picker)
- Modify: `web/src/components/RadialView.tsx`
- Test: `web/src/lib/hfModel.test.ts` (already covers the logic; this task wires it)

**Interfaces:**
- Consumes: `subshellAvailable`, `HF_ORBITAL_CAPTION` from Task 7; store `model`, `hf`, `radial`.
- Produces: no new exported interfaces.

- [ ] **Step 1: Replace the Controls sentence**

In `web/src/components/Controls.tsx`, replace line 127:

```tsx
              : "Self-consistent field, solved per subshell, with no fitted parameters. Every view draws it: cloud, cross-section, radial and surface. What you see is one orbital, not the total density, which for these atoms is exactly spherical."}
```

- [ ] **Step 2: Disable unoccupied subshells**

Find the (n, l) picker in `Controls.tsx` and add to each l option (and each n option where every l under it is unavailable):

```tsx
disabled={!subshellAvailable(hf, model, n, lValue)}
```

with `subshellAvailable` imported from `../lib/hfModel` and `hf` pulled from the store alongside `model` in the existing destructure at line 26. Add the hint beneath the picker, rendered only when something is greyed:

```tsx
{model === "hf" && hf !== null && (
  <p className="panel-hint">
    Greyed subshells are empty in {hf.config}. Hartree-Fock builds one Fock
    operator per occupied subshell, so an empty one has no operator to be an
    eigenfunction of.
  </p>
)}
```

- [ ] **Step 3: Caption the Radial view**

In `web/src/components/RadialView.tsx`, pull `model` from the store in the existing destructure at line 73 and render the claim under the plots:

```tsx
{model === "hf" && <p className="hint-block">{HF_ORBITAL_CAPTION}</p>}
```

with `HF_ORBITAL_CAPTION` imported from `../lib/hfModel`. Do the same in `CloudView.tsx`, `PlaneView.tsx` and the isosurface caption block, placing it beside the existing badge in each.

- [ ] **Step 4: Run the full frontend suite**

Run: `cd web && npm test`
Expected: all pass (246 plus the new hfModel tests).

- [ ] **Step 5: Build**

Run: `cd web && npm run build`
Expected: `tsc --noEmit` clean, `web/dist` written. The server only mounts the app if `web/dist` exists, so this step is not optional.

- [ ] **Step 6: Look at it**

Run: `atomsim serve --port 8001`, then open `http://localhost:8001/?system=ne&n=2&l=1&m=0&view=cloud&model=hf`.
Expected: a Hartree-Fock 2p cloud, an APPROXIMATION badge, the orbital caption, and 3d greyed in the picker. Flip to `&nox=1` and confirm the badge turns COUNTERFACTUAL. Flip to `&nopauli=1` and confirm the picker offers 1s only.

- [ ] **Step 7: Commit**

```bash
git add web/src/components/
git commit -m "Draw the Hartree-Fock orbital in every view, and say what it is not"
```

---

### Task 9: Performance, the cache, and closing out

`solve_hartree_fock` is `lru_cache(maxsize=8)`. All four views share one solve per (atom, configuration, exchange, pauli), but adding two flags and user-chosen configurations multiplies the key space against those eight slots. The phase measures eviction before changing the number, because raising it blind trades memory for a problem nobody has demonstrated.

**Files:**
- Create: `tests/test_hf_view_performance.py`
- Modify: `docs/superpowers/specs/2026-08-03-phase26-hartree-fock-3d-design.md` (status line and a short "what building it changed" section)
- Modify: `src/atomsim/hf_atom.py` only if the measurement says the cache size must move

- [ ] **Step 1: Write the measurement**

Create `tests/test_hf_view_performance.py`:

```python
"""What the picture views actually cost, recorded rather than assumed.

Budgets here are deliberately loose. The number worth guarding is the shape of
the cost - one solve shared by four views, and an isosurface that is grid work
rather than solve work - and a tight budget on a shared CI runner guards the
runner's mood instead.
"""

import time

import numpy as np
import pytest

from atomsim.atoms import aufbau_configuration
from atomsim.hf_atom import solve_hartree_fock
from atomsim.isosurface import hf_isosurface
from atomsim.plane import hf_plane_grid
from atomsim.sampling import sample_hf_density


def test_four_views_share_one_solve():
    """The solve is the expensive part, so it must be paid once.

    Measured through the cache counters rather than a stopwatch: a wall-clock
    assertion on "the second one was faster" passes on a machine where both
    were slow for unrelated reasons.
    """
    solve_hartree_fock.cache_clear()
    config = aufbau_configuration(10)

    sample_hf_density(10, 10, 2, 1, 0, 2_000, config=config)
    hf_plane_grid(10, 10, 2, 1, 0, resolution=32, config=config)
    hf_isosurface(10, 10, 2, 1, 0, resolution=48, config=config)

    info = solve_hartree_fock.cache_info()
    assert info.misses == 1, f"expected one solve, got {info.misses}"
    assert info.hits > 0


def test_the_counterfactual_key_space_fits_the_cache():
    """One atom, both switches, two configurations: does anything evict?

    Eight slots against (atom, configuration, exchange, pauli). This records
    what a user actually reaches by flipping switches on the atom they are
    looking at, and fails if that sequence starts re-solving. If it ever does,
    raise maxsize with this test as the reason rather than as a precaution.
    """
    solve_hartree_fock.cache_clear()
    ground = aufbau_configuration(10)
    excited = aufbau_configuration(10)  # placeholder, replaced below
    from atomsim.atoms import parse_config

    excited = parse_config("1s2 2s2 2p5 3s1")
    collapsed = aufbau_configuration(10, pauli=False)

    keys = [
        (ground, True, True),
        (ground, False, True),
        (collapsed, False, False),
        (excited, True, True),
        (excited, False, True),
    ]
    for config, exchange, pauli in keys:
        solve_hartree_fock(10, 10, config, exchange, pauli)
    first_misses = solve_hartree_fock.cache_info().misses

    # Walk the same set again. Every one should now be a hit.
    for config, exchange, pauli in keys:
        solve_hartree_fock(10, 10, config, exchange, pauli)
    assert solve_hartree_fock.cache_info().misses == first_misses


@pytest.mark.parametrize("resolution", [96])
def test_isosurface_budget(resolution):
    """The expensive path, timed once with the solve already paid.

    96^3 plus the box fit plus the halved grid for the error bar, each point
    through evaluate_hf_state's interpolation. The solve is cached and the
    interpolation is vectorised, so this should sit near the screened path.
    """
    solve_hartree_fock(10, 10, aufbau_configuration(10))  # warm
    t0 = time.monotonic()
    surf = hf_isosurface(10, 10, 2, 1, 0, resolution=resolution)
    elapsed = time.monotonic() - t0
    assert surf.vertices.shape[0] > 0
    assert elapsed < 60.0, f"96^3 Hartree-Fock isosurface took {elapsed:.1f}s"
    print(f"\nHF isosurface {resolution}^3, warm solve: {elapsed:.2f}s")
```

Clean up the placeholder line: delete `excited = aufbau_configuration(10)  # placeholder, replaced below` and move the `parse_config` import to the top of the file with the others before committing.

- [ ] **Step 2: Run it and read the numbers**

Run: `pytest tests/test_hf_view_performance.py -v -s`
Expected: 3 passed. Record the printed isosurface time.

If `test_four_views_share_one_solve` fails with more than one miss, the cause is a caller passing `config=None` beside a caller passing an explicit equal configuration: `None` and the aufbau tuple are different cache keys upstream of `solve_hartree_fock` only if the defaults diverge, so check that `_occupied_orbital` resolves `None` before the call rather than after. Do not raise `maxsize` to paper over it.

If `test_the_counterfactual_key_space_fits_the_cache` fails, five keys evicted inside eight slots, which means something else in the process is populating the cache. Investigate before changing the number, and if the number does move, say in the comment above `lru_cache` which measurement moved it.

- [ ] **Step 3: Full suite**

Run: `pytest -q`
Expected: all pass. Note the wall time; the Hartree-Fock tests are the slow ones and this phase adds several.

- [ ] **Step 4: Lint**

Run: `ruff check .`
Expected: no findings.

- [ ] **Step 5: Update the design spec**

At the top of `docs/superpowers/specs/2026-08-03-phase26-hartree-fock-3d-design.md`, change `Status: designed, not implemented.` to `Status: implemented.` and append a short closing section recording what building it changed about the design. Candidates to check against what actually happened: the fidelity inheritance that section 6 assumed was already in place and was not; whether `_hf_view_target` ended up shared with the radial GET; the measured isosurface time against section 8's expectation that it would match the screened path; and the sulfur and chlorine gap noted at the top of this plan.

- [ ] **Step 6: Commit**

```bash
git add tests/test_hf_view_performance.py docs/superpowers/specs/2026-08-03-phase26-hartree-fock-3d-design.md
git commit -m "Record what the Hartree-Fock views cost and close out Phase 26"
```

- [ ] **Step 7: Confirm the tree is push-ready**

Run: `git status --short`
Expected: empty.

---

## Self-review notes

**Spec coverage.** Section 3 (configuration trap) is Task 1. Section 4 (four fields, `model` on metas, defaults) is Task 5. Section 5's engine list is Tasks 1 to 4; its server list is Tasks 5 and 6; its frontend list is Tasks 7 and 8; its URL line needs no work, confirmed by reading `urlState.ts` (`model`, `config`, `nox`, `nopauli` are all present, including the rule that `nopauli=1` carries `nox`). Section 6's provenance claim is Task 1 step 3 plus Task 8 step 3; its four refusals are Task 5 step 5 and Task 6 step 3. Section 7's tests are distributed across Tasks 1 to 6. Section 8 is Task 9.

**Two places this plan departs from the spec, both deliberate.**

1. Section 6's refusal 2 is written as "`pauli=false` for anything but 1s". This plan makes the check configuration-driven instead, so it refuses whatever the chosen configuration does not occupy and explains it in Pauli terms when the cap is off. That covers the case the spec names and also covers a hand-written collapsed configuration like `1s5 2s3`, which the HF endpoint already accepts and which a blanket 1s-only rule would refuse wrongly.

2. Section 6 says every Hartree-Fock 3-D view carries the orbital claim. This plan attaches it in `hf_radial`, which is upstream of all four views including the radial plot. A single orbital's radial curve is no more an observable than its 3-D shape, so the wider reach is correct rather than a spill.

**One thing the spec asserts that this plan cannot deliver:** section 1's "including sulfur and chlorine" is true of the engine and false of the application, because those two elements are not in `ATOM_KEYS`. Task 8's replacement sentence does not claim them, and Task 9 step 5 records the gap.
