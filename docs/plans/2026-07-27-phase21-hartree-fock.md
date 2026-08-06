# Phase 21: Hartree-Fock Implementation Plan

**Goal:** Replace the fitted Green-Sellin-Zachor screening table with a self-consistent restricted Hartree-Fock solver, giving parameter-free orbitals and a real variational total energy for every atom the engine currently supports plus S and Cl.

**Architecture:** Three new numerics modules plus one top-level orchestrator. The exchange operator is non-local, so the per-`l` eigenproblem stops being tridiagonal and moves to matrix-free LOBPCG preconditioned by the still-tridiagonal local part. An SCF loop with damped linear mixing wraps it, warm-started from GSZ. Total energy is assembled directly and then checked by two routes that share no code with the assembly.

**Tech Stack:** Python 3.12, numpy, scipy (`linalg.eigh_tridiagonal`, `linalg.cholesky_banded`, `linalg.cho_solve_banded`, `sparse.linalg.lobpcg`, `sparse.linalg.LinearOperator`), pytest, ruff.

## Global Constraints

- Hartree atomic units everywhere inside the engine. SI and eV conversions happen at the server boundary only and append to the provenance `method` string.
- Every physical value crossing a module boundary is a `Quantity` or `Field` carrying `Provenance`. A bare `float` crossing a boundary is a bug. Wigner symbols are the one sanctioned exception (see `analytic/wigner.py` docstring) and stay plain floats.
- `l` is the orbital angular-momentum quantum number, never a length. ruff E741 is ignored project-wide for this.
- ruff line-length 100.
- Run everything from the repo root in the conda env `atomsim`.
- Non-convergence raises `HFConvergenceError`. It never returns a result object with a false flag.
- No benchmark energy is written from memory. Every external number is transcribed from a cited source into `src/atomsim/data/` with a retrieval date.
- New physics gets a validation test (analytic ground truth, exact identity, or grid convergence), not a smoke test.
- Commit after every task.

## Deviation from the spec's file plan

The spec (section 9) put the angular coefficients and the SCF loop both in `numerics/hartree_fock.py`. This plan splits them: `numerics/hf_terms.py` holds the angular algebra and Fock-operator assembly, `numerics/hartree_fock.py` holds the preconditioner, eigensolve and SCF loop. Reason: the combined file lands around 400 lines against a repo whose numerics modules run 100 to 150 (`radial_solver.py` is 118, `screening.py` is 106), and the two halves have genuinely different test surfaces. Everything else in the spec's file plan is unchanged.

## The derived equations

The spec (section 2.4) declined to state the angular coefficients and required them to be derived by differentiating the average-of-configuration functional. That derivation is done, and is reproduced here because every task below depends on it. It must be copied into the `hf_terms.py` module docstring.

**Energy functional.** For subshells `a = (n_a, l_a)` with occupancy `q_a`:

```
E = sum_a q_a I(a)
  + sum_a  (q_a (q_a - 1) / 2) [ F0(aa)
        - sum_{k>0} ((2 l_a + 1)/(4 l_a + 1)) * tj(l_a,k,l_a)^2 * Fk(aa) ]
  + sum_{a<b} q_a q_b [ F0(ab)
        - (1/2) sum_k tj(l_a,k,l_b)^2 * Gk(ab) ]
```

writing `tj(l1,k,l2)` for `wigner_3j(l1, k, l2, 0, 0, 0)`. `k` runs over
`|l_a - l_b| .. l_a + l_b` and the 3j vanishes unless `l_a + k + l_b` is even, so
the selection rule is structural and needs no special-casing.

**Fock equation.** Varying `E` with respect to `P_a` and dividing by `2 q_a`:

```
h P_a
  + (q_a - 1) U0[a,a] P_a
  + sum_{b != a} q_b U0[b,b] P_a
  - (q_a - 1) sum_{k>0} ((2 l_a + 1)/(4 l_a + 1)) tj(l_a,k,l_a)^2 Uk[a,a] P_a
  - sum_{b != a} (q_b / 2) sum_k tj(l_a,k,l_b)^2 Uk[a,b] P_b
  = eps_a P_a
```

with `h = -1/2 d2/dr2 + l_a(l_a+1)/(2 r^2) - Z/r` and `U_k` the pair potential of
section 2.3 of the spec.

**Why this is trusted.** Four independent checks, all of which the tasks below turn into tests:

1. **Hydrogen.** `q_a = 1` kills the `(q_a - 1)` direct term, there is no `b`, and `l = 0` admits no `k > 0`. The equation collapses to `h P = eps P`, giving exactly `-0.5` hartree. This is the check that killed the first candidate convention, which left a one-electron atom feeling half its own Hartree potential.
2. **Helium.** `q_a = 2`, `l = 0`: the equation becomes `h P + U0[1s,1s] P = eps P`, exactly one unit of the other electron's Hartree potential and no self-interaction.
3. **Closed-shell agreement.** Deriving the closed-shell functional independently (summing `J - K` over full shells with the 3j sum rules, never averaging) gives coefficients `(2l_a + 1)` and `(2l_b + 1)` where the averaged form gives `(q_a - 1)(2l_a+1)/(4l_a+1)` and `q_b/2`. Substituting `q = 2(2l+1)` makes them identical. Two derivations, same answer.
4. **Beryllium.** Both routes give `4 F0(1s,2s) - 2 G0(1s,2s)` for the interaction between two doubly occupied s shells, which is the textbook `4J - 2K`.

---

### Task 1: Wigner 3j symbols

**Files:**
- Modify: `src/atomsim/analytic/wigner.py`
- Test: `tests/test_wigner_3j.py`

**Interfaces:**
- Consumes: `_doubled`, `_triangular_doubled`, `_delta` (already in the module).
- Produces: `wigner_3j(j1, j2, j3, m1, m2, m3) -> float`. Plain float, no provenance, matching the module's existing rationale for `wigner_6j`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_wigner_3j.py`:

```python
import math

import pytest

from atomsim.analytic.wigner import wigner_3j

def test_trivial_symbol_is_one():
    assert wigner_3j(0, 0, 0, 0, 0, 0) == pytest.approx(1.0)

@pytest.mark.parametrize(
    "args, expected",
    [
        ((1, 1, 0, 0, 0, 0), -1.0 / math.sqrt(3.0)),
        ((1, 0, 1, 0, 0, 0), -1.0 / math.sqrt(3.0)),
        ((2, 2, 0, 0, 0, 0), 1.0 / math.sqrt(5.0)),
        ((1, 2, 1, 0, 0, 0), 2.0 / math.sqrt(30.0)),
        ((2, 2, 2, 0, 0, 0), -math.sqrt(2.0 / 35.0)),
    ],
)
def test_closed_form_values(args, expected):
    """Cross-checked against the closed form for m1=m2=m3=0:

    3j = (-1)^g sqrt( (J-2j1)!(J-2j2)!(J-2j3)! / (J+1)! )
         * g! / [ (g-j1)!(g-j2)!(g-j3)! ],  J = j1+j2+j3 even, g = J/2.
    """
    assert wigner_3j(*args) == pytest.approx(expected, rel=1e-12)

def test_general_m_value():
    # 3j(1,1,0;1,-1,0) = (-1)^(j1-j2-m3)/sqrt(2j3+1) * <1,1;1,-1|0,0>
    #                  = 1 * 1 * (1/sqrt(3))
    assert wigner_3j(1, 1, 0, 1, -1, 0) == pytest.approx(1.0 / math.sqrt(3.0))

def test_zero_when_m_do_not_sum_to_zero():
    assert wigner_3j(1, 1, 0, 1, 0, 0) == 0.0

def test_zero_when_parity_forbids():
    # l1 + k + l2 odd: the (0,0,0) symbol must vanish identically.
    assert wigner_3j(1, 1, 1, 0, 0, 0) == 0.0

def test_zero_when_triangle_fails():
    assert wigner_3j(1, 1, 5, 0, 0, 0) == 0.0

def test_even_permutation_invariance():
    a = wigner_3j(2, 1, 1, 0, 0, 0)
    b = wigner_3j(1, 1, 2, 0, 0, 0)
    assert a == pytest.approx(b)

def test_odd_permutation_sign():
    # Swapping two columns multiplies by (-1)^(j1+j2+j3).
    a = wigner_3j(1, 2, 1, 0, 0, 0)
    b = wigner_3j(2, 1, 1, 0, 0, 0)
    assert a == pytest.approx((-1.0) ** (1 + 2 + 1) * b)

def test_orthogonality_sum_rule():
    # sum over m1,m2 of (2 j3 + 1) |3j(j1 j2 j3; m1 m2 -m1-m2)|^2 = 1
    j1, j2, j3 = 2, 1, 2
    total = 0.0
    for m1 in range(-j1, j1 + 1):
        for m2 in range(-j2, j2 + 1):
            total += (2 * j3 + 1) * wigner_3j(j1, j2, j3, m1, m2, -m1 - m2) ** 2
    assert total == pytest.approx(1.0)

def test_rejects_non_half_integer():
    with pytest.raises(ValueError):
        wigner_3j(0.3, 1, 1, 0, 0, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wigner_3j.py -v`
Expected: FAIL, `ImportError: cannot import name 'wigner_3j'`

- [ ] **Step 3: Write the implementation**

In `src/atomsim/analytic/wigner.py`, extend `__all__` to `["triangular", "wigner_3j", "wigner_6j"]` and add:

```python
def _doubled_m(m: float, name: str) -> int:
    """2m as an exact integer. Unlike _doubled, m may be negative."""
    two_m = round(2 * m)
    if abs(2 * m - two_m) > 1e-9:
        raise ValueError(f"{name} must be integer or half-integer, got {m}")
    return two_m

def wigner_3j(
    j1: float, j2: float, j3: float, m1: float, m2: float, m3: float
) -> float:
    """The 3j symbol (j1 j2 j3; m1 m2 m3), by the Racah formula.

    Returns exactly 0.0 when the projections do not sum to zero, when any
    |m_i| > j_i, or when the triangle condition fails, so the selection rules
    are structural rather than something the caller has to special-case.

    The Hartree-Fock angular coefficients need only the m1=m2=m3=0 case, where
    the symbol also vanishes unless j1+j2+j3 is even. That parity rule is not
    special-cased here: it falls out of the general formula.
    """
    j1_, j2_, j3_ = (_doubled(j, n) for j, n in ((j1, "j1"), (j2, "j2"), (j3, "j3")))
    m1_, m2_, m3_ = (_doubled_m(m, n) for m, n in ((m1, "m1"), (m2, "m2"), (m3, "m3")))

    if m1_ + m2_ + m3_ != 0:
        return 0.0
    for j, m in ((j1_, m1_), (j2_, m2_), (j3_, m3_)):
        if abs(m) > j or (j - m) % 2 != 0:  # m must share j's half-integrality
            return 0.0
    if not _triangular_doubled(j1_, j2_, j3_):
        return 0.0

    prefactor = _delta(j1_, j2_, j3_)
    for j, m in ((j1_, m1_), (j2_, m2_), (j3_, m3_)):
        prefactor *= math.sqrt(
            math.factorial((j + m) // 2) * math.factorial((j - m) // 2)
        )

    # Racah sum: t runs where every factorial argument stays non-negative.
    # The three lower bounds come from t >= 0, (j3-j2+m1)/2 + t >= 0 and
    # (j3-j1-m2)/2 + t >= 0; the three upper bounds from the remaining three
    # factorials.
    lower = max(0, -((j3_ - j2_ + m1_) // 2), -((j3_ - j1_ - m2_) // 2))
    upper = min(
        (j1_ + j2_ - j3_) // 2,
        (j1_ - m1_) // 2,
        (j2_ + m2_) // 2,
    )
    total = 0.0
    for t in range(lower, upper + 1):
        denom = (
            math.factorial(t)
            * math.factorial((j3_ - j2_ + m1_) // 2 + t)
            * math.factorial((j3_ - j1_ - m2_) // 2 + t)
            * math.factorial((j1_ + j2_ - j3_) // 2 - t)
            * math.factorial((j1_ - m1_) // 2 - t)
            * math.factorial((j2_ + m2_) // 2 - t)
        )
        total += (-1.0) ** t / denom

    sign = (-1.0) ** ((j1_ - j2_ - m3_) // 2)
    return sign * prefactor * total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_wigner_3j.py -v`
Expected: all PASS. If `test_orthogonality_sum_rule` fails while the closed-form values pass, the bug is in the `sign` or the summation range, not the prefactor.

- [ ] **Step 5: Lint and run the full suite**

Run: `ruff check . && pytest -q`
Expected: clean, no regressions.

- [ ] **Step 6: Commit**

```bash
git add src/atomsim/analytic/wigner.py tests/test_wigner_3j.py
git commit -m "Add Wigner 3j symbols alongside the existing 6j engine"
```

---

### Task 2: Slater radial integrals

**Files:**
- Create: `src/atomsim/numerics/slater.py`
- Test: `tests/test_slater.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `pair_potential(p_a, p_b, r, k) -> np.ndarray` returning `U_k[a,b](r)`
  - `slater_f(p_a, p_b, r, k) -> float` returning `F^k(ab)`
  - `slater_g(p_a, p_b, r, k) -> float` returning `G^k(ab)`

All take and return plain arrays and floats. This module is quadrature, not physics, so nothing here carries provenance; the caller in `hf_atom.py` assembles it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_slater.py`:

```python
import numpy as np
import pytest

from atomsim.numerics.slater import pair_potential, slater_f, slater_g

def hydrogenic_1s(r: np.ndarray, z: float) -> np.ndarray:
    """P_1s = r R_1s, normalized so that integral P^2 dr = 1."""
    return 2.0 * z**1.5 * r * np.exp(-z * r)

@pytest.fixture
def grid():
    return np.linspace(1e-6, 60.0, 60000)

def test_u0_goes_as_one_over_r_outside_the_charge(grid):
    p = hydrogenic_1s(grid, 1.0)
    u0 = pair_potential(p, p, grid, 0)
    far = grid > 30.0
    assert np.allclose(u0[far], 1.0 / grid[far], rtol=1e-6)

def test_u0_is_finite_at_the_origin(grid):
    p = hydrogenic_1s(grid, 1.0)
    u0 = pair_potential(p, p, grid, 0)
    assert np.isfinite(u0).all()
    assert u0[0] > 0.0

def test_pair_potential_is_symmetric_in_its_orbitals(grid):
    a = hydrogenic_1s(grid, 1.0)
    b = hydrogenic_1s(grid, 2.0)
    assert np.allclose(pair_potential(a, b, grid, 1), pair_potential(b, a, grid, 1))

def test_f0_of_hydrogenic_1s_matches_the_analytic_value(grid):
    # F0(1s,1s) = 5Z/8 hartree for a hydrogenic 1s pair.
    for z in (1.0, 2.0):
        p = hydrogenic_1s(grid, z)
        assert slater_f(p, p, grid, 0) == pytest.approx(5.0 * z / 8.0, rel=1e-5)

def test_g_equals_f_when_both_orbitals_are_the_same(grid):
    p = hydrogenic_1s(grid, 1.5)
    assert slater_g(p, p, grid, 0) == pytest.approx(slater_f(p, p, grid, 0), rel=1e-10)

def test_higher_k_pair_potential_decays_faster(grid):
    p = hydrogenic_1s(grid, 1.0)
    u0 = pair_potential(p, p, grid, 0)
    u2 = pair_potential(p, p, grid, 2)
    far = grid > 20.0
    assert np.all(u2[far] < u0[far])

def test_rejects_negative_k(grid):
    p = hydrogenic_1s(grid, 1.0)
    with pytest.raises(ValueError):
        pair_potential(p, p, grid, -1)
```

The `5Z/8` value is the standard hydrogenic result and is derivable in closed
form, so it is an analytic ground truth rather than a vendored number.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_slater.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'atomsim.numerics.slater'`

- [ ] **Step 3: Write the implementation**

Create `src/atomsim/numerics/slater.py`:

```python
"""Slater radial integrals for the electron-electron interaction.

The pair potential

    U_k[a,b](r) = integral_0^inf ( r_<^k / r_>^(k+1) ) P_a(s) P_b(s) ds
                = r^-(k+1) integral_0^r  s^k     P_a P_b ds
                + r^k      integral_r^inf s^-(k+1) P_a P_b ds

is everything two-electron in Hartree-Fock. Both halves are cumulative
trapezoid integrals, so a whole pair potential costs O(N) and no ODE solve.

Two properties make this self-checking and are asserted in tests: U_0 tends to
1/r beyond the charge for a normalized density, and U_k is symmetric under
exchange of its two orbital arguments.

Hartree atomic units. P = r R(r) throughout, normalized as integral P^2 dr = 1.
"""

import numpy as np

__all__ = ["pair_potential", "slater_f", "slater_g"]

def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative integral of y dx from x[0], same length as x, starting at 0."""
    increments = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    out = np.empty_like(y)
    out[0] = 0.0
    np.cumsum(increments, out=out[1:])
    return out

def pair_potential(
    p_a: np.ndarray, p_b: np.ndarray, r: np.ndarray, k: int
) -> np.ndarray:
    """U_k[a,b](r), the multipole-k potential of the pair density P_a P_b."""
    if k < 0:
        raise ValueError(f"multipole order k must be >= 0, got {k}")
    if not (p_a.shape == p_b.shape == r.shape):
        raise ValueError("orbitals and grid must have the same shape")

    density = p_a * p_b
    inner = _cumulative_trapezoid(density * r**k, r)
    outer_total = _cumulative_trapezoid(density * r ** (-(k + 1)), r)
    outer = outer_total[-1] - outer_total
    return inner / r ** (k + 1) + outer * r**k

def slater_f(p_a: np.ndarray, p_b: np.ndarray, r: np.ndarray, k: int) -> float:
    """F^k(ab) = integral P_a^2(r) U_k[b,b](r) dr."""
    return float(np.trapezoid(p_a**2 * pair_potential(p_b, p_b, r, k), r))

def slater_g(p_a: np.ndarray, p_b: np.ndarray, r: np.ndarray, k: int) -> float:
    """G^k(ab) = integral P_a(r) P_b(r) U_k[a,b](r) dr."""
    return float(np.trapezoid(p_a * p_b * pair_potential(p_a, p_b, r, k), r))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_slater.py -v`
Expected: all PASS. If `test_u0_goes_as_one_over_r_outside_the_charge` fails by a
constant factor, the bug is the `r**k` / `r**-(k+1)` split. If it fails only near
the box edge, the grid does not contain the charge and the fixture needs a larger
`r_max`, not a code change.

- [ ] **Step 5: Lint and commit**

```bash
ruff check .
pytest -q
git add src/atomsim/numerics/slater.py tests/test_slater.py
git commit -m "Add Slater pair potentials and F^k, G^k radial integrals"
```

---

### Task 3: Vendor the Hartree-Fock reference energies

**Files:**
- Create: `src/atomsim/data/hf_reference_energies.json`
- Create: `src/atomsim/data/__init__.py` entry point if the package does not already expose data loading (check `spectra.py` for the existing loader pattern first and reuse it).
- Test: `tests/test_hf_reference_data.py`

**Interfaces:**
- Produces: `load_hf_reference(symbol: str) -> dict` with keys `total_energy_hartree` and `citation`. Later tasks read benchmark energies only through this.

**This task requires the source document.** It is placed before the solver so
the benchmark tests in later tasks have data to read. If the source is not
accessible in the execution environment, complete the loader and the schema,
leave `values` empty, and the benchmark tests in Tasks 7 and 9 will skip with an
explicit reason rather than pass vacuously. Do not invent energies.

- [ ] **Step 1: Retrieve and transcribe the source**

Source: **Bunge, Barrientos and Bunge (1993), "Roothaan-Hartree-Fock ground-state
atomic wave functions: Slater-type orbital expansions and expectation values for
Z = 2 to 54", Atomic Data and Nuclear Data Tables 53, 113 to 162.**

Transcribe the ground-state total energies for He, Be, Ne, Mg and Ar. Record the
retrieval date. Match the metadata shape of the existing NIST files in
`src/atomsim/data/` (`species`, `citation`, `retrieved`, `note`, then the data).

```json
{
  "citation": "C. F. Bunge, J. A. Barrientos, A. V. Bunge, At. Data Nucl. Data Tables 53, 113 (1993)",
  "retrieved": "YYYY-MM-DD",
  "note": "Roothaan-Hartree-Fock (finite Slater-type basis). These lie slightly above the basis-set-free numerical HF limit a grid solver converges to; the gap is far below the 1e-4 relative tolerance the tests use.",
  "units": "hartree",
  "values": {
    "He": {"z": 2, "n_electrons": 2, "total_energy_hartree": null},
    "Be": {"z": 4, "n_electrons": 4, "total_energy_hartree": null},
    "Ne": {"z": 10, "n_electrons": 10, "total_energy_hartree": null},
    "Mg": {"z": 12, "n_electrons": 12, "total_energy_hartree": null},
    "Ar": {"z": 18, "n_electrons": 18, "total_energy_hartree": null}
  }
}
```

- [ ] **Step 2: Write the test**

Create `tests/test_hf_reference_data.py`:

```python
import pytest

from atomsim.hf_reference import HF_REFERENCE, load_hf_reference

def test_metadata_is_present_and_dated():
    assert HF_REFERENCE["citation"]
    assert HF_REFERENCE["retrieved"]
    assert HF_REFERENCE["units"] == "hartree"

@pytest.mark.parametrize("symbol,z", [("He", 2), ("Be", 4), ("Ne", 10),
                                      ("Mg", 12), ("Ar", 18)])
def test_every_benchmark_atom_has_an_entry(symbol, z):
    entry = load_hf_reference(symbol)
    assert entry["z"] == z

@pytest.mark.parametrize("symbol", ["He", "Be", "Ne", "Mg", "Ar"])
def test_energies_are_bound_and_ordered_by_z(symbol):
    entry = load_hf_reference(symbol)
    if entry["total_energy_hartree"] is None:
        pytest.skip("reference energies not yet transcribed from the source")
    assert entry["total_energy_hartree"] < 0.0

def test_energies_decrease_with_z():
    """A heavier atom is more tightly bound. Catches a transcription slip."""
    symbols = ["He", "Be", "Ne", "Mg", "Ar"]
    energies = [load_hf_reference(s)["total_energy_hartree"] for s in symbols]
    if any(e is None for e in energies):
        pytest.skip("reference energies not yet transcribed from the source")
    assert all(a > b for a, b in zip(energies, energies[1:], strict=True))

def test_unknown_symbol_raises():
    with pytest.raises(KeyError):
        load_hf_reference("Xx")
```

`test_energies_decrease_with_z` is the transcription guard: a digit dropped or a
row misread almost always breaks monotonicity.

- [ ] **Step 3: Write the loader**

Create `src/atomsim/hf_reference.py`, following the loader pattern already used
for the NIST files (read it in `spectra.py` first and match it):

```python
"""Vendored Hartree-Fock total energies, for validating the solver.

Never a live query: the file in data/ carries its own citation and retrieval
date, exactly like the NIST line lists.
"""

import json
from pathlib import Path

_PATH = Path(__file__).parent / "data" / "hf_reference_energies.json"

with _PATH.open(encoding="utf-8") as fh:
    HF_REFERENCE = json.load(fh)

__all__ = ["HF_REFERENCE", "load_hf_reference"]

def load_hf_reference(symbol: str) -> dict:
    """Reference entry for an element symbol. Raises KeyError if absent."""
    try:
        return HF_REFERENCE["values"][symbol]
    except KeyError as exc:
        raise KeyError(
            f"no vendored Hartree-Fock reference energy for {symbol!r}; "
            f"available: {sorted(HF_REFERENCE['values'])}"
        ) from exc
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_hf_reference_data.py -v`
Expected: all PASS, or the three energy tests SKIP with the transcription reason
if Step 1 could not be completed.

- [ ] **Step 5: Commit**

```bash
git add src/atomsim/data/hf_reference_energies.json src/atomsim/hf_reference.py tests/test_hf_reference_data.py
git commit -m "Vendor Hartree-Fock reference energies with citation and retrieval date"
```

---

### Task 4: Angular coefficients and the Fock operator

**Files:**
- Create: `src/atomsim/numerics/hf_terms.py`
- Test: `tests/test_hf_terms.py`

**Interfaces:**
- Consumes: `wigner_3j` (Task 1), `pair_potential` (Task 2).
- Produces:
  - `Subshell` frozen dataclass: `n: int`, `l: int`, `q: int`, `p: np.ndarray`
  - `direct_potential(subshells, a_index, r) -> np.ndarray`
  - `exchange_apply(subshells, a_index, psi, r) -> np.ndarray`
  - `exchange_coefficient(l_a, k, l_b, q_b) -> float`
  - `same_shell_coefficient(l_a, k, q_a) -> float`

`exchange_apply` takes an arbitrary trial vector `psi`, not just an occupied
orbital, because LOBPCG applies it to search directions.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hf_terms.py`:

```python
import numpy as np
import pytest

from atomsim.analytic.wigner import wigner_3j
from atomsim.numerics.hf_terms import (
    Subshell,
    direct_potential,
    exchange_apply,
    exchange_coefficient,
    same_shell_coefficient,
)
from atomsim.numerics.slater import pair_potential

def hydrogenic_1s(r, z):
    return 2.0 * z**1.5 * r * np.exp(-z * r)

@pytest.fixture
def grid():
    return np.linspace(1e-6, 40.0, 40000)

def test_hydrogen_feels_no_direct_potential(grid):
    """One electron: (q - 1) = 0, so there is no self-interaction. This is the
    check that rejected the first candidate convention."""
    shells = (Subshell(n=1, l=0, q=1, p=hydrogenic_1s(grid, 1.0)),)
    assert np.allclose(direct_potential(shells, 0, grid), 0.0)

def test_hydrogen_feels_no_exchange(grid):
    shells = (Subshell(n=1, l=0, q=1, p=hydrogenic_1s(grid, 1.0)),)
    psi = hydrogenic_1s(grid, 1.0)
    assert np.allclose(exchange_apply(shells, 0, psi, grid), 0.0)

def test_helium_direct_is_exactly_one_hartree_potential(grid):
    """q = 2, l = 0: the electron sees exactly one unit of U_0, no more."""
    p = hydrogenic_1s(grid, 2.0)
    shells = (Subshell(n=1, l=0, q=2, p=p),)
    assert np.allclose(
        direct_potential(shells, 0, grid), pair_potential(p, p, grid, 0)
    )

def test_helium_has_no_exchange_term(grid):
    """An s shell admits no k > 0, and the k = 0 self-exchange is already
    accounted for by the (q - 1) factor in the direct term."""
    p = hydrogenic_1s(grid, 2.0)
    shells = (Subshell(n=1, l=0, q=2, p=p),)
    assert np.allclose(exchange_apply(shells, 0, p, grid), 0.0)

def test_beryllium_cross_shell_exchange_is_one_unit_of_u0(grid):
    """For 1s2 2s2, the exchange on 1s is (q_b/2) * tj^2 * U_0[1s,2s] P_2s
    with q_b = 2 and tj(0,0,0)^2 = 1, so exactly U_0[1s,2s] P_2s."""
    p1 = hydrogenic_1s(grid, 4.0)
    p2 = np.sqrt(2.0) * grid * (1.0 - grid) * np.exp(-grid)  # a 2s-like trial
    p2 /= np.sqrt(np.trapezoid(p2**2, grid))
    shells = (Subshell(n=1, l=0, q=2, p=p1), Subshell(n=2, l=0, q=2, p=p2))
    got = exchange_apply(shells, 0, p1, grid)
    want = pair_potential(p1, p2, grid, 0) * p2
    assert np.allclose(got, want, rtol=1e-10)

def test_closed_shell_coefficients_match_the_independent_derivation():
    """Averaged coefficients must equal the closed-shell ones when q is full."""
    for l_a in (0, 1, 2):
        q_full = 2 * (2 * l_a + 1)
        for k in range(2, 2 * l_a + 1, 2):
            averaged = same_shell_coefficient(l_a, k, q_full)
            closed = (2 * l_a + 1) * wigner_3j(l_a, k, l_a, 0, 0, 0) ** 2
            assert averaged == pytest.approx(closed)

def test_cross_shell_coefficient_matches_closed_shell_form():
    for l_a, l_b in ((0, 1), (1, 1), (1, 2)):
        q_full_b = 2 * (2 * l_b + 1)
        for k in range(abs(l_a - l_b), l_a + l_b + 1):
            averaged = exchange_coefficient(l_a, k, l_b, q_full_b)
            closed = (2 * l_b + 1) * wigner_3j(l_a, k, l_b, 0, 0, 0) ** 2
            assert averaged == pytest.approx(closed)

def test_p_shell_f2_coefficient_is_the_slater_value():
    """For p^6, ((2l+1)/(4l+1)) * tj(1,2,1)^2 = (3/5) * (2/15) = 2/25 per pair,
    so the Fock-equation coefficient is (q - 1) * 2/25 = 5 * 2/25.

    tj(1,2,1) = 2/sqrt(30) from the closed form, cross-checked in
    tests/test_wigner_3j.py, so this number is derived and not recalled.
    """
    assert same_shell_coefficient(1, 2, 6) == pytest.approx(5.0 * 2.0 / 25.0)

def test_parity_forbidden_multipole_has_zero_coefficient():
    assert exchange_coefficient(0, 0, 1, 6) == 0.0  # l_a + k + l_b odd
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hf_terms.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'atomsim.numerics.hf_terms'`

- [ ] **Step 3: Write the implementation**

Create `src/atomsim/numerics/hf_terms.py`. The module docstring must carry the
derivation from the "The derived equations" section of this plan verbatim.

```python
"""Angular coefficients and Fock-operator terms for average-of-configuration HF.

Derived by varying the average-of-configuration energy functional with respect
to P_a. The functional is

    E = sum_a q_a I(a)
      + sum_a  (q_a (q_a - 1) / 2) [ F0(aa)
            - sum_{k>0} ((2 l_a + 1)/(4 l_a + 1)) tj(l_a,k,l_a)^2 Fk(aa) ]
      + sum_{a<b} q_a q_b [ F0(ab) - (1/2) sum_k tj(l_a,k,l_b)^2 Gk(ab) ]

writing tj(l1,k,l2) for wigner_3j(l1, k, l2, 0, 0, 0). Varying and dividing by
2 q_a gives the Fock equation

    h P_a + (q_a - 1) U0[a,a] P_a + sum_{b != a} q_b U0[b,b] P_a
          - (q_a - 1) sum_{k>0} ((2 l_a + 1)/(4 l_a + 1)) tj(l_a,k,l_a)^2 Uk[a,a] P_a
          - sum_{b != a} (q_b / 2) sum_k tj(l_a,k,l_b)^2 Uk[a,b] P_b
          = eps_a P_a

Four checks pin this, all of them tests in tests/test_hf_terms.py and
tests/test_hartree_fock.py: hydrogen has no self-interaction at all (the
(q_a - 1) factor), helium sees exactly one unit of U_0, the averaged
coefficients reduce to the independently derived closed-shell ones when q is
full, and beryllium gives the textbook 4J - 2K.

This is the part of the phase that fails silently if it is wrong: bad
coefficients produce converged, smooth, believable orbitals with the wrong
energy. Do not adjust a coefficient to make a benchmark match without
re-deriving it.

Hartree atomic units. P = r R(r), normalized as integral P^2 dr = 1.
"""

from dataclasses import dataclass

import numpy as np

from atomsim.analytic.wigner import wigner_3j
from atomsim.numerics.slater import pair_potential

__all__ = [
    "Subshell",
    "direct_potential",
    "exchange_apply",
    "exchange_coefficient",
    "same_shell_coefficient",
]

@dataclass(frozen=True)
class Subshell:
    """One (n, l) subshell with its occupancy and current radial function."""

    n: int
    l: int
    q: int
    p: np.ndarray  # P_nl on the solver grid

def same_shell_coefficient(l_a: int, k: int, q_a: int) -> float:
    """Coefficient of U_k[a,a] P_a in the Fock equation, for k > 0."""
    if k <= 0:
        raise ValueError(f"same-shell exchange needs k > 0, got {k}")
    tj = wigner_3j(l_a, k, l_a, 0, 0, 0)
    return (q_a - 1) * ((2 * l_a + 1) / (4 * l_a + 1)) * tj * tj

def exchange_coefficient(l_a: int, k: int, l_b: int, q_b: int) -> float:
    """Coefficient of U_k[a,b] P_b in the Fock equation, for b != a."""
    tj = wigner_3j(l_a, k, l_b, 0, 0, 0)
    return 0.5 * q_b * tj * tj

def direct_potential(
    subshells: tuple[Subshell, ...], a_index: int, r: np.ndarray
) -> np.ndarray:
    """The local Hartree potential seen by subshell a.

    (q_a - 1) U0[a,a] + sum_{b != a} q_b U0[b,b]. The (q_a - 1) is what makes a
    one-electron atom see nothing at all.
    """
    a = subshells[a_index]
    v = (a.q - 1) * pair_potential(a.p, a.p, r, 0)
    for i, b in enumerate(subshells):
        if i != a_index:
            v = v + b.q * pair_potential(b.p, b.p, r, 0)
    return v

def exchange_apply(
    subshells: tuple[Subshell, ...], a_index: int, psi: np.ndarray, r: np.ndarray
) -> np.ndarray:
    """Apply the non-local exchange operator for subshell a to a trial psi.

    psi is any function in the l_a channel, not only an occupied orbital:
    LOBPCG applies this to its search directions, so the pair potentials are
    rebuilt from psi on every call rather than cached from the last SCF step.
    """
    a = subshells[a_index]
    out = np.zeros_like(psi)

    for k in range(2, 2 * a.l + 1, 2):
        c = same_shell_coefficient(a.l, k, a.q)
        if c:
            out = out + c * pair_potential(a.p, psi, r, k) * a.p

    for i, b in enumerate(subshells):
        if i == a_index:
            continue
        for k in range(abs(a.l - b.l), a.l + b.l + 1):
            c = exchange_coefficient(a.l, k, b.l, b.q)
            if c:
                out = out + c * pair_potential(b.p, psi, r, k) * b.p

    return out
```

Note the same-shell loop starts at `k = 2` and steps by 2: `k = 0` is excluded
because the `(q_a - 1)` factor in `direct_potential` already accounts for it, and
odd `k` vanish by parity. Double-counting `k = 0` here is the most likely way to
break helium.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hf_terms.py -v`
Expected: all PASS.

If `test_hydrogen_feels_no_direct_potential` fails, `(q_a - 1)` was written as
`q_a`. If `test_helium_has_no_exchange_term` fails, the same-shell loop is
including `k = 0`.

- [ ] **Step 5: Lint and commit**

```bash
ruff check .
pytest -q
git add src/atomsim/numerics/hf_terms.py tests/test_hf_terms.py
git commit -m "Derive average-of-configuration Fock terms and pin them on H, He and Be"
```

---

### Task 5: Preconditioned matrix-free channel solve

**Files:**
- Create: `src/atomsim/numerics/hartree_fock.py`
- Test: `tests/test_hf_channel.py`

**Interfaces:**
- Consumes: `Subshell`, `direct_potential`, `exchange_apply` (Task 4).
- Produces:
  - `local_hamiltonian_bands(v_local, l, r) -> tuple[np.ndarray, np.ndarray]` giving the tridiagonal diagonal and off-diagonal
  - `fock_operator(subshells, a_index, v_nuclear, l, r) -> LinearOperator`
  - `solve_channel(subshells, a_index, v_nuclear, l, r, n_states, guess) -> ChannelSolution` with fields `energies`, `orbitals`, `iterations`
  - `ChannelSolution` frozen dataclass

The iteration count is returned rather than recorded in module state: the
preconditioner claim has to be falsifiable, and a module-level counter that
grows for the life of the process is a leak, not a diagnostic.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hf_channel.py`:

```python
import numpy as np
import pytest

from atomsim.numerics.hartree_fock import fock_operator, solve_channel
from atomsim.numerics.hf_terms import Subshell

@pytest.fixture
def grid():
    return np.linspace(1e-5, 40.0, 20000)

def coulomb(z):
    return lambda r: -z / r

def test_one_electron_channel_reproduces_hydrogen(grid):
    """With q = 1 there is no interaction at all, so this must return the
    analytic hydrogen levels: -1/2, -1/8, -1/18."""
    shells = (Subshell(n=1, l=0, q=1, p=np.zeros_like(grid)),)
    sol = solve_channel(shells, 0, coulomb(1.0), l=0, r=grid, n_states=3,
                        guess=None)
    assert sol.energies[0] == pytest.approx(-0.5, rel=2e-4)
    assert sol.energies[1] == pytest.approx(-0.125, rel=2e-4)
    assert sol.energies[2] == pytest.approx(-1.0 / 18.0, rel=2e-3)

def test_returned_orbitals_are_normalized(grid):
    shells = (Subshell(n=1, l=0, q=1, p=np.zeros_like(grid)),)
    sol = solve_channel(shells, 0, coulomb(1.0), l=0, r=grid, n_states=3,
                        guess=None)
    for u in sol.orbitals:
        assert np.trapezoid(u**2, grid) == pytest.approx(1.0, rel=1e-8)

def test_returned_orbitals_are_orthogonal(grid):
    shells = (Subshell(n=1, l=0, q=1, p=np.zeros_like(grid)),)
    sol = solve_channel(shells, 0, coulomb(1.0), l=0, r=grid, n_states=3,
                        guess=None)
    for i in range(3):
        for j in range(i + 1, 3):
            overlap = np.trapezoid(sol.orbitals[i] * sol.orbitals[j], grid)
            assert abs(overlap) < 1e-8

def test_fock_operator_is_symmetric_on_random_vectors(grid):
    """<x, F y> = <F x, y> to quadrature accuracy. The pair-potential
    quadrature is not exactly symmetric, so this is a tolerance, not an
    identity, and the tolerance is what justifies the explicit
    re-orthogonalization in solve_channel."""
    rng = np.random.default_rng(0)
    p = grid * np.exp(-grid)
    p /= np.sqrt(np.trapezoid(p**2, grid))
    shells = (Subshell(n=1, l=0, q=2, p=p),)
    op = fock_operator(shells, 0, coulomb(2.0), l=0, r=grid)
    x = rng.standard_normal(grid.size)
    y = rng.standard_normal(grid.size)
    left = float(x @ op.matvec(y))
    right = float(op.matvec(x) @ y)
    assert left == pytest.approx(right, rel=1e-6)

def test_exchange_actually_changes_the_answer(grid):
    """Guard against an exchange term that is silently zero: helium's 1s
    eigenvalue must sit well above the bare Z=2 hydrogenic -2.0."""
    p = 2.0 * 2.0**1.5 * grid * np.exp(-2.0 * grid)
    shells = (Subshell(n=1, l=0, q=2, p=p),)
    sol = solve_channel(shells, 0, coulomb(2.0), l=0, r=grid, n_states=1,
                        guess=None)
    assert -2.0 < sol.energies[0] < -0.5

def test_warm_start_does_not_cost_more_iterations(grid):
    """The preconditioner claim, made falsifiable: restarting from a converged
    orbital must not take more LOBPCG iterations than starting cold."""
    p = 2.0 * 2.0**1.5 * grid * np.exp(-2.0 * grid)
    shells = (Subshell(n=1, l=0, q=2, p=p),)
    cold = solve_channel(shells, 0, coulomb(2.0), l=0, r=grid, n_states=1,
                         guess=None)
    warm = solve_channel(shells, 0, coulomb(2.0), l=0, r=grid, n_states=1,
                         guess=cold.orbitals)
    assert warm.iterations <= cold.iterations

def test_iteration_count_shows_the_preconditioner_is_working(grid):
    """Unpreconditioned LOBPCG on this operator needs hundreds of iterations.
    If this ever climbs past ~50 the preconditioner has silently stopped
    being applied."""
    p = 2.0 * 2.0**1.5 * grid * np.exp(-2.0 * grid)
    shells = (Subshell(n=1, l=0, q=2, p=p),)
    sol = solve_channel(shells, 0, coulomb(2.0), l=0, r=grid, n_states=1,
                        guess=None)
    assert sol.iterations < 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hf_channel.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

Create `src/atomsim/numerics/hartree_fock.py`:

```python
"""Matrix-free, preconditioned solve of one l channel of the Fock operator.

Exchange is non-local, so the Fock matrix is dense and the tridiagonal
eigensolver that powers radial_solver.py cannot be used. Nor can a dense one:
the grid runs to N ~ 1e4 to 5e4 and a dense symmetric matrix at N = 2e4 is
3.2 GB before any factorization. Matrix-free iteration is mandatory here, not a
preference.

LOBPCG needs a preconditioner. The top of the finite-difference kinetic
spectrum is 2/h^2, which is 8e4 hartree at h = 0.005 and 2e6 at h = 0.001,
against valence level spacings of order 1 hartree; unpreconditioned iteration
stagnates. The preconditioner is free: the LOCAL part of the Fock operator is
still tridiagonal, so shifting it below the lowest sought eigenvalue makes it
positive definite and its inverse is a banded Cholesky solve in O(N). It works
because the local part carries all the high-frequency content while exchange is
a smooth integral kernel with a fast-decaying spectrum.

Hartree atomic units.
"""

from collections.abc import Callable

import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded, eigh_tridiagonal
from scipy.sparse.linalg import LinearOperator, lobpcg

from atomsim.numerics.hf_terms import Subshell, direct_potential, exchange_apply

__all__ = [
    "ChannelSolution",
    "HFConvergenceError",
    "fock_operator",
    "local_hamiltonian_bands",
    "solve_channel",
]

@dataclass(frozen=True)
class ChannelSolution:
    """Eigenpairs for one l channel, plus what it cost to get them."""

    energies: np.ndarray      # shape (n_states,)
    orbitals: np.ndarray      # shape (n_states, len(r))
    iterations: int           # LOBPCG iterations, so the preconditioner
                              # claim in the module docstring is falsifiable

class HFConvergenceError(RuntimeError):
    """The SCF loop or an inner eigensolve failed to converge.

    Raised rather than returning a result with converged=False: a plausible
    unconverged number is exactly the quiet lie the prime directive forbids.
    """

def local_hamiltonian_bands(
    v_local: np.ndarray, l: int, r: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonal and off-diagonal of the tridiagonal local Hamiltonian.

    Same 3-point discretization and same conventions as radial_solver.py, so
    the two engines agree where they overlap.
    """
    h = float(r[1] - r[0])
    inv2m = 0.5
    v_eff = v_local + l * (l + 1) * inv2m / r**2
    diag = 2.0 * inv2m / h**2 + v_eff
    offdiag = np.full(r.size - 1, -inv2m / h**2)
    return diag, offdiag

def fock_operator(
    subshells: tuple[Subshell, ...],
    a_index: int,
    v_nuclear: Callable[[np.ndarray], np.ndarray],
    l: int,
    r: np.ndarray,
) -> LinearOperator:
    """The Fock operator for subshell a as a matrix-free LinearOperator."""
    v_local = np.asarray(v_nuclear(r), dtype=float) + direct_potential(
        subshells, a_index, r
    )
    diag, offdiag = local_hamiltonian_bands(v_local, l, r)

    def matvec(psi: np.ndarray) -> np.ndarray:
        psi = np.asarray(psi, dtype=float).ravel()
        out = diag * psi
        out[:-1] += offdiag * psi[1:]
        out[1:] += offdiag * psi[:-1]
        return out - exchange_apply(subshells, a_index, psi, r)

    return LinearOperator((r.size, r.size), matvec=matvec, rmatvec=matvec,
                          dtype=float)

def _preconditioner(diag: np.ndarray, offdiag: np.ndarray) -> LinearOperator:
    """(H_local - sigma I)^-1 by banded Cholesky, sigma below the spectrum.

    sigma is placed one hartree below the lowest local eigenvalue so the shifted
    matrix is positive definite and the factorization needs no pivoting. The
    factorization is computed once and reused for every LOBPCG iteration.
    """
    lowest = eigh_tridiagonal(
        diag, offdiag, select="i", select_range=(0, 0), eigvals_only=True
    )[0]
    sigma = lowest - 1.0
    ab = np.zeros((2, diag.size))
    ab[0, 1:] = offdiag
    ab[1, :] = diag - sigma
    factor = cholesky_banded(ab, lower=False)

    def apply(x: np.ndarray) -> np.ndarray:
        return cho_solve_banded((factor, False), np.asarray(x, dtype=float).ravel())

    return LinearOperator((diag.size, diag.size), matvec=apply, dtype=float)

def _normalize(u: np.ndarray, r: np.ndarray) -> np.ndarray:
    return u / np.sqrt(np.trapezoid(u**2, r))

def solve_channel(
    subshells: tuple[Subshell, ...],
    a_index: int,
    v_nuclear: Callable[[np.ndarray], np.ndarray],
    l: int,
    r: np.ndarray,
    n_states: int,
    guess: np.ndarray | None = None,
    tol: float = 1e-9,
    maxiter: int = 400,
) -> ChannelSolution:
    """Lowest n_states eigenpairs of the Fock operator in this l channel.

    Orbitals come back shaped (n_states, len(r)), normalized to
    integral P^2 dr = 1, sign-fixed, and explicitly re-orthogonalized. The
    re-orthogonalization is not redundant: the pair
    potentials are quadratures, so the discrete operator is symmetric only to
    O(h^2) and LOBPCG's own orthogonality inherits that error.
    """
    op = fock_operator(subshells, a_index, v_nuclear, l, r)
    v_local = np.asarray(v_nuclear(r), dtype=float) + direct_potential(
        subshells, a_index, r
    )
    diag, offdiag = local_hamiltonian_bands(v_local, l, r)
    precond = _preconditioner(diag, offdiag)

    block = n_states + 2  # guard vectors: LOBPCG is least accurate at the top
    if guess is not None:
        x = np.asarray(guess, dtype=float).reshape(-1, r.size).T
        if x.shape[1] < block:
            extra = eigh_tridiagonal(
                diag, offdiag, select="i", select_range=(0, block - x.shape[1] - 1)
            )[1]
            x = np.hstack([x, extra])
    else:
        x = eigh_tridiagonal(diag, offdiag, select="i",
                             select_range=(0, block - 1))[1]

    eigenvalues, eigenvectors, history = lobpcg(
        op, x, M=precond, tol=tol, maxiter=maxiter, largest=False,
        retResidualNormsHistory=True,
    )
    if len(history) >= maxiter:
        raise HFConvergenceError(
            f"LOBPCG did not converge in {maxiter} iterations for l={l}; "
            f"final residual {float(np.max(history[-1])):.3e}"
        )

    order = np.argsort(eigenvalues)[:n_states]
    out = eigenvectors.T[order]

    # Modified Gram-Schmidt in the trapezoid inner product, then sign-fix,
    # matching radial_solver.py's convention.
    ortho = []
    for u in out:
        for v in ortho:
            u = u - v * np.trapezoid(u * v, r)
        u = _normalize(u, r)
        first = np.argmax(np.abs(u) > 0.01 * np.abs(u).max())
        if u[first] < 0:
            u = -u
        ortho.append(u)

    return ChannelSolution(
        energies=eigenvalues[order],
        orbitals=np.array(ortho),
        iterations=len(history),
    )
```

`dataclass` must be imported at the top of the module alongside the scipy
imports: `from dataclasses import dataclass`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hf_channel.py -v`
Expected: all PASS.

If `test_one_electron_channel_reproduces_hydrogen` fails on the third level only,
the box is too small for `n = 3`, not a code error: widen the fixture grid.

- [ ] **Step 5: Lint and commit**

```bash
ruff check .
pytest -q
git add src/atomsim/numerics/hartree_fock.py tests/test_hf_channel.py
git commit -m "Solve one Fock channel matrix-free with a tridiagonal preconditioner"
```

---

### Task 6: The SCF loop and the total energy, three ways

**Files:**
- Modify: `src/atomsim/numerics/hartree_fock.py`
- Test: `tests/test_hartree_fock.py`

**Interfaces:**
- Consumes: `solve_channel` (Task 5), `slater_f`, `slater_g` (Task 2), coefficients (Task 4).
- Produces:
  - `scf(z, subshells, v_nuclear, r, alpha=0.4, max_iterations=200, tol=1e-8) -> SCFSolution`
  - `SCFSolution` frozen dataclass: `subshells`, `energies`, `iterations`, `residual_history`
  - `total_energy_direct(z, subshells, r) -> float`
  - `total_energy_from_orbitals(subshells, energies, z, r) -> float`
  - `kinetic_and_potential(z, subshells, r) -> tuple[float, float]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_hartree_fock.py`:

```python
import numpy as np
import pytest

from atomsim.numerics.hartree_fock import (
    kinetic_and_potential,
    scf,
    total_energy_direct,
    total_energy_from_orbitals,
)
from atomsim.numerics.hf_terms import Subshell

def coulomb(z):
    return lambda r: -z / r

def start(grid, z, shells):
    """Hydrogenic warm start: P_1s at effective charge z."""
    p = 2.0 * z**1.5 * grid * np.exp(-z * grid)
    p /= np.sqrt(np.trapezoid(p**2, grid))
    return tuple(Subshell(n=n, l=l, q=q, p=p.copy()) for n, l, q in shells)

@pytest.fixture
def grid():
    return np.linspace(1e-5, 30.0, 30000)

def test_hydrogen_is_exactly_minus_one_half(grid):
    """One electron: HF must reduce to the bare Coulomb problem with no
    self-interaction whatsoever. This is the phase's sharpest anchor and it
    costs nothing."""
    sol = scf(1, start(grid, 1.0, [(1, 0, 1)]), coulomb(1.0), grid)
    assert total_energy_direct(1, sol.subshells, grid) == pytest.approx(
        -0.5, rel=2e-4
    )
    assert sol.energies[0] == pytest.approx(-0.5, rel=2e-4)

def test_helium_total_energy_is_physical(grid):
    """No vendored number needed to catch a gross error: helium must sit
    between the non-interacting limit (-4) and the single-ion limit (-2)."""
    sol = scf(2, start(grid, 1.7, [(1, 0, 2)]), coulomb(2.0), grid)
    e = total_energy_direct(2, sol.subshells, grid)
    assert -4.0 < e < -2.0

def test_the_two_energy_routes_agree(grid):
    """Direct assembly and the orbital identity E = 1/2 sum q (I + eps) are
    algebraically identical, so any disagreement is a coding error, not a
    numerical one. Tolerance is tight on purpose."""
    sol = scf(2, start(grid, 1.7, [(1, 0, 2)]), coulomb(2.0), grid)
    direct = total_energy_direct(2, sol.subshells, grid)
    identity = total_energy_from_orbitals(sol.subshells, sol.energies, 2, grid)
    assert direct == pytest.approx(identity, abs=1e-8)

def test_virial_ratio_is_two(grid):
    """At a converged HF solution in a pure Coulomb field, -V/T = 2 exactly.
    Departure measures grid and box error, not model error."""
    sol = scf(2, start(grid, 1.7, [(1, 0, 2)]), coulomb(2.0), grid)
    t, v = kinetic_and_potential(2, sol.subshells, grid)
    assert -v / t == pytest.approx(2.0, rel=1e-3)

def test_energy_equals_minus_kinetic_at_convergence(grid):
    sol = scf(2, start(grid, 1.7, [(1, 0, 2)]), coulomb(2.0), grid)
    t, _ = kinetic_and_potential(2, sol.subshells, grid)
    e = total_energy_direct(2, sol.subshells, grid)
    assert e == pytest.approx(-t, rel=1e-3)

def test_beryllium_converges_and_orders_its_shells(grid):
    sol = scf(4, start(grid, 3.0, [(1, 0, 2), (2, 0, 2)]), coulomb(4.0), grid)
    assert sol.energies[0] < sol.energies[1] < 0.0

def test_residual_history_is_monotone_enough_to_show_convergence(grid):
    sol = scf(2, start(grid, 1.7, [(1, 0, 2)]), coulomb(2.0), grid)
    assert sol.residual_history[-1] < sol.residual_history[0]
    assert sol.residual_history[-1] < 1e-8

def test_non_convergence_raises_rather_than_returning(grid):
    from atomsim.numerics.hartree_fock import HFConvergenceError

    with pytest.raises(HFConvergenceError):
        scf(2, start(grid, 1.7, [(1, 0, 2)]), coulomb(2.0), grid,
            max_iterations=1, tol=1e-14)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hartree_fock.py -v`
Expected: FAIL, `ImportError: cannot import name 'scf'`

- [ ] **Step 3: Write the implementation**

Append to `src/atomsim/numerics/hartree_fock.py`:

```python
@dataclass(frozen=True)
class SCFSolution:
    subshells: tuple[Subshell, ...]
    energies: tuple[float, ...]      # eps_a, aligned with subshells
    iterations: int
    residual_history: tuple[float, ...]

def one_electron_integral(
    subshell: Subshell, z: int, r: np.ndarray
) -> float:
    """I(a) = <P_a| -1/2 d2/dr2 + l(l+1)/(2r^2) - Z/r |P_a>."""
    p = subshell.p
    h = float(r[1] - r[0])
    second = np.zeros_like(p)
    second[1:-1] = (p[2:] - 2.0 * p[1:-1] + p[:-2]) / h**2
    kinetic = -0.5 * np.trapezoid(p * second, r)
    centrifugal = 0.5 * subshell.l * (subshell.l + 1) * np.trapezoid(
        (p / r) ** 2, r
    )
    nuclear = -z * np.trapezoid(p**2 / r, r)
    return float(kinetic + centrifugal + nuclear)

def _interaction_energy(subshells: tuple[Subshell, ...], r: np.ndarray) -> float:
    """The two-electron part of the average-of-configuration functional."""
    total = 0.0
    for i, a in enumerate(subshells):
        total += (a.q * (a.q - 1) / 2.0) * slater_f(a.p, a.p, r, 0)
        for k in range(2, 2 * a.l + 1, 2):
            tj = wigner_3j(a.l, k, a.l, 0, 0, 0)
            coeff = ((2 * a.l + 1) / (4 * a.l + 1)) * tj * tj
            total -= (a.q * (a.q - 1) / 2.0) * coeff * slater_f(a.p, a.p, r, k)
        for b in subshells[i + 1:]:
            total += a.q * b.q * slater_f(a.p, b.p, r, 0)
            for k in range(abs(a.l - b.l), a.l + b.l + 1):
                tj = wigner_3j(a.l, k, b.l, 0, 0, 0)
                total -= 0.5 * a.q * b.q * tj * tj * slater_g(a.p, b.p, r, k)
    return float(total)

def total_energy_direct(
    z: int, subshells: tuple[Subshell, ...], r: np.ndarray
) -> float:
    """Route 1: assemble the energy functional term by term."""
    one = sum(a.q * one_electron_integral(a, z, r) for a in subshells)
    return float(one + _interaction_energy(subshells, r))

def total_energy_from_orbitals(
    subshells: tuple[Subshell, ...],
    energies: tuple[float, ...],
    z: int,
    r: np.ndarray,
) -> float:
    """Route 2: E = 1/2 sum_a q_a ( I(a) + eps_a ).

    Algebraically identical to route 1 but shares no code with it beyond the
    one-electron integral, so a coefficient error in _interaction_energy shows
    up as a disagreement rather than as a wrong number in both.
    """
    return float(
        0.5
        * sum(
            a.q * (one_electron_integral(a, z, r) + e)
            for a, e in zip(subshells, energies, strict=True)
        )
    )

def kinetic_and_potential(
    z: int, subshells: tuple[Subshell, ...], r: np.ndarray
) -> tuple[float, float]:
    """Route 3's inputs: total T and total V, for the virial ratio -V/T = 2."""
    h = float(r[1] - r[0])
    kinetic = 0.0
    for a in subshells:
        second = np.zeros_like(a.p)
        second[1:-1] = (a.p[2:] - 2.0 * a.p[1:-1] + a.p[:-2]) / h**2
        kinetic += a.q * (
            -0.5 * np.trapezoid(a.p * second, r)
            + 0.5 * a.l * (a.l + 1) * np.trapezoid((a.p / r) ** 2, r)
        )
    nuclear = sum(
        -z * a.q * np.trapezoid(a.p**2 / r, r) for a in subshells
    )
    return float(kinetic), float(nuclear + _interaction_energy(subshells, r))

def scf(
    z: int,
    subshells: tuple[Subshell, ...],
    v_nuclear: Callable[[np.ndarray], np.ndarray],
    r: np.ndarray,
    alpha: float = 0.4,
    max_iterations: int = 200,
    tol: float = 1e-8,
) -> SCFSolution:
    """Self-consistent field loop with damped linear mixing.

    Undamped iteration oscillates on atoms with a diffuse valence shell, so the
    new orbitals are mixed into the old at alpha rather than replacing them.
    alpha = 0.4 is a starting value tuned against measured iteration counts, not
    a claim about the physics.

    Raises HFConvergenceError rather than returning an unconverged solution.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"mixing parameter must be in (0, 1], got {alpha}")

    current = tuple(subshells)
    energies = tuple(0.0 for _ in current)
    residuals: list[float] = []

    for iteration in range(1, max_iterations + 1):
        updated: list[Subshell] = []
        new_energies: list[float] = []
        for index, a in enumerate(current):
            k = a.n - a.l - 1
            channel = solve_channel(
                current, index, v_nuclear, a.l, r, n_states=k + 1,
                guess=a.p[None, :],
            )
            mixed = (1.0 - alpha) * a.p + alpha * channel.orbitals[k]
            mixed = _normalize(mixed, r)
            updated.append(Subshell(n=a.n, l=a.l, q=a.q, p=mixed))
            new_energies.append(float(channel.energies[k]))

        residual = max(
            abs(new - old) for new, old in zip(new_energies, energies, strict=True)
        )
        residuals.append(residual)
        current, energies = tuple(updated), tuple(new_energies)
        if residual < tol and iteration > 1:
            return SCFSolution(current, energies, iteration, tuple(residuals))

    raise HFConvergenceError(
        f"SCF did not converge in {max_iterations} iterations for Z={z}; "
        f"last orbital-energy change {residuals[-1]:.3e} hartree"
    )
```

Add the needed imports at the top of the module: `from atomsim.analytic.wigner
import wigner_3j` and `from atomsim.numerics.slater import slater_f, slater_g`.
`dataclass` is already imported for `ChannelSolution` in Task 5.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hartree_fock.py -v`
Expected: all PASS.

`test_the_two_energy_routes_agree` is the important one. If it fails, the bug is
a coefficient in `_interaction_energy`, because route 2 does not use those
coefficients at all. Re-derive; do not tune.

- [ ] **Step 5: Lint and commit**

```bash
ruff check .
pytest -q
git add src/atomsim/numerics/hartree_fock.py tests/test_hartree_fock.py
git commit -m "Add the SCF loop and check the total energy three independent ways"
```

---

### Task 7: Closed-shell atoms end to end, against the vendored energies

**Files:**
- Create: `src/atomsim/hf_atom.py`
- Test: `tests/test_hf_atom.py`

**Interfaces:**
- Consumes: `scf` and the energy routes (Task 6), `load_hf_reference` (Task 3), `Configuration`, `is_ground`, `aufbau_configuration` from `atoms.py`, `screened_potential` from `numerics/screening.py` for the warm start.
- Produces:
  - `HFOrbital`, `HFResult` (exactly as in spec section 4)
  - `solve_hartree_fock(z, n_electrons, config) -> HFResult`
  - `hf_grid(z, n_electrons, n_top) -> tuple[float, int]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_hf_atom.py`:

```python
import pytest

from atomsim.atoms import aufbau_configuration
from atomsim.hf_atom import solve_hartree_fock
from atomsim.hf_reference import load_hf_reference
from atomsim.provenance import Fidelity

CLOSED_SHELL = [("He", 2), ("Be", 4), ("Ne", 10), ("Mg", 12), ("Ar", 18)]

@pytest.fixture(scope="module")
def solved():
    return {
        symbol: solve_hartree_fock(z, z, aufbau_configuration(z))
        for symbol, z in CLOSED_SHELL
    }

@pytest.mark.parametrize("symbol,z", CLOSED_SHELL)
def test_total_energy_matches_the_vendored_reference(solved, symbol, z):
    reference = load_hf_reference(symbol)["total_energy_hartree"]
    if reference is None:
        pytest.skip("reference energies not yet transcribed from the source")
    got = solved[symbol].total_energy.value
    assert got == pytest.approx(reference, rel=1e-4)

@pytest.mark.parametrize("symbol,z", CLOSED_SHELL)
def test_energy_is_a_variational_upper_bound(solved, symbol, z):
    """HF sits above the exact non-relativistic energy, never below. If the
    computed energy drops below the reference by more than the tolerance, the
    functional is wrong, not merely inaccurate."""
    reference = load_hf_reference(symbol)["total_energy_hartree"]
    if reference is None:
        pytest.skip("reference energies not yet transcribed from the source")
    assert solved[symbol].total_energy.value > reference - 1e-3

@pytest.mark.parametrize("symbol,z", CLOSED_SHELL)
def test_virial_ratio_is_near_two(solved, symbol, z):
    assert solved[symbol].virial_ratio.value == pytest.approx(2.0, rel=2e-3)

@pytest.mark.parametrize("symbol,z", CLOSED_SHELL)
def test_total_energy_is_approximation_with_a_numerical_sub_scale(solved, symbol, z):
    energy = solved[symbol].total_energy
    assert energy.provenance.fidelity is Fidelity.APPROXIMATION
    assert energy.provenance.error_estimate is not None
    joined = " ".join(energy.provenance.assumptions)
    assert "correlation" in joined
    assert "variational" in joined

@pytest.mark.parametrize("symbol,z", CLOSED_SHELL)
def test_virial_ratio_is_numerical_not_approximation(solved, symbol, z):
    """A convergence diagnostic is a statement about the solve, not the atom."""
    assert solved[symbol].virial_ratio.provenance.fidelity is Fidelity.NUMERICAL

def test_hydrogen_is_exact_to_the_grid():
    result = solve_hartree_fock(1, 1, aufbau_configuration(1))
    assert result.total_energy.value == pytest.approx(-0.5, rel=1e-4)

def test_orbital_energies_are_ordered_by_shell(solved):
    energies = [o.energy.value for o in solved["Ar"].orbitals if o.occupancy > 0]
    assert energies == sorted(energies)

def test_result_records_its_convergence(solved):
    result = solved["Ne"]
    assert result.converged is True
    assert result.iterations > 1
    assert len(result.residual_history) == result.iterations
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hf_atom.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'atomsim.hf_atom'`

- [ ] **Step 3: Write the implementation**

Create `src/atomsim/hf_atom.py`. Key pieces:

```python
def hf_grid(z: int, n_electrons: int, n_top: int) -> tuple[float, int]:
    """Box radius and point count for a Hartree-Fock solve.

    Two competing scales. The 1s core contracts as 1/Z, so the step must scale
    as 1/Z to hold the relative discretization error fixed: the error in the
    core eigenvalue goes as (h Z)^2. The valence extends as n^2 / Z_net, which
    sets the box.

    This is deliberately much tighter than screened_atom._r_max, whose
    40 (n_top+1)^2 box would spend most of a Hartree-Fock grid on vacuum. It is
    also the reason this phase stops at Z = 18: holding h Z fixed on a UNIFORM
    grid becomes unaffordable well before Z = 54, and a logarithmic mesh is the
    real fix (Phase 22).
    """
    z_net = max(z - n_electrons + 1, 1)
    r_max = min(40.0, 8.0 * (n_top + 1) ** 2 / z_net)
    h = 0.01 / z
    return r_max, int(r_max / h)
```

The rest of the module:

- Build the initial `Subshell` tuple from `config`, warm-starting each `P_nl`
  from the GSZ solution via `screened_potential(z, n_electrons)` and
  `solve_radial` when `gsz_parameters` has an entry, and from a hydrogenic
  `Z_eff = z - n_electrons + 1` guess otherwise (this is what lets S and Cl
  work at all).
- Call `scf`.
- Assemble `total_energy` by `total_energy_direct`, cross-check against
  `total_energy_from_orbitals`, and raise `HFConvergenceError` if they differ by
  more than `1e-6` hartree, since that indicates a coding error rather than a
  physics one.
- Compute `kinetic`, `potential`, `virial_ratio` via `kinetic_and_potential`.
- Re-solve on a doubled grid to get the grid-halving `error_estimate`, matching
  `solve_radial_with_error`'s pattern.

Provenance, exactly per spec section 3:

```python
_TOTAL_ENERGY_PROV = Provenance(
    fidelity=Fidelity.APPROXIMATION,
    method=(
        "self-consistent restricted Hartree-Fock, average of configuration; "
        "matrix-free preconditioned LOBPCG on a uniform radial grid"
    ),
    assumptions=(
        "no electron correlation; variational, so E_HF >= E_exact "
        "(non-relativistic, infinite nuclear mass)",
        "average of configuration: one energy per configuration, not per term",
        "infinite nuclear mass (mu_ratio = 1)",
    ),
    error_estimate=None,  # filled in from grid-halving at construction
    refinement=(
        "configuration interaction or many-body perturbation theory would "
        "recover the correlation energy"
    ),
)
```

`virial_ratio`, `kinetic` and `potential` get `Fidelity.NUMERICAL` with method
`"property of the converged solution, not a claim about the atom"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hf_atom.py -v`
Expected: PASS, with the reference-energy tests skipping if Task 3 could not
transcribe the source.

If argon misses the 1e-4 relative tolerance while helium and beryllium pass, that
is the uniform-grid ceiling from spec section 1.1, not a physics bug. Record the
measured error, and raise it with the user before changing tolerances: the
decision to pull the logarithmic mesh forward from Phase 22 is theirs.

- [ ] **Step 5: Commit**

```bash
git add src/atomsim/hf_atom.py tests/test_hf_atom.py
git commit -m "Solve closed-shell atoms end to end and check them against vendored HF energies"
```

---

### Task 8: The screened-atom API surface, NIST and cross-model checks

**Files:**
- Modify: `src/atomsim/hf_atom.py`
- Modify: `src/atomsim/numerics/screening.py:105` (the `refinement` string)
- Test: `tests/test_hf_atom_api.py`

**Interfaces:**
- Produces, mirroring `screened_atom.py` one for one so callers switch models by swapping a function rather than a call shape:
  - `hf_valence_ionization_energy(result) -> Quantity`
  - `hf_radial(z, n_electrons, n, l, points=400) -> tuple[Field, Field]`
  - `evaluate_hf_state(z, n_electrons, n, l, m, positions, *, basis="complex") -> WavefunctionValues`

- [ ] **Step 1: Write the failing test**

Create `tests/test_hf_atom_api.py`:

```python
import numpy as np
import pytest

from atomsim.atoms import aufbau_configuration
from atomsim.hf_atom import (
    evaluate_hf_state,
    hf_radial,
    hf_valence_ionization_energy,
    solve_hartree_fock,
)
from atomsim.screened_atom import solve_screened_atom, valence_ionization_energy

HARTREE_TO_EV = 27.211386245981

# NIST ASD first ionization energies, retrieved 2026-07-18, already vendored
# for the screened-atom tests.
NIST_IE_EV = {"He": (2, 24.587), "Li": (3, 5.392), "Na": (11, 5.139)}

@pytest.mark.parametrize("symbol", list(NIST_IE_EV))
def test_hartree_fock_is_at_least_as_good_as_gsz_on_nist(symbol):
    """The comparison that carries external weight.

    Note what the GSZ leg of this does NOT mean: Szydlik and Green fitted their
    (d, K) parameters TO Hartree-Fock, so agreement between the two models is a
    check on this implementation, not independent confirmation of the physics.
    The NIST column is the one with outside authority.
    """
    z, reference = NIST_IE_EV[symbol]
    config = aufbau_configuration(z)
    hf = hf_valence_ionization_energy(solve_hartree_fock(z, z, config))
    gsz = valence_ionization_energy(solve_screened_atom(z, z, config))
    hf_error = abs(hf.value * HARTREE_TO_EV - reference)
    gsz_error = abs(gsz.value * HARTREE_TO_EV - reference)
    assert hf_error <= gsz_error * 1.05

def test_hf_radial_returns_fields_with_matching_grids():
    r_field, p_field = hf_radial(2, 2, 1, 0)
    assert r_field.grid.shape == p_field.grid.shape
    assert np.allclose(p_field.values, r_field.grid**2 * r_field.values**2)

def test_radial_density_integrates_to_one():
    r_field, _ = hf_radial(2, 2, 1, 0)
    norm = np.trapezoid((r_field.grid * r_field.values) ** 2, r_field.grid)
    assert norm == pytest.approx(1.0, rel=1e-3)

def test_hf_radial_rejects_n_not_greater_than_l():
    with pytest.raises(ValueError):
        hf_radial(10, 10, 1, 1)

def test_evaluate_hf_state_shape_and_provenance():
    positions = np.array([[0.5, 0.0, 0.0], [0.0, 1.0, 0.0]])
    values = evaluate_hf_state(10, 10, 2, 1, 0, positions)
    assert values.values.shape == (2,)
    assert "Hartree-Fock" in values.provenance.method

def test_screening_refinement_now_points_at_the_implementation():
    from atomsim.numerics.screening import screening_provenance

    refinement = screening_provenance(10, 10).refinement
    assert "hf_atom" in refinement
    assert "a later phase" not in refinement
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hf_atom_api.py -v`
Expected: FAIL on the import of `hf_radial`.

- [ ] **Step 3: Implement, mirroring `screened_atom.py`**

Write `hf_valence_ionization_energy`, `hf_radial` and `evaluate_hf_state` as
direct analogues of `valence_ionization_energy`, `screened_radial` and
`evaluate_screened_state` in `src/atomsim/screened_atom.py:117-279`. Read that
file first and match its structure, including the `lru_cache` pattern for
repeated solves, since an SCF solve is far more expensive than a GSZ one and the
cache matters more here.

Then update `src/atomsim/numerics/screening.py:105`:

```python
        refinement="self-consistent Hartree-Fock in hf_atom.py removes the model error",
```

- [ ] **Step 4: Run tests, then the full suite**

Run: `pytest tests/test_hf_atom_api.py -v && pytest -q`
Expected: all PASS, no regressions in the existing screened-atom tests.

- [ ] **Step 5: Commit**

```bash
git add src/atomsim/hf_atom.py src/atomsim/numerics/screening.py tests/test_hf_atom_api.py
git commit -m "Mirror the screened-atom API on Hartree-Fock and redeem the screening refinement note"
```

---

### Task 9: Open shells, including S and Cl

**Files:**
- Modify: `src/atomsim/hf_atom.py`
- Modify: `src/atomsim/atoms.py` (the `ATOM_KEYS` list and the `NO_GSZ_PARAMETERS` docstring)
- Test: `tests/test_hf_open_shell.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_hf_open_shell.py`:

```python
import pytest

from atomsim.atoms import NO_GSZ_PARAMETERS, aufbau_configuration
from atomsim.hf_atom import solve_hartree_fock

OPEN_SHELL = [("Li", 3), ("B", 5), ("C", 6), ("N", 7), ("O", 8), ("F", 9),
              ("Na", 11), ("Al", 13), ("Si", 14), ("P", 15)]
NO_GSZ = [("S", 16), ("Cl", 17)]

@pytest.mark.parametrize("symbol,z", OPEN_SHELL + NO_GSZ)
def test_open_shell_atoms_converge(symbol, z):
    result = solve_hartree_fock(z, z, aufbau_configuration(z))
    assert result.converged
    assert result.total_energy.value < 0.0

@pytest.mark.parametrize("symbol,z", NO_GSZ)
def test_atoms_gsz_cannot_do_now_work(symbol, z):
    """S and Cl have no Szydlik-Green parameters and the screened model refuses
    them. Hartree-Fock needs no table, so this is the visible payoff."""
    assert z in NO_GSZ_PARAMETERS
    result = solve_hartree_fock(z, z, aufbau_configuration(z))
    assert result.total_energy.value < 0.0

@pytest.mark.parametrize("symbol,z", OPEN_SHELL + NO_GSZ)
def test_total_energy_decreases_monotonically_with_z(symbol, z):
    """A heavier atom binds more tightly. Catches a configuration mis-build."""
    lighter = solve_hartree_fock(z - 1, z - 1, aufbau_configuration(z - 1))
    heavier = solve_hartree_fock(z, z, aufbau_configuration(z))
    assert heavier.total_energy.value < lighter.total_energy.value

@pytest.mark.parametrize("symbol,z", OPEN_SHELL)
def test_open_shell_discloses_the_configuration_average(symbol, z):
    result = solve_hartree_fock(z, z, aufbau_configuration(z))
    joined = " ".join(result.total_energy.provenance.assumptions)
    assert "average of configuration" in joined

def test_closed_shell_does_not_claim_a_term_limitation_it_does_not_have():
    """Neon has no partially filled subshell, so there is nothing to average
    over and the disclosure would be misleading."""
    result = solve_hartree_fock(10, 10, aufbau_configuration(10))
    joined = " ".join(result.total_energy.provenance.assumptions)
    assert "one energy per configuration, not per term" not in joined

def test_virial_ratio_holds_for_open_shells():
    result = solve_hartree_fock(7, 7, aufbau_configuration(7))
    assert result.virial_ratio.value == pytest.approx(2.0, rel=5e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hf_open_shell.py -v`
Expected: FAIL. Most likely SCF non-convergence on the diffuse alkali valence
shells, or a wrong disclosure string.

- [ ] **Step 3: Implement**

The functional and the Fock terms from Task 4 already handle fractional
occupancy: nothing in `hf_terms.py` assumes a full shell. The work here is:

1. Make the provenance assumptions conditional. Add the "average of
   configuration: one energy per configuration, not per term" string only when
   some subshell has `0 < q < 2(2l+1)`.
2. Tune `alpha` per atom if the alkalis oscillate. Start at `0.4`, drop to `0.2`
   for `Z` in `{3, 11}`, and record the value in the provenance `assumptions` so
   the choice is visible rather than hidden.
3. Add S and Cl to `ATOM_KEYS` in `atoms.py`, and extend the
   `NO_GSZ_PARAMETERS` docstring to say the set now bounds only the GSZ model,
   not the engine.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_hf_open_shell.py -v`
Expected: all PASS. These are slow; consider `-n auto` if pytest-xdist is
available, otherwise expect a few minutes.

- [ ] **Step 5: Commit**

```bash
git add src/atomsim/hf_atom.py src/atomsim/atoms.py tests/test_hf_open_shell.py
git commit -m "Extend Hartree-Fock to open shells and unlock sulfur and chlorine"
```

---

### Task 10: Performance guard

**Files:**
- Test: `tests/test_hf_performance.py`

Phase 16 cut screened line strengths from 13.4s to 1.9s by fixing box size and
grid sharing. Hartree-Fock puts an SCF loop on top of the same eigenproblem, so
the same traps are back and larger.

- [ ] **Step 1: Write the test**

```python
import time

import pytest

from atomsim.atoms import aufbau_configuration
from atomsim.hf_atom import solve_hartree_fock

def test_argon_solves_within_the_budget():
    """Generous on purpose: this catches a regression, it is not a benchmark.
    Record the measured time in the failure message so a slow run is diagnosable
    rather than just red."""
    start = time.perf_counter()
    solve_hartree_fock(18, 18, aufbau_configuration(18))
    elapsed = time.perf_counter() - start
    assert elapsed < 60.0, f"argon took {elapsed:.1f}s"

def test_repeated_solves_hit_the_cache():
    solve_hartree_fock(10, 10, aufbau_configuration(10))
    start = time.perf_counter()
    solve_hartree_fock(10, 10, aufbau_configuration(10))
    assert time.perf_counter() - start < 0.1

def test_iteration_counts_stay_bounded():
    """The preconditioner claim, made falsifiable."""
    result = solve_hartree_fock(18, 18, aufbau_configuration(18))
    assert result.iterations < 100
```

- [ ] **Step 2: Run and record**

Run: `pytest tests/test_hf_performance.py -v`
Record the measured argon time in the commit message. If it exceeds 60s, do not
raise the threshold: profile first, and check whether `hf_grid`'s `h = 0.01 / z`
is finer than the accuracy tests actually need.

- [ ] **Step 3: Commit**

```bash
git add tests/test_hf_performance.py
git commit -m "Guard Hartree-Fock solve time and LOBPCG iteration counts"
```

---

### Task 11: Server surface

**Files:**
- Modify: `src/atomsim/server/schemas.py`
- Modify: `src/atomsim/server/app.py`
- Test: `tests/test_server_hf.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from fastapi.testclient import TestClient

from atomsim.server.app import app

client = TestClient(app)

def test_hartree_fock_runs_as_a_job_not_a_blocking_request():
    response = client.post("/api/hf", json={"z": 2, "n_electrons": 2})
    assert response.status_code == 202
    assert "job_id" in response.json()

def test_result_carries_provenance_to_the_browser():
    job = client.post("/api/hf", json={"z": 2, "n_electrons": 2}).json()
    result = client.get(f"/api/hf/{job['job_id']}").json()
    assert result["total_energy"]["provenance"]["fidelity"] == "APPROXIMATION"
    assert result["virial_ratio"]["provenance"]["fidelity"] == "NUMERICAL"

def test_convergence_fields_reach_the_client():
    job = client.post("/api/hf", json={"z": 2, "n_electrons": 2}).json()
    result = client.get(f"/api/hf/{job['job_id']}").json()
    assert result["iterations"] > 0
    assert result["converged"] is True

def test_unsupported_z_is_rejected_with_a_reason():
    response = client.post("/api/hf", json={"z": 30, "n_electrons": 30})
    assert response.status_code == 400
    assert "uniform grid" in response.json()["detail"].lower()
```

The last test matters: refusing Z > 18 with an explanation is the honest
behaviour, exactly as `gsz_parameters` refuses S and Cl rather than inventing
values.

- [ ] **Step 2: Implement**

Add `HFResultModel` and `HFOrbitalModel` to `schemas.py`, mapping `Quantity` and
`Provenance` through the existing converters. Add the `POST /api/hf` and
`GET /api/hf/{job_id}` endpoints to `app.py`, running the solve through
`jobs.py` exactly as sampling and plane grids do. Read the existing job
endpoints first and match them.

- [ ] **Step 3: Run and commit**

```bash
pytest tests/test_server_hf.py -v
ruff check .
git add src/atomsim/server/schemas.py src/atomsim/server/app.py tests/test_server_hf.py
git commit -m "Expose Hartree-Fock as a background job with provenance intact"
```

---

### Task 12: Frontend model selector

**Files:**
- Modify: `web/src/state/store.ts`
- Modify: `web/src/lib/urlState.ts`
- Modify: `web/src/api/` client and the screened-atom views
- Test: `web/src/lib/urlState.test.ts`

- [ ] **Step 1: Write the failing test**

Add to `web/src/lib/urlState.test.ts`:

```typescript
describe('model selection', () => {
  it('round-trips the model key', () => {
    const state = parseUrlState(new URLSearchParams('?z=10&model=hf'))
    expect(state.model).toBe('hf')
    expect(toUrlState(state).get('model')).toBe('hf')
  })

  it('defaults to gsz so existing deep links keep resolving as before', () => {
    expect(parseUrlState(new URLSearchParams('?z=10')).model).toBe('gsz')
  })

  it('omits the default from the serialized URL', () => {
    const state = parseUrlState(new URLSearchParams('?z=10'))
    expect(toUrlState(state).has('model')).toBe(false)
  })

  it('rejects an unknown model rather than silently falling back', () => {
    expect(() => parseUrlState(new URLSearchParams('?model=dft'))).toThrow()
  })
})
```

- [ ] **Step 2: Implement**

Add `model: 'gsz' | 'hf'` to the store. It belongs in the `INVALIDATED` block:
switching models changes the physics, so every derived quantity must be dropped.
This is the opposite of the presentational toggles, which deliberately
invalidate nothing.

The `Badge` shows the fidelity split from spec section 3, and for any atom with a
partially filled subshell it also shows the configuration-average disclosure.
Route it through `lib/liberties.ts` like every other disclosed liberty. Show the
virial ratio as a convergence readout, labelled a diagnostic, never as physics.

- [ ] **Step 3: Test, build and commit**

```bash
cd web && npm test && npm run build && cd ..
git add web/src
git commit -m "Add a Hartree-Fock model selector with its own invalidation and badge"
```

`npm run build` is not optional: `atomsim serve` only mounts `web/dist`, so an
unbuilt frontend change is invisible.

---

### Task 13: Documentation and phase close

**Files:**
- Modify: `docs/specs/2026-07-27-phase21-hartree-fock-design.md`
- Modify: `README.md` if it lists supported atoms, and its layout section

- [ ] **Step 1: Record what the build changed about the design**

Every previous phase spec ends with a "What the build changed" section (see
`2026-07-27-phase20-absorption-spectrum-design.md:117`). Add one here covering
at minimum: the derived coefficients that section 2.4 declined to state, the
`hf_terms.py` / `hartree_fock.py` split, the measured argon grid error against
the uniform-mesh ceiling, and the tuned mixing parameters.

- [ ] **Step 2: Update the architecture notes**

Add `hf_atom.py`, `numerics/hf_terms.py`, `numerics/hartree_fock.py` and
`numerics/slater.py` to the `README.md` layout section, in the style of the
existing entries.

- [ ] **Step 3: Run everything**

```bash
ruff check .
pytest -q
cd web && npm test && npm run build && cd ..
```

- [ ] **Step 4: Commit**

```bash
git add docs README.md
git commit -m "Record what building Phase 21 changed about its design"
```

---

## Self-review notes

**Spec coverage.** Spec section 2.2 equations map to Tasks 4 to 6; 2.3 Slater
integrals to Task 2; 2.4 coefficients to Task 4, with the derivation promoted
into this plan; section 3 fidelity model to Task 7's provenance tests; section 4
modules to Tasks 2, 4, 5, 7 (with the documented file split); section 5 numerics
to Tasks 5 and 6; section 6 three-way energy to Task 6; section 7.1 anchors to
Tasks 4 and 6; 7.2 identities to Tasks 2, 5, 6; 7.3 vendored data to Task 3; 7.4
cross-model to Task 8; 7.5 performance to Task 10; section 8 server and frontend
to Tasks 11 and 12; section 10 milestones map to Tasks 1 to 2, 3 to 7, 9, and 11
to 12 respectively.

**Two spec items intentionally not in a task.** The logarithmic mesh and DIIS are
both deferred in spec section 11 and stay deferred.

**One open dependency.** Task 3 needs a document that may not be reachable from
the execution environment. It is sequenced early so the dependency surfaces on
day one rather than at the benchmark, and every downstream test skips with an
explicit reason instead of passing vacuously.
