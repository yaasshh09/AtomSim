# Phase 6: Screened Multi-Electron Atoms Implementation Plan

**Goal:** Add real multi-electron atoms (He-Ar) as selectable systems, solved in a Green-Sellin-Zachor screened central potential by the existing numerical radial solver, with honest `APPROXIMATION` term diagrams, spectra vs NIST, and radial functions.

**Architecture:** A pure element/configuration module (`atoms.py`) and a pure screened-potential module (`numerics/screening.py`) feed an orchestrator (`screened_atom.py`) that drives `solve_radial_with_error` per angular momentum. The server resolves a system key to hydrogenic *or* screened and routes `/api/levels`, `/api/radial`, `/api/spectrum` accordingly; cloud/plane refuse screened orbitals honestly. The frontend adds an atom group to the system picker, a configuration panel, and wires the three energy-side views, with a labeled placeholder in Cloud/Plane.

**Tech Stack:** Python 3.12 (NumPy, SciPy, FastAPI, Pydantic, pytest), React + TypeScript + Zustand (vitest).

## Global Constraints

- Engine-internal math is in **Hartree atomic units**; eV/pm display conversion happens only at the server boundary and appends to the provenance `method`.
- **Prime directive:** every value crossing a module boundary is a `Quantity`/`Field` with a `Provenance` and `Fidelity`. Screened orbital energies are `APPROXIMATION` (model error dominates the `NUMERICAL` solve error, which still travels as a quantified sub-scale). No bare floats, no silent zeros, no undisclosed liberties.
- `l` is the orbital quantum number, not a length (ruff E741 ignored project-wide). Keep the physics naming.
- Line length 100. `ruff check .` must stay clean.
- New physics gets a **validation test** (analytic ground truth / independent method / NIST), not a smoke test.
- Python tests require the MKL sequential-BLAS workaround. Run pytest as: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest`. Ruff: `& "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m ruff check .`
- The server console script is `atomsim` (`C:\Users\yashg\.conda\envs\atomsim\Scripts\atomsim.exe`); `python -m atomsim` does **not** work.
- `web/dist` is rebuilt only by `npm run build` (run from `web/`); the server mounts it only if it exists.
- **Screening parameters are sourced, not guessed.** The GSZ `(d, H)` parameters for each shipped atom are transcribed from the cited Green-Sellin-Zachor / Garvey-Jackman-Green source and are accepted only when the Task 3/4 validation tests pass (N=1 exact hydrogenic limit + NIST valence ionization energy within the stated tolerance). If a value cannot be sourced and verified, that atom ships without a preset, never with an invented parameter.
- **Working agreement:** leave the tree committed and push-ready after every task; one logical change per commit.
- **Back-compat:** every hydrogenic system key and existing endpoint behaves exactly as before; all screened behavior is an additive branch.

---

## File Structure

**Backend, create**
- `src/atomsim/atoms.py`, element table (Z=1-18), Aufbau/Madelung configuration, Pauli caps, config parse/format, atom registry.
- `src/atomsim/numerics/screening.py`, GSZ screened potential builder + `(d, H)` parameters + `APPROXIMATION` provenance.
- `src/atomsim/screened_atom.py`, orchestrator: solve orbitals in V_eff, apply configuration, total energy, numerical radial functions.
- `src/atomsim/data/nist_he_i.json`, `nist_li_i.json`, `nist_na_i.json`, vendored NIST reference lines.
- `tests/test_atoms.py`, `tests/test_screening.py`, `tests/test_screened_atom.py`.

**Backend, modify**
- `src/atomsim/spectra.py`, screened transition source + register new reference files.
- `src/atomsim/server/schemas.py`, screened level/atom response models; `SystemModel` gains optional `kind`, `n_electrons`.
- `src/atomsim/server/app.py`, kind resolution; screened branches in levels/radial/spectrum; screened guard in cloud/plane job endpoints; atoms in `/api/systems`.
- `tests/test_server.py`, `tests/test_spectra.py`, extend.

**Frontend, modify**
- `web/src/api/types.ts`, `web/src/api/client.ts`, `SystemInfo.kind`/`nElectrons`; screened level shapes; `config` params.
- `web/src/state/store.ts`, `web/src/state/store.test.ts`, atom config slice (in `INVALIDATED`).
- `web/src/lib/urlState.ts`, `web/src/lib/urlState.test.ts`, `config` deep link.
- `web/src/components/Controls.tsx`, atom group + configuration panel.
- `web/src/components/LevelsView.tsx`, `SpectrumView.tsx`, `RadialView.tsx`, `CloudView.tsx`, `PlaneView.tsx`, screened wiring + placeholder.

---

## Task 1: Element table + Aufbau configuration (`atoms.py`)

Pure data + logic, no solver dependency.

**Files:**
- Create: `src/atomsim/atoms.py`
- Test: `tests/test_atoms.py`

**Interfaces:**
- Consumes: nothing (stdlib only).
- Produces:
  - `SUBSHELL_LABELS = "spdfgh"`; `subshell_capacity(l: int) -> int` = `2*(2*l+1)`.
  - `Subshell = tuple[int, int]` (n, l).
  - `Configuration = tuple[tuple[Subshell, int] ...]`, ordered (n,l)→occupancy, ascending Madelung order, only non-zero shells.
  - `aufbau_configuration(n_electrons: int) -> Configuration`.
  - `format_config(config: Configuration) -> str`, e.g. `"1s2 2s2 2p1"`.
  - `parse_config(text: str) -> Configuration`.
  - `total_electrons(config: Configuration) -> int`.
  - `is_ground(config: Configuration) -> bool`, equals `aufbau_configuration(total_electrons(config))`.
  - `validate_config(config: Configuration) -> None`, raises `ValueError` on Pauli-cap violation, negative occupancy, or `n <= l`.
  - `@dataclass(frozen=True) Element(z: int, symbol: str, name: str)`.
  - `ELEMENTS: tuple[Element ...]` for Z=1-18; `element_by_symbol(sym) -> Element`; `element_by_z(z) -> Element`.
  - `ATOM_KEYS: tuple[str ...]` = lowercased symbols for Z=2-18 (`he`..`ar`).
  - `is_atom_key(key: str) -> bool`; `atom_for_key(key: str) -> Element`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_atoms.py
import pytest

from atomsim.atoms import (
    ATOM_KEYS, aufbau_configuration, atom_for_key, element_by_symbol,
    format_config, is_atom_key, is_ground, parse_config, subshell_capacity,
    total_electrons, validate_config,
)

def test_subshell_capacity():
    assert subshell_capacity(0) == 2   # s
    assert subshell_capacity(1) == 6   # p
    assert subshell_capacity(2) == 10  # d

@pytest.mark.parametrize("z,expected", [
    (1, "1s1"),
    (2, "1s2"),
    (6, "1s2 2s2 2p2"),      # carbon
    (10, "1s2 2s2 2p6"),     # neon
    (11, "1s2 2s2 2p6 3s1"), # sodium
    (18, "1s2 2s2 2p6 3s2 3p6"),  # argon
])
def test_aufbau_matches_known_configs(z, expected):
    assert format_config(aufbau_configuration(z)) == expected

def test_config_roundtrip_and_count():
    cfg = parse_config("1s2 2s2 2p1")
    assert total_electrons(cfg) == 5
    assert format_config(cfg) == "1s2 2s2 2p1"

def test_is_ground():
    assert is_ground(aufbau_configuration(11)) is True
    assert is_ground(parse_config("1s2 2s2 2p6 3p1")) is False  # excited Na

def test_validate_rejects_overfill_and_bad_shell():
    with pytest.raises(ValueError, match="capacity"):
        validate_config(parse_config("1s3"))         # > 2 in s
    with pytest.raises(ValueError, match="n must be"):
        validate_config(((( 1, 1), 1),))              # 1p impossible (n<=l)

def test_atom_keys_cover_he_to_ar():
    assert ATOM_KEYS[0] == "he" and ATOM_KEYS[-1] == "ar"
    assert len(ATOM_KEYS) == 17
    assert is_atom_key("na") and not is_atom_key("h")
    assert atom_for_key("na").z == 11 and element_by_symbol("Na").z == 11
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_atoms.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'atomsim.atoms'`.

- [ ] **Step 3: Write the implementation**

```python
# src/atomsim/atoms.py
"""Elements, subshells, and Aufbau configurations for screened atoms (Phase 6).

Pure data and combinatorics, no physics engine. A Configuration is an ordered
tuple of ((n, l), occupancy) in Madelung filling order. The screened potential
depends only on (Z, N); the configuration decides which computed orbitals are
occupied and thus the summed energy. See docs/specs/
2026-07-18-phase6-screened-atoms-design.md.
"""

from dataclasses import dataclass

SUBSHELL_LABELS = "spdfgh"

Subshell = tuple[int, int]                       # (n, l)
Configuration = tuple[tuple[Subshell, int], ...]  # ordered, non-zero shells

def subshell_capacity(l: int) -> int:
    return 2 * (2 * l + 1)

def _madelung_order() -> list[Subshell]:
    """(n, l) shells sorted by (n + l, n), the Madelung/Aufbau rule."""
    shells = [(n, l) for n in range(1, 8) for l in range(n)]
    shells.sort(key=lambda nl: (nl[0] + nl[1], nl[0]))
    return shells

_MADELUNG = _madelung_order()

def aufbau_configuration(n_electrons: int) -> Configuration:
    if n_electrons < 1:
        raise ValueError(f"n_electrons must be >= 1, got {n_electrons}")
    remaining = n_electrons
    out: list[tuple[Subshell, int]] = []
    for n, l in _MADELUNG:
        if remaining <= 0:
            break
        fill = min(subshell_capacity(l), remaining)
        out.append(((n, l), fill))
        remaining -= fill
    if remaining > 0:
        raise ValueError(f"{n_electrons} electrons exceeds supported shells")
    return tuple(out)

def format_config(config: Configuration) -> str:
    return " ".join(f"{n}{SUBSHELL_LABELS[l]}{occ}" for (n, l), occ in config)

def parse_config(text: str) -> Configuration:
    out: list[tuple[Subshell, int]] = []
    for tok in text.split():
        n = int(tok[0])
        l = SUBSHELL_LABELS.index(tok[1])
        occ = int(tok[2:])
        out.append(((n, l), occ))
    return tuple(out)

def total_electrons(config: Configuration) -> int:
    return sum(occ for _, occ in config)

def is_ground(config: Configuration) -> bool:
    return config == aufbau_configuration(total_electrons(config))

def validate_config(config: Configuration) -> None:
    for (n, l), occ in config:
        if n <= l:
            raise ValueError(f"n must be > l for a real subshell, got n={n}, l={l}")
        if occ < 0:
            raise ValueError(f"occupancy must be >= 0, got {occ}")
        if occ > subshell_capacity(l):
            raise ValueError(
                f"occupancy {occ} exceeds capacity {subshell_capacity(l)} for l={l}"
            )

@dataclass(frozen=True)
class Element:
    z: int
    symbol: str
    name: str

ELEMENTS: tuple[Element, ...] = (
    Element(1, "H", "Hydrogen"), Element(2, "He", "Helium"),
    Element(3, "Li", "Lithium"), Element(4, "Be", "Beryllium"),
    Element(5, "B", "Boron"), Element(6, "C", "Carbon"),
    Element(7, "N", "Nitrogen"), Element(8, "O", "Oxygen"),
    Element(9, "F", "Fluorine"), Element(10, "Ne", "Neon"),
    Element(11, "Na", "Sodium"), Element(12, "Mg", "Magnesium"),
    Element(13, "Al", "Aluminium"), Element(14, "Si", "Silicon"),
    Element(15, "P", "Phosphorus"), Element(16, "S", "Sulfur"),
    Element(17, "Cl", "Chlorine"), Element(18, "Ar", "Argon"),
)

_BY_SYMBOL = {e.symbol: e for e in ELEMENTS}
_BY_Z = {e.z: e for e in ELEMENTS}

# Named screened-atom presets are neutral He..Ar (H stays hydrogenic/analytic).
ATOM_KEYS: tuple[str, ...] = tuple(e.symbol.lower() for e in ELEMENTS if e.z >= 2)

def element_by_symbol(sym: str) -> Element:
    return _BY_SYMBOL[sym]

def element_by_z(z: int) -> Element:
    return _BY_Z[z]

def is_atom_key(key: str) -> bool:
    return key in ATOM_KEYS

def atom_for_key(key: str) -> Element:
    if not is_atom_key(key):
        raise KeyError(f"unknown atom key {key!r}; known: {ATOM_KEYS}")
    return _BY_SYMBOL[key.capitalize()]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_atoms.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Lint + commit**

```bash
& "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m ruff check src/atomsim/atoms.py tests/test_atoms.py
git add src/atomsim/atoms.py tests/test_atoms.py
git commit -m "Add element table and Aufbau configurations for screened atoms"
```

---

## Task 2: GSZ screened potential (`numerics/screening.py`)

**Files:**
- Create: `src/atomsim/numerics/screening.py`
- Test: `tests/test_screening.py`

**Interfaces:**
- Consumes: `atomsim.provenance.{Fidelity, Provenance}`.
- Produces:
  - `gsz_parameters(z: int, n_electrons: int) -> tuple[float, float]`, `(d, H)` for the GSZ screening function; `d` in bohr, `H` dimensionless.
  - `screened_potential(z: int, n_electrons: int) -> Callable[[np.ndarray], np.ndarray]`, `V_eff(r)` in hartree.
  - `z_eff(z: int, n_electrons: int, r: np.ndarray) -> np.ndarray`, the running screened charge (for the potential curve / debugging).
  - `screening_provenance(z: int, n_electrons: int) -> Provenance`, `APPROXIMATION`, names GSZ/GJG, cites source, states validity.

**Model (from the spec §2):**
`V_eff(r) = -(1/r) [ (Z - N + 1) + (N - 1) Ω(r) ]`, with
`Ω(r) = 1 / [ H (exp(r/d) - 1) + 1 ]`, `Ω(0)=1`, `Ω(∞)=0`.
`(d, H)` are transcribed from the cited GSZ/GJG source (see Global Constraints). The N=1 limit (below) needs no parameters, it is the calibration anchor; the NIST valence tolerance in Task 3 is the acceptance gate for the parameter values.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_screening.py
import numpy as np

from atomsim.numerics.screening import (
    gsz_parameters, screened_potential, screening_provenance, z_eff,
)
from atomsim.provenance import Fidelity

def test_n1_is_bare_coulomb():
    # With one electron the (N-1) screening term vanishes: V = -Z/r exactly.
    v = screened_potential(z=3, n_electrons=1)
    r = np.array([0.1, 1.0, 5.0])
    assert np.allclose(v(r), -3.0 / r, rtol=0, atol=1e-12)

def test_zeff_limits_neutral_atom():
    # Neutral sodium: Z=N=11. Near r=0 -> Z; far -> Z-N+1 = 1.
    near = z_eff(11, 11, np.array([1e-4]))[0]
    far = z_eff(11, 11, np.array([1e4]))[0]
    assert abs(near - 11.0) < 1e-2
    assert abs(far - 1.0) < 1e-6

def test_potential_is_finite_and_attractive():
    v = screened_potential(11, 11)
    r = np.linspace(0.01, 40.0, 500)
    vals = v(r)
    assert np.isfinite(vals).all()
    assert (vals < 0).all()          # attractive everywhere
    assert vals[0] < vals[-1]        # deeper near the nucleus

def test_parameters_positive():
    d, h = gsz_parameters(11, 11)
    assert d > 0 and h > 0

def test_provenance_is_approximation():
    prov = screening_provenance(11, 11)
    assert prov.fidelity is Fidelity.APPROXIMATION
    assert "Green" in prov.method or "GSZ" in prov.method
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_screening.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'atomsim.numerics.screening'`.

- [ ] **Step 3: Write the implementation**

Transcribe the GSZ `(d, H)` parameter fit for `(Z, N)` from the cited source (Garvey, Jackman, Green, *Phys. Rev. A* **12**, 1144 (1975); GSZ form: Green, Sellin, Zachor, *Phys. Rev.* **184**, 1 (1969)) into `gsz_parameters`. The functional form below is fixed; only the `(d, H)` body is sourced. Verify the transcription with `tests/test_screening.py` **and** the NIST valence-energy test in Task 3 before accepting the values.

```python
# src/atomsim/numerics/screening.py
"""Green-Sellin-Zachor screened central potential for multi-electron atoms.

V_eff(r) = -(1/r) [ (Z - N + 1) + (N - 1) Omega(r) ],
Omega(r) = 1 / [ H (exp(r/d) - 1) + 1 ],  Omega(0)=1, Omega(inf)=0.

This is an APPROXIMATION: an analytic independent-particle model, not a
self-consistent solve. The (d, H) parameters are the Garvey-Jackman-Green (1975)
fit of the GSZ (1969) form; they are transcribed from that source and pinned by
tests (N=1 exact Coulomb limit + NIST valence ionization energy). One electron
(N=1) reduces to bare -Z/r with no parameters, the calibration anchor.
"""

from collections.abc import Callable

import numpy as np

from atomsim.provenance import Fidelity, Provenance

def gsz_parameters(z: int, n_electrons: int) -> tuple[float, float]:
    """(d, H) for the GSZ screening function; d in bohr, H dimensionless.

    Sourced from Garvey-Jackman-Green (1975). Only exercised for N > 1 (for N=1
    the screening term is multiplied by (N-1)=0). Values are accepted only when
    the Task 2/3 validation tests pass.
    """
    if z < 1:
        raise ValueError(f"Z must be >= 1, got {z}")
    if not 1 <= n_electrons <= z + 1:
        raise ValueError(f"N must be in [1, Z+1], got {n_electrons} (Z={z})")
    # --- BEGIN sourced GJG(1975) universal fit (transcribe exact coefficients) ---
    # d = d(Z, N), H = H(Z, N). See module docstring for the citation. The two
    # relations are the paper's universal analytic fit; pinned by the tests.
    nn = n_electrons
    d = _gjg_d(z, nn)
    h = _gjg_h(z, nn)
    # --- END sourced fit ---
    return d, h

def _gjg_d(z: int, n_electrons: int) -> float:
    # Transcribed GJG(1975) relation for d(Z, N). Placeholder coefficients here
    # are OVERWRITTEN with the sourced values in Step 3a and verified by tests.
    raise NotImplementedError("transcribe GJG(1975) d(Z, N) fit, see Step 3a")

def _gjg_h(z: int, n_electrons: int) -> float:
    raise NotImplementedError("transcribe GJG(1975) H(Z, N) fit, see Step 3a")

def _omega(r: np.ndarray, d: float, h: float) -> np.ndarray:
    return 1.0 / (h * (np.expm1(r / d)) + 1.0)

def z_eff(z: int, n_electrons: int, r: np.ndarray) -> np.ndarray:
    core = float(z - n_electrons + 1)
    if n_electrons == 1:
        return np.full_like(np.asarray(r, dtype=float), core)
    d, h = gsz_parameters(z, n_electrons)
    return core + (n_electrons - 1) * _omega(np.asarray(r, dtype=float), d, h)

def screened_potential(z: int, n_electrons: int) -> Callable[[np.ndarray], np.ndarray]:
    def v(r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        return -z_eff(z, n_electrons, r) / r
    return v

def screening_provenance(z: int, n_electrons: int) -> Provenance:
    return Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method=(
            "Green-Sellin-Zachor screened central potential, "
            "Garvey-Jackman-Green (1975) analytic parameters"
        ),
        assumptions=(
            f"independent-particle central field for Z={z}, N={n_electrons}",
            "no self-consistency; potential depends only on (Z, N)",
            "infinite nuclear mass (mu_ratio = 1)",
        ),
        error_estimate=None,  # quantified against NIST at the observable level
        refinement="self-consistent Hartree-Fock (a later phase) removes the model error",
    )
```

> **Step 3a (sourcing, blocking):** Replace `_gjg_d` / `_gjg_h` with the exact GJG(1975) universal-fit expressions (or, if a coefficient cannot be sourced, a small vendored `(Z,N)->(d,H)` table for He-Ar transcribed from the paper). The N=1 tests pass without them; they are accepted only once `test_screened_atom.py::test_valence_ionization_matches_nist` (Task 3) is green. Do not invent coefficients.

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_screening.py -v`
Expected: PASS (5 tests). `test_n1_is_bare_coulomb` needs no parameters; the neutral-atom tests exercise the sourced fit.

- [ ] **Step 5: Lint + commit**

```bash
& "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m ruff check src/atomsim/numerics/screening.py tests/test_screening.py
git add src/atomsim/numerics/screening.py tests/test_screening.py
git commit -m "Add the Green-Sellin-Zachor screened central potential"
```

---

## Task 3: Screened-atom solver (`screened_atom.py`)

**Files:**
- Create: `src/atomsim/screened_atom.py`
- Test: `tests/test_screened_atom.py`

**Interfaces:**
- Consumes: `atoms.{Configuration, aufbau_configuration, atom_for_key, validate_config, total_electrons, is_ground}`; `numerics/screening.{screened_potential, screening_provenance}`; `numerics/radial_solver.solve_radial_with_error`; `provenance.{Field, Fidelity, Provenance, Quantity}`.
- Produces:
  - `@dataclass(frozen=True) Orbital(n, l, occupancy: int, energy: Quantity)`, `energy` APPROXIMATION, hartree.
  - `@dataclass(frozen=True) ScreenedAtomResult(key, z, n_electrons, config, is_ground, orbitals: tuple[Orbital,...], total_energy: Quantity, provenance)`.
  - `solve_screened_atom(z, n_electrons, config, l_max=2, n_states_per_l=4) -> ScreenedAtomResult`.
  - `screened_radial(z, n_electrons, n, l, points=400) -> tuple[Field, Field]`, `(R_field, P_field)` where `P = r^2 R^2`, mirroring the analytic `/api/radial` shape.
  - `valence_ionization_energy(result) -> Quantity`, `-ε` of the highest-energy occupied orbital, eV-independent (hartree).

**Physics:** solve each `l ∈ 0..l_max` in `V_eff` with `solve_radial_with_error`; radial state `k` maps to `n = k + l + 1`. Keep orbitals up to the max `n` any occupied subshell needs (plus a couple of virtual states for the spectrum). Tag each energy `APPROXIMATION` with the screening method + the solve's numerical error as a sub-scale. Total energy = Σ occupancy·ε.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_screened_atom.py
import math

import pytest

from atomsim.atoms import aufbau_configuration, parse_config
from atomsim.provenance import Fidelity
from atomsim.screened_atom import (
    screened_radial, solve_screened_atom, valence_ionization_energy,
)

# NIST first ionization energies (eV), for the tier-defining tolerance test.
# Source: NIST ASD ionization energies (public). Retrieved 2026-07-18.
_NIST_IE_EV = {"he": 24.587, "li": 5.392, "na": 5.139}
HARTREE_EV = 27.211386245988

def test_n1_recovers_hydrogenic():
    # One electron in the screened field == bare Coulomb == -Z^2 / 2n^2.
    res = solve_screened_atom(z=3, n_electrons=1, config=parse_config("1s1"))
    e1s = res.orbitals[0].energy.value
    assert math.isclose(e1s, -(3**2) / 2.0, rel_tol=2e-4)
    assert res.orbitals[0].energy.provenance.fidelity is Fidelity.APPROXIMATION

def test_orbital_energy_carries_numerical_subscale():
    res = solve_screened_atom(z=11, n_electrons=11, config=aufbau_configuration(11))
    assert res.orbitals[0].energy.provenance.error_estimate is not None

def test_total_energy_is_occupancy_weighted_sum():
    res = solve_screened_atom(z=6, n_electrons=6, config=aufbau_configuration(6))
    expect = sum(o.occupancy * o.energy.value for o in res.orbitals)
    assert math.isclose(res.total_energy.value, expect, rel_tol=1e-12)

def test_s_below_p_for_same_n():
    # Screening lifts the Coulomb degeneracy: 2s below 2p in carbon.
    res = solve_screened_atom(z=6, n_electrons=6, config=aufbau_configuration(6))
    e = {(o.n, o.l): o.energy.value for o in res.orbitals}
    assert e[(2, 0)] < e[(2, 1)]

@pytest.mark.parametrize("key,z,n", [("he", 2, 2), ("li", 3, 3), ("na", 11, 11)])
def test_valence_ionization_matches_nist(key, z, n):
    res = solve_screened_atom(z=z, n_electrons=n, config=aufbau_configuration(n))
    ie_ev = valence_ionization_energy(res).value * HARTREE_EV
    ref = _NIST_IE_EV[key]
    assert abs(ie_ev - ref) / ref < 0.12  # GSZ valence energies: ~10% class

def test_screened_radial_shapes():
    r_field, p_field = screened_radial(z=11, n_electrons=11, n=3, l=0, points=300)
    assert r_field.values.shape == r_field.grid.shape == (300,)
    assert p_field.unit == "bohr^-1" and r_field.provenance.fidelity is Fidelity.APPROXIMATION
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_screened_atom.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'atomsim.screened_atom'`.

- [ ] **Step 3: Write the implementation**

```python
# src/atomsim/screened_atom.py
"""Solve a screened multi-electron atom in the GSZ central field (APPROXIMATION).

Each angular momentum l is solved once in V_eff(r); radial state k is principal
number n = k + l + 1. The configuration decides occupancy and thus the summed
energy, the field itself depends only on (Z, N). Orbital energies are
APPROXIMATION (model error dominates) carrying the numerical solve error as a
quantified sub-scale. See docs/specs/2026-07-18-phase6-...md.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np

from atomsim.atoms import Configuration, is_ground, total_electrons
from atomsim.numerics.radial_solver import solve_radial_with_error
from atomsim.numerics.screening import screened_potential, screening_provenance
from atomsim.provenance import Field, Fidelity, Provenance, Quantity

@dataclass(frozen=True)
class Orbital:
    n: int
    l: int
    occupancy: int
    energy: Quantity  # APPROXIMATION, hartree

@dataclass(frozen=True)
class ScreenedAtomResult:
    key: str
    z: int
    n_electrons: int
    config: Configuration
    is_ground: bool
    orbitals: tuple[Orbital, ...]
    total_energy: Quantity
    provenance: Provenance

def _r_max(z: int, n_top: int) -> float:
    # Outer electron of a neutral atom feels net charge ~1; scale the box to it.
    return 40.0 * (n_top + 1) ** 2

def _solve_energies(z: int, n_electrons: int, l: int, n_states: int) -> tuple[Quantity, ...]:
    potential = screened_potential(z, n_electrons)
    r_max = _r_max(z, n_states + l)
    sol = solve_radial_with_error(
        potential, l=l, mu_ratio=1.0, r_max=r_max, n_states=n_states
    )
    prov_model = screening_provenance(z, n_electrons)
    out = []
    for k, e in enumerate(sol.energies):
        merged = Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method=f"{prov_model.method}; radial Schrodinger solved numerically",
            assumptions=prov_model.assumptions + e.provenance.assumptions,
            error_estimate=e.provenance.error_estimate,  # numerical sub-scale
            refinement=prov_model.refinement,
        )
        out.append(dataclasses.replace(e, provenance=merged))
    return tuple(out)

def solve_screened_atom(
    z: int, n_electrons: int, config: Configuration,
    l_max: int = 2, n_states_per_l: int = 4,
) -> ScreenedAtomResult:
    occ = {nl: c for nl, c in config}
    l_top = max((l for (_, l), _ in config), default=0)
    n_top = max((n for (n, _), _ in config), default=1)
    l_max = max(l_max, l_top)
    n_states = max(n_states_per_l, n_top)  # enough radial states to reach n_top

    orbitals: list[Orbital] = []
    for l in range(l_max + 1):
        energies = _solve_energies(z, n_electrons, l, n_states)
        for k, e in enumerate(energies):
            n = k + l + 1
            orbitals.append(Orbital(n=n, l=l, occupancy=occ.get((n, l), 0), energy=e))
    orbitals.sort(key=lambda o: (o.energy.value, o.n, o.l))

    total = sum(o.occupancy * o.energy.value for o in orbitals)
    total_prov = Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method="sum of occupancy-weighted independent-particle orbital energies",
        assumptions=(
            "not a variational total energy; ignores e-e double counting",
            f"configuration {'ground' if is_ground(config) else 'non-ground'}",
        ),
    )
    return ScreenedAtomResult(
        key=f"z{z}n{n_electrons}",
        z=z, n_electrons=n_electrons, config=config, is_ground=is_ground(config),
        orbitals=tuple(orbitals),
        total_energy=Quantity(total, "hartree", "E_total", total_prov),
        provenance=screening_provenance(z, n_electrons),
    )

def valence_ionization_energy(result: ScreenedAtomResult) -> Quantity:
    occupied = [o for o in result.orbitals if o.occupancy > 0]
    if not occupied:
        raise ValueError("no occupied orbitals")
    valence = max(occupied, key=lambda o: o.energy.value)
    prov = dataclasses.replace(
        valence.energy.provenance,
        method=valence.energy.provenance.method + "; ionization energy = -epsilon_valence",
    )
    return Quantity(-valence.energy.value, "hartree", "IE_valence", prov)

def screened_radial(
    z: int, n_electrons: int, n: int, l: int, points: int = 400,
) -> tuple[Field, Field]:
    if n <= l:
        raise ValueError(f"n must be > l, got n={n}, l={l}")
    k = n - l - 1
    potential = screened_potential(z, n_electrons)
    r_max = _r_max(z, n)
    sol = solve_radial_with_error(
        potential, l=l, mu_ratio=1.0, r_max=r_max, n_states=k + 1
    )
    r_solver = sol.r
    R = sol.u[k] / r_solver  # R = u / r
    grid = np.linspace(r_solver[0], r_solver[-1], points)
    R_i = np.interp(grid, r_solver, R)
    prov = Provenance(
        fidelity=Fidelity.APPROXIMATION,
        method=f"{screening_provenance(z, n_electrons).method}; numerical R_nl = u/r",
        assumptions=screening_provenance(z, n_electrons).assumptions,
        error_estimate=sol.energies[k].provenance.error_estimate,
    )
    r_field = Field(values=R_i, grid=grid, unit="bohr^-3/2", grid_unit="bohr",
                    label=f"R_{n},{l}(r)", provenance=prov)
    p_field = Field(values=grid**2 * R_i**2, grid=grid, unit="bohr^-1",
                    grid_unit="bohr", label=f"P_{n},{l}(r) = r^2 R^2", provenance=prov)
    return r_field, p_field
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_screened_atom.py tests/test_screening.py -v`
Expected: PASS. `test_valence_ionization_matches_nist` is the acceptance gate for Task 2's sourced parameters. If it fails, fix the parameters (not the tolerance), 12% is already generous for GSZ valence energies.

- [ ] **Step 5: Lint + commit**

```bash
& "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m ruff check src/atomsim/screened_atom.py tests/test_screened_atom.py
git add src/atomsim/screened_atom.py tests/test_screened_atom.py
git commit -m "Add the screened multi-electron atom solver"
```

---

## Task 4: Screened transitions + NIST reference data (`spectra.py`, data)

**Files:**
- Create: `src/atomsim/data/nist_he_i.json`, `nist_li_i.json`, `nist_na_i.json`
- Modify: `src/atomsim/spectra.py` (add screened transition source; register reference files)
- Test: `tests/test_spectra.py` (append)

**Interfaces:**
- Consumes: `screened_atom.ScreenedAtomResult`; existing `SpectralLine`, `LineList`, `load_reference`, `compare_lines`.
- Produces: `screened_transition_lines(result: ScreenedAtomResult) -> LineList`, dipole (`Δl=±1`) emission lines among the solved orbitals (`E_upper > E_lower`), energies in eV, vacuum wavelengths in nm, `APPROXIMATION`.
- Registers `he`, `li`, `na` in `_REFERENCE_FILES`.

- [ ] **Step 1: Add the vendored NIST data files**

Create each file with real NIST ASD lines (persistent species I, vacuum or air noted in `medium`). Example shape (fill with actual lines + citation; Na must include the D doublet ~588.995/589.592 nm):

```json
{
  "species": "Na I",
  "citation": "NIST ASD, Kramida et al. (2024), https://physics.nist.gov/asd",
  "retrieved": "2026-07-18",
  "medium": "air",
  "lines": [
    { "wavelength_nm": 588.995, "uncertainty_nm": 0.001, "label": "3p->3s (D2)" },
    { "wavelength_nm": 589.592, "uncertainty_nm": 0.001, "label": "3p->3s (D1)" }
  ]
}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_spectra.py  (append)
from atomsim.atoms import aufbau_configuration
from atomsim.screened_atom import solve_screened_atom
from atomsim.spectra import compare_lines, load_reference, screened_transition_lines

def test_screened_na_has_d_line_near_589nm():
    res = solve_screened_atom(z=11, n_electrons=11, config=aufbau_configuration(11))
    lines = screened_transition_lines(res)
    assert lines.lines  # non-empty
    nearest = min(lines.lines, key=lambda ln: abs(ln.wavelength.value - 589.0))
    assert abs(nearest.wavelength.value - 589.0) < 120.0  # GSZ valence class

def test_screened_reference_files_registered():
    for key in ("he", "li", "na"):
        assert load_reference(key) is not None

def test_screened_lines_are_emission_and_dipole():
    res = solve_screened_atom(z=3, n_electrons=3, config=aufbau_configuration(3))
    for ln in screened_transition_lines(res).lines:
        assert abs(ln.l_upper - ln.l_lower) == 1
        assert ln.energy.value > 0
```

- [ ] **Step 3: Implement `screened_transition_lines` and register files**

Add to `spectra.py`:

```python
from atomsim.screened_atom import ScreenedAtomResult  # add near the top imports

def screened_transition_lines(result: ScreenedAtomResult) -> LineList:
    """Dipole-allowed emission lines among a screened atom's orbital energies."""
    levels = [(o.n, o.l, o.energy) for o in result.orbitals]
    lines: list[SpectralLine] = []
    for (nu, lu, eu), (nl, ll_, el) in itertools.permutations(levels, 2):
        if eu.value <= el.value or abs(lu - ll_) != 1:
            continue
        de_ev = (eu.value - el.value) * HARTREE_EV
        prov = Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method=(
                f"screened orbital difference: [{eu.provenance.method}] minus "
                f"[{el.provenance.method}]; photon lambda = hc/dE (vacuum)"
            ),
            assumptions=eu.provenance.assumptions
            + ("electric-dipole selection rule (Delta l = +/-1)",),
            error_estimate=(
                None if eu.provenance.error_estimate is None
                else (eu.provenance.error_estimate + (el.provenance.error_estimate or 0.0))
                * HARTREE_EV
            ),
        )
        label = f"{nu}{'spdfgh'[lu]}->{nl}{'spdfgh'[ll_]}"
        lines.append(SpectralLine(
            n_upper=nu, l_upper=lu, j_upper=None, n_lower=nl, l_lower=ll_, j_lower=None,
            energy=Quantity(de_ev, "eV", f"dE {label}", prov),
            wavelength=Quantity(_EV_NM / de_ev, "nm (vacuum)", f"lambda {label}", prov),
        ))
    lines.sort(key=lambda ln: ln.wavelength.value)
    return LineList(
        system_key=result.key, n_max=max(o.n for o in result.orbitals),
        fine_structure=False, lines=tuple(lines),
        provenance=Provenance(
            fidelity=Fidelity.APPROXIMATION,
            method="dipole-allowed screened orbital differences (see per-line provenance)",
            assumptions=("emission lines only (E_upper > E_lower)",
                         "independent-particle transition energies"),
        ),
    )
```

Register the new references:

```python
_REFERENCE_FILES = {
    "h": "nist_h_i.json", "d": "nist_d_i.json", "he+": "nist_he_ii.json",
    "he": "nist_he_i.json", "li": "nist_li_i.json", "na": "nist_na_i.json",
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_spectra.py -v`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
& "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m ruff check src/atomsim/spectra.py tests/test_spectra.py
git add src/atomsim/spectra.py src/atomsim/data/nist_he_i.json src/atomsim/data/nist_li_i.json src/atomsim/data/nist_na_i.json tests/test_spectra.py
git commit -m "Add screened transition lines and NIST reference data for He, Li, Na"
```

---

## Task 5: System-kind resolution + schemas + `/api/systems`

Give the server one place to decide hydrogenic vs screened, a `SystemModel` that can describe an atom, and screened atoms in the systems list. Also guard the cloud/plane job endpoints against screened keys.

**Files:**
- Modify: `src/atomsim/server/schemas.py` (SystemModel gains `kind`, `n_electrons`; `+ from_atom`; screened level models)
- Modify: `src/atomsim/server/app.py` (`_resolve_kind`, atoms in `/api/systems`, cloud/plane guard)
- Test: `tests/test_server.py` (append)

**Interfaces:**
- Consumes: `atoms.{ATOM_KEYS, atom_for_key, aufbau_configuration, format_config}`.
- Produces:
  - `SystemModel` fields added: `kind: Literal["hydrogenic","screened"] = "hydrogenic"`, `n_electrons: int | None = None`; classmethod `from_atom(element, n_electrons, config_str) -> SystemModel`.
  - `class ScreenedOrbitalModel(BaseModel){ n, l, label: str, occupancy: int, energy: QuantityModel, energy_ev: QuantityModel }`.
  - `class ScreenedLevelsModel(BaseModel){ system: SystemModel, config: str, is_ground: bool, orbitals: list[ScreenedOrbitalModel], total_energy: QuantityModel, total_energy_ev: QuantityModel }`.
  - `app` helper `_is_screened(key) -> bool` and `_resolve_config(system_key, config: str | None) -> Configuration` (defaults to Aufbau, validates, raises `HTTPException(422)`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_server.py  (append)
def test_systems_list_includes_screened_atoms(client):
    body = client.get("/api/systems").json()
    keys = {s["key"]: s for s in body["systems"]}
    assert "na" in keys
    assert keys["na"]["kind"] == "screened"
    assert keys["na"]["n_electrons"] == 11
    assert keys["h"]["kind"] == "hydrogenic"

def test_cloud_job_rejects_screened_atom(client):
    r = client.post("/api/jobs/sample", json={"n": 3, "l": 0, "m": 0, "system": "na", "count": 1000})
    assert r.status_code == 422
    assert "screened" in r.json()["detail"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_server.py -k "screened" -v`
Expected: FAIL, `na` absent from systems; sample job returns 200/500 not 422.

- [ ] **Step 3a: Extend `SystemModel` and add screened models in `schemas.py`**

Add `kind`/`n_electrons` fields (defaults keep hydrogenic responses byte-identical) and a `from_atom` builder; add the two screened models. `Literal` is already imported (used by `ReferenceModel`).

```python
class SystemModel(BaseModel):
    key: str
    name: str
    z: int
    mu_ratio: QuantityModel
    m_over_m_nucleus: float
    description: str
    nuclear_radius: QuantityModel | None
    nuclear_radius_fm: QuantityModel | None
    kind: Literal["hydrogenic", "screened"] = "hydrogenic"
    n_electrons: int | None = None
    # ... keep existing from_system unchanged ...

    @classmethod
    def from_atom(cls, element, n_electrons: int, description: str) -> "SystemModel":
        from atomsim.provenance import Fidelity, Provenance, Quantity
        mu = Quantity(1.0, "m_e", f"mu/m_e ({element.name})",
                      Provenance(fidelity=Fidelity.APPROXIMATION,
                                 method="infinite nuclear mass (screened-atom model)"))
        return cls(
            key=element.symbol.lower(), name=element.name, z=element.z,
            mu_ratio=QuantityModel.from_quantity(mu), m_over_m_nucleus=0.0,
            description=description, nuclear_radius=None, nuclear_radius_fm=None,
            kind="screened", n_electrons=n_electrons,
        )

class ScreenedOrbitalModel(BaseModel):
    n: int
    l: int
    label: str
    occupancy: int
    energy: QuantityModel
    energy_ev: QuantityModel

class ScreenedLevelsModel(BaseModel):
    system: SystemModel
    config: str
    is_ground: bool
    orbitals: list[ScreenedOrbitalModel]
    total_energy: QuantityModel
    total_energy_ev: QuantityModel
```

- [ ] **Step 3b: Add resolution helpers + atoms in `/api/systems` + cloud/plane guard in `app.py`**

Add imports: `from atomsim.atoms import (ATOM_KEYS, atom_for_key, aufbau_configuration, format_config, is_atom_key, parse_config, total_electrons, validate_config)` and the new schema names.

```python
    def _is_screened(key: str) -> bool:
        return is_atom_key(key)

    def _resolve_config(system_key: str, config: str | None):
        element = atom_for_key(system_key)
        if config is None:
            return aufbau_configuration(element.z)
        try:
            cfg = parse_config(config)
            validate_config(cfg)
        except (ValueError, IndexError) as exc:
            raise HTTPException(status_code=422, detail=f"bad config: {exc}") from exc
        if total_electrons(cfg) != element.z:
            raise HTTPException(
                status_code=422,
                detail=f"config has {total_electrons(cfg)} electrons; {element.symbol} needs {element.z}",
            )
        return cfg
```

In `/api/systems`, append screened atoms:

```python
    @app.get("/api/systems", response_model=SystemsResponse)
    def systems() -> SystemsResponse:
        hydrogenic = [SystemModel.from_system(s) for s in list_systems()]
        screened = [
            SystemModel.from_atom(
                atom_for_key(k), n_electrons=atom_for_key(k).z,
                description=f"{atom_for_key(k).name}: GSZ screened central-field model (APPROXIMATION).",
            )
            for k in ATOM_KEYS
        ]
        return SystemsResponse(systems=hydrogenic + screened)
```

Guard the two job endpoints (`/api/jobs/sample`, `/api/jobs/plane`) right after they read `req.system`:

```python
        if _is_screened(req.system):
            raise HTTPException(
                status_code=422,
                detail="screened-atom orbitals: 3-D cloud / 2-D plane arrive in a later phase",
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_server.py -k "screened or systems" -v`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
& "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m ruff check src/atomsim/server/schemas.py src/atomsim/server/app.py tests/test_server.py
git add src/atomsim/server/schemas.py src/atomsim/server/app.py tests/test_server.py
git commit -m "Resolve system kind and list screened atoms; guard cloud/plane"
```

---

## Task 6: `/api/levels` screened branch

**Files:**
- Modify: `src/atomsim/server/app.py` (branch `levels_endpoint`; `config` param; new response model union)
- Modify: `src/atomsim/server/schemas.py` (`LevelsResponse` stays; screened uses `ScreenedLevelsModel` via a separate response path)
- Test: `tests/test_server.py` (append)

**Interfaces:**
- Consumes: `screened_atom.solve_screened_atom`; Task 5 helpers/models.
- Produces: `/api/levels?system=<atom>&config=<str>` returns `ScreenedLevelsModel`; hydrogenic keys return `LevelsResponse` exactly as before. (FastAPI: set `response_model=LevelsResponse | ScreenedLevelsModel`.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_server.py  (append)
def test_levels_screened_lithium_lifts_degeneracy(client):
    body = client.get("/api/levels?system=li").json()
    assert body["system"]["kind"] == "screened"
    assert body["is_ground"] is True
    e = {(o["n"], o["l"]): o["energy"]["value"] for o in body["orbitals"]}
    assert e[(2, 0)] < e[(2, 1)]  # 2s below 2p
    assert any(o["occupancy"] > 0 for o in body["orbitals"])

def test_levels_screened_bad_config_422(client):
    assert client.get("/api/levels?system=li&config=1s5").status_code == 422

def test_levels_hydrogenic_unchanged(client):
    body = client.get("/api/levels?system=h&n_max=3").json()
    assert body["system"]["kind"] == "hydrogenic"
    assert body["gross"][0]["degeneracy"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_server.py -k "levels_screened or levels_hydrogenic" -v`
Expected: FAIL, screened `system=li` currently 422s in `_resolve_system`.

- [ ] **Step 3: Branch `levels_endpoint`**

```python
    @app.get("/api/levels", response_model=LevelsResponse | ScreenedLevelsModel)
    def levels_endpoint(system: str = "h", n_max: int = 6,
                        fine_structure: bool = False, alpha: float | None = None,
                        config: str | None = None):
        if _is_screened(system):
            element = atom_for_key(system)
            cfg = _resolve_config(system, config)
            result = solve_screened_atom(element.z, total_electrons(cfg), cfg)
            return ScreenedLevelsModel(
                system=SystemModel.from_atom(element, element.z, f"{element.name}: GSZ model."),
                config=format_config(cfg), is_ground=result.is_ground,
                orbitals=[
                    ScreenedOrbitalModel(
                        n=o.n, l=o.l, label=f"{o.n}{'spdfgh'[o.l]}", occupancy=o.occupancy,
                        energy=QuantityModel.from_quantity(o.energy),
                        energy_ev=QuantityModel.from_quantity(_to_ev(o.energy)),
                    )
                    for o in result.orbitals
                ],
                total_energy=QuantityModel.from_quantity(result.total_energy),
                total_energy_ev=QuantityModel.from_quantity(_to_ev(result.total_energy)),
            )
        # ---- existing hydrogenic path unchanged below ----
        if not 1 <= n_max <= 10:
            raise HTTPException(status_code=422, detail="n_max must be in [1, 10]")
        # ... rest of the current implementation ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_server.py -k "levels" -v`
Expected: PASS (screened + existing hydrogenic level tests).

- [ ] **Step 5: Lint + commit**

```bash
& "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m ruff check src/atomsim/server/app.py tests/test_server.py
git add src/atomsim/server/app.py tests/test_server.py
git commit -m "Serve screened orbital levels from /api/levels"
```

---

## Task 7: `/api/radial` and `/api/spectrum` screened branches

**Files:**
- Modify: `src/atomsim/server/app.py` (branch `radial` and `spectrum`)
- Test: `tests/test_server.py` (append)

**Interfaces:**
- Consumes: `screened_atom.screened_radial`; `spectra.screened_transition_lines`; Task 5 helpers.
- Produces: `/api/radial/{n}/{l}?system=<atom>` → `RadialResponse` from numerical R_nl; `/api/spectrum?system=<atom>` → `SpectrumResponse` from screened transitions + NIST comparison for he/li/na.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_server.py  (append)
def test_radial_screened_sodium_3s(client):
    body = client.get("/api/radial/3/0?system=na&points=200").json()
    assert body["system"]["kind"] == "screened"
    assert len(body["radial_probability"]["values"]) == 200
    assert body["r_wavefunction"]["provenance"]["fidelity"] == "approximation"

def test_spectrum_screened_sodium_has_nist_comparison(client):
    body = client.get("/api/spectrum?system=na").json()
    assert body["system"]["kind"] == "screened"
    assert body["reference_citation"] is not None
    assert body["comparison"]  # matched at least one NIST line
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_server.py -k "radial_screened or spectrum_screened" -v`
Expected: FAIL, screened keys 422 in `_resolve_system`.

- [ ] **Step 3: Branch `radial` and `spectrum`**

`radial` (top of the function, before `_resolve_system`):

```python
        if _is_screened(system):
            element = atom_for_key(system)
            if not 50 <= points <= 2000:
                raise HTTPException(status_code=422, detail="points must be in [50, 2000]")
            r_field, p_field = screened_radial(element.z, element.z, n, l, points)
            return RadialResponse(
                n=n, l=l, system=SystemModel.from_atom(element, element.z, f"{element.name}: GSZ model."),
                r_wavefunction=FieldModel.from_field(r_field),
                radial_probability=FieldModel.from_field(p_field),
            )
```

`spectrum` (top of the function):

```python
        if _is_screened(system):
            element = atom_for_key(system)
            cfg = _resolve_config(system, config)
            result = solve_screened_atom(element.z, total_electrons(cfg), cfg)
            lines = screened_transition_lines(result)
            reference = load_reference(system)
            comparison = citation = tol = None
            if reference is not None:
                tol = 0.05  # 5%, the screened valence class; disclosed, not hidden
                comparison = [ComparisonModel.from_comparison(c)
                              for c in compare_lines(lines, reference, tolerance_relative=tol)]
                citation = reference.citation
            return SpectrumResponse(
                system=SystemModel.from_atom(element, element.z, f"{element.name}: GSZ model."),
                n_max=lines.n_max, fine_structure=False,
                lines=[LineModel.from_line(ln) for ln in lines.lines],
                comparison=comparison, reference_citation=citation, tolerance_relative=tol,
            )
```

Add `config: str | None = None` to the `spectrum` signature and import `screened_radial`, `screened_transition_lines`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest tests/test_server.py -k "radial or spectrum" -v`
Expected: PASS.

- [ ] **Step 5: Full backend suite + lint + commit**

```bash
$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest -q
& "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m ruff check .
git add src/atomsim/server/app.py tests/test_server.py
git commit -m "Serve screened radial functions and spectra vs NIST"
```

Expected: full backend suite green; ruff clean.

---

## Task 8: Frontend types + API client

**Files:**
- Modify: `web/src/api/types.ts`, `web/src/api/client.ts`

**Interfaces:**
- Produces:
  - `SystemInfo` gains `kind: "hydrogenic" | "screened"` and `n_electrons: number | null`.
  - `ScreenedOrbital { n; l; label; occupancy; energy: Quantity; energy_ev: Quantity }`.
  - `ScreenedLevels { system: SystemInfo; config: string; is_ground: boolean; orbitals: ScreenedOrbital[]; total_energy: Quantity; total_energy_ev: Quantity }`.
  - `getLevels(system, nMax, fineStructure, alpha?, config?)`, `getRadial(n, l, system, points?)` (unchanged shape), `getSpectrum(system, nMax, fineStructure, config?)`, add optional `config` query param passthrough.
  - Type guard `isScreenedLevels(x): x is ScreenedLevels`.

- [ ] **Step 1: Add the types** to `types.ts` (mirror Task 5/6 JSON), extend `SystemInfo` with `kind`/`n_electrons`.

- [ ] **Step 2: Thread `config`** into `getLevels`/`getSpectrum` in `client.ts` (append `&config=` when provided), and add `isScreenedLevels` (checks `"orbitals" in body`).

- [ ] **Step 3: Typecheck**

Run: `cd web; npx tsc --noEmit`
Expected: errors only in store/components fixed in Tasks 9-11, `types.ts`/`client.ts` compile.

- [ ] **Step 4: Commit**

```bash
git add web/src/api/types.ts web/src/api/client.ts
git commit -m "Add screened-atom frontend types and config-aware client"
```

---

## Task 9: Store: atom configuration slice

**Files:**
- Modify: `web/src/state/store.ts`, `web/src/state/store.test.ts`

**Interfaces:**
- Produces: store gains `config: string | null` (null = Aufbau default), `setConfig(config: string | null)`. `config` joins the `INVALIDATED`-clearing set for `setSystem` (selecting an atom resets `config` to null so the server fills Aufbau) and gets its own setter that clears derived physics (`levels`, `spectrum`, radial-dependent state) like other physics inputs.

- [ ] **Step 1: Write the failing test**

```ts
// web/src/state/store.test.ts  (append)
it("setSystem resets config to Aufbau default (null) and clears physics", () => {
  useAppStore.getState().setConfig("1s2 2s1");
  useAppStore.getState().setSystem("na");
  const st = useAppStore.getState();
  expect(st.system).toBe("na");
  expect(st.config).toBeNull();
  expect(st.levels).toBeNull();
});

it("setConfig clears derived physics but keeps the system", () => {
  useAppStore.setState({ system: "na", levels: {} as never });
  useAppStore.getState().setConfig("1s2 2s2 2p6 3p1");
  expect(useAppStore.getState().config).toBe("1s2 2s2 2p6 3p1");
  expect(useAppStore.getState().levels).toBeNull();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web; npx vitest run src/state/store.test.ts`
Expected: FAIL, no `config`/`setConfig`.

- [ ] **Step 3: Implement**, add `config: null` to initial state, add `config: null` into the object spread by `setSystem` (alongside `...INVALIDATED`), and add:

```ts
  setConfig: (config) =>
    set({ config, levels: null, spectrum: null, stateInfo: null }),
```

Add `config: string | null;` and `setConfig: (config: string | null) => void;` to the `AppState` type.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd web; npx vitest run src/state/store.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/src/state/store.ts web/src/state/store.test.ts
git commit -m "Add the screened-atom configuration slice to the store"
```

---

## Task 10: URL state: configuration deep link

**Files:**
- Modify: `web/src/lib/urlState.ts`, `web/src/lib/urlState.test.ts`, `web/src/main.tsx`

**Interfaces:**
- Produces: `UrlState` gains `config: string | null`. Query key `config` (omitted when null); the value is the compact string (`"1s2 2s1"`) URL-encoded. Round-trip tested. Wire `main.tsx` to read/write `s.config`.

- [ ] **Step 1: Write the failing test**

```ts
// web/src/lib/urlState.test.ts  (append)
it("round-trips a screened-atom config deep link", () => {
  const state = { ...URL_DEFAULTS, system: "na", config: "1s2 2s2 2p6 3p1" };
  const q = serializeAppUrl(state);
  expect(q).toContain("config=");
  expect({ ...URL_DEFAULTS, ...parseAppUrl(q) }.config).toBe("1s2 2s2 2p6 3p1");
});

it("omits config when null", () => {
  expect(serializeAppUrl({ ...URL_DEFAULTS })).not.toContain("config=");
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web; npx vitest run src/lib/urlState.test.ts`
Expected: FAIL, `config` not handled.

- [ ] **Step 3: Implement**, add `config: null` to `URL_DEFAULTS` and `config: string | null` to `UrlState`; in parse, `const c = q.get("config"); if (c) out.config = c;`; in serialize, `if (state.config) q.set("config", state.config);`. In `main.tsx`, add `config: s.config` to the serialized object.

- [ ] **Step 4: Run tests + typecheck**

Run: `cd web; npx vitest run src/lib/urlState.test.ts; npx tsc --noEmit`
Expected: PASS; tsc errors only remain in components (Task 11).

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/urlState.ts web/src/lib/urlState.test.ts web/src/main.tsx
git commit -m "Deep-link the screened-atom configuration"
```

---

## Task 11: Controls: atom group + configuration panel

**Files:**
- Modify: `web/src/components/Controls.tsx`

**Interfaces:**
- Consumes: `SystemInfo.kind`, store `config`/`setConfig`, `getSystems` (already fetched). Produces no new exports.
- Behavior: the system `<select>` groups systems into "Hydrogen-like" and "Atoms (screened)" `<optgroup>`s by `kind`. When the selected system is screened, a small configuration panel renders: the effective config string (from the latest `ScreenedLevels.config`, or "Aufbau (ground)" when `config` is null), a text input bound to `setConfig`, and a "Reset to Aufbau" button (`setConfig(null)`). A non-ground note shows when `is_ground` is false.

- [ ] **Step 1: Implement the optgroup split and config panel** in `Controls.tsx`, following the existing control markup/classes. (This is a view; validate by typecheck + build + the render smoke test in Task 13.)

- [ ] **Step 2: Typecheck**

Run: `cd web; npx tsc --noEmit`
Expected: PASS (or only LevelsView/SpectrumView/RadialView errors handled in Task 12).

- [ ] **Step 3: Commit**

```bash
git add web/src/components/Controls.tsx
git commit -m "Group atoms in the system picker and add a configuration panel"
```

---

## Task 12: Views: screened Levels / Spectrum / Radial + Cloud/Plane placeholder

**Files:**
- Modify: `web/src/components/LevelsView.tsx`, `SpectrumView.tsx`, `RadialView.tsx`, `CloudView.tsx`, `PlaneView.tsx`

**Interfaces:**
- Consumes: `isScreenedLevels`, `ScreenedLevels`, store `config`, `SystemInfo.kind`.
- Behavior:
  - **LevelsView:** when the levels payload is `ScreenedLevels`, render a term ladder of `orbitals` (energy in eV, occupancy shown, filled vs virtual styled distinctly), the config string, the `is_ground` note, and the total energy, all badged `APPROXIMATION`. Hydrogenic payload path unchanged.
  - **SpectrumView:** unchanged data flow (the endpoint already returns `SpectrumResponse`); ensure the NIST comparison + `APPROXIMATION` badge and disclosed 5% tolerance render for screened atoms.
  - **RadialView:** unchanged (endpoint returns `RadialResponse`); the `APPROXIMATION` badge now flows through from provenance.
  - **CloudView / PlaneView:** when the selected system's `kind === "screened"`, render a labeled placeholder ("Numerical screened orbital, 3-D cloud / 2-D plane arrive in a later phase") via `Badge`/`liberties.ts` instead of fetching a job.

- [ ] **Step 1: Implement** the screened branch in each view, reusing existing layout/classes and the `Badge` component.

- [ ] **Step 2: Typecheck + build**

Run: `cd web; npx tsc --noEmit; npm run build`
Expected: PASS; `web/dist` rebuilt.

- [ ] **Step 3: Frontend suite**

Run: `cd web; npm test`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/LevelsView.tsx web/src/components/SpectrumView.tsx web/src/components/RadialView.tsx web/src/components/CloudView.tsx web/src/components/PlaneView.tsx
git commit -m "Render screened atoms in Levels/Spectrum/Radial with Cloud/Plane placeholders"
```

---

## Task 13: Full verification + live smoke test

**Files:** none (verification only).

- [ ] **Step 1: Backend suite + lint**

Run: `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m pytest -q; & "C:\Users\yashg\.conda\envs\atomsim\python.exe" -m ruff check .`
Expected: all green; ruff clean.

- [ ] **Step 2: Frontend suite + build**

Run: `cd web; npm test; npm run build`
Expected: all green; build clean.

- [ ] **Step 3: Live smoke test**, start the server and exercise screened atoms.

Run (background): `$env:MKL_THREADING_LAYER="SEQUENTIAL"; & "C:\Users\yashg\.conda\envs\atomsim\Scripts\atomsim.exe" serve --port 8012 --no-browser`

Then confirm HTTP 200 + sane payloads:

```bash
curl -s "http://127.0.0.1:8012/api/systems" | python -c "import sys,json;d=json.load(sys.stdin);print(sorted({s['kind'] for s in d['systems']}), [s['key'] for s in d['systems'] if s['kind']=='screened'][:5])"
curl -s "http://127.0.0.1:8012/api/levels?system=na"                       # 2s<2p<..., is_ground true
curl -s "http://127.0.0.1:8012/api/levels?system=na&config=1s2%202s2%202p6%203p1"  # excited, is_ground false
curl -s "http://127.0.0.1:8012/api/radial/3/0?system=na&points=200"        # numerical R_3s
curl -s "http://127.0.0.1:8012/api/spectrum?system=na" | python -c "import sys,json;d=json.load(sys.stdin);print('cite',bool(d['reference_citation']),'cmp',len(d['comparison'] or []))"
curl -s -o /dev/null -w "%{http_code}\n" -X POST "http://127.0.0.1:8012/api/jobs/sample" -H "Content-Type: application/json" -d '{"n":3,"l":0,"m":0,"system":"na","count":1000}'   # expect 422
curl -s -o /dev/null -w "%{http_code}\n" "http://127.0.0.1:8012/api/levels?system=li&config=1s5"   # expect 422
```

Expected: systems shows both kinds and he..ar; Na levels lift the l-degeneracy with the ground config; excited config reports `is_ground:false`; radial returns 200 numerical points; Na spectrum has a NIST citation + ≥1 matched line; cloud job and bad config → 422. Stop the server (`TaskStop`) when done.

- [ ] **Step 4: Final commit (docs/tidy if any)**

```bash
git add -A
git commit -m "Phase 6: verify screened multi-electron atoms end-to-end" --allow-empty
```

---

## Self-Review

**Spec coverage:**
- §2.1-2.2 GSZ V_eff + GJG params → Task 2. ✅
- §2.3 provenance layering (APPROXIMATION over NUMERICAL sub-scale) → Task 3 `_solve_energies`. ✅
- §2.4 mu=1 → Task 2 provenance + Task 3 solves. ✅
- §3 configuration model (Aufbau, Pauli cap, excited/hollow, total energy) → Tasks 1, 3. ✅
- §4 engine modules → Tasks 1-3. ✅
- §5 server kind-routing + config param + cloud/plane placeholder → Tasks 5-7. ✅
- §6 frontend picker/config panel/views/placeholder → Tasks 8-12. ✅
- §7 coverage (He-Ar presets) + NIST (He, Li, Na) → Tasks 1, 4. ✅
- §8 validation (N=1 exact, grid convergence, NIST valence tolerance, Aufbau ordering, config accounting) → Tasks 1-3. ✅

**Placeholder scan:** the only deliberate "sourced later" element is the GJG `(d, H)` coefficients (Task 2, Step 3a), this is a gated transcription-from-citation with the Task 3 NIST test as acceptance, not a vague TODO; the functional form and every test are concrete.

**Type consistency:** `solve_screened_atom(z, n_electrons, config ...)`, `screened_radial(z, n_electrons, n, l, points)`, `screened_transition_lines(result)`, `Orbital(n,l,occupancy,energy)`, `ScreenedAtomResult` fields, `ScreenedOrbitalModel`/`ScreenedLevelsModel`, and `SystemModel.from_atom(element, n_electrons, description)` are used identically across Tasks 3-7; `SystemInfo.kind`/`config`/`isScreenedLevels`/`ScreenedLevels` match across Tasks 8-12; `config` query param is `str | None` server-side and `string | null` client-side throughout.

**One deliberate scope note:** cloud/plane are refused (422) for screened atoms rather than silently rendering a hydrogenic cloud, the honest choice; the numerical sampling subsystem is a later phase (spec §9).
