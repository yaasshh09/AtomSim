# Phase 11: Stark Effect (parabolic manifold) Design

**Phase:** 11
**Branch:** `phase11-stark-effect` (cut from `phase10-zeeman-field`)
**Status:** approved by standing order (user picked "continue with recommended" while away; Stark was named the Phase 11 successor in the Phase 10 spec §7 and §8)
**Parent spec:** `2026-07-04-atom-sim-requirements-design.md` (§3.1 "Zeeman and Stark effects ... external fields as first-class perturbations" / §9 Phase 3 list) · **Sibling:** `2026-07-23-phase10-zeeman-field-design.md` · **Builds on:** `analytic/hydrogen.py`, the coarse levels in the Levels view.

---

## 1. The one-sentence payload

An external electric field splits each degenerate hydrogen n-shell into a symmetric fan of sublevels labelled by the parabolic quantum numbers (n₁, n₂, m); the splitting is **linear** in the field, a signature that hydrogen's accidental l-degeneracy lets a permanent-looking dipole appear at first order, unlike every non-degenerate atom whose lowest Stark shift is quadratic.

## 2. Physics

### 2.1 Why parabolic, not spherical

A uniform field along z breaks rotational symmetry but keeps axial symmetry, so m stays good while l does not. The hydrogen Hamiltonian plus a linear potential (−F z, F the field in atomic units) separates exactly in **parabolic coordinates**, giving two new non-negative integers n₁, n₂ with

    n = n₁ + n₂ + |m| + 1.

The "electric quantum number" is k = n₁ − n₂ ∈ {−(n−1), …, +(n−1)}. This is the deliberate contrast with Phase 10: Zeeman split the *fine* levels by mⱼ; Stark splits the *gross* n-shell by k, needs no spin, and does not touch the fine-structure toggle.

### 2.2 The closed-form shift (what the engine computes)

Second-order perturbation theory in the degenerate n-manifold is closed form for hydrogen. In atomic units, for nuclear charge Z and reduced-mass ratio μ (= `mu_ratio`), with F the field magnitude in atomic units:

    E(n, n₁, n₂, m; F) = −μZ²/(2n²)                                   # Bohr level (EXACT part)
        + (3/2) · n · k · F / (Zμ)                                    # linear (first order)
        − (1/16) · n⁴ · [17n² − 3k² − 9m² + 19] · F² / (Z⁴μ³)         # quadratic (second order)

The Z and μ scaling follows from the exact scaling of the hydrogenic Stark problem (energy unit μZ², length unit 1/(Zμ), field unit μ²Z³): the scaled problem is plain hydrogen, so both terms are the textbook hydrogen coefficients divided by the appropriate power of Zμ. Verified: n=1 (k=m=0) gives −(9/4)F²/Z⁴, i.e. polarizability α = 9/(2Z⁴) a.u., the known hydrogen value at Z=1. `m_over_M` does not enter (Stark is a non-relativistic gross-structure effect).

Total sublevel count per shell = n² (spinless orbital manifold): Σ over m=−(n−1)…(n−1) of (n−|m|) = n². States with ±m are exactly degenerate (energy depends on m²).

### 2.3 Fidelity

`APPROXIMATION`, always. It is second-order perturbation theory: it neglects third-and-higher orders and, crucially, the Stark manifold is not truly bound, a static field ionizes the atom (the true spectrum is a set of resonances, not eigenvalues). The perturbation series is asymptotic and diverges as F approaches the classical field-ionization threshold

    F_ion ≈ Z³μ² / (16 n⁴)   (atomic units).

There is **no COUNTERFACTUAL tier here**: the non-relativistic Stark shift is independent of α, so altering α (the What-If knob) does not change it. The engine takes no `alpha` argument. This is itself an honesty point worth stating in the badge, not a gap.

**Error estimate** attached to every sublevel: the size of the leading neglected physics, taken as the retained quadratic term scaled by the perturbation parameter F/F_ion,

    err ≈ |E⁽²⁾| · (F / F_ion) = |E⁽²⁾| · 16 n⁴ F / (Z³μ²).

This is ∝ F³ (the correct cubic order of the first neglected term), is small deep in the perturbative regime, and grows to order |E⁽²⁾| exactly as the field nears ionization and the series breaks down. It is labelled an order-of-magnitude scale, not an exact third-order coefficient, the engine never reports a higher-order *energy* it is unsure of.

## 3. Scope

**In:** a new closed-form engine `analytic/stark.py`; `/api/levels?...&e_field=<MV/m>` attaching parabolic `sublevels` to each **gross** level; the Levels view gains an F (electric-field) slider and a zoomed Stark-manifold column for the selected n; deep-link param `e`; validation tests at every layer.

**Out (YAGNI, named as next rungs):**
- Stark line-splitting in the Spectrum view.
- Cloud/plane rendering of parabolic eigenstates (mixed l needs new wavefunction plumbing).
- The Stark-vs-fine-structure low-field crossover (quadratic→linear as F overtakes the fine-structure splitting), this phase acts on the gross Bohr shells only, disclosed in the caption.
- Simultaneous B and E fields (crossed/parallel fields is a distinct hard problem); the Stark zoom column and the Zeeman zoom column are mutually exclusive in the view (Stark takes the right column whenever e_field > 0).
- Field ionization *rates* (complex-energy resonances); we only mark where perturbation theory is trustworthy.

## 4. Key decisions

| Question | Decision |
|---|---|
| Coordinates | Parabolic (n₁, n₂, m); closed-form linear + quadratic. Rejected: numerical diagonalization of the n-manifold in the spherical basis, the closed form is exact-of-model with zero numerical error, matching the Zeeman module's spirit. |
| Order | Through second order. Linear is the headline (the degeneracy signature); quadratic gives the fan its curvature and the n=1 polarizability check. Third order is used only as an error scale, never reported. |
| Attachment point | The **gross** levels (`GrossLevelModel.sublevels`), not the fine levels. Stark is a gross-structure effect that mixes l; hanging it on (n,l,j) would misrepresent the physics. |
| Fine-structure gating | **None.** The F slider is always active. Contrast with Zeeman, which required fine structure. Disclosed in the caption. |
| Fidelity | `APPROXIMATION` always; no α dependence, so no COUNTERFACTUAL branch. |
| Field unit / range | Slider in MV/m (megavolt per metre), 0-100 MV/m, carried end-to-end (store, URL, API) exactly as Tesla was for Zeeman. Engine converts MV/m→a.u. via the atomic-unit-of-electric-field anchor, appended to provenance `method`. Range chosen so the n=2-4 fans are clearly visible and the n=4 error estimate visibly balloons as its F_ion (~125 MV/m) is approached. |
| Screened atoms | Ignored (the screened branch returns `ScreenedLevelsModel`, no gross manifold), same as Zeeman. |

## 5. Engine (`analytic/stark.py`)

- `E0_V_PER_M` anchor added to `constants.py` (atomic unit of electric field, V/m).
- `@dataclass(frozen=True) StarkSublevel(n1: int, n2: int, m: int, k: int, energy: Quantity)`.
- `stark_sublevels(n, Z=1, mu_ratio=1.0, field_mv_per_m=0.0) -> list[StarkSublevel]`.
- Enumerates every parabolic state of shell n (n² of them), energy = Bohr + linear + quadratic, provenance `APPROXIMATION` with the F/F_ion error scale and the field-ionization refinement note.

## 6. Server (`/api/levels`)

- `StarkSublevelModel(n1, n2, m, k, energy, energy_ev)`.
- `GrossLevelModel.sublevels: list[StarkSublevelModel] | None`.
- `LevelsResponse.e_field: float = 0.0` (MV/m).
- `e_field` query param; reject `< 0` with 422; attach sublevels to each gross level when `e_field > 0`; independent of `fine_structure`/`dirac`.

## 7. Web

- Types: `StarkSublevel`, `GrossLevel.sublevels`, `LevelsResponse.e_field`.
- `getLevels(..., eField = 0)`; URL param `ef` (serialize whenever `eField > 0`, no fine-structure gate). Note: the short name is `ef`, not `e`, because `e` is already the charge-multiplier param in the What-If lab (`CONST_PARAMS`); reusing it would corrupt those deep links. The server/API param stays `e_field`.
- Store `eField`/`setEField` (clears `levels`); `loadLevels` threads it; refetch on `eField` change.
- Levels view: an always-on F slider (MV/m) beside the B slider; a zoomed Stark-manifold column for the selected n when `eField > 0` (reference line at the Bohr E_n, sublevels fanned by E−E_n, extreme-k states labelled), taking the right column ahead of the Zeeman/fine zoom; caption naming the linear-fan degeneracy signature and the perturbative/ionization caveat.

## 8. Tests

- **Engine** (`tests/test_stark.py`): zero-field recovers the Bohr energy for every sublevel; count = n²; parabolic constraint n=n₁+n₂+|m|+1 holds; low-field slope dE/dF = (3/2)nk/(Zμ); linear fan is traceless (Σ k = 0 over a shell); n=1 has zero linear term and quadratic coefficient −9/4 (polarizability 9/2); ±m degeneracy; Z scaling (He⁺ linear shift = ½ of H); reduced-mass scaling; provenance `APPROXIMATION`, error > 0 and monotonically increasing in F; negative field rejected.
- **Server** (`tests/test_server.py`): e_field splits gross levels (n=2 → 4 sublevels, fidelity approximation); absent without field (`e_field == 0.0`, sublevels None); negative → 422; works with `fine_structure=false` (independence); ignored for screened.
- **Web** (`urlState.test.ts`, `store.test.ts`): `e` round-trips and is omitted at 0; `setEField` clears cached levels.

## 9. Self-review anchors

- n=1 polarizability 9/2 a.u. (Z=1), the load-bearing numeric check.
- MV/m→a.u.: F[a.u.] = F[MV/m]·1e6 / 5.14220674763e11. At 100 MV/m, F = 1.944e-4 a.u.; n=2 k=1 linear shift = 3F = 5.83e-4 a.u. = 15.9 meV, visible.
- F_ion(n=2) ≈ 1/(16·16) = 3.9e-3 a.u. = 2.0e3 MV/m; F_ion(n=4) ≈ 1/(16·256) = 2.4e-4 a.u. = 125 MV/m, so the 0-100 MV/m range stays perturbative for n≤3 and honestly stresses n=4.
- Sublevel count per shell = n² (1, 4, 9, 16 for n=1..4).
