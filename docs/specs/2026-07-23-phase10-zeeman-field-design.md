# Phase 10: Zeeman Field (Breit-Rabi crossover) (design)

**Date:** 2026-07-23
**Branch:** `phase10-zeeman-field`
**Status:** approved (user picked Zeeman-first, full Breit-Rabi crossover, Levels-only, fine-structure-gated; then said "continue till tokens run out" while away)
**Parent spec:** `2026-07-04-atom-sim-requirements-design.md` (§3.1 "Zeeman and Stark effects ... external fields as first-class perturbations" / §9 Phase 3 list) · **Builds on:** `analytic/fine_structure.py`, `analytic/dirac.py`, `analytic/hydrogen.py`, the Levels view.

---

## 1. Goal and success criteria

An external magnetic field **B** breaks the rotational symmetry that makes mⱼ degenerate. Phase 10
turns on that field for hydrogen-like atoms in the Levels view and shows the **full crossover**:
at low B each fine-structure (n, l, j) level fans into 2j+1 equally-spaced mⱼ sublevels (the
anomalous Zeeman effect, spacing set by the Landé g_J), and as B rises the levels reorganize
continuously into the Paschen-Back pattern where the good labels become (m_l, m_s). That handoff of
"good quantum number" from j to (m_l, m_s) **is** the teaching payload.

The parent spec claims fields were designed as "first-class perturbations from day one." That is
aspirational: there is no perturbation-Hamiltonian machinery in the engine, only closed-form
energies (`hydrogen.py` lists "no external fields" as an assumption). Phase 10 builds the field
model the same way every other tier was built: a small, closed-form, provenance-carrying module.

**Done when:**

1. `analytic/zeeman.py` computes, for a shell n and field B, the exact eigenvalues of the
   {fine structure + linear Zeeman, gₛ=2} model. For each (l, mⱼ) the two j = l±½ states form a
   2×2 block whose eigenvalues are the closed-form Breit-Rabi roots; stretched states
   (|mⱼ| = l+½) and all of l=0 are 1×1 blocks, exactly linear in B. Each sublevel is a `Quantity`
   tagged with (l, mⱼ, branch) carrying **both** the low-field label j and the high-field label
   (m_l, m_s).
2. The diagonal source follows the level model already selected: α² fine structure by default, or
   the exact Dirac E(n, j) when Dirac mode is on. The Zeeman coupling is identical either way.
3. Fidelity is `APPROXIMATION` at real α (linear Zeeman, gₛ=2, no diamagnetic B² term), with
   `error_estimate` carrying the neglected diamagnetic scale and gₛ−2; `COUNTERFACTUAL` when α is
   altered (same seam as fine structure). In Dirac mode the exact diagonal is split by a
   perturbative Zeeman term, so the combined tier stays `APPROXIMATION`, labeled as such.
4. Validation proves the model behaves: B=0 recovers the underlying fine-structure (or Dirac)
   levels bit-for-bit; the low-field slope of each sublevel equals g_J μ_B mⱼ (Landé check); the
   high-field slopes approach integer (m_l + 2 m_s); the trace of every block is conserved for all
   B; stretched states are exactly linear.
5. `/api/levels` gains `b_field` (Tesla, default 0). When B > 0 **and** fine structure is on, each
   fine level carries a `sublevels` array of {mⱼ, branch, energy}. Tesla→atomic-units conversion
   happens at the server boundary and is appended to the provenance `method`.
6. The Levels view shows a B slider (0-20 T), active only when fine structure is on and disabled
   with a hint otherwise. Each j-level fans into its mⱼ sublevels; sweeping B shows the anomalous →
   Paschen-Back reorganization. The caption names the Breit-Rabi crossover, shows the current
   μ_B B in µeV, and names the omitted diamagnetic term.
7. The B value is deep-linkable (`b=` in Tesla) and round-trip tested.
8. Suite green in CI: engine analytic validation, server route, web logic + URL round-trip.

## 2. Physics (the exact core of the model)

Work in Hartree atomic units (ℏ = mₑ = e = 1). The Bohr magneton is μ_B = ½ a.u.; one atomic unit
of magnetic field is B₀ = 2.35051757e5 T, so a field of `B_tesla` contributes
`μ_B · B = 0.5 · (B_tesla / B₀)` in hartree per unit of the dimensionless operator below.

For a fixed shell n, the perturbation is fine structure (diagonal in l, j) plus the linear Zeeman
operator

```
H_Z / (μ_B B) = (L_z + gₛ S_z) / ℏ = J_z + S_z        (gₛ = 2 exactly)
```

In the coupled basis |l, j, mⱼ⟩, J_z = mⱼ is diagonal and S_z has the standard matrix elements
(a function of l and mⱼ only):

```
⟨l, j=l+½, mⱼ | S_z | l, j=l+½, mⱼ⟩ = + mⱼ / (2l+1)
⟨l, j=l−½, mⱼ | S_z | l, j=l−½, mⱼ⟩ = − mⱼ / (2l+1)
⟨l, j=l+½, mⱼ | S_z | l, j=l−½, mⱼ⟩ =  √((l+½)² − mⱼ²) / (2l+1)
```

So for a fixed (n, l, mⱼ) with both j present (|mⱼ| ≤ l−½), the 2×2 block is

```
        [ E_up + μ_B B · mⱼ (2l+2)/(2l+1),     μ_B B · √((l+½)²−mⱼ²)/(2l+1) ]
  H  =  [                                                                   ]
        [ μ_B B · √((l+½)²−mⱼ²)/(2l+1),     E_dn + μ_B B · mⱼ (2l)/(2l+1)   ]
```

with `E_up = E_diag(n, l, j=l+½)` and `E_dn = E_diag(n, l, j=l−½)`. Its eigenvalues are the
closed-form Breit-Rabi roots

```
E± = (H₀₀ + H₁₁)/2 ± √( ((H₀₀ − H₁₁)/2)² + H₀₁² )
```

For the stretched states |mⱼ| = l+½ (only j = l+½ exists) and for every mⱼ = ±½ of l=0, the block
is 1×1: `E = E_diag(n, l, l+½) + μ_B B · mⱼ (2l+2)/(2l+1)`, exactly linear in B. (For l=0 this is
`E_diag + μ_B B · 2 mⱼ = E_diag ± μ_B B`.)

Because the eigenvalues are a closed-form root of a 2×2, **no numerical eigensolver is used**, the
result is exact-of-the-model with zero numerical error, and the module stays pure closed form like
`fine_structure.py` and `dirac.py`.

### 2.1 The two limits (what the crossover interpolates)

- **Low field** (μ_B B ≪ fine-structure splitting): degenerate perturbation theory within a j-level
  gives `ΔE = g_J μ_B B mⱼ` with the Landé factor `g_J = 1 + [j(j+1) + s(s+1) − l(l+1)] / [2j(j+1)]`
  (s = ½). This is the anomalous Zeeman effect. The block's low-B slope reproduces g_J μ_B mⱼ, the
  first validation check.
- **High field** (μ_B B ≫ fine-structure splitting, Paschen-Back): L and S decouple, energies →
  `μ_B B (m_l + 2 m_s)` plus the (now-small) fine-structure average. The block's high-B slopes
  approach the integers m_l + 2 m_s ∈ {…, −1, 0, +1, …}.

The **good quantum number** hands off from j (low B) to (m_l, m_s) (high B). Each sublevel carries
both labels so the view can show the handoff rather than assert it.

### 2.2 Fidelity: why `APPROXIMATION`, and what it omits

The eigenvalues are exact **for the model**, but the model itself is perturbative, so the honest
tier is `APPROXIMATION` (a truncation of reality, exactly like `fine_structure.py`). The
`error_estimate` carries the scale of the omitted physics, dominated by:

- the **diamagnetic** term `e² B² ⟨r²⟩ / (8 mₑ)`, quadratic in B, kept out of the linear model
  (grows with B and with n⁴, so it is named explicitly and its scale reported);
- the QED anomalous moment gₛ−2 (≈ 0.1% of the spin part), same omission `fine_structure.py` makes;
- coupling to other n manifolds (negligible for the fields modeled).

`refinement` points at "diamagnetic (B²) term, then Paschen-Back beyond the two-effect model." When
α is altered the tier is `COUNTERFACTUAL`, threaded through the same `alpha` seam the fine-structure
and Dirac modules already use.

### 2.3 Dirac diagonal

When Dirac mode is on, `E_diag(n, l, j) = dirac_energy(n, j)` (exact, l-independent) instead of the
α² `level_energy`. The Zeeman off-diagonal is unchanged. The exact diagonal split by the
perturbative Zeeman term yields a combined `APPROXIMATION`, labeled "exact Dirac levels split by a
perturbative linear-Zeeman model." This lets the two exact-level toggles (Dirac, altered α) compose
with the field cleanly rather than being mutually exclusive.

## 3. Decisions locked in

| Question | Decision |
|---|---|
| Effect | Zeeman only this phase. Stark deferred to Phase 11 (distinct quantum-number story, parabolic coordinates). Keeps one clear teaching payload per phase, matching phases 6-9. |
| Regime | Full Breit-Rabi crossover (anomalous → Paschen-Back), not just the low-field Landé formula. The crossover is the physics; the closed-form 2×2 gives it exactly with no extra machinery. |
| Eigenvalues | Closed-form 2×2 roots, not `np.linalg.eigvalsh`. Exact-of-model, zero numerical error, no MKL/LAPACK exposure, keeps the module closed form. Rejected: numerical eigensolve, overkill for 2×2 and adds a NUMERICAL provenance wrinkle. |
| Fidelity | `APPROXIMATION` at real α; `COUNTERFACTUAL` when α altered. In Dirac mode the exact diagonal + perturbative Zeeman is still `APPROXIMATION`, labeled honestly. Rejected: `EXACT`, the linear-Zeeman model omits the diamagnetic term and gₛ−2; those are approximation error, not just model scope. |
| Diagonal source | Follows the selected level model: α² fine structure, or exact Dirac E(n,j) when Dirac is on. Same Zeeman coupling both ways. |
| Toggle gating | B slider active only when fine structure is on (it supplies the j-levels the crossover reorganizes). Disabled with a hint otherwise. No spinless "normal Zeeman" fallback this phase (deferred with Stark). |
| View scope | Levels view only. Spectrum line-splitting and cloud/plane symmetry-breaking deferred (YAGNI), named as next rungs. |
| Field units / range | Slider in Tesla, 0-20 T (the n=2 crossover sits near ~1 T, so the full reorganization is visible in range). Engine works in atomic units; Tesla→a.u. conversion at the server boundary, appended to provenance `method`, matching the SI-at-boundary convention. |
| Server API | Add `b_field: float = 0.0` to `/api/levels`. When B > 0 and fine structure is on, each fine level gains a `sublevels` array. Rejected: a separate `/api/zeeman` endpoint, needless churn to the stable levels contract. |
| Response shape | Additive: each existing `FineLevelModel` gains an optional `sublevels: list[ZeemanSublevelModel]`. B=0 or fine-structure-off responses are byte-identical to today (field omitted). |

## 4. Engine: `analytic/zeeman.py` (new)

```
zeeman_sublevels(
    n: int, l: int, Z: int = 1, mu_ratio: float = 1.0, m_over_M: float = 0.0,
    alpha: float = ALPHA, b_tesla: float = 0.0, dirac: bool = False,
) -> list[ZeemanSublevel]
```

- Validate `n >= 1`, `0 <= l < n`, `Z >= 1`, `b_tesla >= 0`.
- Build the diagonal energies `E_diag(n, l, j)` for j = l±½ from `level_energy` (perturbative) or
  `dirac_energy` (when `dirac`). For l=0 only j=½ exists.
- Convert `μ_B B = 0.5 * b_tesla / B0_TESLA` (B0 a new constant in `constants.py`).
- For each mⱼ in {−(l+½), …, +(l+½)} (step 1): form the 1×1 or 2×2 block per §2, take the
  closed-form eigenvalue(s), and emit one `ZeemanSublevel` per eigenvalue with:
  - `energy`: `Quantity` in hartree, tier per §2.2, `method` naming the Breit-Rabi model and the B
    value, assumptions listing the omissions, `error_estimate` = diamagnetic + gₛ−2 scale.
  - `m_j`: float; `branch`: "upper"/"lower" (which eigenvalue, by energy) or "single" for 1×1;
    `j_label`: the low-field j the eigenvector is closest to at this B (continuity-tracked from B=0
    so the label is stable); `high_field_label`: (m_l, m_s) the state approaches at large B.
- `ZeemanSublevel` is a small frozen dataclass: `(m_j, branch, energy, j_label, high_field_label)`.
- Helper `lande_g(l, j)` returns g_J (used by the module for `j_label` slope reporting and by a
  validation test). Export only what a test needs.

Units: hartree internally; eV/µeV conversion stays at the server boundary as everywhere else.

`constants.py`: add `B0_TESLA = 2.35051757e5` (atomic unit of magnetic field, CODATA) with a source
comment, alongside the existing SI anchors.

## 5. Server: `server/app.py` + `schemas.py`

- `/api/levels` gains `b_field: float = 0.0` (Tesla). Inert unless `b_field > 0` **and**
  `fine_structure` (or `dirac`) is on and the system is non-screened.
- When active: for each `(n, l, j)` fine entry, call `zeeman_sublevels(n, l, …, b_tesla=b_field,
  dirac=dirac)` and attach the resulting `sublevels` to the `FineLevelModel`. The parent fine
  entry keeps its B=0 energy (so the un-split reference line is still there to fan out from).
- `schemas.py`: new `ZeemanSublevelModel { m_j: float, branch: str, j_label: float,
  high_field_label: str, energy: QuantityModel }`. `FineLevelModel` gains
  `sublevels: list[ZeemanSublevelModel] | None = None`. `LevelsResponse` gains `b_field: float`
  (echo).
- Tesla→a.u. conversion and the `μ_B B` value appear in each sublevel `energy.provenance.method`.
- Validation: `b_field < 0` → 422. Screened systems ignore `b_field` (return `ScreenedLevelsModel`
  before this branch, as with `dirac`).

## 6. Web: `web/src`

- **`state/store.ts`**: add `bField: number` (default 0) + `setBField`. Setting it clears cached
  levels (stale physics never renders) and `loadLevels` passes `b_field`. Purely a Levels-fetch
  input, like `dirac`.
- **`api/client.ts` + `api/types.ts`**: `getLevels` gains `bField`; `LevelsResponse` type gains
  `b_field`; `FineLevel` gains optional `sublevels`; a `ZeemanSublevel` type mirrors the schema.
- **`components/LevelsView.tsx`**: when `fineStructure` is on, render a B slider (0-20 T, step 0.1)
  in the view header, with the live μ_B B readout in µeV. When B > 0, draw each fine level's
  `sublevels` fanned around the parent line, faint connectors from parent to sublevel; label the
  extreme sublevels with mⱼ. Caption: "A magnetic field splits each j-level into 2j+1 mⱼ sublevels
  (anomalous Zeeman, spacing g_J μ_B B). As B rises they reorganize into the Paschen-Back pattern
  where (m_l, m_s) become the good labels. Linear model: the diamagnetic B² term is omitted." When
  fine structure is off, the slider is disabled with "turn on fine structure to add a field."
- **`lib/urlState.ts`**: serialize `b` (Tesla, trimmed number) only when `bField > 0` and
  `fineStructure` on; parse hard (clamp to ≥ 0); round-trip tested. Add to `URL_DEFAULTS` and
  `main.tsx`'s serialized state.

## 7. Honesty / provenance (the heart)

- The field-split levels are `APPROXIMATION`: the eigenvalues are exact for the model, but the model
  drops the diamagnetic B² term and uses gₛ = 2, both named in assumptions with a quantified scale.
  "Exact 2×2 eigenvalues" is never overread as "exact physics."
- The crossover is shown, not hidden: both the low-field (j) and high-field (m_l, m_s) labels ride
  on every sublevel, so the handoff of good quantum number is visible.
- The low-field limit is cross-checked against the textbook Landé g_J formula and the high-field
  limit against integer (m_l + 2 m_s) in the test suite, so the model is provably consistent with
  both known regimes, not merely plausible between them.
- Dirac + Zeeman composes honestly: an exact diagonal split by a perturbative field is labeled as
  the `APPROXIMATION` it is.

## 8. Out of scope (YAGNI)

- Stark effect (Phase 11); the spinless "normal Zeeman" contrast (deferred with Stark).
- Zeeman line-splitting in the Spectrum view (Δmⱼ = 0, ±1 selection rules) and cloud/plane
  symmetry-breaking (mixed eigenstates need new wavefunction plumbing).
- The diamagnetic B² term and Paschen-Back physics beyond the two-effect model (named, scaled, not
  built).
- Hyperfine / nuclear Zeeman; screened-atom Zeeman.
- B field in the What-If constants lab beyond the α seam already threaded.

## 9. Testing (validation, not smoke)

- **Engine (`tests/test_zeeman.py`)**:
  - B=0 recovery: every sublevel energy at `b_tesla=0` equals the underlying `level_energy`
    (perturbative) / `dirac_energy` (Dirac) for its (n, l, j), to tight rtol.
  - Landé low-field slope: for a representative (n, l, j), the finite-difference slope
    `dE/dB` at small B for each mⱼ sublevel equals `lande_g(l, j) * μ_B * m_j` to O(B).
  - Paschen-Back high-field slope: at large B the sublevel slopes approach integer
    `(m_l + 2 m_s) * μ_B` (check the multiset of asymptotic slopes for a shell).
  - Trace invariance: for every (n, l, mⱼ) block and several B, the sum of block eigenvalues equals
    the block trace (`E_up + E_dn + diagonal Zeeman`), i.e. the field only redistributes.
  - Stretched states linear: |mⱼ| = l+½ and all l=0 sublevels are exactly linear in B (second
    difference ≈ 0).
  - Count: a shell (n, l) yields exactly `2(2l+1) = 4l+2` sublevels for l≥1 (both j = l±½ present:
    2l+2 from j=l+½ and 2l from j=l−½) and 2 for l=0.
  - Dirac diagonal: with `dirac=True`, B=0 recovers `dirac_energy(n, j)`.
  - Provenance: tier is `APPROXIMATION` at real α, `COUNTERFACTUAL` at altered α, `error_estimate`
    > 0 and grows with B (diamagnetic scale).
  - Supercritical guard inherited from `dirac_energy` when `dirac=True` and Z too large → `ValueError`.
- **Server (`tests/test_server.py`)**: `/api/levels?system=h&n_max=3&fine_structure=true&b_field=2`
  → 200, `b_field` echoed, a fine level with l≥1 carries `sublevels` of the right length, each
  sublevel `energy.provenance.fidelity == "approximation"`, and `b_field=0` responses have no
  `sublevels`. `b_field` on a screened system is ignored. Negative `b_field` → 422.
- **Web (`*.test.ts`)**: `getLevels` includes `b_field` in the URL when set; URL round-trip for
  `fs=1&b=2.5`; the B=0 default omits `b`; `bField` change clears the levels cache in the store.
