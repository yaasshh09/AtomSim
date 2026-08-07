# Research paper: plan, outline, and figure list

Date: 2026-08-07
Author: Yash Gupta (solo)
Status: outline for review, nothing written yet

## 0. The "one paper for everything" strategy

You cannot literally submit one file to every venue. SoftwareX enforces a rigid
section template, JOSS caps at roughly a thousand words, and every journal
forbids simultaneous submission. What you can do is write one master manuscript
at the longest useful length, structured so that every shorter version is a
deletion rather than a rewrite. Down-cutting is mechanical. Up-cutting is a new
paper.

    master manuscript (~6500 words, EJP/AJP shaped)
      |
      +-- arXiv preprint            verbatim, no changes, week 1 deliverable
      +-- EJP or AJP submission     verbatim, primary target
      +-- SoftwareX fallback        down-cut to ~3000, template remap only
      +-- JOSS                      NOT compatible, see below

Order of operations: arXiv first (it is compatible with all journals and
timestamps the work), then one journal at a time.

JOSS is dropped on purpose. Likely (unverified): JOSS declines software that is
already described in a full paper elsewhere, so it conflicts with the EJP route
rather than complementing it. It is also the weakest venue for this particular
argument, since JOSS explicitly does not assess novelty and novelty is the whole
claim.

Likely (unverified) venue lengths, to confirm against current author guides on
day 1: AJP and EJP roughly 4000 to 8000 words; SoftwareX roughly 3000 with a
fixed template; JOSS 250 to 1000.

## 1. What the paper claims

Not new physics. Every model implemented is textbook: hydrogenic analytics,
exact Dirac levels, GSZ screening, average-of-configuration Hartree-Fock, Wigner
6j algebra, Saha-Boltzmann LTE, Voigt profiles, curve of growth. Correctly
implemented textbook is worth publishing as a tool, but it is not a discovery,
and claiming otherwise would violate the project's own prime directive.

Four contributions, in descending order of strength:

1. **Fidelity as a typed, boundary-enforced property.** Every physical value
   crossing a module boundary carries a tier stating how it was computed and how
   far it can be trusted, and that tier survives the solver, the HTTP layer, and
   lands as a badge next to the picture.
2. **Validated error bars.** The quoted uncertainty is itself under test against
   independent published references. This is the most defensible original claim
   in the paper and the least common practice in the field.
3. **Counterfactual physics as a first-class, labeled tier.** Exclusion off,
   exchange off, arbitrary force laws, free-form V(r), each computed rigorously
   and marked as deliberately altered rather than wrong.
4. **A working reference implementation**, public, tested, and deployed.

**Assumption:** contribution 1 is genuinely unclaimed in the literature. If the
day-1 survey finds prior art, the paper reframes around contribution 2, which
survives independently. See the self-attack in section 6 of this plan.

## 2. Master manuscript outline

Working title: *Honest by Construction: Provenance-Tracked Fidelity in an
Interactive Atomic-Structure Simulator*

| # | Section | Words | Notes |
|---|---------|-------|-------|
| | Abstract | 200 | |
| 1 | Introduction | 900 | The problem: codes silently blend exact, converged, approximate, and decorative numbers. A student cannot tell a solution from a cartoon. |
| 2 | Related work | 700 | Visualizers, structure codes, and the units/uncertainty/workflow-provenance literature. Establishes the gap. |
| 3 | **The fidelity model** | 1100 | Core. Five tiers, the carriers, the boundary rule, composition, error semantics, the refinement field. |
| 4 | Physics implementation | 1400 | Six subsections, one per tier grouping. |
| 5 | **Validation** | 1000 | Core. Four validation modes, the error-bar honesty result, CI enforcement. |
| 6 | The interface | 800 | Badges at point of display, the invalidation invariant, single color authority, URL state as reproducibility. |
| 7 | Pedagogical use | 900 | Counterfactual teaching, guided tours, disclosure as the actual lesson. |
| 8 | Limitations | 400 | Load-bearing for reviewer trust. Do not trim this. |
| 9 | Conclusion | 300 | |
| A | Appendix: tier assignment decision table | | |
| B | Appendix: reproducing every figure | | URLs and commands |

Target 7700 draft, trim to 6500.

Section 4 breakdown:

- 4.1 Analytic hydrogenic (EXACT): energies, radial functions, complex and real
  angular bases as equal citizens, exact Dirac levels
- 4.2 Numerical radial solver (NUMERICAL): finite differences on an exponential
  mesh, grid-halving error, arbitrary central potentials
- 4.3 Many-electron (APPROXIMATION): GSZ screening against average-of-
  configuration Hartree-Fock, Slater integrals, the SCF loop
- 4.4 Perturbations (APPROXIMATION): fine structure, Zeeman through the full
  Breit-Rabi crossover, parabolic Stark, hyperfine
- 4.5 Radiative (mixed tiers): oscillator strengths, Einstein A, 6j-resolved
  rates, LTE populations, Voigt profiles, optical depth
- 4.6 Counterfactual (COUNTERFACTUAL): exchange off, Pauli off, force law, V(r)

## 3. Figures

Each figure names the code that produces it. Nothing is drawn by hand except F1.

| ID | Figure | Source |
|----|--------|--------|
| F1 | Architecture: a value's journey from solver to badge, tier annotated at each boundary | hand-drawn SVG, the thesis in one picture |
| F2 | The five tiers, one real example of each, side by side | `plane.py`, `sampling.py`, composite |
| F3 | Grid convergence: observed order matches theory, log-log | `numerics/analysis.py`, `test_radial_solver.py` |
| F4 | Computed vs NIST wavelengths, residuals colored by tier | `spectra.py`, `data/nist_*.json` |
| F5 | GSZ vs HF total radial density for argon, one axis | `density_compare.py`, `hf_atom.py` |
| **F6** | **Quoted error bar vs actual deviation, one point per orbital, y=x line** | **promoted from `test_hf_atom.py:59,128`** |
| F7 | Counterfactual: argon with and without exclusion, density collapse to 1s^18 | `test_hf_pauli.py` path |
| F8 | Counterfactual: helium Hartree vs Hartree-Fock, and the exact two-electron zero | `test_hf_exchange.py` path |
| F9 | Annotated UI: badge, assumptions, error bar, refinement hint | screenshot |
| F10 | Tier propagation along a pipeline: exact levels, approximate populations, numerical profile | `populations.py`, `broadening.py`, `transfer.py` |

**F6 is the paper.** Every point must sit below the diagonal, which shows the
uncertainties are conservative rather than decorative. I have not seen this
figure in a physics software paper. If only one figure survives review, keep it.

Submit 7 to 8 figures. F2, F9, F10 are the first cuts, moved to supplement.

## 4. Tables

| ID | Table | Source |
|----|-------|--------|
| T1 | Five tiers: definition, invariant, example, what escalates it | `provenance.py` |
| T2 | Validation matrix: module, method, ground truth, tolerance, test file | test suite |
| T3 | HF total energies vs Bunge for He, Be, Ne, Mg, Ar, with error bars, deviations, virial ratios | `hf_reference_energies.json` |
| T4 | Hydrogenic lines vs NIST, with tier per row | `spectra.py` |
| T5 | Code metadata block | needed by SoftwareX, harmless in master |

Confirmed numbers already available in the suite, to be measured properly for
T3 and T4 rather than transcribed from tolerances:

- HF total energies within `rel=1e-4` of Bunge for all five atoms
- Hydrogen through the numerical solver: -0.5 hartree to `rel=1e-4`
- Lyman-alpha 121.567 nm to 2e-3 nm
- Fine-structure Lyman-alpha doublet 5.4e-4 nm to 5 percent
- H/D isotope shift 0.033 nm to 5 percent
- Positronium Lyman-alpha 243.0 nm to 0.1 nm
- Virial ratio 2.0 to `rel=2e-3`
- GSZ helium inside its disclosed 5 percent, Na D line inside 10 percent

## 5. Week schedule

| Day | Work | Deliverable |
|-----|------|-------------|
| 1 | Request arXiv endorsement. Prior-art survey. Lock title, abstract, contribution claims. | Novelty claim confirmed or reframed |
| 2 | `scripts/paper_data.py` emits every table from live code as CSV/JSON | No number ever typed by hand |
| 3 | All figures, vector, colorblind-safe, reusing the LUT authority | Figure set complete |
| 4 | Draft sections 3, 4, 5 | Technical core |
| 5 | Draft sections 1, 2, 6, 7, 8, 9 | Full draft |
| 6 | Number re-derivation pass, references, AI disclosure, limitations honesty | Submission-ready |
| 7 | Format, arXiv, then journal | Submitted |

Rule for day 2: every table is generated by a script that reads the live engine.
Nothing is transcribed. A hand-copied number in a paper about not lying about
numbers is the one error the paper cannot survive.

## 6. Self-attack on the main conclusion

The claim most likely to be wrong is that provenance-tracked fidelity is novel.

The attack: provenance is a mature field. W3C PROV and ProvONE standardize
lineage, VisTrails and Taverna implement it, `astropy.units` and Pint handle
dimensional correctness, and the `uncertainties` package propagates error bars
through arbitrary expressions. A reviewer will ask, reasonably, whether Fidelity
is a rebranding of work already done.

The response, which needs confirming on day 1 rather than assuming: workflow
provenance records lineage, meaning what ran, in what order, on what inputs. It
does not record epistemic status. Uncertainty libraries propagate the error bar
of a number but say nothing about whether the model itself is right, and a
COUNTERFACTUAL or VISUAL_LIBERTY value has no meaningful error bar at all, so it
falls outside what those libraries can express. The tier is orthogonal to both.

This is a real objection, not a dismissible one, so it moves the prior-art survey
to day 1 and the novelty claim stays provisional until the survey clears it.
Contribution 2, the validated error bars, does not depend on this and carries the
paper on its own if contribution 1 has to be downgraded.

## 7. Risks

- **Wrong if arXiv endorsement is required and slow, then the week-1 arXiv
  deliverable slips.** Likely (unverified): new submitters to physics categories
  need endorsement from an established author, which can take days to arrange and
  is harder solo. Start it on day 1, before any writing.
- **Wrong if the prior-art survey finds the fidelity idea published, then**
  the paper reframes around validated error bars and the deployed tool. Costs
  roughly a day, does not kill the paper.
- **Wrong if AI assistance is not disclosed, then** the paper is retractable
  later. The code was written largely with an AI assistant. Journals require a
  declaration and forbid AI authorship. This is one acknowledgments paragraph,
  written on day 6, not optional.
- **Wrong if figures are regenerated after the tables, then** they can disagree.
  Generate both from the same day-2 artifacts, never independently.
