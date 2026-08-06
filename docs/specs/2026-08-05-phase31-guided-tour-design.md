# Phase 31: The guided tour

Status: designed, not implemented.
Predecessors: every phase since Phase 1. The deep-link schema in
`web/src/lib/urlState.ts` has been called "the tour hook" in six prior specs and
now carries 30+ parameters; this is the phase that spends it.

## 1. What this is

The master spec (`2026-07-04-atom-sim-requirements-design.md`, section 2) names
one flagship guided tour, **"The Hydrogen Atom, Honestly"**, and says it doubles
as the demo script for portfolio viewers. It has been the last unbuilt v1 item
since Phase 2 and the schedule guardrail lists "tour length" as the first thing
to cut, which is why it kept losing to physics.

That spec was written when the app was hydrogen only. It now spans screened and
Hartree-Fock atoms, Dirac, Zeeman, Stark, hyperfine, line strengths, LTE
populations, Voigt profiles, optical depth, absorption, isosurfaces and the
counterfactual lab. So this phase builds the flagship the spec names **plus
three short side tours**, off one engine, because a portfolio viewer who lands
on hydrogen will never discover that chlorine solves.

The tour drives the **real** application state. It does not render its own
mock-ups of the views. A tour that drew its own pictures would drift from the
app within one phase, and it would be the app lying about itself, which is the
one thing this project does not do.

## 2. What a tour is

Content is data, not code, and it lives in `web/src/tours/*.json`: one file per
tour, read by **both** halves of the project.

That one file, read twice, is the load-bearing decision in this phase. The claims
(section 5) have to be checked against the engine, and the engine is Python; the
rendering is TypeScript. A shared JSON file lets pytest read exactly what the
browser renders, with no server running, no cross-language build step, and no
duplicated content to fall out of sync. The alternatives were all worse: TS
content plus a pytest that shells out to node adds a Node dependency to the
Python suite, and claims restated in Python break single-source-of-truth on the
very thing being checked.

It sits under `web/src/` rather than in a neutral top-level `data/` because the
friction is asymmetric. Vite imports JSON from inside its own root with zero
configuration, while a path outside it needs `server.fs.allow` handling; pytest
opening a file by repo-relative path needs nothing at all. So the shared file
lives where the side with the constraint wants it. Note this is *not* next to the
vendored NIST tables in `src/atomsim/data/`, those are Python package data that
ships with an install, and tour content is not.

```json
{
  "id": "hydrogen-honestly",
  "title": "The Hydrogen Atom, Honestly",
  "blurb": "One electron, one proton, and every liberty the picture takes.",
  "steps": [
    {
      "id": "accidental-degeneracy",
      "title": "Same energy, different shape",
      "body": [
        "The 2s and the 2p sit at exactly the same -3.40 eV, and nothing about their shapes says they should.",
        "This is the accidental degeneracy, and it is accidental: it belongs to the 1/r potential and to nothing else. Screen the nucleus even slightly and it breaks."
      ],
      "state": { "n": 2, "l": 1, "m": 0, "view": "levels" },
      "spotlight": "l-picker",
      "claims": [
        { "of": "energy_eV", "n": 2, "l": 1, "is": -3.4, "tol": 0.01 },
        { "of": "energy_eV", "n": 2, "l": 0, "is": -3.4, "tol": 0.01 }
      ]
    }
  ]
}
```

### 2.1 `state`

A **partial** `UrlState`. Reusing that vocabulary means a step can name any
state the app can reach, and it inherits `parseAppUrl`'s existing hard
validation for free rather than growing a second validator that drifts.

It is applied as `{ ...URL_DEFAULTS ...step.state }`, a **full reset every
step**, never a patch onto whatever the previous step left behind. Two reasons.
A step's picture is then reproducible from that step's own data, which is what
makes the claims in section 5 mean anything. And stepping backward from step 8
to step 7 cannot leave step 8's fine structure switched on, which is the bug this
design would otherwise ship.

### 2.2 `claims`

Optional. A claim carries its own inputs and inherits the step's `state` for
anything it does not name, so it is checkable standing alone. The resolver reads
exactly one set of keys and ignores the rest of `state`: `system`, `n`, `l`, `m`,
`model`, `fineStructure`, `dirac`, `exchange`, `pauli`, `n_upper`, `n_lower`.
Anything a claim needs that is not in that list must be named on the claim
itself, so a step changing a display toggle can never quietly change what a
claim asserts. Four kinds at the start:

| `of` | Resolves to | Needs |
|---|---|---|
| `energy_eV` | level energy | system, n, (l, j when fine structure is on), model |
| `mean_r_pm` | ⟨r⟩ | system, n, l |
| `wavelength_nm` | transition wavelength | system, n_upper, n_lower |
| `ionization_eV` | valence ionization energy | system, model |

Adding a kind is one function in a dispatch table. Deliberately started narrow:
a resolver that silently returns the wrong quantity reports green on prose that
lies, which is worse than having no test, so each kind gets its own test against
a known analytic value before any tour uses it.

### 2.3 `spotlight`

Optional. Names a `data-tour` anchor on a control. If the anchor is missing or
scrolled out of view, the card renders docked with **no ring** rather than
pointing at empty space.

## 3. The frontend

| File | Job |
|---|---|
| `web/src/tours/*.json` | content, shared with pytest |
| `web/src/tours/types.ts` | `Tour`, `TourStep`, `Claim` |
| `web/src/tours/registry.ts` | imports the JSON, validates shape at module load |
| `web/src/tours/step.ts` | the pure helpers: `stepState`, `clampStep`, `spotlightBox` |
| `web/src/components/TourPanel.tsx` | the card: title, body, back/next, close |
| `web/src/components/TourSpotlight.tsx` | the ring, positioned off a `data-tour` anchor |
| `web/src/components/TourMenu.tsx` | the four tours, with blurbs |
| `web/src/state/store.ts` | `tourId`, `stepIndex`, `savedState` |
| `web/src/lib/urlState.ts` | `tour`, `step` |

The `step.ts` split is not decoration. This repo's vitest runs in the node
environment with no jsdom and no testing-library, so every existing frontend test
is a pure-function test and components are covered through the helpers they call.
The tour follows that: `stepState(step)` returning a full `UrlState`,
`clampStep(tour, i)`, and the spotlight's rectangle arithmetic are all pure and
tested, and the `.tsx` files stay thin enough that reading them is enough.

Store slice, and the one invariant it must hold: entering a tour snapshots the
reader's current `UrlState` into `savedState`, and exiting restores it. A reader
three minutes into a chlorine Hartree-Fock solve should not lose it by clicking
a tour out of curiosity.

`?tour=hydrogen-honestly&step=4` is a real address, so the demo script is a URL
you can send someone. `step` is clamped to the tour's length and an unknown
`tour` drops, matching how every other parameter in that module already behaves.

### 3.1 Behaviour

Controls stay **live** during a tour. `next` and `back` re-assert the step's
state, and there is no divergence detection or "you have changed something"
warning: this is an explorable instrument, and the whole point of driving the
real store is that fiddling mid-step is a feature. A reader who wanders off and
hits `next` simply gets the next step's state, cleanly.

No auto-advance. It fights every reader who wants to look at the picture, and
buying it back needs pause, resume and a progress meter to not feel hostile.

## 4. Content

### 4.1 Flagship: "The Hydrogen Atom, Honestly" (10 steps)

1. **One electron, one proton, an exact answer.** The 1s cloud. Why this is the
   only atom with a closed form, and why the whole app is built around that.
2. **A probability, not a shell.** True-scale nucleus. The atom is mostly nothing
   and the marker you normally see is a `VISUAL LIBERTY`.
3. **n sets the size, l sets the shape.** 2s against 2p.
4. **The accidental degeneracy.** Levels view: 2s and 2p at one energy. It
   belongs to 1/r and nothing else.
5. **m, and a basis is a choice.** Complex `Y_lm` against the real chemistry
   orbitals. Both first-class, and the basis is provenance-visible.
6. **A surface is a contour, not a boundary.** The 90% isosurface, and the 10%
   of the electron that is outside it.
7. **So the degeneracy was the model's lie.** Fine structure, α² shifts,
   `APPROXIMATION` with its three neglected scales quantified.
8. **Dirac, exactly, and the gap it still leaves.** The Lamb shift, which this
   engine does not model and says so.
9. **Levels become lines.** The spectrum against vendored NIST.
10. **What the picture cost.** The liberties ledger: every disclosed liberty
    currently on screen, in one place.

Opens on the one atom we can solve and closes on the project's actual thesis.
That shape is what makes it work as a portfolio demo as well as a lesson.

### 4.2 Side tours

- **Break the physics** (4 steps): α up, force law morphed, exchange off, Pauli
  off. Everything `COUNTERFACTUAL`, computed rigorously under the altered rules.
- **More than one electron** (4 steps): GSZ against Hartree-Fock, sulfur and
  chlorine which GSZ cannot do at all, the density comparison on one axis.
- **A spectrum you could observe** (5 steps): line strengths, LTE populations,
  Voigt profiles, the curve of growth, absorption against a continuum.

## 5. Tests

Three checks, matching how the views already earn their captions.

**vitest, structural.** Every step's `state` parses and round-trips through
`parseAppUrl`/`serializeAppUrl`. Every tour and step `id` is unique. Every
claim's `of` names a kind the Python resolver implements, checked against a
list the resolver exports, so a typo cannot silently skip a check.

Every `spotlight` names an anchor that exists. With no jsdom there is nothing to
query, so this test reads the component sources and extracts the `data-tour="…"`
literals, then asserts the tours name a subset. A source scan is the right tool
anyway: it catches an anchor deleted in a refactor, which is the failure this is
guarding, and it cannot be fooled by a component that happens not to render
under test.

**pytest, physical.** `tests/test_tour_claims.py` reads the same JSON, resolves
every claim against the real engine, and asserts it inside the claim's stated
tolerance. A step saying "-3.40 eV" fails CI the day that stops being true.
Each resolver kind additionally gets its own test against a known analytic
value, because the resolver is the piece whose failure mode is a green tick on
prose that lies.

**vitest, prose lint.** A numeral adjacent to a unit token (eV, nm, pm, bohr, K,
%, s⁻¹, m⁻²) in a `body` must be covered by a claim on that step. Targeting
measurements specifically is what keeps "four lobes", "2p" and "the fifth d
state" from tripping it.

## 6. Staging

The flagship's ten steps are enumerated above because they are the deliverable
people will actually judge. The side tours are named by content but not yet
step by step; those get settled in the plan.

The natural cut, if this has to ship in pieces, is that everything in sections 2,
3 and 5 plus the flagship is one coherent increment, and the three side tours
are content-only afterwards: new JSON files, no new code, no new resolver kinds
unless a step wants a quantity the four do not cover. Building the flagship
first is also what proves the claims resolver on the tier where the right answer
is known in closed form, before any tour leans on it for an approximation.

## 7. Out of scope

- Auto-advance and any timed playback.
- Narration audio, video export, screenshots.
- Branching tours, quizzes, progress persistence across sessions.
- Localisation. Bodies are English strings in the JSON.
- A tour authoring UI. Tours are hand-written JSON, reviewed like code.
- Rewriting the existing `ShowPhysics` layer. The tour links to views; it does
  not replace the per-view math disclosure.
