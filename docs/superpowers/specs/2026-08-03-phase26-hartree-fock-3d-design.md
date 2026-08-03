# Phase 26: Hartree-Fock in three dimensions

Status: implemented. See section 10 for what building it changed.
Predecessors: Phase 21 (Hartree-Fock atoms), Phase 22 (exchange off), Phase 24
(Pauli off), Phase 25 (isosurfaces).

## 1. What this is

The Hartree-Fock stack solves real atoms, including sulfur and chlorine, which
GSZ cannot do at all because they have no fitted screening parameters. It has
been able to do this since Phase 21. What it cannot do is show you one.

`Controls.tsx` says so to the user's face today:

> Self-consistent field, solved per subshell. Reaches the Energy levels view
> only, the cloud, cross-section and radial views are still the screened model,
> and say so on their own badges.

This phase deletes that sentence. Hartree-Fock reaches the Cloud, the 2-D
cross-section, the Radial plot and the isosurface, and the two counterfactual
switches built in Phases 22 and 24 reach them with it.

Half of the engine work is already done and unreachable. `hf_isosurface` was
built and tested in Phase 25 with no route to it. `hf_radial` has existed since
Phase 21 and nothing on the server calls it. What is genuinely missing is two
engine functions, four request fields, the client branches that send them, and
one correctness problem described in section 3 that would otherwise make all of
it lie.

## 2. What gets drawn, and what that is not

The views draw the orbital, `|psi_nlm|^2` for the chosen subshell, exactly as
the screened model does. The two many-electron models stay swappable and stay
comparable, which is most of the point of having both.

The orbital is not an observable. The total electron density is, and for every
atom this solver can produce, that density is exactly spherically symmetric.
The orbitals are central-field, `P_nl(r)/r` times `Y_lm`; a filled subshell sums
over m to `(2l+1)/4pi` by Unsold's theorem; and the average-of-configuration
functional spreads a partially filled subshell equally over m, so the m-sum is
spherical there too. Carbon, oxygen and chlorine are all perfect balls. That is
not a defect in the model, it is what the model is.

So drawing the observable instead would produce a sphere for every atom in the
application, honestly and uselessly. The decision is to draw the orbital and
state the sphere in words rather than pixels. Section 6 gives the exact claim.

The counterweight belongs beside it and is also true: the angular factor here is
not a convenience or a borrowed shape. Restricted Hartree-Fock on a spherically
averaged configuration leaves the angular dependence exactly `Y_lm`, so the
lobes are the model's own answer, not hydrogen's answer reused.

## 3. The configuration trap, which is task one

`evaluate_hf_state` reaches `_occupied_orbital`, and `_occupied_orbital` calls

    solve_hartree_fock(z, n_electrons, aufbau_configuration(n_electrons))

with the configuration hardcoded and `exchange` and `pauli` left at their real
physics defaults. Nothing on the server calls that path today, so nothing lies
today. Wire it up unchanged and a user who sets `1s2 2s2 2p5 3s1` in the Levels
view gets the aufbau orbital drawn, under a Hartree-Fock badge, silently. A user
who switches exchange off gets the exchange-on orbital under a `COUNTERFACTUAL`
badge, which is worse: the badge would be advertising a departure the picture
does not contain.

`config`, `exchange` and `pauli` are threaded through `_occupied_orbital`,
`hf_radial`, `evaluate_hf_state` and `hf_isosurface` before anything renders.
The regression test in section 7 fails today and is what makes this task one
rather than a detail.

## 4. How a job asks for Hartree-Fock

`SampleRequest`, `PlaneRequest` and `IsoRequest` each gain four fields:

    model: Literal["gsz", "hf"] = "gsz"
    config: str | None = None
    exchange: bool = True
    pauli: bool = True

The system key still names the atom, and `_resolve_config` already turns a key
plus an optional config string into a configuration. The server branches on
`model` to `sample_hf_density`, `hf_plane_grid` or `hf_isosurface`. Meta models
gain `model` so the browser can say what it is looking at.

Defaults are the real, screened, already-shipped behaviour, so a client that has
never heard of any of these fields cannot accidentally ask for Hartree-Fock or
for a counterfactual. This mirrors the reasoning already written into
`HFRequest`.

Two alternatives were considered and rejected. Separate job kinds taking the raw
(Z, electron count, configuration) triple, which is what Phase 25's deferral
note assumed, would buy ions that have no preset; but a picture cannot be asked
for an atom the UI cannot select, and the Levels view already serves those ions.
It costs three endpoints, three meta models and a second client path for that.
Encoding the model into the system key as `"ar:hf"` costs no schema change and
overloads a URL contract this project treats as stable; the codebase already has
scars from key encoding, in the `"he "` and `1e+22` bugs.

## 5. Surfaces

**Engine.**

- `hf_atom.py`: `_occupied_orbital`, `hf_radial` and `evaluate_hf_state` gain
  `config`, `exchange` and `pauli` and forward them to the cached solve.
- `sampling.py`: `sample_hf_density`, beside `sample_screened_density` and with
  the same signature shape, drawing the radial part from `hf_radial`.
- `plane.py`: `hf_plane_grid`, beside `screened_plane_grid`, over the
  `evaluate_hf_state` evaluator.
- `isosurface.py`: `hf_isosurface` gains the same three parameters.

Nothing else in the engine changes shape. Both new functions follow the
evaluator route their screened twins already take, so the sampling and plane
machinery itself is untouched.

**Server.** Four fields on three existing request models, one branch each, and
`model` on the three meta models. Four refusals, described in section 6.

**Frontend.** The Cloud, Plane and Surface job payloads carry the four fields
from the store. `RadialView` learns the Hartree-Fock branch. The (n, l) picker
disables subshells the current configuration does not occupy, reading the
orbital list already present in the store's `hf` meta; a view under
`model: "hf"` triggers the Hartree-Fock solve first, the way `LevelsView` does,
rather than firing a job it can already tell will be refused. The
`Controls.tsx` sentence quoted in section 1 goes.

**URL.** No work. `model`, `config`, `exchange` as `nox=1` and `pauli` as
`nopauli=1` are all in `urlState.ts` already, including the rule that
`nopauli=1` carries `nox` with it whether or not the link says so.

## 6. Provenance

Hartree-Fock orbital pictures are `APPROXIMATION`, which is what `hf_isosurface`
already chose and for the reason `screened_isosurface` gives: take the weaker of
the two tiers, because the model's distance from the real atom is larger than
the grid's distance from the model. With `exchange=false` or `pauli=false` they
are `COUNTERFACTUAL`, matching the Hartree-Fock job.

Every Hartree-Fock 3-D view carries this claim, in the provenance assumptions
and in short form on the view itself:

> This is one orbital of a self-consistent field, and an orbital is not an
> observable. The total density of this atom is exactly spherical, so the shape
> you are looking at is a basis choice rather than a photograph.

It is a claim about the physics rather than about the rendering, so it lives in
provenance and not in `lib/liberties.ts`. The presentational disclosures already
attached to the cloud and the surface are unaffected and still apply.

The neglected correlation, and the relativity `hf_atom` already quantifies from
Z >= 9 upward, flow through unchanged from the solve. The isosurface keeps the
grid and volume error bars Phase 25 built, including the fact that the enclosed
fraction's own bar is nearly blind and the volume's is the one that moves.

Four refusals, all 422 carrying their reason, three of them reusing text that
already exists:

1. **Unoccupied subshell**, reusing `_occupied_orbital`'s existing refusal:
   Hartree-Fock builds one Fock operator per occupied subshell, so an empty one
   has no operator to be an eigenfunction of, and borrowing another subshell's
   operator would answer a different question silently.
2. **`pauli=false` for anything but 1s.** New text, same pattern. With the
   occupancy cap gone the configuration is `1s^N` and no other orbital exists.
3. **`pauli=false` with `exchange=true`**, reusing `HFRequest`'s validator.
4. **Z or outermost n outside the solver's exercised range**, reusing
   `_validate_hf_request`.

`HFConvergenceError` raised inside a job fails the job with the message intact,
as the job protocol already does.

## 7. Tests, and two exact ground truths

This tier rarely has closed-form checks available. Two are, and they are worth
spending.

**Hartree-Fock must reduce to hydrogen.** At Z=1, N=1 there is no other electron
and the Fock operator is the bare Coulomb Hamiltonian. So `sample_hf_density`
gets a KS test against the closed-form 1s CDF, exactly as the analytic sampler
does, and the Hartree-Fock 1s isosurface at 90% must land on the 2.6612 bohr
that Phase 25 validated against the closed form.

**Helium's exchange is bit-exactly zero.** Phase 22 established this. So
helium's Hartree-Fock and Hartree surfaces must come back identical to the bit,
and a multi-shell atom's must differ.

The remaining checks:

- **The configuration trap.** `evaluate_hf_state` with an explicit non-aufbau
  configuration must differ from the aufbau result. Fails today.
- **The Pauli collapse, cross-checked rather than asserted.** No direction is
  claimed for the collapsed 1s radius without deriving the sign first. Instead
  the 90% enclosure radius and the independently computed `hf_mean_radius` must
  order the collapsed and the real atom the same way, which puts two separate
  code paths on one claim.
- **Exchange off, likewise.** Assert that a multi-shell atom's two surfaces
  differ and that the provenance flips to `COUNTERFACTUAL`. No direction
  asserted without a derivation.
- **`hf_plane_grid` against `evaluate_hf_state`** on the same points, and psi
  real on the y=0 plane where the basis makes it so.
- **Server**, mirroring `test_server_iso.py` for the Hartree-Fock branch, plus
  the four refusals of section 6.
- **Frontend vitest**: the disabled-subshell logic, and that the job payload
  actually carries model, config, exchange and pauli.

## 8. Performance

`solve_hartree_fock` is `lru_cache(maxsize=8)`, so all four views share one
solve per (atom, configuration, exchange, pauli). Adding two counterfactual
flags and user-chosen configurations multiplies the key space against those
eight slots. The phase measures eviction behaviour before changing the number,
because raising it blind trades memory for a problem nobody has demonstrated.

The isosurface is the expensive path: a 96^3 grid plus the box fit plus the
halved grid for the error bar, each point going through `evaluate_hf_state`'s
interpolation. The solve is cached and the interpolation is vectorised, so the
cost profile should match the screened path; a budget test lands beside the
existing Hartree-Fock ones and records what it actually is.

## 9. Out of scope

- **Total density as a drawable quantity**, in 3-D or as a radial curve.
  Section 2 gives the reason for 3-D. The radial curve is genuinely interesting,
  since the shell structure lives there, but it is a new quantity in a view this
  phase is otherwise only branching, and it can be its own small piece later.

  *Landed immediately afterwards as Phase 27, in
  `docs/superpowers/plans/2026-08-03-phase27-total-density.md`. The Radial view
  now draws D(r) under Hartree-Fock, so the caption this phase added ("an
  orbital is not an observable") points at something the reader can see.*
- **Ions with no preset.** Section 4.
- **Showing the screened and Hartree-Fock orbitals in one camera.** The model
  selector switches between them. A comparison view is a different design.

## 10. What building it changed

Six things the design did not have right.

**The tier was hardcoded one layer below where section 6 assumed.** Section 6
says the pictures go `COUNTERFACTUAL` with a switch thrown, and reads as though
that follows from the solve. It did not. `evaluate_hf_state` wrote
`fidelity=Fidelity.APPROXIMATION` as a literal in its `Provenance` constructor,
and `hf_isosurface` passed the same literal to `_build`. Threading the flags
through without touching those two lines would have produced exactly the badge
section 3 warns about, from the opposite direction: a Hartree orbital under the
real atom's tier. Both now read the tier off the solve they came out of, and
`hf_plane_grid` and `sample_hf_density` were built that way from the start.
Three engine tests assert the flip rather than the value.

**The Pauli cross-check compared two different questions.** Section 7 proposed
ordering the collapsed and real atoms by the 90% enclosure radius and by
`hf_mean_radius`, on the grounds that two code paths agreeing on a sign is
worth more than an asserted direction. The two disagreed, and the code was
right both times: for beryllium the whole atom shrinks under the collapse
(`<r>` 1.53 to 0.53 bohr, because the 2s stops existing) while its 1s swells
(0.42 to 0.53 bohr, because four electrons in one orbital see three others
each instead of one). `hf_mean_radius` is a whole-atom average and the surface
draws one orbital. The check now runs the surface against a 1-D quadrature on
the same subshell, which is still two independent paths, and a second test
pins both facts side by side so the next reader does not call one of them a
bug. The design's instinct was right and its comparator was wrong.

**Refusal 2 is configuration-driven, not a 1s rule.** Section 6 states it as
"`pauli=false` for anything but 1s". Implemented that way it would wrongly
refuse a hand-written collapsed configuration like `1s5 2s3`, which the HF
endpoint already accepts. The check asks the actual configuration instead and
switches only its explanation on `pauli`, which covers the case section 6 names
and one it did not.

**The claim attaches in `hf_radial`, and so reaches four views from one edit.**
Section 6 scopes it to the 3-D views. `hf_radial` is upstream of all of them
plus the radial plot, and a single orbital's radial curve is no more an
observable than its lobes, so that is where it lives.

**Two frontend sentences had gone stale and one had to be re-scoped.** The
`Controls.tsx` line section 1 quotes was the known one. `InfoPanel.tsx` also
carried "other views still use the screened field", which this phase makes
false everywhere; a correction that has itself gone stale is worse than none,
because the reader was told to trust it. Separately, the subshell picker needed
the solve in order to grey anything, and under the Cloud view nothing requests
it, since sampling is a button. `Controls` now asks for the solve itself
whenever the model is Hartree-Fock, rather than relying on whichever view
happens to be open.

**Performance matched the prediction and the cache did not move.** Section 8
expected the isosurface cost profile to match the screened path: a 96^3
Hartree-Fock surface is 1.92s with the solve warm, against roughly 1.9s for the
screened one. The five cache keys a user reaches by flipping switches on one
atom fit inside `maxsize=8` with no eviction, so the number stays where it is,
now with `tests/test_hf_view_performance.py` recording why.

**Still not reachable: sulfur and chlorine.** Section 1 is right that the
engine solves them and GSZ cannot, but `ATOM_KEYS` excludes Z = 16 and 17
because they have no GSZ parameters, so neither the picture views nor the
Levels view can select them. Giving them keys touches every screened code path
and is its own change. The shipped `Controls.tsx` copy does not claim them.
