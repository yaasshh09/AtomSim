# 2026-08-04: Sulfur, chlorine, and the screened density

Not a numbered phase. Five pieces of follow-on work after Phase 27, each one
found by using what Phases 26 and 27 shipped rather than planned in advance.
Written after the fact, because that is what happened; the commits carry the
per-change reasoning and this collects what a reader needs next.

## 1. Sulfur and chlorine exist now

`ATOM_KEYS` was answering two questions with one list: which keys name an atom,
and which atoms the GSZ screened model has parameters for. Those were the same
list because GSZ used to be the only model that could draw anything, and the
cost was that sulfur and chlorine were absent from the whole application,
including the Levels view, which runs on Hartree-Fock and has never needed a
fitted table.

- `ATOM_KEYS` is now all 17 atoms He..Ar. `GSZ_ATOM_KEYS` is the 15 the screened
  model can speak for. `has_gsz_parameters(z)` is the predicate.
- `server/app.py:_gsz_element(key)` sits at the top of all seven screened
  branches and returns 400 naming Szydlik and Green and redirecting to
  `model='hf'`. The Hartree-Fock branches keep the bare `atom_for_key`.
  **Which of the two you call is the capability claim**, so check before adding
  a branch.
- `SystemModel.has_gsz` is derived inside `from_atom`, never passed in. The
  picker disables the GSZ radio with the reason beside it, and `resolveModel`
  moves the selection to Hartree-Fock for an atom the screened model cannot
  serve.
- `ensureHF` waits for the systems table before deciding, because a deep link
  `?system=s&model=gsz` is only knowably wrong once the table has landed. Before
  that, the browser showed exactly one 400 racing the table.

## 2. The thumbnail strip greys what it cannot draw

The n and l pickers already refused an empty subshell under Hartree-Fock and
the strip beneath them still offered the same states, so a 3d tile for chlorine
was the refused request by a second route. Same `subshellAvailable` predicate,
so there is one answer rather than two.

## 3. Two layout bugs, both hiding disclosures

- `.center-col` declared its rows and left its one column implicit, which makes
  it auto-sized. The track grew to its content, 848 px inside a 680 px element,
  and every plot painted 144 px of itself under the right panel.
  `grid-template-columns: minmax(0, 1fr)` fixes it. `min-width: 0` on either
  element does nothing and is not the fix; both were already the right size.
- `.plane-frame` shrank as a flex item while the square canvas inside kept its
  size and spilled over the caption below it, which is the line disclosing the
  gamma compression. `flex-shrink: 0`, the same call `.levels-svg` already made.

## 4. The screened model gets its total density

`screened_total_radial_density` is the GSZ counterpart of Phase 27's
`hf_total_radial_density`, on the same endpoint and the same plot.

**A density needs a different box from an orbital.** `_r_max` sizes a box
around the orbital being asked for, and for one orbital that is fine because
box and orbital scale together. A density has to hold the outermost occupied
shell while resolving the innermost, and those are Z apart. Measured on neutral
argon: the `_r_max` box is 640 bohr, the default 48000 points put h at 0.013,
and the 1s peaks at 0.055. The density then **loses 0.13 of an electron and
splits the K shell into two maxima at 0.054 and 0.066 bohr**, a fourth shell
argon does not have, smooth and plausible and wrong. Neither survives being
looked for on the display grid, because neither is a display problem.

`_density_grid` sizes the box to the valence (`4(n+1)^2 / Z_net`) and the
spacing to the core (`h = 1/40Z`). The tests have teeth: restore the old box and
the electron count fails at 0.010 against a 5e-3 bar and the peak count returns
4 instead of 3.

Unlike Phase 27, the closure residual here settles at a floor rather than
converging to zero, because interpolating u and then squaring sits under the
true u^2 between solver nodes. So the test asserts the residual does not grow,
and leans on peak counting for the failure mode a tolerance cannot see.

## 5. Two wrong claims in the Phase 27 caption

Both found by reading the caption over the picture it describes.

- "The area under it is how many electrons that shell holds" is true of the
  integral in r and **not** of the visible area on the log axis this plot uses,
  which stretches the inner shell and squeezes the outer one. The caption now
  says which reading is the true one and which way the axis distorts the other.
- The count is approximate even as an integral, because shells overlap and the
  dip between two of them is not a wall. Cut argon at its minima and K holds
  2.17 electrons, not 2, under both models. A test pins the number, because a
  figure written into the interface has to be checkable.

## Open, and deliberately not acted on

**The screened solver's box costs the deep orbitals.** Measured across He..Ar:
argon's 1s comes out -111.88 hartree where the converged value in the same
model is -114.05; sodium -0.30 out, silicon -0.78, phosphorus -1.03. Halving
the box again moves argon 0.005, so the narrow box is the converged one.

Two reasons it was left alone rather than fixed:

1. **The valence energies do not care.** Under 0.06 eV across He..Ar, so every
   ionization energy the app compares against NIST is unaffected. The GSZ model
   error there is 2 to 24 percent and swamps the grid completely.
2. **It is not a quiet lie.** The grid-halving bar already reports it: argon's
   1s ships as `-108.96 +/- 11.94` hartree and sodium's as `-37.24 +/- 1.97`,
   and the converged values sit well inside both. The model says out loud that
   its deep numbers are coarse.

Tightening `_r_max` would be an accuracy improvement that moves every screened
number in the app, not a correctness fix. That is a call for whoever owns the
benchmarks, not one to make in passing.

**A side-by-side of the two densities** is the obvious next thing and is not
built. Both models now answer the same question on the same axes, the README
says the disagreement is the point, and comparing them currently means flipping
a radio button and remembering. It needs real design decisions about a second
series, so it wants a spec rather than a commit.
