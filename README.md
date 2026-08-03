# atomsim

A physically rigorous, deeply customizable quantum-mechanical atom model and
visualization platform — portfolio project, teaching tool, and self-directed
physics sandbox.

![CI](https://github.com/yaasshh09/atomsim/actions/workflows/ci.yml/badge.svg)

**Prime directive: the model never quietly lies about physics.** Every quantity
carries provenance:

| Badge | Meaning |
|---|---|
| `EXACT` | Closed-form solution of the stated model |
| `NUMERICAL` | Converged numerical solution with quantified error |
| `APPROXIMATION` | Honest simplified model, assumptions stated |
| `COUNTERFACTUAL` | Deliberately altered physics, computed rigorously |
| `VISUAL LIBERTY` | Purely presentational choice, disclosed |

## Status — Phase 1 complete (M4: polish)

- **Nucleus scale modes**: draw the nucleus at **true scale** (the measured
  r_rms, honestly subpixel — the caption says that IS the physics) or as a
  disclosed **visible marker** whose live magnification factor is stated
  (VISUAL LIBERTY badge). CODATA charge radii for p/d/α, cited literature for
  the triton; positronium draws nothing — a point lepton has no size, and the
  UI says so.
- **Deep links**: every app state is addressable by URL — the demo-script
  hooks for the Phase 2 guided tour.
- **Perf polish**: app chunk 60 kB after vendor splitting (three.js/katex
  cache independently); measured-FPS readout keeps rendering honest.

Everything below landed in M1–M3:

- **`atomsim serve` opens the full "Hydrogen, Honestly" app** — five views over
  any hydrogen-like (n, l, m) state, each labeled with exactly what it plots:
  1. **3D point cloud** — Monte-Carlo |ψ|² samples; colour by solid accent,
     density (inferno), or **phase as hue** (complex basis), with a live legend
     and a measured-FPS readout
  2. **2D cross-section** — y=0 plane as inferno |ψ|² or diverging signed ψ
     (honest label: ψ is real on that plane, e^{imφ} = ±1)
  3. **Radial plots** — R(r) and P(r) = r²R² with the ⟨r⟩ marker
  4. **Energy levels** — gross levels with degeneracies plus a zoomed µeV
     fine-structure column (two scales, labeled, never blended)
  5. **Spectrum** — computed emission lines vs vendored NIST ASD reference with
     a residual band at the stated tolerance
- **What-If constants lab** (Phase 2, sixth view): a continuous α slider and an
  integer-Z stepper drive a real-vs-altered energy-level diagram. Altering α
  flips the fine-structure levels to `COUNTERFACTUAL` provenance (behind a
  banner) while the Pauli-approximation error stays disclosed — and the view
  says out loud where the perturbative model breaks down instead of quietly
  lying past its validity. Deep-linkable (`?view=whatif&alpha=0.02&z=3`)
- **Thumbnail gallery** of every (l, m) state in the shell — server-rendered
  matplotlib PNGs, disclosed as gamma-brightened navigation aids
- **Layered math**: a per-view "Show the physics" KaTeX expander — the app is
  fully usable with it closed
- Honesty UI throughout: γ-compression, point glow, and axis choices are
  disclosed `VISUAL LIBERTY` provenances; what is plotted is labeled exactly
  (|ψ|² vs ψ — the fix over the classic poster's contradictory colorbar)
- One colour authority: client LUTs are generated from matplotlib
  (`scripts/gen_luts.py`), so densities look identical in thumbnails, the 2D
  canvas, and the 3D cloud
- Readouts with provenance: E (hartree/eV), ⟨R⟩ (a₀/pm), |L| (ℏ), radial +
  angular node counts, fine-structure shifts (µeV)
- Provenance system: every boundary-crossing value is a `Quantity` (scalar),
  a `Field` (array), or a container carrying its own `Provenance`
- Exact hydrogen-like physics: energies + radial wavefunctions, reduced-mass
  exact (deuterium, muonic hydrogen, positronium, He+, ...)
- Real AND complex angular bases engine-wide (chemistry p_x/d_xy orbitals vs
  L_z eigenstates — the basis choice is labeled, never hidden)
- Perturbative fine structure (spin-orbit + relativistic + Darwin) with honest
  error scales: alpha^4, nuclear recoil, and electron g-2 all quantified
- Spectral line lists with selection rules, compared against vendored NIST ASD
  reference wavelengths in CI (citation + retrieval date in-repo)
- System presets: H, D, T, muonic hydrogen, positronium, He+, generic Z
- Many-electron atoms under **two** models you can switch between, because they
  disagree and the disagreement is the point: the fitted Green-Sellin-Zachor
  screened central field (He through Ar, no S or Cl — nobody published
  parameters for them), and a self-consistent Hartree-Fock solve that needs no
  fitted parameters and so covers S and Cl too, along with ions that have no
  named preset. Both are `APPROXIMATION`, of different things, and each says
  which on its own badge
- The total radial density D(r) beside the orbitals, under both models, with
  the electron count as its error bar. This is the observable one: the orbital
  plots are a basis choice and the shell peaks are not, which is why neon has
  two of them and argon three
- Numerical radial solver for **arbitrary central potentials** (the engine that
  will power real atoms, screened models, and counterfactual force laws alike),
  validated against closed-form hydrogen and harmonic-oscillator solutions —
  see [docs/phase0-convergence.md](docs/phase0-convergence.md)
- Numerical energies ship with grid-halving error estimates
- Monte-Carlo sampler validated by Kolmogorov–Smirnov tests against analytic
  radial CDFs and angular moments

## Quickstart (Windows, native — no WSL)

Prerequisites: [docs/SETUP.md](docs/SETUP.md). From the **Miniforge Prompt**,
in the cloned repo:

    conda env create -f environment.yml
    conda activate atomsim
    powershell -ExecutionPolicy Bypass -File scripts\setup_web_node_modules.ps1
    cd web
    npm ci
    npm run build
    cd ..
    atomsim serve

Your browser opens the app: pick a state (n, l, m), press **Sample**, and explore
a Monte-Carlo point cloud of |ψ|² — click any provenance badge to see the method,
assumptions, and error scale behind the number it labels.

Every app state is addressable by URL (deep links — also the demo-script hooks
for the Phase 2 guided tour), e.g.:

    http://127.0.0.1:8000/?n=3&l=1&m=-1&system=mu-h&view=plane&plane=psi
    http://127.0.0.1:8000/?n=2&l=1&m=1&color=phase&nucleus=true-scale

To run the validation suites:

    pytest          # physics + server (from the repo root)
    cd web && npm test   # frontend

## Roadmap

Full specification: [docs/superpowers/specs/2026-07-04-atom-sim-requirements-design.md](docs/superpowers/specs/2026-07-04-atom-sim-requirements-design.md).
Phase 1 — "Hydrogen, Honestly" — is underway
([design](docs/superpowers/specs/2026-07-05-phase1-hydrogen-honestly-design.md)):
milestones M1 (walking skeleton), M2 (engine depth: real orbitals, fine
structure, spectra vs NIST) and M3 (UI depth: five views, colour modes,
gallery, layered math — deps added: matplotlib, d3-scale, katex) of 4 are
done. Next: M4 polish. Poster mode lands in Phase 2 and will reuse the M3
plane-grid and thumbnail machinery.

## License

MIT
