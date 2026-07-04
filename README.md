# atomsim

A physically rigorous, deeply customizable quantum-mechanical atom model and
visualization platform — portfolio project, teaching tool, and self-directed
physics sandbox.

![CI](https://github.com/OWNER/atom_sim/actions/workflows/ci.yml/badge.svg)

**Prime directive: the model never quietly lies about physics.** Every quantity
carries provenance:

| Badge | Meaning |
|---|---|
| `EXACT` | Closed-form solution of the stated model |
| `NUMERICAL` | Converged numerical solution with quantified error |
| `APPROXIMATION` | Honest simplified model, assumptions stated |
| `COUNTERFACTUAL` | Deliberately altered physics, computed rigorously |
| `VISUAL LIBERTY` | Purely presentational choice, disclosed |

## Status — Phase 0 complete

- Provenance system: every physical quantity is a `Quantity` with `Provenance`
- Exact hydrogen-like physics: energies + radial wavefunctions, reduced-mass
  exact (deuterium, muonic hydrogen, positronium, He+, ...)
- Numerical radial solver for **arbitrary central potentials** (the engine that
  will power real atoms, screened models, and counterfactual force laws alike),
  validated against closed-form hydrogen and harmonic-oscillator solutions —
  see [docs/phase0-convergence.md](docs/phase0-convergence.md)
- Numerical energies ship with grid-halving error estimates

## Quickstart (Windows, native — no WSL)

Prerequisites: [docs/SETUP.md](docs/SETUP.md). Then:

    conda env create -f environment.yml
    conda activate atomsim
    pytest

## Roadmap

Full specification: [docs/superpowers/specs/2026-07-04-atom-sim-requirements-design.md](docs/superpowers/specs/2026-07-04-atom-sim-requirements-design.md).
Next: Phase 1 — "Hydrogen, Honestly": 3D orbital visualization (point cloud +
isosurfaces, complex phase as hue), fine structure, spectra vs NIST, and the
honesty-badge UI, served from a local Python engine into a WebGL browser front
end.

## License

MIT
