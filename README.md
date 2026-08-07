# atomsim

**A quantum-mechanical atom model that never quietly lies about its physics.**

[![CI](https://github.com/yaasshh09/AtomSim/actions/workflows/ci.yml/badge.svg)](https://github.com/yaasshh09/AtomSim/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A hand-written atomic-physics engine and the instrument that reads it. Solve
hydrogen in closed form, run self-consistent Hartree-Fock on real atoms,
compare the resulting spectrum against NIST, then break the rules on purpose
and watch what the altered physics actually does.

![The atomsim instrument showing a 4f hydrogen orbital as a Monte-Carlo point cloud](docs/images/instrument.png)

## The prime directive

Every physical value that crosses a module boundary carries a `Provenance`
saying how it was obtained. Not a comment, not documentation: a data structure
the engine threads through the server and out to the browser, where the UI
renders it as a badge you can click.

| Tier | Meaning |
|---|---|
| `EXACT` | Closed-form solution of the stated model |
| `NUMERICAL` | Converged numerical solution, with quantified error |
| `APPROXIMATION` | Honest simplified model, assumptions stated |
| `COUNTERFACTUAL` | Deliberately altered physics, computed rigorously under the new rules |
| `VISUAL LIBERTY` | A presentational choice with no physical content, disclosed |

The rule this enforces is narrow and strict: a plain `float` crossing a
boundary, a silent zero, or an undisclosed presentational choice is a bug. When
the nucleus is drawn 76,000 times its true size so you can see it at all, the
picture says so.

## What it computes

**One electron, exactly.** Analytic hydrogen-like states for any Z with exact
reduced-mass treatment, which delivers deuterium and tritium isotope shifts,
muonic hydrogen, positronium, and hydrogenic ions from the same formulas.
Complex `Y_lm` and real chemistry orbitals are both first-class, and which
basis you are looking at is part of the provenance.

**Beyond the gross structure.** Perturbative fine structure (spin-orbit,
relativistic kinetic, Darwin), the exact Dirac solution to compare it against,
hyperfine splitting including the 21 cm line, and Zeeman and Stark effects with
the full Breit-Rabi crossover.

**Many electrons.** A hand-written radial Hartree-Fock stack on an exponential
mesh, self-consistent, no fitted parameters, solving neutral atoms through
argon and ions beyond it. Alongside it a GSZ screened-potential model for 15 of
those elements, so you can put two different approximations of the same atom on
one axis and see where they disagree.

**Light.** Transition strengths from dipole matrix elements (oscillator
strengths, Einstein A, lifetimes) with a Wigner 6j engine for j-resolved rates,
LTE level populations via Boltzmann and Saha, Voigt line profiles, optical
depth and the curve of growth, and synthetic absorption spectra.

**The What-If Lab.** Vary ℏ, e, mₑ, ε₀, or c and watch which dimensionless
combinations actually changed, because doubling e while quadrupling ε₀ changes
nothing observable and that is the lesson. Swap the Coulomb law for Yukawa, a
power law, a harmonic well, or an expression you type in. Switch off exchange.
Switch off Pauli exclusion and watch chemistry collapse into 1s^N. Every one of
these runs the real consequences of the altered rule under a `COUNTERFACTUAL`
banner.

<table>
<tr>
<td width="50%"><img src="docs/images/spectrum.png" alt="Computed hydrogen emission lines against NIST reference values, with a residual panel"></td>
<td width="50%"><img src="docs/images/plane.png" alt="Signed wavefunction of the 3p state on the y=0 plane"></td>
</tr>
<tr>
<td>Computed lines against vendored NIST reference data, with the fractional residual against the stated tolerance underneath.</td>
<td>ψ is real on the y=0 plane, so a signed plot is honest there. On any other plane it would not be, and the caption says which one you are looking at.</td>
</tr>
</table>

## Validation is the feature

The test suite is the argument that the physics is real, so it is written to be
readable as evidence rather than as coverage:

- The numerical solver is checked against closed-form hydrogen at production
  grid resolution, with stated tolerances, plus normalization, orthogonality,
  node counts, virial theorem, and hydrogenic ⟨r⟩ formulas.
- Monte-Carlo sampling is validated by Kolmogorov-Smirnov tests against the
  analytic CDFs, not by eyeballing a picture.
- Grid-dependent results assert a **convergence rate** under grid halving, not
  a tolerance a finer mesh could accidentally satisfy.
- Computed spectra are compared against vendored NIST Atomic Spectra Database
  values (H, D, He, Li, Na) with citation and retrieval date in-repo. No live
  queries, so the comparison is reproducible.
- Hartree-Fock energies are checked against published reference values.

That comparison runs inside pytest, which means **a physics regression fails
CI**, on every push, on a Windows runner.

## Quickstart

Windows-native, no WSL. Prerequisites are in [docs/SETUP.md](docs/SETUP.md).
From the Miniforge Prompt, in a clone of this repo:

```
conda env create -f environment.yml
conda activate atomsim
powershell -ExecutionPolicy Bypass -File scripts\setup_web_node_modules.ps1
cd web && npm ci && npm run build && cd ..
atomsim serve
```

`atomsim serve` only mounts the app if `web/dist` exists, so rebuild the
frontend after changing anything under `web/src`.

Every state is addressable by URL, which makes any moment of a session
shareable: `?n=3&l=1&m=-1&system=mu-h&view=plane&plane=psi`.

Run the suites with `pytest` from the repo root and `npm test` from `web/`.

Design notes are under [docs/specs/](docs/specs/) and the phase-by-phase
implementation plans under [docs/plans/](docs/plans/).

## License

MIT
