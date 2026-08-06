# Setup: What to Install on a Fresh Device

Everything below is native Windows. No WSL2, no Docker, no compilers.

## Required (manual installs: only 3 things)

| Tool | Why | Get it |
|---|---|---|
| **Git for Windows** | Version control; includes Git Bash | <https://git-scm.com/download/win> |
| **Miniforge** (conda) | Manages the Python scientific environment. Required because **Psi4 only ships via conda-forge**, a plain `pip install` cannot provide it. Do NOT use full Anaconda; Miniforge defaults to conda-forge, which is exactly what we need | <https://conda-forge.org/download/> |
| **Node.js LTS** | Builds the TypeScript/WebGL browser frontend (`npm`) | <https://nodejs.org/> (choose LTS) |

A modern browser with WebGL2 is also required, Edge (preinstalled on Windows 11) or Chrome both work.

## Everything else installs itself

You do **not** manually install Python, NumPy, SciPy, FastAPI, pytest, Psi4, or (later) QuTiP.
They are pinned in the project's `environment.yml` and created in one step:

```powershell
conda env create -f environment.yml   # one command, one isolated env
conda activate atom_sim
```

Frontend dependencies likewise come from `package.json`:

```powershell
npm install
```

> A system-wide Python (e.g., 3.14 from python.org) can coexist harmlessly, the
> project never uses it. The conda env pins its own Python version compatible
> with Psi4.

## Optional (nice to have)

| Tool | Why |
|---|---|
| **VS Code** | Recommended editor, <https://code.visualstudio.com/> |
| **GitHub CLI (`gh`)** | Easier repo publishing, PRs, CI checks from the terminal, <https://cli.github.com/> |

## Verify your installs

Open a fresh terminal after installing and check:

```powershell
git --version     # any recent version
conda --version   # from Miniforge (if not on PATH, use the "Miniforge Prompt" from the Start menu)
node --version    # v20+ LTS
```

If `conda` isn't recognized in a normal terminal, that's expected with default
Miniforge settings, use the **Miniforge Prompt** from the Start menu, or invoke
it by full path (default all-users install: `C:\ProgramData\miniforge3\condabin\conda.bat`).

## Psi4 on Windows: resolved

The spec's open question (does Psi4 ship native Windows builds?) was answered
during Phase 0: **yes, Psi4 1.11 on conda-forge, win-64, Python 3.10-3.14.**
Details in [psi4-windows-status.md](psi4-windows-status.md). Psi4 is deliberately
NOT part of the current environment; it arrives in the Hartree-Fock phase.
