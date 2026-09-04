# Setup: what to install on a fresh device

Everything below is native Windows. No WSL2, no Docker, no compilers.

## Required: three manual installs

| Tool | Why | Get it |
|---|---|---|
| **Git for Windows** | Version control, and it brings Git Bash along | <https://git-scm.com/download/win> |
| **Miniforge** (conda) | Runs the Python scientific environment. It has to be conda, because **Psi4 only ships via conda-forge** and no `pip install` will get you there. Skip full Anaconda; Miniforge already defaults to conda-forge | <https://conda-forge.org/download/> |
| **Node.js LTS** | Builds the TypeScript/WebGL frontend (`npm`) | <https://nodejs.org/> (pick LTS) |

You also need a browser with WebGL2. Edge ships with Windows 11, and Chrome
works too.

## Everything else installs itself

Don't hand-install Python, NumPy, SciPy, FastAPI, pytest, Psi4, or (later)
QuTiP. They're pinned in `environment.yml` and arrive in one step:

```powershell
conda env create -f environment.yml   # one command, one isolated env
conda activate atomsim
```

The frontend needs one extra step first. `node_modules` must not live under
OneDrive, where sync churn causes file-lock errors, so a script junctions it
out to local storage:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_web_node_modules.ps1
cd web
npm ci
```

> A system-wide Python (3.14 from python.org, say) can sit there harmlessly.
> The project never touches it. The conda env pins its own Python, one Psi4 is
> happy with.

## Optional (nice to have)

| Tool | Why |
|---|---|
| **VS Code** | The editor this was built in, <https://code.visualstudio.com/> |
| **GitHub CLI (`gh`)** | Publishing, PRs and CI checks without leaving the terminal, <https://cli.github.com/> |

## Check the installs

Open a fresh terminal afterwards and run:

```powershell
git --version     # any recent version
conda --version   # from Miniforge (if not on PATH, use the "Miniforge Prompt" from the Start menu)
node --version    # v22 LTS, which is what CI builds on
```

If a normal terminal doesn't recognise `conda`, nothing is broken. That's what
default Miniforge settings do. Use the **Miniforge Prompt** from the Start menu,
or call it by full path (all-users install:
`C:\ProgramData\miniforge3\condabin\conda.bat`).

## Psi4 on Windows: settled

The spec left one question open. Does Psi4 ship native Windows builds? Phase 0
answered it: **yes, Psi4 1.11 on conda-forge, win-64, Python 3.10 to 3.14.**
The probe is in [notes/psi4-windows-status.md](notes/psi4-windows-status.md).
Psi4 stays out of the current environment on purpose. It arrives with the
Hartree-Fock phase.
