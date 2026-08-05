# atomsim

A physically rigorous, deeply customizable quantum-mechanical atom model and
visualization platform — portfolio project, teaching tool, and self-directed
physics sandbox.

| Badge | Meaning |
|---|---|
| `EXACT` | Closed-form solution of the stated model |
| `NUMERICAL` | Converged numerical solution with quantified error |
| `APPROXIMATION` | Honest simplified model, assumptions stated |
| `COUNTERFACTUAL` | Deliberately altered physics, computed rigorously |
| `VISUAL LIBERTY` | Purely presentational choice, disclosed |

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

To run the validation suites:

    pytest          # physics + server (from the repo root)
    cd web && npm test   # frontend

## License

MIT
