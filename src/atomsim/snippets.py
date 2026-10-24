"""The Python that reproduces what a view is showing, written for the reader to copy.

Every snippet calls the same engine function the server called to draw the
view, with the same arguments, and prints the result together with its
fidelity tier, so a pasted snippet cannot quietly come back as a bare number.
The test suite executes every branch here: a snippet that does not run is
worse than none, because it reads as though it does.
"""

import re

from atomsim.atoms import atom_for_key, has_gsz_parameters, is_atom_key

_Z_KEY = re.compile(r"^z(\d+)$")


def _system_lines(system: str) -> list[str]:
    z = _Z_KEY.match(system)
    if z:
        return [
            "from atomsim.systems import hydrogen_like",
            "",
            f"system = hydrogen_like({int(z.group(1))})",
        ]
    return ["from atomsim.systems import get_system", "", f'system = get_system("{system}")']


def _atom_snippet(system: str, model: str) -> str:
    el = atom_for_key(system)
    if model == "hf" or not has_gsz_parameters(el.z):
        return "\n".join([
            "from atomsim.atoms import aufbau_configuration",
            "from atomsim.hf_atom import solve_hartree_fock",
            "",
            f"# {el.name}, self-consistent Hartree-Fock, ground configuration",
            f"result = solve_hartree_fock({el.z}, {el.z}, aufbau_configuration({el.z}))",
            'print("E_total =", result.total_energy.value, result.total_energy.unit,',
            "      result.total_energy.provenance.fidelity.name)",
            "for orbital in result.orbitals:",
            '    print(orbital.n, "spdf"[orbital.l], orbital.energy.value, "hartree")',
        ])
    return "\n".join([
        "from atomsim.atoms import aufbau_configuration",
        "from atomsim.screened_atom import solve_screened_atom",
        "",
        f"# {el.name}, the GSZ screened central field, ground configuration",
        f"result = solve_screened_atom({el.z}, {el.z}, aufbau_configuration({el.z}))",
        'print("E_total =", result.total_energy.value, result.total_energy.unit,',
        "      result.total_energy.provenance.fidelity.name)",
    ])


def snippet(
    view: str,
    n: int = 1,
    l: int = 0,
    m: int = 0,
    system: str = "h",
    basis: str = "complex",
    model: str = "gsz",
    hybrid: str | None = None,
    plane: str = "density",
    count: int = 100_000,
    sup: list[str] | None = None,
) -> str:
    """Python source reproducing `view` for this state."""
    if is_atom_key(system):
        return _atom_snippet(system, model)
    head = _system_lines(system)
    zmu = "Z=system.Z, mu_ratio=system.mu_ratio.value"

    if hybrid is not None and view in ("cloud", "plane"):
        if view == "cloud":
            body = [
                "from atomsim.hybrids import sample_hybrid",
                *head,
                f'cloud = sample_hybrid("{hybrid}", {count}, {zmu}, seed=0)',
                "print(cloud.positions.shape, cloud.provenance.fidelity.name)",
            ]
        else:
            body = [
                "from atomsim.hybrids import hybrid_plane_grid",
                *head,
                f'grid = hybrid_plane_grid("{hybrid}", "density", {zmu}, resolution=512)',
                "print(grid.label, grid.values.max(), grid.provenance.fidelity.name)",
            ]
        return "\n".join(body)

    if view == "cloud":
        body = [
            "from atomsim.sampling import sample_density",
            *head,
            f'cloud = sample_density({n}, {l}, {m}, {count}, {zmu}, seed=0, basis="{basis}")',
            "print(cloud.positions.shape, cloud.provenance.fidelity.name)",
        ]
    elif view == "plane":
        quantity = "density" if plane == "current" else plane
        imports = (
            "from atomsim.plane import current_through_plane, plane_grid"
            if plane == "current" else "from atomsim.plane import plane_grid"
        )
        body = [
            imports,
            *head,
            f'grid = plane_grid({n}, {l}, {m}, quantity="{quantity}", basis="{basis}",',
            f"                  {zmu}, resolution=512)",
        ]
        if plane == "current":
            body.append("grid = current_through_plane(grid)")
        body.append("print(grid.label, grid.unit, grid.provenance.fidelity.name)")
    elif view == "radial":
        body = [
            "import numpy as np",
            "from atomsim.analytic.hydrogen import radial_wavefunction",
            *head,
            f"r = np.linspace(1e-3, {6 * n * n}, 2000) / (system.Z * system.mu_ratio.value)",
            f"R = radial_wavefunction({n}, {l}, r, {zmu})",
            "P = r**2 * R.values**2  # the radial probability density",
            'print("integral of P dr =", np.trapezoid(P, r), R.provenance.fidelity.name)',
        ]
    elif view == "spectrum":
        body = [
            "from atomsim.spectra import transition_lines",
            *head,
            "lines = transition_lines(system, 6, intensities=True)",
            "for line in lines.lines[:5]:",
            "    print(line.n_upper, '->', line.n_lower, line.wavelength.value, 'nm',",
            "          line.einstein_a.value if line.einstein_a else None, 's^-1')",
        ]
    elif view == "evolve":
        comps = []
        for raw in sup or ["1_0_0_1_0", "2_1_0_1_0"]:
            cn, cl, cm, w, ph = raw.split("_")
            comps.append(
                f"Component({int(cn)}, {int(cl)}, {int(cm)}, {float(w)!r}, "
                f"{float(ph)!r} * math.pi)"
            )
        body = [
            "import math",
            "from atomsim.superposition import Component, evolve_density, superposition_grid",
            *head,
            "mix = [" + ", ".join(comps) + "]",
            f'grid = superposition_grid(mix, basis="{basis}", {zmu}, resolution=256)',
            "rho = evolve_density(grid, t=10.0)  # |Psi|^2 at t = 10 atomic units of time",
            "print(rho.shape, grid.beat_frequencies(), grid.provenance.fidelity.name)",
        ]
    else:  # levels, and any view that is about the energy ladder
        body = [
            "from atomsim.analytic.hydrogen import energy",
            *head,
            "for n in range(1, 7):",
            f"    e = energy(n, {zmu})",
            "    print(n, e.value, e.unit, e.provenance.fidelity.name)",
        ]
    return "\n".join(body)
