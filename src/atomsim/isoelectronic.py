"""Isoelectronic sequences: one electron count, the nuclear charge turned up.

He, Li+, Be2+, B3+ all carry two electrons. Along such a sequence the energy
has an expansion in 1/Z that perturbation theory hands over almost for free:

    E(Z) = E0 Z^2 + E1 Z + E2 + E3 / Z + ...

E0 is the bare-nucleus energy, -sum q / (2 n^2), with the electrons ignoring
each other. E1 is the first-order repulsion, <V_ee> taken over those same
hydrogenic orbitals at Z = 1, and it is the same number for Hartree-Fock and
for the exact atom, because first-order perturbation theory never sees
correlation. Everything the two models disagree about lives in E2 and below,
which is why a Hartree-Fock solve minus E0 Z^2 + E1 Z should settle onto a
constant as Z grows (for the helium sequence the literature value is
-0.111003 hartree; the exact atom's is -0.157666, and the gap, -0.0467, is
the correlation energy of a two-electron ion in the high-Z limit).

E0 is EXACT. E1 is a closed-form integral this module evaluates by
quadrature, so it is NUMERICAL with the two-mesh spread as its error bar; the
tests pin it to 5/8 for helium and to the closed forms for lithium and
beryllium. For an open shell, E1 is the average over the configuration's
terms, the same average Hartree-Fock solves for here.

The measured ionization energies the sequence is held against are vendored
in data/nist_isoelectronic.json, never queried live.
"""

import json
from dataclasses import dataclass
from importlib import resources

from atomsim.analytic.hydrogen import _radial_eval
from atomsim.atoms import Configuration, aufbau_configuration
from atomsim.numerics.hartree_fock import _interaction_energy
from atomsim.numerics.hf_terms import Subshell
from atomsim.numerics.mesh import mesh_for_atom_at_step
from atomsim.provenance import Fidelity, Provenance, Quantity

NIST_ISOELECTRONIC = json.loads(
    resources.files("atomsim.data")
    .joinpath("nist_isoelectronic.json")
    .read_text(encoding="utf-8")
)

#: The electron counts with a vendored sequence: He, Li, Be, Ne, Na-like.
SEQUENCE_ELECTRONS = tuple(sorted(int(k) for k in NIST_ISOELECTRONIC["sequences"]))

_E0_METHOD = (
    "the bare-nucleus limit: every electron in a hydrogenic orbital of charge Z, "
    "E0 = -sum q / (2 n^2) per Z^2"
)
_E1_METHOD = (
    "first-order perturbation theory: the average-of-configuration electron "
    "repulsion over hydrogenic Z = 1 orbitals, Slater integrals by cumulative "
    "quadrature on an exponential mesh"
)
_E1_ASSUMPTIONS = (
    "identical for Hartree-Fock and the exact atom: correlation first enters at E2",
    "for an open shell this is the average over the configuration's terms",
    "non-relativistic, clamped nucleus",
)
#: Box and step for the E1 quadrature. The outermost orbital supported is
#: n = 3 at Z = 1, whose density is below 1e-20 by 80 bohr.
_E1_BOX = 120.0
_E1_STEP = 0.01


@dataclass(frozen=True)
class SequenceCoefficients:
    n_electrons: int
    config: Configuration
    e0: Quantity  # hartree per Z^2
    e1: Quantity  # hartree per Z


def hydrogenic_coefficient(config: Configuration) -> Quantity:
    """E0 = -sum q / (2 n^2): the energy per Z^2 with the repulsion switched off."""
    value = -sum(q / (2.0 * n * n) for (n, _), q in config)
    return Quantity(value, "hartree", "E0", Provenance(
        fidelity=Fidelity.EXACT, method=_E0_METHOD,
        assumptions=("non-relativistic, clamped nucleus",),
    ))


def _repulsion_on_hydrogenic(config: Configuration, step: float) -> float:
    r = mesh_for_atom_at_step(1, _E1_BOX, step).r
    shells = tuple(
        Subshell(n=n, l=l, q=q, p=r * _radial_eval(n, l, r, 1.0))
        for (n, l), q in config
    )
    return _interaction_energy(shells, r)


def first_order_coefficient(config: Configuration) -> Quantity:
    """E1: the first-order electron repulsion per unit Z, in hartree."""
    if max(n for (n, _), _ in config) > 3:
        raise ValueError("E1 is evaluated for configurations up to n = 3")
    fine = _repulsion_on_hydrogenic(config, _E1_STEP / 2)
    coarse = _repulsion_on_hydrogenic(config, _E1_STEP)
    return Quantity(fine, "hartree", "E1", Provenance(
        fidelity=Fidelity.NUMERICAL, method=_E1_METHOD, assumptions=_E1_ASSUMPTIONS,
        error_estimate=abs(fine - coarse),
        refinement="a finer exponential mesh; the closed forms are rational numbers",
    ))


def sequence_coefficients(n_electrons: int) -> SequenceCoefficients:
    """E0 and E1 for the ground configuration of an n_electrons sequence."""
    config = aufbau_configuration(n_electrons)
    return SequenceCoefficients(
        n_electrons=n_electrons,
        config=config,
        e0=hydrogenic_coefficient(config),
        e1=first_order_coefficient(config),
    )


def sequence_reference(n_electrons: int) -> dict:
    """The vendored NIST entry for one sequence: ``isoelectronic_with`` and ``ions``."""
    try:
        return NIST_ISOELECTRONIC["sequences"][str(n_electrons)]
    except KeyError as exc:
        raise KeyError(
            f"no vendored isoelectronic sequence for N = {n_electrons}; "
            f"available: {list(SEQUENCE_ELECTRONS)}"
        ) from exc
