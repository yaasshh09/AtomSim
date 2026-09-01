"""How I check a guided tour's numbers against myself.

Tour content is data (``web/src/tours/*.json``) precisely so I can read exactly
what the browser renders. My alternative, restating each claim in Python, would
break single-source-of-truth on the very thing I am checking.

I keep the dispatch deliberately narrow. A resolver that silently hands back the
wrong quantity puts a green tick on prose that lies, which is worse than no test
at all, so every kind here is pinned to a closed-form value in
``tests/test_tour_claims.py`` before any tour leans on it. Adding a kind means
adding a function, an entry in ``_RESOLVERS``, an entry in ``CLAIM_KINDS`` in
``web/src/tours/types.ts``, and a test against a value known independently.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from atomsim.analytic.hydrogen import energy, mean_radius
from atomsim.atoms import ATOM_KEYS, atom_for_key, aufbau_configuration
from atomsim.constants import BOHR_RADIUS_PM, HARTREE_EV
from atomsim.hf_atom import hf_valence_ionization_energy, solve_hartree_fock
from atomsim.spectra import transition_lines
from atomsim.systems import get_system

CLAIM_KINDS: tuple[str, ...] = (
    "energy_eV",
    "mean_r_pm",
    "wavelength_nm",
    "ionization_eV",
)

#: Repo root, from ``src/atomsim/tour_claims.py``.
_TOUR_DIR = Path(__file__).resolve().parents[2] / "web" / "src" / "tours"


def load_tours() -> list[dict[str, Any]]:
    """Every tour I know of, read from the same JSON the browser bundles."""
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(_TOUR_DIR.glob("*.json"))]


def iter_claims() -> Iterator[tuple[str, str, dict[str, Any]]]:
    """(tour id, step id, claim) for every claim I find in every tour.

    A claim inherits its step's ``state`` for anything it does not name itself,
    and I restrict that to the keys my resolvers read. Restricted rather than
    merged wholesale, so a step changing a display toggle can never quietly
    change what one of its claims asserts.
    """
    inherit = (
        "system",
        "n",
        "l",
        "m",
        "model",
        "fineStructure",
        "dirac",
        "exchange",
        "pauli",
    )
    for tour in load_tours():
        for step in tour["steps"]:
            state = step.get("state", {})
            for claim in step.get("claims", []):
                merged = {k: state[k] for k in inherit if k in state}
                merged.update(claim)
                yield tour["id"], step["id"], merged


def _system_of(claim: dict[str, Any]):
    return get_system(claim.get("system", "h"))


def _energy_ev(claim: dict[str, Any]) -> float:
    system = _system_of(claim)
    q = energy(claim["n"], Z=system.Z, mu_ratio=system.mu_ratio.value)
    return q.value * HARTREE_EV


def _mean_r_pm(claim: dict[str, Any]) -> float:
    system = _system_of(claim)
    q = mean_radius(claim["n"], claim["l"], Z=system.Z, mu_ratio=system.mu_ratio.value)
    return q.value * BOHR_RADIUS_PM


def _wavelength_nm(claim: dict[str, Any]) -> float:
    system = _system_of(claim)
    n_up = claim["n_upper"]
    n_lo = claim["n_lower"]
    lines = transition_lines(system, n_max=max(n_up, n_lo), fine_structure=False)
    for line in lines.lines:
        if line.n_upper == n_up and line.n_lower == n_lo:
            return line.wavelength.value
    raise ValueError(f"no {n_up} -> {n_lo} line in {system.key}")


def _ionization_ev(claim: dict[str, Any]) -> float:
    """The Koopmans ionization energy I get for a neutral Hartree-Fock atom.

    Neutral only: the tours name elements, not ions, and inventing a charge from
    a key that does not carry one would have me answering a question nobody
    asked.
    """
    key = claim.get("system", "h")
    if key not in ATOM_KEYS:
        raise ValueError(f"ionization_eV needs a many-electron atom, got {key!r}")
    z = atom_for_key(key).z
    pauli = claim.get("pauli", True)
    exchange = claim.get("exchange", True)
    config = aufbau_configuration(z, pauli=pauli)
    result = solve_hartree_fock(z, z, config, exchange, pauli)
    return hf_valence_ionization_energy(result).value * HARTREE_EV


_RESOLVERS = {
    "energy_eV": _energy_ev,
    "mean_r_pm": _mean_r_pm,
    "wavelength_nm": _wavelength_nm,
    "ionization_eV": _ionization_ev,
}


def resolve_claim(claim: dict[str, Any]) -> float:
    """My own answer to one claim, in the unit the claim states.

    I raise rather than default on a missing input: defaulting ``n`` to 1 would
    let a claim about the 3d silently check the 1s and pass.
    """
    kind = claim.get("of")
    if kind not in _RESOLVERS:
        raise ValueError(f"unknown claim kind {kind!r}; known: {', '.join(CLAIM_KINDS)}")
    return _RESOLVERS[kind](claim)
