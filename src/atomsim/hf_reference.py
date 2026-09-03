"""The vendored Hartree-Fock total energies, for validating the SCF solver.

These are never queried live: the file in ``data/`` carries its own citation,
retrieval date and transcription trail, exactly like the NIST line lists.

They are reference *data*, not a computed result, so no `Quantity` is produced
here. The fidelity claim attaches at the point of comparison: the solver result
is `NUMERICAL`, and this file is what says how far off it is allowed to be.
Read `note` in the JSON before treating any of these as a physical total
energy: they are non-relativistic, correlation-free, clamped-nucleus numbers.
"""

import json
from importlib import resources

_FILENAME = "hf_reference_energies.json"

HF_REFERENCE = json.loads(
    resources.files("atomsim.data").joinpath(_FILENAME).read_text(encoding="utf-8")
)

__all__ = ["HF_REFERENCE", "load_hf_reference"]


def load_hf_reference(symbol: str) -> dict:
    """The reference entry for an element symbol. Raises KeyError when there is none.

    The returned dict carries ``z``, ``n_electrons``, ``configuration``,
    ``term`` and ``total_energy_hartree``.
    """
    try:
        return HF_REFERENCE["values"][symbol]
    except KeyError as exc:
        raise KeyError(
            f"no vendored Hartree-Fock reference energy for {symbol!r}; "
            f"available: {sorted(HF_REFERENCE['values'])}"
        ) from exc
