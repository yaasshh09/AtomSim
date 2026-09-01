"""The Hartree-Fock total energies I vendor, for validating my SCF solver.

I never query these live: the file in ``data/`` carries its own citation,
retrieval date and transcription trail, exactly like my NIST line lists.

These are reference *data*, not something I computed, so I produce no
`Quantity` here. My fidelity claim attaches at the point of comparison: my
solver result is `NUMERICAL`, and this file is what says how far off I am
allowed to be. Read `note` in the JSON before treating any of these as a
physical total energy: they are non-relativistic, correlation-free,
clamped-nucleus numbers.
"""

import json
from importlib import resources

_FILENAME = "hf_reference_energies.json"

HF_REFERENCE = json.loads(
    resources.files("atomsim.data").joinpath(_FILENAME).read_text(encoding="utf-8")
)

__all__ = ["HF_REFERENCE", "load_hf_reference"]


def load_hf_reference(symbol: str) -> dict:
    """My reference entry for an element symbol. I raise KeyError when I have none.

    The dict I return carries ``z``, ``n_electrons``, ``configuration``,
    ``term`` and ``total_energy_hartree``.
    """
    try:
        return HF_REFERENCE["values"][symbol]
    except KeyError as exc:
        raise KeyError(
            f"I have no vendored Hartree-Fock reference energy for {symbol!r}; "
            f"I hold: {sorted(HF_REFERENCE['values'])}"
        ) from exc
