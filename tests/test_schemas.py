import inspect
import json
import typing

import numpy as np
import pytest

from atomsim.provenance import Fidelity, Field, Provenance, Quantity
from atomsim.server.schemas import FieldModel, ProvenanceModel, QuantityModel


def _prov():
    return Provenance(
        fidelity=Fidelity.NUMERICAL,
        method="test method",
        assumptions=("a1", "a2"),
        error_estimate=1e-6,
        refinement="refine",
    )


#: Floats allowed to reach the browser with no provenance on themselves or on any
#: model enclosing them. None is a computed result: each is a label, an input
#: echoed back, or a ratio whose sibling Quantity carries the provenance. The
#: reason is the entry, so adding one means writing down why it is not physics.
_NO_PROVENANCE_NEEDED = {
    ("FineLevelModel", "j"): "quantum-number label",
    ("LevelModel", "j"): "quantum-number label",
    ("LineModel", "j_upper"): "quantum-number label",
    ("LineModel", "j_lower"): "quantum-number label",
    ("HyperfineLevelModel", "F"): "quantum-number label",
    ("HyperfineShellModel", "I"): "nuclear spin label",
    ("ZeemanSublevelModel", "m_j"): "quantum-number label",
    ("ZeemanSublevelModel", "j_label"): "quantum-number label",
    ("LevelsResponse", "alpha"): "the fine-structure constant the caller asked for",
    ("LevelsResponse", "b_field"): "the field the caller asked for",
    ("LevelsResponse", "e_field"): "the field the caller asked for",
    ("ThermalModel", "temperature_k"): "the temperature the caller asked for",
    ("ThermalModel", "electron_density_cm3"): "the density the caller asked for",
    ("ForceLawModel", "params"): "the force-law parameters the caller asked for",
    ("SpectrumResponse", "tolerance_relative"): "the disclosed pass bar, not a result",
    ("JobModel", "progress"): "job progress, not physics",
    ("SystemModel", "m_over_m_nucleus"): "CODATA input; the mu_ratio Quantity beside it "
                                          "carries the provenance",
    ("DerivedObservableModel", "ratio"): "ratio of the sibling quantity to its real value; "
                                         "that Quantity carries the provenance",
    ("DiracComponentModel", "j"): "quantum-number label",
    ("FiniteSizeLevelModel", "fraction"): "ratio of the sibling shift to E_n; that Quantity "
                                          "carries the provenance",
    ("FiniteSizeModel", "coefficient_published"): "vendored published value; citation and "
                                                  "retrieval date sit beside it",
    ("FiniteSizeModel", "coefficient_published_uncertainty"): "vendored published value",
    ("LambPredictionModel", "radius_fm"): "vendored published radius; the parent carries "
                                          "the citation",
    ("LambPredictionModel", "radius_uncertainty_fm"): "vendored published value",
}


def _models(annotation) -> list:
    from pydantic import BaseModel

    found = [m for arg in typing.get_args(annotation) for m in _models(arg)]
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        found.append(annotation)
    return found


def _uncovered(model, covered: bool, out: set, seen: set) -> None:
    if (model, covered) in seen:
        return
    seen.add((model, covered))
    covered = covered or bool({"provenance", "column_provenance"} & set(model.model_fields))
    for name, f in model.model_fields.items():
        if not covered and "float" in str(f.annotation):
            out.add((model.__name__, name))
        for sub in _models(f.annotation):
            _uncovered(sub, covered, out, seen)


def test_every_float_a_route_returns_is_covered_by_a_provenance():
    """The prime directive at the HTTP boundary, checked rather than hoped for.

    Every response model of every route is walked, and each float in it must sit
    under a provenance, its own or an enclosing model's. The NIST comparison rows
    failed this until they carried one, and with it the fact that the screened
    references are air wavelengths compared against vacuum ones.
    """
    from atomsim.server.app import create_app

    uncovered: set = set()
    for route in create_app().routes:
        for model in _models(getattr(route, "response_model", None)):
            _uncovered(model, False, uncovered, set())
    assert uncovered - set(_NO_PROVENANCE_NEEDED) == set()


def test_provenance_round_trip():
    model = ProvenanceModel.from_provenance(_prov())
    data = json.loads(model.model_dump_json())
    assert data["fidelity"] == "numerical"
    assert data["assumptions"] == ["a1", "a2"]
    assert data["error_estimate"] == pytest.approx(1e-6)
    assert data["refinement"] == "refine"


def test_quantity_serializes_with_provenance():
    q = Quantity(value=-0.5, unit="hartree", label="E_1", provenance=_prov())
    data = json.loads(QuantityModel.from_quantity(q).model_dump_json())
    assert data["value"] == -0.5
    assert data["unit"] == "hartree"
    assert data["provenance"]["method"] == "test method"


def test_field_serializes_values_and_grid():
    r = np.linspace(0.0, 1.0, 5)
    f = Field(values=2.0 * r, grid=r, unit="u", grid_unit="bohr", label="f", provenance=_prov())
    data = json.loads(FieldModel.from_field(f).model_dump_json())
    assert data["grid"] == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert data["values"] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])


def test_field_model_rejects_stacked_values():
    r = np.linspace(0.0, 1.0, 4)
    f = Field(values=np.zeros((2, 4)), grid=r, unit="", grid_unit="bohr",
              label="u", provenance=_prov())
    with pytest.raises(ValueError, match="1-D"):
        FieldModel.from_field(f)


def test_every_fidelity_value_is_representable():
    for fid in Fidelity:
        p = Provenance(fidelity=fid, method="m")
        assert ProvenanceModel.from_provenance(p).fidelity == fid.value
