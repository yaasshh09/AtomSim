"""Provenance: every scalar (`Quantity`) or array (`Field`) I hand out says how I
computed it and how far I trust it.

This is how I mechanically enforce my prime directive: I never quietly lie
about physics.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Fidelity(Enum):
    EXACT = "exact"                    # I solved the stated model in closed form
    NUMERICAL = "numerical"            # I converged numerically, and I quantify the error
    APPROXIMATION = "approximation"    # an honest simplified model, and I state my assumptions
    COUNTERFACTUAL = "counterfactual"  # I altered the physics deliberately, then solved it rigorously
    VISUAL_LIBERTY = "visual_liberty"  # a purely presentational choice of mine, disclosed


@dataclass(frozen=True)
class Provenance:
    fidelity: Fidelity
    method: str
    assumptions: tuple[str, ...] = field(default=())
    error_estimate: float | None = None  # in the same unit as the quantity it describes
    refinement: str | None = None        # what would make me more accurate


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str
    label: str
    provenance: Provenance


@dataclass(frozen=True)
class Field:
    """An array-valued physical quantity: samples of a function on a 1-D grid.

    This completes my boundary rule: every physical value I let cross a module
    boundary is a Quantity (scalar), a Field (array), or a container carrying
    its own Provenance.
    """

    values: np.ndarray
    grid: np.ndarray
    unit: str
    grid_unit: str
    label: str
    provenance: Provenance

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        grid = np.asarray(self.grid)
        if grid.ndim != 1:
            raise ValueError(f"grid must be 1-D, got shape {grid.shape}")
        if values.shape[-1] != grid.shape[0]:
            raise ValueError(
                f"values last axis ({values.shape[-1]}) must match "
                f"grid length ({grid.shape[0]})"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "grid", grid)
