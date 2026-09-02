"""Provenance: every scalar (`Quantity`) and array (`Field`) carries a record of
how it was computed and how far it can be trusted.

That is what enforces the prime directive mechanically: nothing here quietly
lies about physics.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Fidelity(Enum):
    EXACT = "exact"                    # solved the stated model in closed form
    NUMERICAL = "numerical"            # converged numerically, with the error quantified
    APPROXIMATION = "approximation"    # an honest simplified model, assumptions stated
    COUNTERFACTUAL = "counterfactual"  # physics deliberately altered, then solved rigorously
    VISUAL_LIBERTY = "visual_liberty"  # a purely presentational choice, disclosed


@dataclass(frozen=True)
class Provenance:
    fidelity: Fidelity
    method: str
    assumptions: tuple[str, ...] = field(default=())
    error_estimate: float | None = None  # in the same unit as the quantity it describes
    refinement: str | None = None        # what would make this more accurate


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str
    label: str
    provenance: Provenance


@dataclass(frozen=True)
class Field:
    """An array-valued physical quantity: samples of a function on a 1-D grid.

    This completes the boundary rule: every physical value that crosses a module
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
