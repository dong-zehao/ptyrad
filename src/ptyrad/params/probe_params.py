"""Configuration for the optional aberration-parametrized probe."""

import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from ptyrad.optics.aberrations import Aberrations


def coefficient_order(name):
    match = re.fullmatch(r"C([1-9])(\d+)([ab]?)", name)
    if match is None:
        raise ValueError(f"Expected a Cartesian aberration coefficient, got {name!r}")
    n, m, component = int(match[1]), int(match[2]), match[3]
    if not Aberrations._is_valid_nm(n, m) or bool(m) != bool(component):
        raise ValueError(f"Invalid Cartesian aberration coefficient {name!r}")
    return n, m, component


class CoefficientParams(BaseModel):
    model_config = {"extra": "forbid"}
    trainable: bool = True
    lr: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)


class ProbeParams(BaseModel):
    model_config = {"extra": "forbid"}
    parametrize: bool = False
    lr_gamma: float = Field(default=0.25, ge=0, allow_inf_nan=False)
    coefficients: dict[str, CoefficientParams] = Field(default_factory=dict)

    @field_validator("coefficients")
    @classmethod
    def validate_names(cls, values):
        for name in values:
            coefficient_order(name)
        return values
