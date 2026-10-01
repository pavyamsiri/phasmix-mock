"""Shared typed structures used by the recipe format parsers."""

from __future__ import annotations

from typing import Final, Literal, NotRequired, TypedDict


class _Axis(TypedDict):
    min: float
    max: float
    bins: int


class _Grid(TypedDict):
    x: _Axis
    y: _Axis


class _GaussianParameters(TypedDict):
    x_scale: float
    y_scale: float
    amplitude: float
    variance: float
    x_offset: float
    y_offset: float


class _AlinderParameters(TypedDict):
    alpha: float
    b: float
    c: float
    theta0: float
    scale_factor: float
    rho: float
    winding: Literal[-1, 1]
    flattening_strength: float


class _GaussianEntry(TypedDict):
    type: Literal["gaussian-v1"]
    parameters: _GaussianParameters


class _AlinderEntry(TypedDict):
    type: Literal["alinder-v1"]
    parameters: _AlinderParameters


type ComponentEntry = _GaussianEntry | _AlinderEntry


class _Sampling(TypedDict):
    count: int
    seed: int
    bit_generator: Literal["PCG64"]


class ParsedRecipeData(TypedDict):
    """Typed dictionary returned by a validated recipe parser."""

    format: Literal["phasmix"]
    version: Literal[1]
    sampler: Literal["grid-jitter-v1"]
    grid: _Grid
    background: list[ComponentEntry]
    signal: list[ComponentEntry]
    sampling: _Sampling
    metadata: NotRequired[dict[str, str]]


__all__: Final[list[str]] = ["ComponentEntry", "ParsedRecipeData"]
