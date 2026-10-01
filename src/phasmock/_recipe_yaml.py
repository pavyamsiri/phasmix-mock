"""YAML-specific schema and loading for recipes."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

import strictyaml
from strictyaml.validators import Validator

if TYPE_CHECKING:
    from phasmock.recipe import MockRecipe

    from strictyaml.yamllocation import YAMLChunk


class _ComponentSchema(Validator):
    """Select a component's parameter schema using its required type tag."""

    def __init__(self, parameters: dict[str, strictyaml.Map]) -> None:
        self._parameters = parameters
        self._envelope = strictyaml.Map(
            {"type": strictyaml.Enum(list(parameters)), "parameters": strictyaml.Any()}
        )

    def __call__(self, chunk: YAMLChunk) -> strictyaml.YAML:
        component = self._envelope(chunk)
        component["parameters"].revalidate(self._parameters[component["type"].data])
        return component


def recipe_schema() -> strictyaml.Map:
    """Build the version-1 recipe schema for ``strictyaml.load``."""
    component = _ComponentSchema(
        {
            "gaussian-v1": strictyaml.Map(
                {
                    "x_scale": strictyaml.Float(),
                    "y_scale": strictyaml.Float(),
                    "amplitude": strictyaml.Float(),
                    "variance": strictyaml.Float(),
                    strictyaml.Optional("x_offset", default=0.0): strictyaml.Float(),
                    strictyaml.Optional("y_offset", default=0.0): strictyaml.Float(),
                }
            ),
            "alinder-v1": strictyaml.Map(
                {
                    "alpha": strictyaml.Float(),
                    "b": strictyaml.Float(),
                    "c": strictyaml.Float(),
                    "theta0": strictyaml.Float(),
                    "scale_factor": strictyaml.Float(),
                    "rho": strictyaml.Float(),
                    strictyaml.Optional("winding", default=1): strictyaml.Enum(
                        [-1, 1], item_validator=strictyaml.Int()
                    ),
                    strictyaml.Optional("flattening_strength", default=0.1): (
                        strictyaml.Float()
                    ),
                }
            ),
        }
    )
    axis = strictyaml.Map(
        {"min": strictyaml.Float(), "max": strictyaml.Float(), "bins": strictyaml.Int()}
    )
    return strictyaml.Map(
        {
            "format": strictyaml.Enum(["phasmix"]),
            "version": strictyaml.Enum([1], item_validator=strictyaml.Int()),
            "sampler": strictyaml.Enum(["grid-jitter-v1"]),
            "grid": strictyaml.Map({"x": axis, "y": axis}),
            "background": strictyaml.Seq(component),
            "signal": strictyaml.Seq(component) | strictyaml.EmptyList(),
            "sampling": strictyaml.Map(
                {
                    "count": strictyaml.Int(),
                    "seed": strictyaml.Int(),
                    "bit_generator": strictyaml.Enum(["PCG64"]),
                }
            ),
            strictyaml.Optional("metadata"): strictyaml.MapPattern(
                strictyaml.Str(), strictyaml.Str()
            ),
        }
    )


def from_yaml(path: str | PathLike[str]) -> MockRecipe:
    """Load and validate a YAML recipe."""
    path = Path(path)
    with path.open(mode="r") as file:
        contents = file.read()

    data = strictyaml.load(contents, recipe_schema()).data
    del data
    raise NotImplementedError
