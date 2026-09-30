"""Portable, versioned recipes for (mostly)-deterministic mock generation."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

import numpy as np
import strictyaml
from strictyaml.validators import Validator

from phasmix.mock import MockModel, MockParticles

if TYPE_CHECKING:
    from typing import Final, Literal

    from optype import numpy as onp
    from strictyaml.yamllocation import YAMLChunk


type SupportedRng = Literal["PCG64"]


class _ComponentSchema(Validator):
    """Select a component's parameter schema using its required type tag."""

    def __init__(self, parameters: dict[str, strictyaml.Map]) -> None:
        self._parameters = parameters
        self._envelope = strictyaml.Map(
            {
                "type": strictyaml.Enum(list(parameters)),
                "parameters": strictyaml.Any(),
            }
        )

    def __call__(self, chunk: YAMLChunk) -> strictyaml.YAML:
        component = self._envelope(chunk)
        component["parameters"].revalidate(self._parameters[component["type"].data])
        return component


def recipe_schema() -> strictyaml.Map:
    """Build the version-1 recipe schema for ``strictyaml.load``.

    Both component sequences accept ``gaussian-v1`` and ``alinder-v1``.
    Each type requires its own parameters and rejects unknown parameters.
    Optional component parameters default to the corresponding dataclass values.
    Numerical domain constraints remain the component constructors' responsibility.
    """
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
        {
            "min": strictyaml.Float(),
            "max": strictyaml.Float(),
            "bins": strictyaml.Int(),
        }
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


@dataclass(frozen=True)
class PhasmixRng:
    kind: SupportedRng
    seed: int

    def __post_init__(self) -> None:
        if self.kind not in ("PCG64",):
            msg = f"{self.kind} is not a supported RNG."
            raise ValueError(msg)

    def to_generator(self) -> np.random.Generator:
        match self.kind:
            case "PCG64":
                return np.random.Generator(np.random.PCG64(self.seed))
            case kind:  # pyright: ignore[reportUnnecessaryComparison]
                assert_never(kind)


class MockRecipe:
    def __init__(
        self,
        *,
        model: MockModel,
        num_samples: int,
        x_edges: onp.Array1D[np.float64],
        y_edges: onp.Array1D[np.float64],
        rng: PhasmixRng,
    ) -> None:
        self._model: Final[MockModel] = model
        self._num_samples: Final[int] = num_samples
        self._x_edges: Final[onp.Array1D[np.float64]] = x_edges
        self._y_edges: Final[onp.Array1D[np.float64]] = y_edges
        self._rng: Final[PhasmixRng] = rng

    def generate(self) -> MockParticles:
        return self._model.mock_particles(
            self._num_samples,
            self._x_edges,
            self._y_edges,
            rng=self._rng.to_generator(),
        )

    @staticmethod
    def from_yaml(path: str | PathLike[str]) -> MockRecipe:
        path = Path(path)

        with path.open(mode="r") as file:
            contents = file.read()

        schema = recipe_schema()
        data = strictyaml.load(contents, schema).data
        print(f"hello {str(data)}")
        raise NotImplementedError


if __name__ == "__main__":
    rec = MockRecipe.from_yaml(
        "/home/bumi/hebi/phd/phasmix/examples/two-arm_short.yaml"
    )
    print(rec)
