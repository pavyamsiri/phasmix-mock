"""Portable, versioned recipes for (mostly)-deterministic mock generation."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import TYPE_CHECKING, assert_never, override

import numpy as np

from phasmock.component import AlinderComponent, Component, GaussianComponent
from phasmock.mock import MockModel, MockParticles

if TYPE_CHECKING:
    from typing import Final, Literal

    from optype import numpy as onp

    from phasmock._recipe_types import ComponentEntry, ParsedRecipeData

type SupportedRng = Literal["PCG64"]


@dataclass(frozen=True)
class RngSpec:
    """Specification of the RNG used for mock data generators.

    Attributes
    ----------
    kind : SupportedRng
        The type of RNG.
    seed : int
        The RNG seed.

    """

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
        rng: RngSpec,
        description: str | None,
    ) -> None:
        self._model: Final[MockModel] = model
        self._num_samples: Final[int] = num_samples
        self._x_edges: Final[onp.Array1D[np.float64]] = x_edges
        self._y_edges: Final[onp.Array1D[np.float64]] = y_edges
        self._rng: Final[RngSpec] = rng
        self._description: Final[str | None] = description

    @override
    def __str__(self) -> str:
        x_grid = (
            f"{self._x_edges[0]:g}..{self._x_edges[-1]:g}"
            f" ({len(self._x_edges) - 1} bins)"
        )
        y_grid = (
            f"{self._y_edges[0]:g}..{self._y_edges[-1]:g}"
            f" ({len(self._y_edges) - 1} bins)"
        )
        background = ", ".join(map(repr, self._model.background))
        signal = ", ".join(map(repr, self._model.signal))

        buffer: str = ""
        buffer += f"{type(self).__name__}("
        buffer += f"num_samples={self._num_samples}, "
        buffer += f"grid=(x: {x_grid}, y: {y_grid}), "
        buffer += f"rng={self._rng.kind}(seed={self._rng.seed}), "
        buffer += f"background=[{background}], signal=[{signal}]"
        if self._description is not None:
            buffer += f', description="{self._description}")'
        else:
            buffer += ")"

        return buffer

    def generate(self) -> MockParticles:
        return self._model.mock_particles(
            self._num_samples,
            self._x_edges,
            self._y_edges,
            rng=self._rng.to_generator(),
        )

    @classmethod
    def from_yaml(cls, path: str | PathLike[str]) -> MockRecipe:
        from phasmock._recipe_yaml import from_yaml

        return cls._from_parsed(from_yaml(path))

    @classmethod
    def from_json(cls, path: str | PathLike[str]) -> MockRecipe:
        from phasmock._recipe_json import from_json

        return cls._from_parsed(from_json(path))

    @classmethod
    def _from_parsed(cls, data: ParsedRecipeData) -> MockRecipe:
        x_edges: onp.Array1D[np.float64] = np.linspace(
            data["grid"]["x"]["min"],
            data["grid"]["x"]["max"],
            data["grid"]["x"]["bins"] + 1,
            dtype=np.float64,
        )
        y_edges: onp.Array1D[np.float64] = np.linspace(
            data["grid"]["y"]["min"],
            data["grid"]["y"]["max"],
            data["grid"]["y"]["bins"] + 1,
            dtype=np.float64,
        )
        model = MockModel(
            signal=[_make_component(component) for component in data["signal"]],
            background=[_make_component(component) for component in data["background"]],
        )
        sampling = data["sampling"]
        metadata = data.get("metadata", {})

        return cls(
            model=model,
            num_samples=sampling["count"],
            x_edges=x_edges,
            y_edges=y_edges,
            rng=RngSpec(kind=sampling["bit_generator"], seed=sampling["seed"]),
            description=metadata.get("description"),
        )


def _make_component(component: ComponentEntry) -> Component:
    match component["type"]:
        case "gaussian-v1":
            return GaussianComponent(**component["parameters"])
        case "alinder-v1":
            return AlinderComponent(**component["parameters"])
        case component_type:  # pyright: ignore[reportUnnecessaryComparison]
            assert_never(component_type)
