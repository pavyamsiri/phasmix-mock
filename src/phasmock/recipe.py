"""Portable, versioned recipes for (mostly)-deterministic mock generation."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, assert_never, override

import numpy as np

from phasmock.component import AlinderComponent, Component, GaussianComponent
from phasmock.mock import MockModel, MockParticles

if TYPE_CHECKING:
    from typing import Final, Literal

    from optype import numpy as onp

    from phasmock._recipe_types import AxisData, ComponentEntry, ParsedRecipeData

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

        # Derived quantities
        self._x_centres: Final[onp.Array1D[np.float64]] = 0.5 * (
            x_edges[:-1] + x_edges[1:]
        )
        self._y_centres: Final[onp.Array1D[np.float64]] = 0.5 * (
            y_edges[:-1] + y_edges[1:]
        )
        x_mesh, y_mesh = np.meshgrid(self._x_centres, self._y_centres)
        self._x_mesh: Final[onp.Array2D[np.float64]] = x_mesh
        self._y_mesh: Final[onp.Array2D[np.float64]] = y_mesh

    @property
    def model(self) -> MockModel:
        """MockModel: The mock data model."""
        return self._model

    @property
    def num_samples(self) -> int:
        """int: The number of samples."""
        return self._num_samples

    @property
    def x_edges(self) -> onp.Array1D[np.float64]:
        """Array1D[f64]: The edges of the bins along the x-axis."""
        return self._x_edges

    @property
    def y_edges(self) -> onp.Array1D[np.float64]:
        """Array1D[f64]: The edges of the bins along the y-axis."""
        return self._y_edges

    @property
    def x_centres(self) -> onp.Array1D[np.float64]:
        """Array1D[f64]: The centres of the bins along the x-axis."""
        return self._x_centres

    @property
    def y_centres(self) -> onp.Array1D[np.float64]:
        """Array1D[f64]: The centres of the bins along the y-axis."""
        return self._y_centres

    @property
    def x_mesh(self) -> onp.Array2D[np.float64]:
        """Array2D[f64]: The 2D mesh of the x-centres of the bins."""
        return self._x_mesh

    @property
    def y_mesh(self) -> onp.Array2D[np.float64]:
        """Array2D[f64]: The 2D mesh of the y-centres of the bins."""
        return self._y_mesh

    @property
    def rng(self) -> RngSpec:
        """RngSpec: The RNG spec."""
        return self._rng

    @property
    def description(self) -> str | None:
        """str | None: The description if set."""
        return self._description

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

    def save_yaml(self, path: str | PathLike[str]) -> None:
        """Save this recipe, using the path extension to choose a format by default.

        This function will auto-detect which format to save in from the path's file extension.
        Pass `format` to override this functionality. Unrecognised file extensions require
        an explicit `format`.

        Parameters
        ----------
        path : str  | PathLike[str]
            The path to save to.
        format : "yaml" | "json" | None
            Set this argument to either "yaml" or "json" to explicitly determine which format the recipe will
            be saved in. If this argument is `None` then the format will be detected from the extension.

        """
        from phasmock._recipe_yaml import save_yaml

        save_yaml(self._to_parsed(), path)

    def save_json(self, path: str | PathLike[str]) -> None:
        """Save this recipe as JSON.

        JSON serialization has not been implemented yet.
        """
        _ = path
        raise NotImplementedError

    def save(
        self,
        path: str | PathLike[str],
        *,
        format: Literal["yaml", "json"] | None = None,
    ) -> None:
        """Save this recipe, using the path extension to choose a format by default.

        This function will auto-detect which format to save in from the path's file extension.
        Pass `format` to override this functionality. Unrecognised file extensions require
        an explicit `format`.

        Parameters
        ----------
        path : str  | PathLike[str]
            The path to save to.
        format : "yaml" | "json" | None
            Set this argument to either "yaml" or "json" to explicitly determine which format the recipe will
            be saved in. If this argument is `None` then the format will be detected from the extension.

        """
        selected_format = format
        if selected_format is None:
            match Path(path).suffix.lower():
                case ".yaml" | ".yml":
                    selected_format = "yaml"
                case ".json":
                    selected_format = "json"
                case suffix:
                    msg = (
                        f"Cannot infer recipe format from extension {suffix!r}; "
                        "pass format='yaml' or format='json'."
                    )
                    raise ValueError(msg)

        match selected_format:
            case "yaml":
                self.save_yaml(path)
            case "json":
                self.save_json(path)
            case format_value:  # pyright: ignore[reportUnnecessaryComparison]
                assert_never(format_value)

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

    def _to_parsed(self) -> ParsedRecipeData:
        data: ParsedRecipeData = {
            "format": "phasmix",
            "version": 1,
            "sampler": "grid-jitter-v1",
            "grid": {
                "x": _axis_to_data(self._x_edges),
                "y": _axis_to_data(self._y_edges),
            },
            "background": [
                _component_to_data(component) for component in self._model.background
            ],
            "signal": [
                _component_to_data(component) for component in self._model.signal
            ],
            "sampling": {
                "count": self._num_samples,
                "seed": self._rng.seed,
                "bit_generator": self._rng.kind,
            },
        }
        if self._description is not None:
            data["metadata"] = {"description": self._description}
        return data


def _make_component(component: ComponentEntry) -> Component:
    match component["type"]:
        case "gaussian-v1":
            return GaussianComponent(**component["parameters"])
        case "alinder-v1":
            return AlinderComponent(**component["parameters"])
        case component_type:  # pyright: ignore[reportUnnecessaryComparison]
            assert_never(component_type)


def _axis_to_data(edges: onp.Array1D[np.float64]) -> AxisData:
    if len(edges) < 2:
        msg = "A recipe grid axis must contain at least two edges."
        raise ValueError(msg)

    widths = np.diff(edges)
    if not np.all(widths > 0.0) or not np.allclose(widths, widths[0]):  # pyright: ignore[reportAny]
        msg = "Recipe grid edges must be strictly increasing and evenly spaced."
        raise ValueError(msg)

    return {"min": float(edges[0]), "max": float(edges[-1]), "bins": len(edges) - 1}  # pyright: ignore[reportAny]


def _component_to_data(component: Component) -> ComponentEntry:
    if isinstance(component, GaussianComponent):
        return {
            "type": "gaussian-v1",
            "parameters": {
                "x_scale": component.x_scale,
                "y_scale": component.y_scale,
                "amplitude": component.amplitude,
                "variance": component.variance,
                "x_offset": component.x_offset,
                "y_offset": component.y_offset,
            },
        }
    if isinstance(component, AlinderComponent):
        return {
            "type": "alinder-v1",
            "parameters": {
                "alpha": component.alpha,
                "b": component.b,
                "c": component.c,
                "theta0": component.theta0,
                "scale_factor": component.scale_factor,
                "rho": component.rho,
                "winding": component.winding,
                "flattening_strength": component.flattening_strength,
            },
        }

    msg = f"Cannot serialize unsupported component type {type(component).__name__}."
    raise TypeError(msg)
