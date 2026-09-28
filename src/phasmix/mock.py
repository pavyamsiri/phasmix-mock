"""The data model used to generate mocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from phasmix.component import Component

if TYPE_CHECKING:
    from collections.abc import Sequence

    import optype as op
    from optype import numpy as onp


@dataclass(frozen=True)
class MockGrid:
    """Generated mock data returned in the form of a 2D grid of total number count and background number count.

    Attributes
    ----------
    density : Array2D[f64]
        The number count.
    background : Array2D[f64]
        The background number count.

    """

    density: onp.Array2D[np.float64]
    background: onp.Array2D[np.float64]

    def __post_init__(self) -> None:
        """Validate attributes."""

        _validate_grid_ndim("density", self.density, ndim=2)
        _validate_grid_ndim("background", self.background, ndim=2)

        _validate_grid_dtype("density", self.density, dtype=np.float64)
        _validate_grid_dtype("background", self.background, dtype=np.float64)

        _validate_common_shape(
            ("density", self.density), ("background", self.background)
        )


@dataclass(frozen=True)
class MockParticles:
    """Generated mock data returned in the form of particle phase space coordinates, 2D number count array and 2D background number count array.

    Attributes
    ----------
    x : Array1D[f64]
        The x coordinate/z coordinate.
    y : Array1D[f64]
        The y coordinate/Vz velocity.
    density : Array2D[f64]
        The number count.
    background : Array2D[f64]
        The background number count.

    """

    x: onp.Array1D[np.float64]
    y: onp.Array1D[np.float64]
    density: onp.Array2D[np.float64]
    background: onp.Array2D[np.float64]

    def __post_init__(self) -> None:
        """Validate attributes."""

        _validate_grid_ndim("x", self.x, ndim=1)
        _validate_grid_ndim("y", self.y, ndim=1)
        _validate_grid_ndim("density", self.density, ndim=2)
        _validate_grid_ndim("background", self.background, ndim=2)

        _validate_grid_dtype("x", self.x, dtype=np.float64)
        _validate_grid_dtype("y", self.y, dtype=np.float64)
        _validate_grid_dtype("density", self.density, dtype=np.float64)
        _validate_grid_dtype("background", self.background, dtype=np.float64)

        _validate_common_shape(("x", self.x), ("y", self.y))
        _validate_common_shape(
            ("density", self.density), ("background", self.background)
        )


class MockModel:
    """The data model used during mock data generation."""

    def __init__(
        self, signal: Sequence[Component], background: Sequence[Component]
    ) -> None:
        """Create mock data model.

        Parameters
        ----------
        signal : Sequence[Component]
            The components that comprise the signal/perturbation.
        background : Sequence[Component]
            The components that comprise the background.

        """
        self._signal: Sequence[Component] = signal
        self._background: Sequence[Component] = background

    def mock_grid(
        self, x_edges: onp.Array1D[np.float64], y_edges: onp.Array1D[np.float64]
    ) -> MockGrid:
        """Generate mock data and return the data in grid form.

        Parameters
        ----------
        x_edges : Array1D[f64]
            The bin edges in the x-axis; assumed to be monotonically increasing.
        y_edges : Array1D[f64]
            The bin edges in the y-axis; assumed to be monotonically increasing.

        Returns
        -------
        result : MockGrid
            The mock data generated in grid form.

        """
        x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
        x_mesh, y_mesh = np.meshgrid(x_centres, y_centres)
        num_x_bins: int = len(x_centres)
        num_y_bins: int = len(y_centres)

        background: onp.Array2D[np.float64] = np.zeros(
            (num_y_bins, num_x_bins), dtype=np.float64
        )

        for comp in self._background:
            background += comp(x_mesh, y_mesh)

        signal: onp.Array2D[np.float64] = np.full(
            (num_y_bins, num_x_bins), -np.inf, dtype=np.float64
        )

        for comp in self._signal:
            signal = np.maximum(signal, comp(x_mesh, y_mesh))
        signal[~np.isfinite(signal)] = 1.0

        density = background * signal
        norm = np.sum(density)

        return MockGrid(density=density / norm, background=background / norm)

    def mock_particles(
        self,
        num_samples: op.CanInt,
        x_edges: onp.Array1D[np.float64],
        y_edges: onp.Array1D[np.float64],
        *,
        rng: np.random.Generator | op.CanInt | None = None,
    ) -> MockParticles:
        """Generate mock data and return the data in particle form.

        Parameters
        ----------
        num_samples : int
            The number of particles to sample.
        x_edges : Array1D[f64]
            The bin edges in the x-axis; assumed to be monotonically increasing.
        y_edges : Array1D[f64]
            The bin edges in the y-axis; assumed to be monotonically increasing.
        rng : np.random.Generator | int | None
            The RNG to use if given or the seed to use with the default generator or a non-seeded RNG if none is given.

        Returns
        -------
        result : MockParticles
            The mock data generated in particle form.

        """
        num_samples = int(num_samples)

        dx: float = float(np.mean(np.diff(x_edges)))
        dy: float = float(np.mean(np.diff(y_edges)))
        mock_grid = self.mock_grid(x_edges, y_edges)
        background = num_samples * mock_grid.background
        density = num_samples * mock_grid.density

        if not isinstance(rng, np.random.Generator):
            if rng is not None:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(rng)

        density_flat = density.ravel()
        density_norm = density_flat / np.sum(density_flat)
        cdf_unorm = np.cumsum(density_norm)
        cdf_norm = cdf_unorm / np.sum(cdf_unorm)

        sample_indices = rng.choice(
            np.arange(len(cdf_norm)),
            size=num_samples,
            p=density_norm,
        )

        sample_indices_y, sample_indices_x = np.unravel_index(
            sample_indices, density.shape
        )

        x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])

        jitter_x = rng.uniform(-dx / 2, dx / 2, size=num_samples)
        jitter_y = rng.uniform(-dy / 2, dy / 2, size=num_samples)
        sample_x = x_centres[sample_indices_x] + jitter_x
        sample_y = y_centres[sample_indices_y] + jitter_y

        return MockParticles(
            x=sample_x, y=sample_y, density=density, background=background
        )


def _validate_grid_ndim(name: str, arr: np.ndarray, *, ndim: int) -> None:
    if arr.ndim != ndim:
        msg = f"`{name}` must be a {ndim}-dim array: {arr.ndim} != {ndim}"
        raise ValueError(msg)


def _validate_grid_dtype[T: np.generic](
    name: str, arr: np.ndarray, *, dtype: np.dtype[T] | type[T]
) -> None:
    if arr.dtype != dtype:
        msg = f"`{name}` must be an array with dtype {dtype}: {arr.dtype} != {dtype}"
        raise ValueError(msg)


def _validate_common_shape(*arrays: tuple[str, np.ndarray]) -> None:
    shape: tuple[int, ...] | None = None
    for name, arr in arrays:
        if shape is not None:
            if arr.shape != shape:
                msg = f"`{name}` should the same shape as {shape} but it has shape {arr.shape}"
                raise ValueError(msg)
        else:
            shape = arr.shape
