"""Portable, versioned recipes for (mostly)-deterministic mock generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never

import numpy as np

from phasmix.mock import MockModel, MockParticles

if TYPE_CHECKING:
    from typing import Final, Literal

    from optype import numpy as onp


type SupportedRng = Literal["PCG64"]


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
