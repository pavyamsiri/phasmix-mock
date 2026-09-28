"""Components that compose to form a model to generate mock data from."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    cast,
    override,
    runtime_checkable,
)

import numpy as np
from scipy import special

if TYPE_CHECKING:
    from optype import numpy as onp


@runtime_checkable
class Component(Protocol):
    """A component of either the background or signal."""

    def __call__[ShapeT: tuple[Any, ...]](
        self, x: onp.ArrayND[np.float64, ShapeT], y: onp.ArrayND[np.float64, ShapeT], /
    ) -> onp.ArrayND[np.float64, ShapeT]:
        """Given x and y, return the component's contribution to the background or signal.

        Parameters
        ----------
        x : ArrayND[f64, S]
            The x coordinate.
        y : ArrayND[f64, S]
            The y coordinate.

        Returns
        -------
        value : ArrayND[f64, S]
            The component's contribution.

        """
        ...


@dataclass(frozen=True)
class GaussianComponent(Component):
    """A 2D Gaussian component.

    Attributes
    ----------
    x_scale : float
        The scale length along the x-axis.
    y_scale : float
        The scale length along the y-axis.
    amplitude : float
        The component's amplitude.
    variance : float
        The variance (square of width/standard deviation).
    x_offset : float
        The offset to the x coordinate, sets the central x-coordinate.
        Defaults to 0.
    y_offset : float
        The offset to the y coordinate, sets the central x-coordinate.
        Defaults to 0.

    """

    x_scale: float
    y_scale: float
    amplitude: float
    variance: float
    x_offset: float = 0
    y_offset: float = 0

    def __post_init__(self) -> None:
        """Validate attributes."""

        # Ensure lengths are positive
        _validate_positive("x_scale", self.x_scale)
        _validate_positive("y_scale", self.y_scale)
        _validate_positive("variance", self.variance)

        # Ensure amplitude is not negative
        _validate_nonnegative("amplitude", self.amplitude)

        # Ensure offsets are at least finite
        _validate_finite("x_offset", self.x_offset)
        _validate_finite("y_offset", self.y_offset)

    @override
    def __call__[ShapeT: tuple[Any, ...]](
        self, x: onp.ArrayND[np.float64, ShapeT], y: onp.ArrayND[np.float64, ShapeT], /
    ) -> onp.ArrayND[np.float64, ShapeT]:
        rxy = np.hypot(x / self.x_scale, y / self.y_scale).astype(np.float64)
        result = self.amplitude * np.exp(-np.square(rxy) / self.variance).astype(
            np.float64
        )
        return cast("onp.ArrayND[np.float64, ShapeT]", result)


@dataclass(frozen=True)
class AlinderComponent(Component):
    """A phase spiral component.

    Attributes
    ----------
    alpha : float
        The spiral amplitude.
    b : float
        The linear winding amplitude.
    c : float
        The quadratic winding amplitude.
    theta0 : float
        The angle offset in radians.
    scale_factor : float
        The scale factor.
    rho : float
        The flattening function distance in phase radius units.
    flattening_strength : float
        The strength of the flattening in phase radius units.
    winding : -1 or +1
        The winding direction i.e., the sign of phi_s(r).

    """

    alpha: float
    b: float
    c: float
    theta0: float
    scale_factor: float
    rho: float
    winding: Literal[-1, 1] = 1
    flattening_strength: float = 0.1

    def __post_init__(self) -> None:
        """Validate attributes."""

        _validate_positive("b", self.b)
        _validate_positive("scale_factor", self.scale_factor)
        _validate_positive("flattening_strength", self.flattening_strength)

        _validate_nonnegative("alpha", self.alpha)
        _validate_nonnegative("c", self.c)
        _validate_nonnegative("rho", self.rho)

        _validate_finite("theta0", self.theta0)

        if self.winding not in (-1, 1):
            msg = "`winding` must be either -1 or 1."
            raise ValueError(msg)

    @override
    def __call__[ShapeT: tuple[Any, ...]](
        self, z: onp.ArrayND[np.float64, ShapeT], vz: onp.ArrayND[np.float64, ShapeT], /
    ) -> onp.ArrayND[np.float64, ShapeT]:
        assert z.shape == vz.shape

        scaled_z = z * self.scale_factor
        scaled_vz = vz / self.scale_factor
        r_mesh = np.hypot(z, scaled_vz).astype(np.float64)
        theta_mesh = np.arctan2(vz, scaled_z)

        phase = self.spiral_phase(r_mesh)

        flattening = special.expit((r_mesh - self.rho) / self.flattening_strength)
        pert = 1.0 + self.alpha * flattening * np.cos(
            self.winding * theta_mesh - phase - self.theta0
        )
        return cast("onp.ArrayND[np.float64, ShapeT]", pert)

    def spiral_phase[ShapeT: tuple[Any, ...]](
        self, r: onp.ArrayND[np.float64, ShapeT]
    ) -> onp.ArrayND[np.float64, ShapeT]:
        """Compute the phase angle of the spiral phi_s(r).

        Parameters
        ----------
        r : ArrayND[f64, S]
            The phase distance.

        Returns
        -------
        phase : ArrayND[f64, S]
            The spiral phase in radians.

        """
        # phi_s(r) = (-b/2c + sqrt((b/2c)^2 + r/c))
        if self.c != 0.0:
            half_b_over_c = 0.5 * self.b / self.c
            phase = -half_b_over_c + np.sqrt(half_b_over_c**2 + r / self.c)
        # phi_s(r) = r / b
        else:
            phase = r / self.b
        return cast("onp.ArrayND[np.float64, ShapeT]", phase)

    def model_phase(self, r_test: float = 0.5) -> float:
        """Calculate the model phase angle.

        Parameters
        ----------
        r_test : float
            The reference phase distance to calculate the angle at.

        Returns
        -------
        model_phase : float
            The model phase angle in radians.

        """
        r_test_arr: onp.Array1D[np.float64] = np.array([r_test], dtype=np.float64)
        # NOTE: `__getitem__` returns `Any` even though for Array1D[T] it should be T
        phase = self.spiral_phase(r_test_arr)[0]  # pyright: ignore[reportAny]
        return float(cast("np.float64", phase) + self.theta0)


def _validate_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        msg = f"`{name}` must be positive: {value} !> 0.0"
        raise ValueError(msg)


def _validate_nonnegative(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0.0:
        msg = f"`{name}` must be non-negative: {value} !>= 0.0"
        raise ValueError(msg)


def _validate_finite(name: str, value: float) -> None:
    if not np.isfinite(value):
        msg = f"`{name}` must be finite: {value} is not finite."
        raise ValueError(msg)


if __name__ == "__main__":
    prot: Component = AlinderComponent(
        alpha=1.0,
        b=0.04,
        c=0.0,
        theta0=0.0,
        scale_factor=40.0,
        rho=0.05,
        winding=1,
        flattening_strength=0.1,
    )
