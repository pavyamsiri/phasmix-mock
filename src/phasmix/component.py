"""Components that compose to form a model to generate mock data from."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, override, runtime_checkable, Any, Literal
from scipy import special
import numpy as np

if TYPE_CHECKING:
    from optype import numpy as onp


@runtime_checkable
class Component(Protocol):
    def __call__[ShapeT: tuple[Any, ...]](
        self, x: onp.ArrayND[np.float64, ShapeT], y: onp.ArrayND[np.float64, ShapeT]
    ) -> onp.ArrayND[np.float64, ShapeT]: ...


@dataclass
class GaussianComponent(Component):
    x_scale: float
    y_scale: float
    amplitude: float
    variance: float
    x_offset: float = 0
    y_offset: float = 0

    @override
    def __call__[ShapeT: tuple[Any, ...]](
        self, x: onp.ArrayND[np.float64, ShapeT], y: onp.ArrayND[np.float64, ShapeT]
    ) -> onp.ArrayND[np.float64, ShapeT]:
        rxy: onp.ArrayND[np.float64, ShapeT] = np.hypot(
            x / self.x_scale, y / self.y_scale
        )
        result: onp.ArrayND[np.float64, ShapeT] = self.amplitude * np.exp(
            -np.square(rxy) / self.variance
        )
        return result


@dataclass
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

    @override
    def __call__[ShapeT: tuple[Any, ...]](
        self, z: onp.ArrayND[np.float64, ShapeT], vz: onp.ArrayND[np.float64, ShapeT]
    ) -> onp.ArrayND[np.float64, ShapeT]:
        """Calculate the contribution to the perturbation from this component.

        The form being f(r, theta) = 1 + alpha * flattening(r, rho) * cos(theta - phi_s(r) - theta0).

        Parameters
        ----------
        z_mesh : ArrayND[f64, S]
            The z coordinates.
        vz_mesh : ArrayND[f64, S]
            The vz coordinates.

        Returns
        -------
        perturbation : ArrayND[f64, S]
            The perturbation.

        """
        assert z.shape == vz.shape

        scaled_z = np.multiply(z, self.scale_factor)
        scaled_vz = vz * np.reciprocal(self.scale_factor)
        r_mesh = np.hypot(z, scaled_vz)
        theta_mesh = np.arctan2(vz, scaled_z)

        phase = self.spiral_phase(r_mesh)

        flattening = special.expit((r_mesh - self.rho) / self.flattening_strength)
        pert = 1.0 + self.alpha * flattening * np.cos(
            self.winding * theta_mesh - phase - self.theta0
        )
        return pert

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
        abs_b: np.float64 = np.abs(self.b).astype(np.float64)
        abs_c: np.float64 = np.abs(self.c)
        # phi_s(r) = (-b/2c + sqrt((b/2c)^2 + r/c))
        if abs_c != 0.0:
            half_b_over_c = 0.5 * abs_b / abs_c
            phase = -half_b_over_c + np.sqrt(np.square(half_b_over_c) + r / abs_c)
        # phi_s(r) = r / b
        else:
            phase = r / abs_b
        return phase

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
        phase = float(self.spiral_phase(np.array(r_test))[0])
        return phase + self.theta0
