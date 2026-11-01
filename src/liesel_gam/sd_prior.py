"""Scale-dependent priors for the stochastic part of Gaussian smooths."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
from typing import Any, Protocol

import jax.numpy as jnp
import liesel.goose as gs
import numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from scipy.optimize import brentq

from .basis import Basis
from .var import CatVar

type _VariancePriorFactory = Callable[[float], tfd.Distribution]


def _weibull_variance(scale: float) -> tfd.Distribution:
    return tfd.Weibull(concentration=0.5, scale=scale)


@dataclass(frozen=True)
class CalibrationResult:
    """Immutable output of :meth:`SDPrior.calibrate <liesel_gam.SDPrior.calibrate>`.

    Results describe one calculation. They contain no model variables and do not
    change when subsequently constructing terms. Diagnostic mappings are copied.
    """

    prior_scale: float
    """Calibrated argument supplied to the variance-prior factory."""
    variance_priors: tuple[tfd.Distribution, ...]
    """Instantiated distributions on tau squared for this calculation."""
    initial_scales: tuple[float, ...]
    """Median-based suggested tau starts, before any explicit value override."""
    diagnostics: Mapping[str, float | int | str]
    """Read-only numerical, validation, and evaluation-grid information."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )


class _CalibrationTarget(Protocol):
    @property
    def threshold(self) -> float: ...

    @property
    def probability(self) -> float: ...

    @property
    def grid_size(self) -> int | None: ...

    @property
    def seed(self) -> int: ...

    @property
    def draws(self) -> int: ...

    def statistic(self, effects: np.ndarray) -> np.ndarray: ...


def _validate_target(
    target: _CalibrationTarget, *, threshold_name: str = "threshold"
) -> None:
    if not np.isfinite(target.threshold) or target.threshold <= 0:
        raise ValueError(f"{threshold_name} must be finite and positive.")
    if not np.isfinite(target.probability) or not 0 < target.probability < 1:
        raise ValueError("probability must be finite and strictly between 0 and 1.")
    for name, value, minimum in (
        ("grid_size", target.grid_size, 2),
        ("draws", target.draws, 2),
        ("seed", target.seed, 0),
    ):
        if name == "grid_size" and value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}.")


@dataclass(frozen=True)
class MaxAbsEffect:
    r"""Define a simultaneous stochastic-effect bound on the predictor scale.

    The target is :math:`P(\max_{x \in G}|f_\mathrm{stochastic}(x)| \le c)=p`.
    This bounds the penalized part on a finite grid, excluding unpenalized
    directions. It is neither a response-scale nor a pointwise probability bound.

    Parameters
    ----------
    max_abs_effect
        Finite positive bound ``c`` in additive predictor units.
    probability
        Target probability ``p``, strictly between zero and one.
    grid_size
        Points over the scalar input's setup range, including endpoints.
        ``None`` selects 129 points. An explicit integer must be at least two.
        Finite categorical domains always evaluate every declared level.
    seed
        Nonnegative integer seed for independent fit and validation draws.
    draws
        Number of draws in each set; must be an integer of at least two.

    Notes
    -----
    This target supplies a per-draw statistic and its allowable-variance limits.
    :meth:`SDPrior.calibrate <liesel_gam.SDPrior.calibrate>` performs the shared
    sampling, CDF integration, search, validation, and result construction.

    See Also
    --------
    liesel_gam.SDPrior.from_max_abs_effect : Create a prior recipe directly.
    """

    max_abs_effect: float
    """Maximum absolute stochastic effect on the predictor scale."""
    probability: float
    """Requested simultaneous grid probability."""
    grid_size: int | None = None
    """Points per scalar input axis; None selects 129."""
    seed: int = 42
    """Seed used to generate independent fit and validation streams."""
    draws: int = 4096
    """Gaussian draws in each stream."""

    def __post_init__(self) -> None:
        _validate_target(self, threshold_name="max_abs_effect")

    @property
    def threshold(self) -> float:
        """Effect bound, equal to ``max_abs_effect``."""
        return self.max_abs_effect

    def statistic(self, effects: np.ndarray) -> np.ndarray:
        """Return the maximum absolute effect for each predictive draw.

        ``effects`` has shape ``(draws, evaluation_points)``. The returned array
        has shape ``(draws,)``.
        """
        return np.max(np.abs(effects), axis=-1)

    def variance_limits(self, values: np.ndarray) -> np.ndarray:
        """Return the largest allowed variance for each unit-scale statistic.

        For unit-Gaussian maxima ``M``, this is ``(max_abs_effect / M)**2``.
        Zero maxima allow infinite variance. The shared workflow uses these
        limits for the single-variance CDF shortcut.
        """
        return (
            np.divide(
                self.max_abs_effect,
                values,
                out=np.full_like(values, np.inf, dtype=float),
                where=values > 0,
            )
            ** 2
        )


@dataclass(frozen=True)
class SDPrior:
    r"""Reusable recipe for a basis-dependent prior on :math:`\tau^2`.

    Pass this configuration as ``scale`` to a compatible term constructor
    on :class:`TermBuilder <liesel_gam.TermBuilder>`, or opt in for a builder
    instance with ``default_scale_fn``. Every independent term receives fresh scale
    variables; reusing the recipe does not share random variables. Creating the
    recipe constructs no model variables.

    Parameters
    ----------
    calibration
        Target definition, such as :class:`MaxAbsEffect <liesel_gam.MaxAbsEffect>`,
        carrying the statistic, threshold, probability, and numerical settings.
        The target supplies no calibration algorithm.
    variance_prior
        Callable, or a one-element tuple of callables, mapping a positive scalar
        to a scalar TFD prior on tau squared.
        The default is Weibull with concentration 0.5 and that scalar as its TFD
        scale. The current single-variance rule requires CDF and median-quantile
        methods. This argument is not a prior on tau itself.
    value
        Explicit finite positive initial tau. ``None`` initializes tau at
        ``sqrt(median(tau**2))`` under the calibrated variance prior.
    bijector
        Transform from an unconstrained parameter to tau squared; default Exp.
    inference
        Inference specification for the unconstrained variance parameter. Defaults
        to :class:`MCMCSpec <liesel.goose.MCMCSpec>` with
        :class:`NUTSKernel <liesel.goose.NUTSKernel>`.
        With the default Exp bijector, NUTS samples log variance. Explicit ``None``
        disables automatic sampling; an explicit specification overrides NUTS.

    See Also
    --------
    liesel_gam.SDPrior.calibrate : Inspect a pure calibration result.
    liesel_gam.scale_wb : Construct an explicitly specified variance prior.

    Examples
    --------
    >>> import numpy as np
    >>> import liesel_gam as gam
    >>> prior = gam.SDPrior.from_max_abs_effect(3.0, probability=0.95, draws=128)
    >>> tb = gam.TermBuilder.from_dict({"age": np.linspace(0.0, 1.0, 20)})
    >>> smooth = tb.ps("age", k=6, scale=prior)
    >>> result = prior.calibrate(smooth.basis)
    >>> result.diagnostics["evaluation_points"]
    129
    """

    calibration: _CalibrationTarget
    """Effect criterion and numerical settings."""
    variance_prior: _VariancePriorFactory | tuple[_VariancePriorFactory, ...] = field(
        default_factory=lambda: _weibull_variance
    )
    """Factory producing a scalar prior on tau squared."""
    value: float | None = None
    """Explicit initial tau, or None for the calibrated prior median."""
    bijector: tfb.Bijector = field(default_factory=tfb.Exp)
    """Bijector from the unconstrained parameter to tau squared."""
    inference: Any = field(
        default_factory=lambda: gs.MCMCSpec(gs.NUTSKernel), hash=False
    )
    """Inference specification for the unconstrained variance parameter."""

    def __post_init__(self) -> None:
        if self.value is not None and (not np.isfinite(self.value) or self.value <= 0):
            raise ValueError("value must be finite and positive on the tau scale.")
        if not isinstance(self.bijector, tfb.Bijector):
            raise ValueError(
                f"bijector must be a TFP bijector instance, got {self.bijector!r}."
            )

    @property
    def _variance_factories(self) -> tuple[_VariancePriorFactory, ...]:
        return (
            self.variance_prior
            if isinstance(self.variance_prior, tuple)
            else (self.variance_prior,)
        )

    @classmethod
    def from_max_abs_effect(
        cls,
        max_abs_effect: float,
        *,
        probability: float,
        variance_prior: _VariancePriorFactory
        | tuple[_VariancePriorFactory, ...] = _weibull_variance,
        grid_size: int | None = None,
        seed: int = 42,
        draws: int = 4096,
        value: float | None = None,
        bijector: tfb.Bijector = tfb.Exp(),
        inference: Any = gs.MCMCSpec(gs.NUTSKernel),
    ) -> "SDPrior":
        """Create a recipe with :class:`MaxAbsEffect <liesel_gam.MaxAbsEffect>`.

        ``max_abs_effect``, ``probability``, ``grid_size``, ``seed``, and ``draws``
        configure that rule. Remaining arguments configure
        :class:`SDPrior <liesel_gam.SDPrior>`.
        """
        return cls(
            calibration=MaxAbsEffect(
                max_abs_effect, probability, grid_size, seed, draws
            ),
            variance_prior=variance_prior,
            value=value,
            bijector=bijector,
            inference=inference,
        )

    def calibrate(self, basis: Basis) -> CalibrationResult:
        """Run shared calibration without constructing or updating model variables.

        Parameters
        ----------
        basis
            Fitted :class:`Basis <liesel_gam.Basis>` with its coefficient penalty.
            Numeric scalar inputs (vectors or one-column matrices) require known
            row-wise evaluation. Categorical inputs evaluate every declared code
            once; indexed outputs use equivalent indicator geometry. Constraints,
            scaling, and reparameterizations are retained, including for MRF fields.
            Singular PSD penalties calibrate only their stochastic subspace.

        Returns
        -------
        liesel_gam.CalibrationResult
            Immutable priors, suggested starts, and diagnostics for this calculation.
            An explicit recipe ``value`` changes constructed starts, not these
            median-based suggestions.

        Raises
        ------
        ValueError
            If the fitted geometry is invalid, has no stochastic effect on the
            grid, or the target cannot be calibrated within finite prior parameters.

        Notes
        -----
        The workflow normalizes variance-prior factories, prepares the fitted
        stochastic geometry, draws unit-Gaussian effects, solves for the prior
        parameter, and constructs the validated result. Targets supply a
        ``statistic(effects)`` returning one finite value per draw. The scalar
        CDF path also requires a target ``variance_limits(statistics)`` capability
        returning one nonnegative allowable variance per draw; positive infinity
        is allowed. The workflow integrates the variance prior at those limits.

        Fitting draws stay fixed during search; validation uses an independent
        random stream. Grid and Monte Carlo approximations do not certify
        continuous-domain coverage. Penalty eigenvalues within
        ``n * eps * max(abs(eigenvalues))`` of zero count as null directions,
        where ``eps`` is the penalty's floating precision (at least float32).
        The search spans normal positive prior parameters in the model's default
        floating precision, with log-scale tolerance ``1e-7``.
        """
        factories = self._variance_factories
        target = self.calibration
        _validate_target(target)
        if len(factories) != 1:
            raise ValueError(
                "A single-variance basis requires exactly one variance-prior factory."
            )
        if isinstance(basis.x, CatVar):
            size = len(basis.x.mapping.labels_to_codes_map)
            grid = jnp.arange(size)
        else:
            size = 129 if target.grid_size is None else target.grid_size
            x = np.asarray(basis.x.value)
            scalar_input = x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1)
            if (
                basis.row_wise is not True
                or not scalar_input
                or not x.size
                or not np.issubdtype(x.dtype, np.number)
                or not np.isfinite(x).all()
            ):
                raise ValueError(
                    "Automatic grids require a finite numeric scalar row-wise basis."
                )
            grid = jnp.asarray(np.linspace(x.min(), x.max(), size))
            if x.ndim == 2:
                grid = grid[:, None]
        design = np.asarray(basis._exact_basis_fn(grid, **basis._basis_kwargs_values()))
        if isinstance(basis.x, CatVar) and design.ndim == 1:
            # Indexed category codes represent indicator rows without storing them.
            design = np.eye(size)[design]
        design = np.asarray(design, dtype=float)
        if (
            design.ndim != 2
            or not design.size
            or design.shape[0] != size
            or not np.isfinite(design).all()
        ):
            raise ValueError(
                "Calibration requires a finite nonempty matrix-valued basis."
            )
        penalty_node = basis.penalty
        raw_penalty = (
            np.eye(design.shape[1])
            if penalty_node is None
            else np.asarray(penalty_node.value)
        )
        penalty = np.asarray(raw_penalty, dtype=float)
        if penalty.shape != (design.shape[1], design.shape[1]):
            raise ValueError("The penalty must match the design columns.")
        if not np.isfinite(penalty).all():
            raise ValueError("The penalty must be finite.")
        relative_tolerance = np.finfo(
            np.result_type(raw_penalty.dtype, np.float32)
        ).eps * len(penalty)
        if not np.allclose(
            penalty,
            penalty.T,
            rtol=0.0,
            atol=relative_tolerance * np.abs(penalty).max(),
        ):
            raise ValueError("The penalty must be symmetric.")
        penalty = (penalty + penalty.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(penalty)
        tolerance = relative_tolerance * np.max(np.abs(eigenvalues))
        if eigenvalues.min() < -tolerance:
            raise ValueError("The penalty must be positive semidefinite.")
        positive = eigenvalues > tolerance
        if not positive.any():
            raise ValueError("Calibration requires a nonzero stochastic penalty.")
        unit_design = (design @ eigenvectors[:, positive]) / np.sqrt(
            eigenvalues[positive]
        )
        if not np.any(unit_design):
            raise ValueError(
                "The evaluation grid has no stochastic effect to calibrate."
            )
        rank = int(positive.sum())
        variance_limits = getattr(target, "variance_limits", None)
        if not callable(variance_limits):
            raise ValueError(
                "This target does not provide a single-variance CDF shortcut."
            )
        fit_rng, validation_rng = np.random.default_rng(target.seed).spawn(2)

        def predictive_limits(rng: np.random.Generator) -> np.ndarray:
            coefficients = rng.standard_normal((target.draws, rank))
            values = np.asarray(target.statistic(coefficients @ unit_design.T))
            if values.shape != (target.draws,) or not np.isfinite(values).all():
                raise ValueError(
                    "The target statistic must return one finite value per draw."
                )
            limits = np.asarray(variance_limits(values))
            if (
                limits.shape != values.shape
                or np.isnan(limits).any()
                or np.any(limits < 0)
            ):
                raise ValueError(
                    "Target variance limits must be nonnegative values, one per draw."
                )
            return limits

        fit, validation = predictive_limits(fit_rng), predictive_limits(validation_rng)
        factory = factories[0]

        def distribution_at(scale: float) -> tfd.Distribution:
            distribution = factory(scale)
            if not isinstance(distribution, tfd.Distribution):
                raise TypeError(
                    "A variance-prior factory must return a TFD distribution."
                )
            if tuple(distribution.batch_shape) or tuple(distribution.event_shape):
                raise ValueError("Each variance prior must be scalar.")
            return distribution

        def coverage(log_scale: float, limits: np.ndarray) -> np.ndarray:
            unlimited = np.isposinf(limits)
            # Avoid distribution-specific NaNs at infinity for extreme scales.
            cdf_limits = np.where(unlimited, 0.0, limits)
            probabilities = np.asarray(
                distribution_at(float(np.exp(log_scale))).cdf(cdf_limits)
            )
            if probabilities.shape != limits.shape:
                raise ValueError(
                    "The variance-prior CDF must return one probability per limit."
                )
            rounding = 4 * np.finfo(np.result_type(probabilities.dtype, np.float32)).eps
            if not np.isfinite(probabilities).all() or np.any(
                (probabilities < -rounding) | (probabilities > 1 + rounding)
            ):
                raise ValueError(
                    "The variance-prior CDF must return finite probabilities in [0, 1]."
                )
            return np.where(unlimited, 1.0, np.clip(probabilities, 0.0, 1.0))

        # Search the normal floating-point range used by the model's prior nodes.
        precision = np.finfo(jnp.asarray(1.0).dtype)
        lower, upper = np.log(float(precision.tiny)), np.log(float(precision.max))

        def residual(value: float) -> float:
            return float(coverage(value, fit).mean()) - target.probability

        if residual(lower) * residual(upper) > 0:
            raise ValueError(
                "Cannot calibrate the target within finite positive prior parameters."
            )
        log_scale = brentq(residual, lower, upper, xtol=1.0e-7)
        scale = float(np.exp(log_scale))
        distribution = distribution_at(scale)
        median = np.asarray(distribution.quantile(0.5))
        if median.ndim or not np.isfinite(median) or median <= 0:
            raise ValueError("The variance-prior median must be finite and positive.")
        achieved = coverage(log_scale, validation)
        return CalibrationResult(
            prior_scale=scale,
            variance_priors=(distribution,),
            initial_scales=(float(np.sqrt(median)),),
            diagnostics={
                "method": "integrated_variance_cdf",
                "target_probability": target.probability,
                "validation_probability": float(achieved.mean()),
                "validation_mc_se": float(achieved.std(ddof=1) / np.sqrt(target.draws)),
                "threshold": target.threshold,
                "seed": target.seed,
                "draws": target.draws,
                "rank": rank,
                "nullity": design.shape[1] - rank,
                "grid_size": size,
                "evaluation_points": len(design),
            },
        )
