"""Scale-dependent priors for the stochastic part of Gaussian smooths."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import liesel.goose as gs
import numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from scipy.optimize import brentq

from .basis import Basis

type _VariancePriorFactory = Callable[[float], tfd.Distribution]


def _weibull_variance(scale: float) -> tfd.Distribution:
    return tfd.Weibull(concentration=0.5, scale=scale)


class _VarianceSampleError(ValueError):
    """Variance draws are unavailable at a requested factory argument."""


def _is_real(values: np.ndarray, *, allow_infinite: bool = False) -> bool:
    try:
        if np.iscomplexobj(values) or not np.issubdtype(
            np.result_type(values.dtype, np.float32), np.number
        ):
            return False
        return bool(
            not np.isnan(values).any() if allow_infinite else np.isfinite(values).all()
        )
    except TypeError:
        return False


def _variance_samples(
    distribution: tfd.Distribution,
    draws: int,
    seed: jax.Array,
    *,
    allow_infinite: bool = False,
) -> np.ndarray:
    values = np.asarray(distribution.sample(draws, seed=seed))
    if (
        values.shape != (draws,)
        or not _is_real(values, allow_infinite=allow_infinite)
        or np.any(values < 0)
    ):
        raise _VarianceSampleError(
            "Variance samples must be finite and positive scalar draws."
        )
    if np.any(values == 0):
        dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.float32
        values = np.where(values == 0, np.finfo(dtype).tiny, values)
    return values


def _variance_median(distribution: tfd.Distribution) -> float:
    samples = _variance_samples(
        distribution, 4096, jax.random.PRNGKey(0), allow_infinite=True
    )
    try:
        median = np.asarray(distribution.quantile(0.5))
    except NotImplementedError:
        median = np.median(samples)
    if np.ndim(median) or not _is_real(np.asarray(median)) or median <= 0:
        raise ValueError("The variance-prior median must be finite and positive.")
    return float(median)


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
    diagnostics: Mapping[str, Any]
    """Read-only numerical, validation, and evaluation-grid information."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )


class CalibrationTarget(Protocol):
    """Structural interface for a criterion evaluated on predictive effects.

    Supply the attributes below and :meth:`statistic
    <liesel_gam.CalibrationTarget.statistic>` on an ordinary object; subclassing
    or registration is unnecessary. :meth:`SDPrior.calibrate
    <liesel_gam.SDPrior.calibrate>` owns sampling, prior construction, solving,
    independent validation, and results.

    A scalar fitted basis is evaluated at equally spaced locations from its
    setup-input minimum to its maximum, including endpoints. ``grid_size=None``
    selects 129 locations. Explicit geometry uses design-row order. Targets that
    need locations store matching domain context themselves.

    A target may additionally supply ``variance_limits(values)`` for the
    single-variance CDF shortcut. Each limit must be nonnegative and describe
    the largest variance satisfying the target for that unit-Gaussian statistic;
    positive infinity is allowed. Without this capability, calibration uses
    predictive sampling and the requested statistic quantile.

    See Also
    --------
    liesel_gam.MaxAbsEffect : A target with the optional variance-limit shortcut.
    liesel_gam.GaussianGeometry : Explicit evaluation locations and penalties.
    """

    @property
    def threshold(self) -> float:
        """Finite positive upper bound on the statistic."""
        ...

    @property
    def probability(self) -> float:
        """Requested probability that the statistic is at most the threshold."""
        ...

    @property
    def grid_size(self) -> int | None:
        """Scalar basis-grid count, at least two; None selects the default."""
        ...

    @property
    def seed(self) -> int:
        """Nonnegative seed for independent fitting and validation streams."""
        ...

    @property
    def draws(self) -> int:
        """Number of draws in each stream, at least two."""
        ...

    def statistic(self, effects: np.ndarray) -> np.ndarray:
        """Return one finite real statistic per predictive effect.

        ``effects`` has shape ``(batch, evaluation_points)``. Evaluate rows
        independently: batches may contain fewer rows than ``draws``. Targets
        can close over domain-specific context and reshape evaluation locations.
        """
        ...


@dataclass(frozen=True, eq=False)
class GaussianGeometry:
    """Evaluated Gaussian effect with one or more coefficient penalties.

    Parameters
    ----------
    design
        Finite nonempty matrix with one row per evaluation location and one
        column per coefficient. Include fitted constraints and transformations.
    penalties
        One symmetric PSD matrix per independent variance parameter, matching
        the design columns. Conditional precision is their sum after division
        by those variances. None supplies one iid identity penalty; an explicit
        zero penalty remains zero. Empty tuples are invalid.
    """

    design: np.ndarray | jax.Array
    """Evaluated mapping from coefficients to effect values."""
    penalties: tuple[np.ndarray | jax.Array, ...] | None = None
    """Penalty matrices in variance-factory order, or None for iid precision."""


@dataclass(frozen=True)
class _PreparedGaussian:
    design: np.ndarray
    penalties: tuple[np.ndarray, ...]
    unit_design: np.ndarray | None
    nullity: int
    grid_shape: tuple[int, ...]

    @property
    def rank(self) -> int:
        return self.design.shape[1]

    def statistics(
        self, target: CalibrationTarget, normals: np.ndarray, variances: np.ndarray
    ) -> np.ndarray:
        values = []
        for start in range(0, len(normals), 128):
            stop = start + 128
            z, v = normals[start:stop], variances[start:stop]
            if self.unit_design is not None:
                effects = (z @ self.unit_design.T) * np.sqrt(v[:, :1])
            else:
                precision = sum(
                    penalty[None, :, :] / v[:, i, None, None]
                    for i, penalty in enumerate(self.penalties)
                )
                try:
                    factor = np.linalg.cholesky(precision)
                    coefficients = np.linalg.solve(
                        factor.swapaxes(-1, -2), z[..., None]
                    )[..., 0]
                except np.linalg.LinAlgError as error:
                    raise ValueError(
                        "Conditional precision must be positive definite on the "
                        "common stochastic subspace."
                    ) from error
                effects = coefficients @ self.design.T
            values.append(_target_statistics(target, effects))
        return np.concatenate(values)


def _prepare_gaussian(
    basis: Basis | GaussianGeometry, grid_size: int | None
) -> _PreparedGaussian:
    if isinstance(basis, Basis):
        size = 129 if grid_size is None else grid_size
        x = np.asarray(basis.x.value)
        if (
            basis.row_wise is not True
            or x.ndim != 1
            or not x.size
            or not np.issubdtype(x.dtype, np.number)
            or np.iscomplexobj(x)
            or not np.isfinite(x).all()
        ):
            raise ValueError(
                "Automatic grids require a finite numeric scalar row-wise basis."
            )
        grid = jnp.asarray(np.linspace(x.min(), x.max(), size))
        design = np.asarray(basis._exact_basis_fn(grid, **basis._basis_kwargs_values()))
        raw_penalties = None if basis.penalty is None else (basis.penalty.value,)
        grid_shape = (size,)
    elif isinstance(basis, GaussianGeometry):
        design = np.asarray(basis.design)
        raw_penalties = basis.penalties
        grid_shape = None
    else:
        raise TypeError("Expected a fitted Basis or evaluated GaussianGeometry.")
    if design.ndim != 2 or not design.size or not _is_real(design):
        raise ValueError(
            "Calibration requires a finite nonempty real matrix-valued basis."
        )
    design = np.asarray(design, dtype=float)
    if grid_shape is None:
        grid_shape = (len(design),)
    if raw_penalties is None:
        raw_penalties = (np.eye(design.shape[1]),)
    if not isinstance(raw_penalties, tuple):
        raise TypeError("penalties must be a tuple of matrices or None.")
    if not raw_penalties:
        raise ValueError("Gaussian geometry requires at least one penalty.")
    penalties = []
    normalized_penalties = []
    rank_relative_tolerance = 0.0
    for raw in raw_penalties:
        raw = np.asarray(raw)
        if np.iscomplexobj(raw):
            raise ValueError("The penalty must be real.")
        penalty = np.asarray(raw, dtype=float)
        if penalty.shape != (design.shape[1], design.shape[1]):
            raise ValueError("The penalty must match the design columns.")
        if not np.isfinite(penalty).all():
            raise ValueError("The penalty must be finite.")
        relative_tolerance = np.finfo(np.result_type(raw.dtype, np.float32)).eps * len(
            penalty
        )
        if not np.allclose(
            penalty,
            penalty.T,
            rtol=0.0,
            atol=relative_tolerance * np.abs(penalty).max(),
        ):
            raise ValueError("The penalty must be symmetric.")
        penalty = (penalty + penalty.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(penalty)
        scale = np.max(np.abs(eigenvalues))
        if eigenvalues.min() < -relative_tolerance * scale:
            raise ValueError("The penalty must be positive semidefinite.")
        if scale > 0:
            normalized_penalties.append(penalty / scale)
            rank_relative_tolerance = max(rank_relative_tolerance, relative_tolerance)
        penalties.append(penalty)
    if not normalized_penalties:
        raise ValueError("Calibration requires a nonzero stochastic penalty.")
    if len(penalties) > 1:
        eigenvalues, eigenvectors = np.linalg.eigh(sum(normalized_penalties))
    positive = eigenvalues > rank_relative_tolerance * np.max(np.abs(eigenvalues))
    if not positive.any():
        raise ValueError("Calibration requires a nonzero stochastic penalty.")
    rotation = eigenvectors[:, positive]
    design = design @ rotation
    if not np.any(design):
        raise ValueError("The evaluation grid has no stochastic effect to calibrate.")
    return _PreparedGaussian(
        design=design,
        penalties=tuple(rotation.T @ penalty @ rotation for penalty in penalties),
        unit_design=design / np.sqrt(eigenvalues[positive])
        if len(penalties) == 1
        else None,
        nullity=len(eigenvalues) - int(positive.sum()),
        grid_shape=tuple(grid_shape),
    )


def _target_statistics(target: CalibrationTarget, effects: np.ndarray) -> np.ndarray:
    values = np.asarray(target.statistic(effects))
    if values.shape != (len(effects),) or not _is_real(values):
        raise ValueError("The target statistic must return one finite value per draw.")
    return values


def _calibration_root(objective: Callable[[float], float]) -> float:
    precision = np.finfo(jnp.asarray(1.0).dtype)
    lower, upper = np.log(float(precision.tiny)), np.log(float(precision.max))
    center = objective(0.0)
    if center == 0.0:
        return 0.0
    radius = 1.0
    active = [True, True]
    while any(active):
        for side, trial in enumerate((max(lower, -radius), min(upper, radius))):
            if not active[side]:
                continue
            try:
                value = objective(trial)
            except _VarianceSampleError:
                active[side] = False
                continue
            if center * value <= 0:
                return brentq(objective, min(0.0, trial), max(0.0, trial), xtol=1e-7)
            if trial in (lower, upper):
                active[side] = False
        radius *= 2
    raise ValueError(
        "Cannot calibrate the target within finite positive prior parameters."
    )


def _validate_target(
    target: CalibrationTarget, *, threshold_name: str = "threshold"
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

        ``effects`` has shape ``(batch, evaluation_points)``. The returned array
        has shape ``(batch,)``.
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

    Pass this configuration as ``scale`` to a compatible scalar spline constructor
    on :class:`TermBuilder <liesel_gam.TermBuilder>`, or opt in for a builder
    instance with ``default_scale_fn``. Every independent term receives fresh scale
    variables; reusing the recipe does not share random variables. Creating the
    recipe constructs no model variables.

    Parameters
    ----------
    calibration
        :class:`CalibrationTarget <liesel_gam.CalibrationTarget>` definition,
        such as :class:`MaxAbsEffect <liesel_gam.MaxAbsEffect>`,
        carrying the statistic, threshold, probability, and numerical settings.
        The target supplies no calibration algorithm.
    variance_prior
        Callable, or a tuple matching the geometry's distinct variances, mapping
        a positive scalar to a scalar TFD prior on tau squared. A single callable
        is repeated independently when there are several variances. Every
        factory receives the same calibrated scalar; fixed relative factors
        belong inside the factories.
        The default is Weibull with concentration 0.5 and that scalar as its TFD
        scale. Priors must be sampleable; CDF and quantile implementations are
        optional. See :meth:`calibrate <liesel_gam.SDPrior.calibrate>` for numeric
        draw handling. This argument is a prior on tau squared.
    value
        Explicit finite positive initial tau. ``None`` initializes tau at
        ``sqrt(median(tau**2))`` under the calibrated variance prior. If quantile
        is unavailable, the median uses 4,096 seed-zero samples independently
        of calibration randomness and variance ordering.
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

    calibration: CalibrationTarget
    """Effect criterion and numerical settings."""
    variance_prior: _VariancePriorFactory | tuple[_VariancePriorFactory, ...] = field(
        default_factory=lambda: _weibull_variance
    )
    """Scalar tau-squared prior factory, repeated independently for each variance.

    A tuple supplies one factory per penalty and must match their number exactly.
    All factories receive the same calibrated argument.
    """
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

    def _variance_factories(self, count: int) -> tuple[_VariancePriorFactory, ...]:
        factories = (
            self.variance_prior
            if isinstance(self.variance_prior, tuple)
            else (self.variance_prior,) * count
        )
        if any(not callable(factory) for factory in factories):
            raise TypeError("variance_prior must be a callable or tuple of callables.")
        if len(factories) != count:
            raise ValueError(
                f"Expected {count} variance-prior factories, got {len(factories)}."
            )
        return factories

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

    def calibrate(self, basis: Basis | GaussianGeometry) -> CalibrationResult:
        """Run shared calibration without constructing or updating model variables.

        Parameters
        ----------
        basis
            Fitted numeric scalar, known row-wise :class:`Basis <liesel_gam.Basis>`
            or evaluated :class:`GaussianGeometry <liesel_gam.GaussianGeometry>`.
            Constraints and reparameterizations are retained. Singular PSD
            penalties are supported; multiple penalties use their common
            stochastic subspace.

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
        For multiple penalties, their sum after individual spectral normalization
        identifies the common null space; the original penalties determine
        conditional precision.
        Both searches double a log-scale bracket around ``s=1``, bounded by
        normal positive prior parameters in the model's default floating
        precision, with log-scale tolerance ``1e-7``. When
        either shortcut capability is unavailable, fixed independent variance
        and Gaussian draws supply a sampled statistic quantile. That search
        uses linear sample-quantile interpolation. Fitting coverage must be
        within two draw counts of the target;
        targets with incompatible atoms or unattainable probabilities fail.

        Predictive variance draws must be finite and nonnegative; exact zeros
        are replaced by the smallest positive normal value of their draw dtype.
        Support checks and median estimation also allow positive-infinite draws,
        provided the resulting median is finite and positive. A quantile median
        is preferred; otherwise 4,096 seed-zero draws supply a deterministic median.
        Search expansion stops on a side whose variance draws become invalid,
        while continuing on the other side. On the sampled path, draws at
        ``s=1`` must be valid.
        Common random numbers stabilize the objective but do not establish monotonicity
        for arbitrary factories or criteria. Use a factory argument whose effect
        on the target is meaningful over the searched range. Diagnostics report
        either ``integrated_variance_cdf`` or ``sampled_variance_quantile``.
        """
        target = self.calibration
        _validate_target(target)
        geometry = _prepare_gaussian(basis, target.grid_size)
        n_variances = len(geometry.penalties)
        factories = self._variance_factories(n_variances)
        rank = geometry.rank
        variance_limits = getattr(target, "variance_limits", None)
        fit_rng, validation_rng = np.random.default_rng(target.seed).spawn(2)
        fit_normals = fit_rng.standard_normal((target.draws, rank))
        validation_normals = validation_rng.standard_normal((target.draws, rank))

        def distribution_at(
            scale: float, factory: _VariancePriorFactory = factories[0]
        ) -> tfd.Distribution:
            distribution = factory(scale)
            if not isinstance(distribution, tfd.Distribution):
                raise TypeError(
                    "A variance-prior factory must return a TFD distribution."
                )
            if tuple(distribution.batch_shape) or tuple(distribution.event_shape):
                raise ValueError("Each variance prior must be scalar.")
            return distribution

        use_cdf = n_variances == 1 and callable(variance_limits)
        if use_cdf:
            try:
                distribution_at(1.0).cdf(np.ones(target.draws))
            except NotImplementedError:
                use_cdf = False

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

        if use_cdf and callable(variance_limits):

            def predictive_limits(values: np.ndarray) -> np.ndarray:
                limits = np.asarray(variance_limits(values))
                if (
                    limits.shape != values.shape
                    or not _is_real(limits, allow_infinite=True)
                    or np.any(limits < 0)
                ):
                    raise ValueError(
                        "Target variance limits must be nonnegative real values, "
                        "one per draw."
                    )
                return limits

            fit = predictive_limits(
                geometry.statistics(target, fit_normals, np.ones((target.draws, 1))),
            )
            validation = predictive_limits(
                geometry.statistics(
                    target, validation_normals, np.ones((target.draws, 1))
                ),
            )

            def residual(value: float) -> float:
                return float(coverage(value, fit).mean()) - target.probability

            log_scale = _calibration_root(residual)
            achieved = coverage(log_scale, validation)
            method = "integrated_variance_cdf"
        else:
            fit_key = jax.random.PRNGKey(int(fit_rng.integers(0, 2**32)))
            validation_key = jax.random.PRNGKey(int(validation_rng.integers(0, 2**32)))

            def sampled_statistics(
                value: float, normals: np.ndarray, key: jax.Array
            ) -> np.ndarray:
                try:
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        variances = np.column_stack(
                            [
                                _variance_samples(
                                    distribution_at(float(np.exp(value)), factory),
                                    target.draws,
                                    jax.random.fold_in(key, index),
                                )
                                for index, factory in enumerate(factories)
                            ]
                        )
                except FloatingPointError as error:
                    raise _VarianceSampleError(
                        "Variance samples must be finite and positive scalar draws."
                    ) from error
                return geometry.statistics(target, normals, variances)

            def quantile_residual(value: float) -> float:
                values = sampled_statistics(value, fit_normals, fit_key)
                return float(np.quantile(values, target.probability)) - target.threshold

            log_scale = _calibration_root(quantile_residual)
            fitting_probability = np.mean(
                sampled_statistics(log_scale, fit_normals, fit_key) <= target.threshold
            )
            if abs(fitting_probability - target.probability) > 2.0 / target.draws:
                raise ValueError(
                    "Cannot calibrate the requested probability with this target."
                )
            achieved = (
                sampled_statistics(log_scale, validation_normals, validation_key)
                <= target.threshold
            )
            method = "sampled_variance_quantile"
        scale = float(np.exp(log_scale))
        distributions = tuple(distribution_at(scale, factory) for factory in factories)
        return CalibrationResult(
            prior_scale=scale,
            variance_priors=distributions,
            initial_scales=tuple(
                float(np.sqrt(_variance_median(d))) for d in distributions
            ),
            diagnostics={
                "method": method,
                "target_probability": target.probability,
                "validation_probability": float(achieved.mean()),
                "validation_mc_se": float(achieved.std(ddof=1) / np.sqrt(target.draws)),
                "threshold": target.threshold,
                "seed": target.seed,
                "draws": target.draws,
                "rank": rank,
                "nullity": geometry.nullity,
                "grid_size": geometry.grid_shape[0],
                "grid_shape": geometry.grid_shape,
                "evaluation_points": int(np.prod(geometry.grid_shape)),
            },
        )
