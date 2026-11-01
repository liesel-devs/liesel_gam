from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
import liesel.model as lsl
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from liesel.types import Position
from scipy.stats import t

import liesel_gam as gam

from .test_sd_prior import _scalar_coverage


class _SampleMedianWeibull(tfd.Weibull):
    def _quantile(self, value, **kwargs):
        raise NotImplementedError("No quantile implementation")


class _SampleOnlyWeibull(_SampleMedianWeibull):
    def _cdf(self, y, **kwargs):
        raise NotImplementedError("No CDF implementation")


@dataclass(frozen=True)
class _WeightedRMS:
    threshold: float = 2.0
    probability: float = 0.9
    grid_size: int = 65
    seed: int = 42
    draws: int = 8192
    weight: float = 2.0

    def statistic(self, effects):
        return self.weight * np.sqrt(np.mean(effects**2, axis=-1))


def test_missing_quantile_uses_the_fixed_sample_median_for_results_and_starts():
    prior = gam.SDPrior.from_max_abs_effect(
        3.0,
        probability=0.95,
        draws=256,
        variance_prior=lambda s: _SampleMedianWeibull(0.5, s),
    )
    builder = gam.TermBuilder.from_dict({"x": np.linspace(0.0, 1.0, 30)})

    term = builder.ps("x", k=6, scale=prior)

    result = prior.calibrate(term.basis)
    expected = np.sqrt(
        np.median(result.variance_priors[0].sample(4096, seed=jax.random.PRNGKey(0)))
    )
    assert result.initial_scales == pytest.approx((expected,))
    assert term.scale.value == pytest.approx(expected)
    assert result.diagnostics["method"] == "integrated_variance_cdf"
    assert np.isfinite(lsl.Model(term).log_prob)


def test_target_without_variance_limits_reuses_the_sampled_gaussian_workflow():
    target = _WeightedRMS()
    basis = gam.Basis(
        jnp.linspace(-1.0, 1.0, 20),
        basis_fn=lambda x: x[:, None],
        penalty=jnp.eye(1),
        xname="x",
        row_wise=True,
        use_callback=False,
    )

    result = gam.SDPrior(calibration=target).calibrate(basis)

    # For this linear effect the event is |beta| <= threshold / RMS(weight * x).
    factor = target.weight * np.sqrt(np.mean(np.linspace(-1.0, 1.0, 65) ** 2))
    assert _scalar_coverage(result.prior_scale, factor) == pytest.approx(0.9, abs=0.02)
    assert result.diagnostics["validation_probability"] == pytest.approx(0.9, abs=0.02)
    assert result.diagnostics["method"] == "sampled_variance_quantile"


@pytest.mark.parametrize("absolute_normal", [False, True])
def test_sample_only_variance_prior_builds_a_smooth_and_keeps_results_pure(
    absolute_normal,
):
    def factory(s):
        if absolute_normal:
            return tfd.TransformedDistribution(tfd.Normal(0.0, s), tfb.AbsoluteValue())
        return _SampleOnlyWeibull(0.5, s)

    prior = gam.SDPrior.from_max_abs_effect(
        3.0,
        probability=0.95,
        draws=4096,
        variance_prior=factory,
    )
    builder = gam.TermBuilder.from_dict({"x": np.linspace(0.0, 1.0, 40)})

    term = builder.ps("x", k=6, scale=prior)
    original_start = float(term.scale.value)
    result = prior.calibrate(term.basis)
    repeated = prior.calibrate(term.basis)

    assert result.diagnostics["method"] == "sampled_variance_quantile"
    assert result.diagnostics["validation_probability"] == pytest.approx(0.95, abs=0.02)
    assert result.prior_scale == repeated.prior_scale
    assert result.initial_scales == repeated.initial_scales
    assert result.diagnostics == repeated.diagnostics
    assert float(term.scale.value) == original_start
    assert np.isfinite(lsl.Model(term).log_prob)
    assert term.coef.dist_node is not None
    coefficients = term.coef.dist_node.init_dist().sample(
        8192, seed=jax.random.key(975)
    )
    variances = result.variance_priors[0].sample(8192, seed=jax.random.key(836))
    coefficients = coefficients / term.scale.value * jnp.sqrt(variances[:, None])
    effects = term.predict(
        Position({term.coef.name: coefficients}),
        newdata=Position({"x": np.linspace(0.0, 1.0, 129)}),
    )
    assert np.mean(np.max(np.abs(effects), axis=-1) <= 3.0) == pytest.approx(
        0.95, abs=0.025
    )


def test_explicit_geometry_calibrates_different_independent_variance_families():
    geometry = gam.GaussianGeometry(
        design=np.eye(3),
        penalties=(np.diag([1.0, 0.0, 0.0]), np.diag([0.0, 2.0, 0.0])),
    )
    prior = gam.SDPrior.from_max_abs_effect(
        2.0,
        probability=0.9,
        draws=8192,
        variance_prior=(
            lambda s: tfd.Weibull(0.5, 0.4 * s),
            lambda s: tfd.InverseGamma(3.0, 2.0 * s),
        ),
    )

    result = prior.calibrate(geometry)

    # Independent coefficient 1 has a Weibull mixture; coefficient 2 is Student-t.
    second = t(df=6, scale=np.sqrt(result.prior_scale / 3.0))
    coverage = _scalar_coverage(0.4 * result.prior_scale) * (
        second.cdf(2.0) - second.cdf(-2.0)
    )
    assert coverage == pytest.approx(0.9, abs=0.02)
    weibull, inverse_gamma = result.variance_priors
    assert isinstance(weibull, tfd.Weibull)
    assert isinstance(inverse_gamma, tfd.InverseGamma)
    assert weibull.scale == pytest.approx(0.4 * result.prior_scale)
    assert inverse_gamma.scale == pytest.approx(2.0 * result.prior_scale)
    assert len(result.initial_scales) == 2
    assert result.diagnostics["rank"] == 2
    assert result.diagnostics["nullity"] == 1
    assert result.diagnostics["evaluation_points"] == 3
    assert result.diagnostics["method"] == "sampled_variance_quantile"


def test_a_generic_target_with_an_unattainable_probability_fails_clearly():
    class ConstantTarget(_WeightedRMS):
        def statistic(self, effects):
            return np.full(len(effects), self.threshold)

    geometry = gam.GaussianGeometry(np.ones((1, 1)))
    prior = gam.SDPrior(calibration=ConstantTarget(draws=64))

    with pytest.raises(ValueError, match="Cannot calibrate.*probability"):
        prior.calibrate(geometry)


def test_repeated_factory_uses_independent_variances_in_the_sum_of_precisions():
    geometry = gam.GaussianGeometry(
        np.ones((1, 1)), penalties=(np.ones((1, 1)), np.full((1, 1), 2.0))
    )

    def factory(s):
        return tfd.Weibull(0.5, s)

    prior = gam.SDPrior.from_max_abs_effect(
        2.0, probability=0.9, draws=8192, variance_prior=factory
    )

    result = prior.calibrate(geometry)

    # Independent scalar oracle: precision is 1/v1 + 2/v2, not two added effects.
    rng = np.random.default_rng(975)
    v1, v2 = result.prior_scale * rng.exponential(size=(2, 60000)) ** 2
    coefficients = rng.normal(size=60000) / np.sqrt(1.0 / v1 + 2.0 / v2)
    assert np.mean(np.abs(coefficients) <= 2.0) == pytest.approx(0.9, abs=0.015)
    assert len(result.variance_priors) == 2
    assert result.initial_scales[0] == result.initial_scales[1]
    tuple_result = gam.SDPrior.from_max_abs_effect(
        2.0, probability=0.9, draws=8192, variance_prior=(factory, factory)
    ).calibrate(geometry)
    assert result.prior_scale == tuple_result.prior_scale


@pytest.mark.parametrize("count", [0, 1, 3])
def test_explicit_geometry_rejects_a_mismatched_factory_tuple(count):
    geometry = gam.GaussianGeometry(
        np.eye(2), penalties=(np.diag([1.0, 0.0]), np.diag([0.0, 1.0]))
    )
    prior = gam.SDPrior.from_max_abs_effect(
        2.0,
        probability=0.9,
        variance_prior=(lambda s: tfd.Weibull(0.5, s),) * count,
    )
    with pytest.raises(ValueError, match="Expected 2 variance-prior factories"):
        prior.calibrate(geometry)


@pytest.mark.parametrize("integrated", [False, True])
def test_variance_priors_with_negative_support_are_rejected(integrated):
    target = (
        gam.MaxAbsEffect(2.0, 0.9, draws=256) if integrated else _WeightedRMS(draws=256)
    )
    prior = gam.SDPrior(
        calibration=target, variance_prior=lambda s: tfd.Normal(loc=s, scale=s)
    )
    with pytest.raises(ValueError, match="Variance samples.*positive"):
        prior.calibrate(gam.GaussianGeometry(np.ones((1, 1))))


@pytest.mark.parametrize("dtype", [str, complex])
def test_generic_targets_must_return_real_numeric_statistics(dtype):
    class InvalidTarget(_WeightedRMS):
        def statistic(self, effects):
            return np.ones(len(effects)).astype(dtype)

    prior = gam.SDPrior(calibration=InvalidTarget(draws=64))
    with pytest.raises(ValueError, match="target statistic.*finite"):
        prior.calibrate(gam.GaussianGeometry(np.ones((1, 1))))


@pytest.mark.parametrize("specification", [3, (3,), [tfd.HalfNormal]])
def test_variance_factories_must_be_a_callable_or_tuple_of_callables(specification):
    prior = gam.SDPrior.from_max_abs_effect(
        2.0, probability=0.9, variance_prior=cast(Any, specification)
    )
    with pytest.raises(TypeError, match="variance_prior.*callable"):
        prior.calibrate(gam.GaussianGeometry(np.ones((1, 1))))


def test_noncommuting_penalties_match_an_independent_covariance_reference():
    design = np.array([[1.0, 0.0], [1.0, 1.0]])
    k1 = np.array([[1.0, 0.9], [0.9, 1.0]])
    k2 = np.diag([1.0, 0.1])
    geometry = gam.GaussianGeometry(design, penalties=(k1, k2))
    prior = gam.SDPrior.from_max_abs_effect(
        2.0,
        probability=0.9,
        draws=8192,
        variance_prior=lambda s: tfd.InverseGamma(4.0, 3.0 * s),
    )

    result = prior.calibrate(geometry)

    rng = np.random.default_rng(853)
    v1, v2 = 3.0 * result.prior_scale / rng.gamma(4.0, size=(2, 60000))
    precision = k1[None] / v1[:, None, None] + k2[None] / v2[:, None, None]
    covariance = np.linalg.inv(precision)
    coefficients = np.linalg.cholesky(covariance) @ rng.normal(size=(60000, 2, 1))
    effects = coefficients[..., 0] @ design.T
    assert np.mean(np.max(np.abs(effects), axis=-1) <= 2.0) == pytest.approx(
        0.9, abs=0.015
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("seed", [2, 42])
def test_sample_median_initialization_is_independent_of_seed_and_variance_order(
    reverse, seed
):
    penalties = (np.diag([1.0, 0.0]), np.diag([0.0, 1.0]))
    factories = (
        lambda s: tfd.TransformedDistribution(
            tfd.Normal(0.0, 0.5 * s), tfb.AbsoluteValue()
        ),
        lambda s: tfd.TransformedDistribution(
            tfd.Normal(0.0, 2.0 * s), tfb.AbsoluteValue()
        ),
    )
    if reverse:
        penalties, factories = penalties[::-1], factories[::-1]
    result = gam.SDPrior.from_max_abs_effect(
        2.0, probability=0.9, draws=512, seed=seed, variance_prior=factories
    ).calibrate(gam.GaussianGeometry(np.eye(2), penalties=penalties))

    base_median = np.median(
        np.abs(tfd.Normal(0.0, 1.0).sample(4096, seed=jax.random.key(0)))
    )
    for initial, distribution in zip(
        result.initial_scales, result.variance_priors, strict=True
    ):
        assert isinstance(distribution, tfd.TransformedDistribution)
        normal = distribution.distribution
        assert isinstance(normal, tfd.Normal)
        assert initial / np.sqrt(normal.scale) == pytest.approx(
            np.sqrt(base_median), rel=1e-6
        )


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"design": np.ones(2)}, "matrix-valued"),
        ({"design": np.array([[np.nan]])}, "finite"),
        ({"design": np.eye(2, dtype=complex)}, "real"),
        ({"penalties": (np.eye(2, dtype=complex),)}, "real"),
        ({"penalties": ()}, "at least one penalty"),
        ({"penalties": (np.eye(3),)}, "design columns"),
        ({"penalties": (np.array([[1.0, 1.0], [0.0, 1.0]]),)}, "symmetric"),
        ({"penalties": (np.diag([1.0, -1.0]), np.eye(2))}, "positive semidefinite"),
        ({"penalties": (np.zeros((2, 2)),)}, "nonzero stochastic"),
    ],
)
def test_explicit_geometry_rejects_invalid_evaluation_and_penalties(settings, message):
    arguments = {"design": np.eye(2)} | settings
    prior = gam.SDPrior.from_max_abs_effect(2.0, probability=0.9, draws=64)
    with pytest.raises(ValueError, match=message):
        prior.calibrate(gam.GaussianGeometry(**arguments))


def test_explicit_geometry_preserves_its_evaluated_layout():
    design = np.ones((6, 1))
    original = design.copy()
    geometry = gam.GaussianGeometry(design)
    result = gam.SDPrior.from_max_abs_effect(
        2.0, probability=0.9, draws=64, grid_size=17
    ).calibrate(geometry)
    assert result.diagnostics["evaluation_points"] == 6
    assert result.diagnostics["grid_shape"] == (6,)
    assert result.diagnostics["grid_size"] == 6
    np.testing.assert_array_equal(design, original)


def test_inverse_gamma_factory_constructs_a_calibrated_scale_prior():
    prior = gam.SDPrior.from_max_abs_effect(
        2.0,
        probability=0.9,
        draws=256,
        variance_prior=lambda s: tfd.InverseGamma(4.0, 3.0 * s),
    )
    builder = gam.TermBuilder.from_dict({"x": np.linspace(0.0, 1.0, 30)})
    term = builder.ps("x", k=6, scale=prior)
    result = prior.calibrate(term.basis)
    variance = term.scale.value_node[0]
    actual = variance.bijected_var.dist_node.init_dist().distribution.distribution
    assert isinstance(actual, tfd.InverseGamma)
    assert actual.scale == pytest.approx(3.0 * result.prior_scale)
    assert result.diagnostics["method"] == "integrated_variance_cdf"
    assert np.isfinite(lsl.Model(term).log_prob)


@pytest.mark.parametrize(
    "integrated, invalid",
    [
        (integrated, invalid)
        for integrated in [False, True]
        for invalid in [-1.0, np.nan, 1.0 + 1.0j]
    ]
    + [(False, np.inf)],
)
def test_nonpositive_or_nonfinite_variance_samples_fail_in_both_methods(
    invalid, integrated
):
    class InvalidSamples(tfd.Normal):
        def _sample_n(self, n, seed=None):
            return jnp.full((n,), invalid)

    target = (
        gam.MaxAbsEffect(2.0, 0.9, draws=64) if integrated else _WeightedRMS(draws=64)
    )
    prior = gam.SDPrior(
        calibration=target, variance_prior=lambda s: InvalidSamples(loc=s, scale=s)
    )
    with pytest.raises(ValueError, match="Variance samples.*finite.*positive"):
        prior.calibrate(gam.GaussianGeometry(np.ones((1, 1))))


def test_generic_calibration_rejects_a_factory_that_cannot_reach_the_target():
    prior = gam.SDPrior(
        calibration=_WeightedRMS(threshold=1e-6, draws=64),
        variance_prior=lambda _: tfd.Weibull(0.5, 1.0),
    )
    with pytest.raises(ValueError, match="Cannot calibrate.*finite positive"):
        prior.calibrate(gam.GaussianGeometry(np.ones((1, 1))))


def test_heavy_tailed_cdf_prior_uses_its_finite_quantile_despite_overflowing_draws():
    prior = gam.SDPrior.from_max_abs_effect(
        3.0,
        probability=0.95,
        variance_prior=lambda s: tfd.InverseGamma(0.05, s),
    )
    builder = gam.TermBuilder.from_dict({"x": np.linspace(0.0, 1.0, 30)})

    term = builder.ps("x", k=6, scale=prior)

    result = prior.calibrate(term.basis)
    assert result.diagnostics["method"] == "integrated_variance_cdf"
    assert np.isfinite(term.scale.value) and term.scale.value > 0
    median = result.variance_priors[0].quantile(0.5)
    assert result.initial_scales == pytest.approx((np.sqrt(median),))
    assert np.isfinite(lsl.Model(term).log_prob)


@pytest.mark.parametrize("multiple", [False, True])
def test_rounded_zero_weibull_draw_does_not_fail_an_otherwise_valid_calibration(
    multiple,
):
    geometry = gam.GaussianGeometry(
        np.random.default_rng(0).normal(size=(20, 4)),
        penalties=(np.eye(4), np.eye(4)) if multiple else None,
    )
    prior = gam.SDPrior(
        calibration=_WeightedRMS(
            threshold=2.0, probability=0.9, draws=4096, seed=79, weight=1.0
        )
    )

    result = prior.calibrate(geometry)

    assert result.diagnostics["method"] == "sampled_variance_quantile"
    assert result.diagnostics["validation_probability"] == pytest.approx(0.9, abs=0.02)
    assert all(np.isfinite(value) and value > 0 for value in result.initial_scales)


@pytest.mark.parametrize("floor, factor", [(True, 1.0), (False, 1.0), (True, 3.0)])
def test_unreachable_sampled_targets_report_calibration_failure(floor, factor):
    class FlooredRMS(_WeightedRMS):
        def statistic(self, effects):
            return np.maximum(5.0, super().statistic(effects))

    target = (
        FlooredRMS(draws=256) if floor else _WeightedRMS(threshold=1e-30, draws=256)
    )
    basis = gam.TermBuilder.from_dict({"x": np.linspace(0.0, 1.0, 30)}).bases.ps(
        "x", k=6
    )
    prior = gam.SDPrior(
        calibration=target, variance_prior=lambda s: tfd.Weibull(0.5, factor * s)
    )
    with pytest.raises(ValueError, match="Cannot calibrate.*finite positive"):
        prior.calibrate(basis)


def test_an_invalid_trial_does_not_hide_a_reachable_root_on_the_other_side():
    geometry = gam.GaussianGeometry(np.random.default_rng(0).normal(size=(20, 4)))
    prior = gam.SDPrior(
        calibration=_WeightedRMS(threshold=1.0, weight=1.0, draws=512),
        variance_prior=lambda s: tfd.Weibull(0.5, 1e-20 * s),
    )
    result = prior.calibrate(geometry)
    assert result.prior_scale > 1e15
    assert result.diagnostics["validation_probability"] == pytest.approx(0.9, abs=0.04)


def test_penalty_rescaling_preserves_the_common_stochastic_subspace():
    first = np.diag(np.r_[np.ones(25), np.zeros(25)]).astype(np.float32)
    second = np.diag(np.r_[np.zeros(25), np.ones(25)]).astype(np.float32)

    def calibrate(factor):
        geometry = gam.GaussianGeometry(np.eye(50), penalties=(first, factor * second))
        return gam.SDPrior.from_max_abs_effect(
            2.0,
            probability=0.9,
            draws=128,
            variance_prior=(
                lambda s: tfd.Weibull(0.5, s),
                lambda s: tfd.Weibull(0.5, factor * s),
            ),
        ).calibrate(geometry)

    scaled, unscaled = calibrate(1e-6), calibrate(1.0)

    assert scaled.diagnostics["rank"] == unscaled.diagnostics["rank"] == 50
    assert scaled.diagnostics["nullity"] == unscaled.diagnostics["nullity"] == 0
    assert scaled.prior_scale == pytest.approx(unscaled.prior_scale, rel=1e-5)


def test_a_bare_penalty_matrix_has_a_clear_collection_error():
    geometry = gam.GaussianGeometry(np.eye(2), penalties=cast(Any, np.eye(2)))
    with pytest.raises(TypeError, match="penalties.*tuple"):
        gam.SDPrior.from_max_abs_effect(2.0, probability=0.9).calibrate(geometry)


def test_string_designs_raise_a_clear_geometry_error():
    with pytest.raises(ValueError, match="real matrix"):
        gam.SDPrior.from_max_abs_effect(2.0, probability=0.9).calibrate(
            gam.GaussianGeometry(np.array([["a"]]))
        )


def test_complex_variance_limits_raise_a_clear_target_error():
    class ComplexLimits(gam.MaxAbsEffect):
        def variance_limits(self, values):
            return np.ones_like(values, dtype=complex)

    with pytest.raises(ValueError, match="variance limits.*real"):
        gam.SDPrior(calibration=ComplexLimits(2.0, 0.9)).calibrate(
            gam.GaussianGeometry(np.ones((1, 1)))
        )
