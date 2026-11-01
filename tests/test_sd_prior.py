from dataclasses import FrozenInstanceError, dataclass
from operator import setitem
from typing import Any, cast

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from liesel.types import Position
from scipy.integrate import quad

import liesel_gam as gam


@dataclass(frozen=True)
class _RootMeanSquare:
    threshold: float = 2.0
    probability: float = 0.9
    grid_size: int = 65
    seed: int = 42
    draws: int = 8192

    def statistic(self, effects):
        return np.sqrt(np.mean(effects**2, axis=-1))

    def variance_limits(self, values):
        return (self.threshold / values) ** 2


def _ps_basis() -> gam.Basis:
    return gam.TermBuilder.from_dict({"x": np.linspace(0.0, 1.0, 20)}).bases.ps(
        "x", k=6
    )


def _scalar_coverage(prior_scale: float, factor: float = 1.0) -> float:
    # Integrate the exact Weibull-times-Normal density independently of calibration.
    tail, _ = quad(
        lambda z: (
            np.exp(-2.0 / (np.sqrt(prior_scale) * factor * z))
            * np.sqrt(2.0 / np.pi)
            * np.exp(-z * z / 2.0)
        ),
        0.0,
        np.inf,
    )
    return 1.0 - tail


def test_explicit_calibration_bounds_a_known_scalar_effect() -> None:
    basis = gam.Basis(
        jnp.linspace(-1.0, 1.0, 20),
        basis_fn=lambda x: x[:, None],
        penalty=jnp.eye(1),
        xname="x",
        use_callback=False,
        row_wise=True,
    )
    prior = gam.SDPrior.from_max_abs_effect(2.0, probability=0.9, draws=8192)

    result = prior.calibrate(basis)

    # Here max |f| = |beta|, beta = sqrt(s) * Exp(1) * Normal(0, 1).
    assert _scalar_coverage(result.prior_scale) == pytest.approx(0.9, abs=0.01)
    assert result.diagnostics["evaluation_points"] == 129
    assert result.diagnostics["rank"] == 1
    distribution = result.variance_priors[0]
    assert isinstance(distribution, tfd.Weibull)
    assert distribution.concentration == pytest.approx(0.5)
    assert distribution.scale == pytest.approx(result.prior_scale)
    assert result.initial_scales[0] == pytest.approx(
        np.sqrt(result.prior_scale) * np.log(2.0)
    )


@pytest.mark.parametrize("convenience", [False, True])
def test_ps_constructs_a_variance_prior_under_ordinary_singular_defaults(
    convenience,
) -> None:
    tb = gam.TermBuilder.from_dict({"x": np.linspace(0.0, 1.0, 30)})
    prior = (
        gam.SDPrior.from_max_abs_effect(3.0, probability=0.95, draws=512)
        if convenience
        else gam.SDPrior(calibration=gam.MaxAbsEffect(3.0, probability=0.95, draws=512))
    )

    term = tb.ps("x", k=6, scale=prior)

    result = prior.calibrate(term.basis)
    variance = term.scale.value_node[0]
    parameter = variance.bijected_var
    transformed = parameter.dist_node.init_dist()
    actual = transformed.distribution.distribution
    assert actual.concentration == pytest.approx(0.5)
    assert actual.scale == pytest.approx(result.prior_scale)
    assert term.scale.value == pytest.approx(result.initial_scales[0])
    assert variance.value == pytest.approx(float(term.scale.value) ** 2)
    assert parameter.inference.kernel is gs.NUTSKernel
    assert parameter.value == pytest.approx(float(jnp.log(variance.value)))
    assert result.diagnostics["nullity"] == 1
    assert term.scale.name == r"$\tau_{ps(x)}$"
    assert np.isfinite(lsl.Model(term).log_prob)


def test_opt_in_default_reuses_settings_but_creates_independent_ps_and_np() -> None:
    prior = gam.SDPrior.from_max_abs_effect(3.0, probability=0.95, draws=256)
    tb = gam.TermBuilder.from_dict(
        {"x": np.linspace(0.0, 1.0, 30)}, default_scale_fn=prior
    )

    ps = tb.ps("x", k=6)
    np_term = tb.np("x", k=8)

    assert ps.scale is not np_term.scale
    assert ps.scale.value_node[0] is not np_term.scale.value_node[0]
    assert ps.scale.value_node[0].bijected_var is not (
        np_term.scale.value_node[0].bijected_var
    )
    assert float(ps.scale.value) != float(np_term.scale.value)
    assert np.isfinite(lsl.Model([ps, np_term]).log_prob)


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"max_abs_effect": 0.0}, "max_abs_effect"),
        ({"max_abs_effect": -1.0}, "max_abs_effect"),
        ({"max_abs_effect": np.inf}, "max_abs_effect"),
        ({"probability": 0.0}, "probability"),
        ({"probability": 1.0}, "probability"),
        ({"probability": np.nan}, "probability"),
        ({"grid_size": 1}, "grid_size"),
        ({"grid_size": 4.5}, "grid_size"),
        ({"grid_size": True}, "grid_size"),
        ({"draws": 1}, "draws"),
        ({"draws": 2.5}, "draws"),
        ({"draws": True}, "draws"),
        ({"seed": -1}, "seed"),
        ({"seed": 1.5}, "seed"),
    ],
)
def test_invalid_calibration_settings_fail_clearly(settings, message) -> None:
    options = {"max_abs_effect": 3.0, "probability": 0.95, "draws": 32} | settings
    with pytest.raises(ValueError, match=message):
        gam.MaxAbsEffect(**options)


@pytest.mark.parametrize(
    "penalty, message",
    [
        (np.zeros((2, 2)), "stochastic"),
        (np.diag([1.0, -0.1]), "positive semidefinite"),
        (np.array([[1.0, 1.0], [0.0, 1.0]]), "symmetric"),
        (np.array([[1.0, 1e-3], [0.0, 1.0]]), "symmetric"),
        (np.diag([1.0, np.nan]), "finite"),
    ],
)
def test_invalid_penalties_fail_before_calibration(penalty, message) -> None:
    basis = gam.Basis(
        jnp.linspace(0.0, 1.0, 10),
        basis_fn=lambda x: jnp.column_stack((jnp.ones_like(x), x)),
        penalty=penalty,
        xname="x",
        use_callback=False,
        row_wise=True,
    )
    prior = gam.SDPrior.from_max_abs_effect(3.0, probability=0.95, draws=32)
    with pytest.raises(ValueError, match=message):
        prior.calibrate(basis)


def test_zero_effect_on_the_grid_cannot_be_calibrated() -> None:
    basis = gam.Basis(
        jnp.zeros(10),
        basis_fn=lambda x: x[:, None],
        penalty=jnp.eye(1),
        xname="x",
        use_callback=False,
        row_wise=True,
    )
    with pytest.raises(ValueError, match="stochastic effect"):
        gam.SDPrior.from_max_abs_effect(3.0, probability=0.95).calibrate(basis)


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_explicit_start_must_be_finite_and_positive(value) -> None:
    with pytest.raises(ValueError, match="value.*finite.*positive"):
        gam.SDPrior.from_max_abs_effect(3.0, probability=0.95, value=value)


@pytest.mark.parametrize("row_wise", [None, False])
def test_automatic_grids_require_a_known_scalar_row_wise_basis(row_wise) -> None:
    basis = gam.Basis(
        jnp.linspace(0.0, 1.0, 10),
        basis_fn=lambda x: x[:, None],
        penalty=jnp.eye(1),
        xname="x",
        use_callback=False,
        row_wise=row_wise,
    )
    with pytest.raises(ValueError, match="scalar.*row-wise"):
        gam.SDPrior.from_max_abs_effect(3.0, probability=0.95).calibrate(basis)


@pytest.mark.parametrize("bound", [1.0e-8, 1.0e8])
def test_calibration_is_not_limited_to_an_arbitrary_search_bracket(bound) -> None:
    tb = gam.TermBuilder.from_dict({"x": np.linspace(0.0, 1.0, 20)})
    basis = tb.bases.ps("x", k=6)
    baseline = gam.SDPrior.from_max_abs_effect(1.0, probability=0.95, draws=128)
    scaled = gam.SDPrior.from_max_abs_effect(bound, probability=0.95, draws=128)

    result = scaled.calibrate(basis)

    assert result.prior_scale == pytest.approx(
        baseline.calibrate(basis).prior_scale * bound**2, rel=1.0e-5, abs=0.0
    )


@pytest.mark.parametrize("diagonal_penalty", [True, False])
def test_constructed_prior_has_the_requested_prior_predictive_coverage(
    diagonal_penalty,
) -> None:
    tb = gam.TermBuilder.from_dict({"x": np.linspace(0.0, 1.0, 40)})
    prior = gam.SDPrior.from_max_abs_effect(2.0, probability=0.9, draws=2048)
    term = tb.ps("x", k=8, diagonal_penalty=diagonal_penalty, scale=prior)
    variance_prior = prior.calibrate(term.basis).variance_priors[0]

    assert term.coef.dist_node is not None
    conditional = term.coef.dist_node.init_dist()
    coefficients = conditional.sample(8192, seed=jax.random.key(975))
    coefficients = coefficients / term.scale.value
    variances = variance_prior.sample(8192, seed=jax.random.key(836))
    coefficients = coefficients * jnp.sqrt(variances[:, None])
    effects = term.predict(
        Position({term.coef.name: coefficients}),
        newdata=Position({"x": np.linspace(0.0, 1.0, 129)}),
    )

    coverage = np.mean(np.max(np.abs(effects), axis=-1) <= 2.0)
    assert coverage == pytest.approx(0.9, abs=0.025)


@pytest.mark.parametrize("malformed", ["vector", "nonfinite"])
def test_calibration_requires_a_finite_matrix_valued_basis(malformed) -> None:
    function = (
        (lambda x: x)
        if malformed == "vector"
        else (lambda x: jnp.full((len(x), 1), jnp.nan))
    )
    basis = gam.Basis(
        jnp.linspace(0.0, 1.0, 10),
        basis_fn=function,
        penalty=jnp.eye(1),
        xname="x",
        use_callback=False,
        row_wise=True,
    )
    with pytest.raises(ValueError, match="finite.*matrix"):
        gam.SDPrior.from_max_abs_effect(3.0, probability=0.95).calibrate(basis)


def test_explicit_results_are_pure_reproducible_and_immutable() -> None:
    tb = gam.TermBuilder.from_dict({"x": np.linspace(0.0, 1.0, 20)})
    prior = gam.SDPrior.from_max_abs_effect(
        3.0, probability=0.95, draws=128, grid_size=65
    )
    term = tb.np("x", k=6, scale=prior)
    model = lsl.Model(term)
    before = {name: np.array(var.value, copy=True) for name, var in model.vars.items()}

    result = prior.calibrate(term.basis)
    repeated = prior.calibrate(term.basis)

    assert result.prior_scale == repeated.prior_scale
    assert result.diagnostics == repeated.diagnostics
    assert result.diagnostics["evaluation_points"] == 65
    for name, value in before.items():
        np.testing.assert_array_equal(model.vars[name].value, value)
    with pytest.raises(FrozenInstanceError):
        setattr(result, "prior_scale", 1.0)
    with pytest.raises(TypeError):
        setitem(cast(Any, result.diagnostics), "rank", 0)
    copied_diagnostics = dict(result.diagnostics)
    copied = gam.CalibrationResult(
        result.prior_scale,
        result.variance_priors,
        result.initial_scales,
        copied_diagnostics,
    )
    copied_diagnostics["rank"] = 0
    assert copied.diagnostics["rank"] == result.diagnostics["rank"]


@pytest.mark.parametrize("method", ["ps", "np"])
@pytest.mark.parametrize("factor_scale", [False, True])
def test_explicit_start_bijector_and_inference_are_preserved(
    method, factor_scale
) -> None:
    tb = gam.TermBuilder.from_dict({"x": np.linspace(0.0, 1.0, 20)})
    inference = gs.MCMCSpec(gs.HMCKernel)
    prior = gam.SDPrior(
        calibration=gam.MaxAbsEffect(3.0, probability=0.95, draws=128),
        value=0.7,
        bijector=tfb.Softplus(),
        inference=inference,
    )

    term = getattr(tb, method)("x", k=6, scale=prior, factor_scale=factor_scale)

    variance = term.scale.value_node[0]
    parameter = variance.bijected_var
    assert term.scale.value == pytest.approx(0.7)
    assert variance.value == pytest.approx(0.49)
    assert parameter.inference is inference
    assert float(parameter.value) == pytest.approx(float(tfb.Softplus().inverse(0.49)))
    assert np.isfinite(lsl.Model(term).log_prob)


@pytest.mark.parametrize("convenience", [False, True])
def test_explicit_none_disables_the_sd_variance_sampler(convenience) -> None:
    prior = (
        gam.SDPrior.from_max_abs_effect(
            3.0, probability=0.95, draws=128, inference=None
        )
        if convenience
        else gam.SDPrior(
            calibration=gam.MaxAbsEffect(3.0, probability=0.95, draws=128),
            inference=None,
        )
    )
    term = gam.TermBuilder.from_dict({"x": np.linspace(0.0, 1.0, 20)}).np(
        "x", k=6, scale=prior
    )
    variance = term.scale.value_node[0]
    assert variance.bijected_var.inference is None
    assert np.isfinite(lsl.Model(term).log_prob)


def test_library_default_and_ordinary_explicit_scales_remain_unchanged() -> None:
    data = {"x": np.linspace(0.0, 1.0, 20)}
    default_term = gam.TermBuilder.from_dict(data).ps("x", k=6)
    assert isinstance(default_term.scale, gam.ScaleIG)
    assert default_term.scale.value == pytest.approx(1.0)

    prior = gam.SDPrior.from_max_abs_effect(3.0, probability=0.95, draws=128)
    tb = gam.TermBuilder.from_dict(data, default_scale_fn=prior)
    fixed = tb.np("x", k=6, scale=0.4)
    supplied = gam.scale_wb(0.5, 2.0, name="supplied")
    custom = tb.ps("x", k=6, scale=supplied)
    assert fixed.scale.value == pytest.approx(0.4)
    assert custom.scale is supplied
    assert np.isfinite(lsl.Model([fixed, custom]).log_prob)

    callable_tb = gam.TermBuilder.from_dict(
        data, default_scale_fn=lambda: gam.scale_wb(0.3, 2.0)
    )
    assert callable_tb.ps("x", k=6).scale.value == pytest.approx(0.3)


def test_explicit_iid_basis_uses_identity_precision() -> None:
    basis = gam.Basis(
        jnp.linspace(-1.0, 1.0, 10),
        basis_fn=lambda x: x[:, None],
        penalty=None,
        xname="x",
        use_callback=False,
        row_wise=True,
    )
    prior = gam.SDPrior.from_max_abs_effect(3.0, probability=0.95, draws=128)
    result = prior.calibrate(basis)
    assert result.diagnostics["rank"] == 1
    assert result.diagnostics["nullity"] == 0
    assert result.prior_scale > 0


@pytest.mark.parametrize("k, order", [(20, 2), (40, 2), (40, 3)])
def test_float32_penalty_roundoff_does_not_reject_valid_smooths(k, order) -> None:
    x = (
        np.linspace(0.0, 1.0, 200)
        if k == 20
        else np.random.default_rng(1).normal(size=200)
    )
    tb = gam.TermBuilder.from_dict({"x": x})
    prior = gam.SDPrior.from_max_abs_effect(3.0, probability=0.95, draws=64)
    term = tb.ps("x", k=k, penalty_order=order, diagonal_penalty=False, scale=prior)
    assert np.isfinite(lsl.Model(term).log_prob)


def test_float32_half_normal_cdf_roundoff_allows_calibration() -> None:
    prior = gam.SDPrior.from_max_abs_effect(
        3.0, probability=0.95, variance_prior=lambda s: tfd.HalfNormal(s)
    )
    builder = gam.TermBuilder.from_dict({"x": np.linspace(0.0, 1.0, 100)})

    term = builder.ps("x", k=10, scale=prior)

    result = prior.calibrate(term.basis)
    assert isinstance(result.variance_priors[0], tfd.HalfNormal)
    assert result.diagnostics["validation_probability"] == pytest.approx(0.95, abs=0.02)
    assert np.isfinite(lsl.Model(term).log_prob)


@pytest.mark.parametrize("use_default", [False, True])
def test_multivariate_terms_reject_univariate_sd_calibration(use_default) -> None:
    prior = gam.SDPrior.from_max_abs_effect(3.0, probability=0.95, draws=64)
    builder = gam.MVTermBuilder.from_dict(
        {"x": np.linspace(0.0, 1.0, 20)},
        jnp.eye(2),
        default_scale_fn=cast(
            Any, prior if use_default else gam.VarIGPrior(1.0, 0.005)
        ),
    )
    with pytest.raises(ValueError, match="SDPrior.*multivariate"):
        builder.ps(
            "x", k=6, dimension_scale=1.0, **({} if use_default else {"scale": prior})
        )
    # An explicit ordinary scale remains usable under an opted-in scalar default.
    term = builder.ps("x", k=6, dimension_scale=1.0, scale=1.0)
    assert np.isfinite(lsl.Model(term).log_prob)


@pytest.mark.parametrize("multivariate", [False, True])
def test_callable_defaults_must_return_scale_variables(multivariate) -> None:
    prior = gam.SDPrior.from_max_abs_effect(3.0, probability=0.95, draws=64)
    data = {"x": np.linspace(0.0, 1.0, 20)}
    invalid_default = cast(Any, lambda: prior)
    builder = (
        gam.MVTermBuilder.from_dict(data, jnp.eye(2), default_scale_fn=invalid_default)
        if multivariate
        else gam.TermBuilder.from_dict(data, default_scale_fn=invalid_default)
    )
    with pytest.raises(TypeError, match="default_scale_fn.*lsl.Var.*ScaleIG"):
        if isinstance(builder, gam.MVTermBuilder):
            builder.ps("x", k=6, dimension_scale=1.0)
        else:
            builder.ps("x", k=6)


@pytest.mark.parametrize("convenience", [False, True])
def test_none_bijector_is_rejected_consistently(convenience) -> None:
    with pytest.raises(ValueError, match="bijector"):
        if convenience:
            gam.SDPrior.from_max_abs_effect(
                3.0, probability=0.95, bijector=cast(Any, None)
            )
        else:
            gam.SDPrior(
                calibration=gam.MaxAbsEffect(3.0, probability=0.95),
                bijector=cast(Any, None),
            )


@pytest.mark.parametrize("method", ["cr", "cs", "cc", "bs", "cp"])
def test_compatible_splines_use_the_sd_builder_default(method) -> None:
    prior = gam.SDPrior.from_max_abs_effect(3.0, probability=0.95, draws=64)
    builder = gam.TermBuilder.from_dict(
        {"x": np.linspace(0.0, 1.0, 30)}, default_scale_fn=prior
    )
    term = getattr(builder, method)("x", k=6)
    variance = term.scale.value_node[0]
    assert variance.bijected_var.inference.kernel is gs.NUTSKernel
    assert np.isfinite(lsl.Model(term).log_prob)


def test_unsupported_default_term_identifies_the_term_and_explicit_override() -> None:
    builder = gam.TermBuilder.from_dict(
        {"group": ["a", "b"] * 10},
        default_scale_fn=gam.SDPrior.from_max_abs_effect(3.0, probability=0.95),
    )
    with pytest.raises(ValueError, match=r"ri\(group\).*explicit, non-SDPrior scale"):
        builder.ri("group")
    term = builder.ri("group", scale=1.0)
    assert np.isfinite(lsl.Model(term).log_prob)


def test_invalid_cdf_is_not_reported_as_an_unattainable_target() -> None:
    basis = _ps_basis()
    prior = gam.SDPrior.from_max_abs_effect(
        3.0,
        probability=0.95,
        draws=64,
        variance_prior=lambda s: tfd.Weibull(0.5, np.nan),
    )
    with pytest.raises(ValueError, match="CDF.*finite"):
        prior.calibrate(basis)


def test_a_valid_factory_with_an_unattainable_target_fails_clearly() -> None:
    basis = _ps_basis()
    prior = gam.SDPrior.from_max_abs_effect(
        3.0,
        probability=0.95,
        draws=64,
        variance_prior=lambda _: tfd.Weibull(0.5, 1.0),
    )
    with pytest.raises(ValueError, match="Cannot calibrate.*finite positive"):
        prior.calibrate(basis)


def test_single_variance_calibration_rejects_wrong_factory_counts() -> None:
    factories = (lambda s: tfd.Weibull(0.5, s),) * 2
    prior = gam.SDPrior.from_max_abs_effect(
        3.0, probability=0.95, draws=64, variance_prior=cast(Any, factories)
    )
    basis = _ps_basis()
    with pytest.raises(ValueError, match="Expected 1 variance-prior factories"):
        prior.calibrate(basis)


@pytest.mark.parametrize("event_shape", [False, True])
def test_variance_prior_must_be_scalar(event_shape) -> None:
    def factory(s):
        distribution = tfd.Weibull(0.5, jnp.array([s, 2 * s]))
        return tfd.Independent(distribution, 1) if event_shape else distribution

    prior = gam.SDPrior.from_max_abs_effect(
        3.0, probability=0.95, draws=64, variance_prior=factory
    )
    basis = _ps_basis()
    with pytest.raises(ValueError, match="scalar"):
        prior.calibrate(basis)


@pytest.mark.parametrize("median", [np.nan, np.inf, 0.0, -1.0])
def test_invalid_variance_prior_medians_are_rejected(median) -> None:
    class InvalidMedian(tfd.Weibull):
        def _quantile(self, value, **kwargs):
            return jnp.asarray(median)

    prior = gam.SDPrior.from_max_abs_effect(
        3.0,
        probability=0.95,
        draws=64,
        variance_prior=lambda s: InvalidMedian(0.5, s),
    )
    basis = _ps_basis()
    with pytest.raises(ValueError, match="median.*finite.*positive"):
        prior.calibrate(basis)


def test_single_factory_tuple_constructs_the_same_prior() -> None:
    prior = gam.SDPrior.from_max_abs_effect(
        3.0,
        probability=0.95,
        draws=64,
        variance_prior=cast(Any, (lambda s: tfd.Weibull(0.5, s),)),
    )
    term = gam.TermBuilder.from_dict({"x": np.linspace(0.0, 1.0, 20)}).ps(
        "x", k=6, scale=prior
    )
    assert np.isfinite(lsl.Model(term).log_prob)


def test_default_prior_recipe_is_usable_as_a_dictionary_key() -> None:
    prior = gam.SDPrior.from_max_abs_effect(3.0, probability=0.95)
    recipes = {prior: "age"}
    assert recipes[prior] == "age"


def test_shared_workflow_calibrates_a_target_without_its_own_algorithm() -> None:
    target = _RootMeanSquare()
    basis = gam.Basis(
        jnp.linspace(-1.0, 1.0, 20),
        basis_fn=lambda x: x[:, None],
        penalty=jnp.eye(1),
        xname="x",
        use_callback=False,
        row_wise=True,
    )

    result = gam.SDPrior(calibration=target).calibrate(basis)

    rms_x = np.sqrt(np.mean(np.linspace(-1.0, 1.0, 65) ** 2))
    assert _scalar_coverage(result.prior_scale, rms_x) == pytest.approx(0.9, abs=0.01)
    assert result.diagnostics["method"] == "integrated_variance_cdf"
    assert result.diagnostics["evaluation_points"] == 65


def test_maximum_effect_retains_the_single_variance_cdf_method() -> None:
    basis = _ps_basis()
    result = gam.SDPrior.from_max_abs_effect(
        3.0, probability=0.95, draws=512
    ).calibrate(basis)
    assert result.diagnostics["method"] == "integrated_variance_cdf"
    assert abs(float(result.diagnostics["validation_probability"]) - 0.95) < 0.025


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"probability": 1.0}, "probability"),
        ({"threshold": np.nan}, "threshold"),
        ({"draws": 1}, "draws"),
        ({"seed": -1}, "seed"),
    ],
)
def test_shared_workflow_validates_settings_for_target_only_objects(
    settings, message
) -> None:
    target = _RootMeanSquare(**settings)
    basis = _ps_basis()
    with pytest.raises(ValueError, match=message):
        gam.SDPrior(calibration=target).calibrate(basis)


@pytest.mark.parametrize("invalid", ["shape", "range"])
def test_shared_workflow_rejects_malformed_cdf_probabilities(invalid) -> None:
    class InvalidCDF(tfd.Weibull):
        def _cdf(self, y, **kwargs):
            return jnp.asarray(0.5) if invalid == "shape" else jnp.full_like(y, 2.0)

    prior = gam.SDPrior.from_max_abs_effect(
        3.0,
        probability=0.95,
        draws=64,
        variance_prior=lambda s: InvalidCDF(0.5, s),
    )
    basis = _ps_basis()
    with pytest.raises(ValueError, match="CDF"):
        prior.calibrate(basis)


def test_cdf_shortcut_handles_a_target_with_zero_unit_statistics() -> None:
    class PositiveMaximum(gam.MaxAbsEffect):
        def statistic(self, effects):
            return np.maximum(0.0, np.max(effects, axis=-1))

    basis = gam.Basis(
        jnp.linspace(0.0, 1.0, 20),
        basis_fn=lambda x: x[:, None],
        xname="x",
        row_wise=True,
        use_callback=False,
    )
    result = gam.SDPrior(calibration=PositiveMaximum(2.0, 0.9, draws=8192)).calibrate(
        basis
    )
    # Half the unit draws have a nonpositive slope and always satisfy this event.
    assert 0.5 + 0.5 * _scalar_coverage(result.prior_scale) == pytest.approx(
        0.9, abs=0.01
    )


@pytest.mark.parametrize(
    "invalid", ["statistic_nan", "statistic_shape", "limits_negative", "limits_shape"]
)
def test_shared_workflow_rejects_invalid_target_outputs(invalid) -> None:
    class InvalidTarget(_RootMeanSquare):
        def statistic(self, effects):
            if invalid == "statistic_nan":
                return np.full(len(effects), np.nan)
            if invalid == "statistic_shape":
                return np.asarray(1.0)
            return super().statistic(effects)

        def variance_limits(self, values):
            if invalid == "limits_negative":
                return -np.ones_like(values)
            if invalid == "limits_shape":
                return np.asarray(1.0)
            return super().variance_limits(values)

    basis = _ps_basis()
    with pytest.raises(ValueError, match="(statistic|variance limits)"):
        gam.SDPrior(calibration=InvalidTarget(draws=64)).calibrate(basis)
