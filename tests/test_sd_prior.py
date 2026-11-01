from dataclasses import FrozenInstanceError
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
    # Integrating the exact scalar density is independent of the MC calibration.
    tail, _ = quad(
        lambda z: (
            np.exp(-2.0 / (np.sqrt(result.prior_scale) * z))
            * np.sqrt(2.0 / np.pi)
            * np.exp(-z * z / 2.0)
        ),
        0.0,
        np.inf,
    )
    assert 1.0 - tail == pytest.approx(0.9, abs=0.01)
    assert result.diagnostics["evaluation_points"] == 129
    assert result.diagnostics["rank"] == 1
    distribution = result.variance_priors[0]
    assert isinstance(distribution, tfd.Weibull)
    assert distribution.concentration == pytest.approx(0.5)
    assert distribution.scale == pytest.approx(result.prior_scale)
    assert result.initial_scales[0] == pytest.approx(
        np.sqrt(result.prior_scale) * np.log(2.0)
    )


def test_ps_constructs_a_variance_prior_under_ordinary_singular_defaults() -> None:
    tb = gam.TermBuilder.from_dict({"x": np.linspace(0.0, 1.0, 30)})
    prior = gam.SDPrior.from_max_abs_effect(3.0, probability=0.95, draws=512)

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
    assert parameter.inference is None
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
    inference = gs.MCMCSpec(gs.NUTSKernel)
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
