import inspect

import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_gam as gam


def _variance_and_bijected(scale: lsl.Var) -> tuple[lsl.Var, lsl.Var]:
    variance = scale.value_node[0]
    assert isinstance(variance, lsl.Var)

    bijected = variance.bijected_var
    assert isinstance(bijected, lsl.Var)
    return variance, bijected


def _base_prior(scale: lsl.Var) -> tfd.Distribution:
    _, bijected = _variance_and_bijected(scale)
    assert bijected.dist_node is not None
    prior = bijected.dist_node.init_dist()
    assert isinstance(prior, tfd.TransformedDistribution)
    assert isinstance(prior.distribution, tfd.Independent)
    assert prior.distribution.reinterpreted_batch_ndims == 0
    return prior.distribution.distribution


@pytest.mark.parametrize(
    ("function", "parameter"),
    (
        (gam.scale_wb, "value"),
        (gam.scale_wb, "scale"),
        (gam.scale_ig, "value"),
        (gam.scale_ig, "concentration"),
        (gam.scale_ig, "scale"),
    ),
)
def test_value_and_prior_parameters_have_no_defaults(function, parameter) -> None:
    signature = inspect.signature(function)
    assert signature.parameters[parameter].default is inspect.Parameter.empty


def test_scale_wb_places_prior_on_variance() -> None:
    inference = gs.MCMCSpec(gs.HMCKernel)
    scale = gam.scale_wb(0.5, 2.0, name="tau", inference=inference)
    variance, bijected = _variance_and_bijected(scale)
    assert bijected.dist_node is not None
    prior = bijected.dist_node.init_dist()

    assert scale.value == pytest.approx(0.5)
    assert scale.name == "tau"
    assert variance.value == pytest.approx(0.25)
    assert variance.name == "tau^2"
    assert isinstance(prior, tfd.TransformedDistribution)
    base_prior = _base_prior(scale)
    assert isinstance(base_prior, tfd.Weibull)
    assert base_prior.concentration == pytest.approx(0.5)
    assert base_prior.scale == pytest.approx(2.0)
    assert bijected.value == pytest.approx(jnp.log(0.25))
    assert bijected.inference is inference


def test_scale_ig_places_prior_on_variance() -> None:
    scale = gam.scale_ig(0.5, 3.0, 0.1, name="tau")
    variance, bijected = _variance_and_bijected(scale)
    assert bijected.dist_node is not None
    prior = bijected.dist_node.init_dist()

    assert scale.value == pytest.approx(0.5)
    assert scale.name == "tau"
    assert variance.value == pytest.approx(0.25)
    assert variance.name == "tau^2"
    assert isinstance(prior, tfd.TransformedDistribution)
    base_prior = _base_prior(scale)
    assert isinstance(base_prior, tfd.InverseGamma)
    assert base_prior.concentration == pytest.approx(3.0)
    assert base_prior.scale == pytest.approx(0.1)
    assert bijected.value == pytest.approx(jnp.log(0.25))


@pytest.mark.parametrize(
    ("scale_var", "unconstrained_value"),
    (
        (gam.scale_wb(0.5, 0.05, name="tau"), jnp.float32(2.8017025)),
        (gam.scale_ig(0.5, 3.0, 0.1, name="tau"), jnp.float32(-20.0)),
    ),
)
def test_transformed_scale_prior_log_prob_is_stable(
    scale_var: lsl.Var,
    unconstrained_value,
) -> None:
    variance, bijected = _variance_and_bijected(scale_var)
    assert bijected.dist_node is not None
    transformed_prior = bijected.dist_node.init_dist()
    base_prior = _base_prior(scale_var)

    actual = transformed_prior.log_prob(unconstrained_value)
    variance_value = jnp.exp(unconstrained_value)
    expected = base_prior.log_prob(variance_value) + unconstrained_value

    assert jnp.isfinite(expected)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("function", (gam.scale_wb, gam.scale_ig))
def test_scale_can_be_used_in_model(function) -> None:
    if function is gam.scale_wb:
        scale = function(0.5, 2.0, name="tau")
    else:
        scale = function(0.5, 3.0, 0.1, name="tau")

    model = lsl.Model([scale])

    assert "tau" in model.vars
    assert "tau^2" in model.vars
    assert jnp.isfinite(model.log_prob)


def test_empty_scale_name_keeps_variance_unnamed() -> None:
    scale = gam.scale_wb(0.5, 2.0, name="")
    variance, bijected = _variance_and_bijected(scale)

    assert not scale.name
    assert not variance.name
    assert not bijected.name


@pytest.mark.parametrize("value", (0.0, -0.5))
@pytest.mark.parametrize("function", (gam.scale_wb, gam.scale_ig))
def test_scale_value_must_be_positive(function, value) -> None:
    if function is gam.scale_wb:
        with pytest.raises(ValueError, match="positive"):
            function(value, 2.0)
    else:
        with pytest.raises(ValueError, match="positive"):
            function(value, 3.0, 0.1)
