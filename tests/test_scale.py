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
    scale = gam.scale_wb(0.25, 2.0, name="tau", inference=inference)
    variance, bijected = _variance_and_bijected(scale)
    assert bijected.dist_node is not None
    prior = bijected.dist_node.init_dist()

    assert scale.value == pytest.approx(0.5)
    assert scale.name == "tau"
    assert variance.value == pytest.approx(0.25)
    assert variance.name == "tau^2"
    assert isinstance(prior, tfd.TransformedDistribution)
    assert isinstance(prior.distribution, tfd.Weibull)
    assert prior.distribution.concentration == pytest.approx(0.5)
    assert prior.distribution.scale == pytest.approx(2.0)
    assert bijected.value == pytest.approx(jnp.log(0.25))
    assert bijected.inference is inference


def test_scale_ig_places_prior_on_variance() -> None:
    scale = gam.scale_ig(0.25, 3.0, 0.1, name="tau")
    variance, bijected = _variance_and_bijected(scale)
    assert bijected.dist_node is not None
    prior = bijected.dist_node.init_dist()

    assert scale.value == pytest.approx(0.5)
    assert scale.name == "tau"
    assert variance.value == pytest.approx(0.25)
    assert variance.name == "tau^2"
    assert isinstance(prior, tfd.TransformedDistribution)
    assert isinstance(prior.distribution, tfd.InverseGamma)
    assert prior.distribution.concentration == pytest.approx(3.0)
    assert prior.distribution.scale == pytest.approx(0.1)
    assert bijected.value == pytest.approx(jnp.log(0.25))


@pytest.mark.parametrize("function", (gam.scale_wb, gam.scale_ig))
def test_scale_can_be_used_in_model(function) -> None:
    if function is gam.scale_wb:
        scale = function(0.25, 2.0, name="tau")
    else:
        scale = function(0.25, 3.0, 0.1, name="tau")

    model = lsl.Model([scale])

    assert "tau" in model.vars
    assert "tau^2" in model.vars
    assert jnp.isfinite(model.log_prob)


def test_empty_scale_name_keeps_variance_unnamed() -> None:
    scale = gam.scale_wb(0.25, 2.0, name="")
    variance, bijected = _variance_and_bijected(scale)

    assert not scale.name
    assert not variance.name
    assert not bijected.name
