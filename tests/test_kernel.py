import jax
import jax.numpy as jnp
import liesel.model as lsl
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd
from ryp import r, to_py

import liesel_gam as gam
import liesel_gam.term_builder as gb


@pytest.fixture(scope="module")
def columb():
    r("library(mgcv)")
    r("data(columb)")
    columb = to_py("columb", format="pandas")
    return columb


class TestGibbsKernel:
    def test_no_model(self, columb):
        tb = gb.TermBuilder.from_df(columb)

        term = tb.ps("x", k=10)
        with pytest.raises(ValueError):
            gam.init_star_ig_gibbs(
                position_keys=[term.scale.value_node[0].name],
                coef=term.coef,
                scale=term.scale,
                penalty=term.basis.penalty.value,
            )

    def test_penalty_array(self, columb):
        tb = gb.TermBuilder.from_df(columb)

        term = tb.ps("x", k=10)
        model = lsl.Model([term])
        tau2_name = term.scale.value_node[0].name
        kernel = gam.init_star_ig_gibbs(
            position_keys=[tau2_name],
            coef=term.coef,
            scale=term.scale,
            penalty=term.basis.penalty.value,
        )
        pos = kernel._transition_fn(jax.random.key(1), model.state)

        assert not jnp.isnan(pos[tau2_name])

    def test_position_key(self, columb):
        tb = gb.TermBuilder.from_df(columb)

        term = tb.ps("x", k=10)
        _ = lsl.Model([term])
        with pytest.raises(ValueError):
            gam.init_star_ig_gibbs(
                position_keys=["test"],
                coef=term.coef,
                scale=term.scale,
                penalty=term.basis.penalty.value,
            )

        tau2_name = term.scale.value_node[0].name
        with pytest.raises(ValueError):
            gam.init_star_ig_gibbs(
                position_keys=[tau2_name, tau2_name],
                coef=term.coef,
                scale=term.scale,
                penalty=term.basis.penalty.value,
            )

    def test_penalty_none(self, columb):
        tb = gb.TermBuilder.from_df(columb)

        term = tb.ps("x", k=10)
        model = lsl.Model([term])
        tau2_name = term.scale.value_node[0].name
        kernel = gam.init_star_ig_gibbs(
            position_keys=[tau2_name],
            coef=term.coef,
            scale=term.scale,
            penalty=None,
        )
        pos = kernel._transition_fn(jax.random.key(1), model.state)

        assert not jnp.isnan(pos[tau2_name])

    def test_penalty_none_dist_without_penalty(self, columb):
        tb = gb.TermBuilder.from_df(columb)

        term = tb.lin(
            "x",
            prior=lsl.Dist(tfd.Normal, loc=0.0, scale=gam.ScaleIG(1.0, 1.0, 0.005)),
        )
        model = lsl.Model([term])
        tau2_name = term.coef.dist_node["scale"].value_node[0].name
        kernel = gam.init_star_ig_gibbs(
            position_keys=[tau2_name],
            coef=term.coef,
            scale=term.coef.dist_node["scale"],
            penalty=None,
        )
        pos = kernel._transition_fn(jax.random.key(1), model.state)

        assert not jnp.isnan(pos[tau2_name])
