import jax.numpy as jnp
import jax.random as jrd
import tensorflow_probability.substrates.jax.distributions as tfd
from liesel.contrib import splines
from liesel.distributions import MultivariateNormalDegenerate as MVND
from tensorflow_probability.substrates.jax import tf2jax as tf

import liesel_gam as gam


class TestMultivariateNormalSingular:
    def test_log_prob(self) -> None:
        pen = splines.pspline_penalty(d=10, diff=2)
        mvns = gam.MultivariateNormalSingular(
            loc=0.0, scale=1.0, penalty=pen, penalty_rank=8
        )
        mvnd = MVND.from_penalty(loc=0.0, var=1.0, pen=pen)

        x = jrd.normal(jrd.key(1), (20, 10))

        assert mvns.log_prob(x).shape == (20,)

        impl_diffs = jnp.diff(mvnd.log_prob(x) - mvns.log_prob(x))
        assert jnp.allclose(impl_diffs, 0.0, atol=1e-5)

    def test_event_shape(self) -> None:
        pen = splines.pspline_penalty(d=10, diff=2)
        mvns = gam.MultivariateNormalSingular(
            loc=0.0, scale=1.0, penalty=pen, penalty_rank=8
        )

        assert mvns.event_shape == tf.TensorShape([10])
        assert mvns.event_shape_tensor() == 10

    def test_samples(self) -> None:
        pen = splines.pspline_penalty(d=10, diff=2)
        mvns = gam.MultivariateNormalSingular(
            loc=0.0, scale=1.0, penalty=pen, penalty_rank=8
        )
        mvnd = MVND.from_penalty(loc=0.0, var=1.0, pen=pen)

        dist = tfd.MultivariateNormalFullCovariance(
            loc=0.0, covariance_matrix=jnp.eye(10)
        )

        dist.sample((1,), seed=jrd.key(1))

        x1 = mvns.sample((1,), seed=jrd.key(1))
        x2 = mvnd.sample((1,), seed=jrd.key(1))
        assert jnp.allclose(x1, x2)
