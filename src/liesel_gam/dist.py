from functools import cached_property

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax import tf2jax as tf
from tensorflow_probability.substrates.jax.internal.parameter_properties import (
    ParameterProperties,
)

Array = jax.typing.ArrayLike


class MultivariateNormalSingular(tfd.Distribution):
    def __init__(
        self,
        loc: Array,
        scale: Array,
        penalty: Array,
        penalty_rank: Array,
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "MultivariateNormalSingular",
    ):
        parameters = dict(locals())

        self._loc = jnp.asarray(loc)
        self._scale = jnp.asarray(scale)
        self._penalty = jnp.asarray(penalty)
        self._penalty_rank = jnp.asarray(penalty_rank)

        super().__init__(
            dtype=self._loc.dtype,
            reparameterization_type=tfd.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name,
        )

    @classmethod
    def _parameter_properties(cls, dtype=jnp.float32, num_classes=None):
        return dict(
            loc=ParameterProperties(event_ndims=1),
            scale=ParameterProperties(event_ndims=0),
            penalty=ParameterProperties(event_ndims=2),
            penalty_rank=ParameterProperties(event_ndims=0),
        )

    def _event_shape(self):
        return tf.TensorShape((jnp.shape(self._penalty)[-1],))

    def _event_shape_tensor(self):
        return jnp.array((jnp.shape(self._penalty)[-1],), dtype=self._penalty.dtype)

    def _log_prob(self, x: Array) -> Array:
        x_centered = x - self._loc

        # The following lines illustrate what the jnp.einsum call is conceptually
        # doing.
        # xt = jnp.expand_dims(x, axis=-2) # [batch_dims, 1, event_dim]
        # x = jnp.swapaxes(xt, -2, -1) # [batch_dims, event_dim, 1]
        # quad_form = jnp.squeeze((xt @ self._penalty @ x))
        quad_form = jnp.einsum(
            "...i,...ij,...j->...", x_centered, self._penalty, x_centered
        )

        neg_kernel = 0.5 * quad_form * jnp.power(self._scale, -2.0)

        return -(jnp.log(self._scale) * self._penalty_rank + neg_kernel)

    def _sample_n(self, n, seed=None) -> Array:
        shape = [n] + self.batch_shape + self.event_shape

        # The added dimension at the end here makes sure that matrix multiplication
        # with the "sqrt pcov" matrices works out correctly.
        z = jax.random.normal(key=seed, shape=shape + [1])

        # Add a dimension at 0 for the sample size.
        sqrt_cov = jnp.expand_dims(self._sqrt_cov, 0)
        centered_samples = jnp.reshape(sqrt_cov @ z, shape)

        # Add a dimension at 0 for the sample size.
        loc = jnp.expand_dims(self._loc, 0)
        scale = jnp.expand_dims(self._scale, 0)

        return scale * centered_samples + loc

    @cached_property
    def _sqrt_cov(self) -> Array:
        eigenvalues, evecs = jnp.linalg.eigh(self._penalty)
        sqrt_eval = jnp.sqrt(1 / eigenvalues)
        sqrt_eval = sqrt_eval.at[: -self._penalty_rank].set(0.0)

        event_shape = sqrt_eval.shape[-1]
        shape = sqrt_eval.shape + (event_shape,)

        r = tuple(range(event_shape))
        diags = jnp.zeros(shape).at[..., r, r].set(sqrt_eval)
        return evecs @ diags
