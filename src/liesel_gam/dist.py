from collections.abc import Callable, Sequence
from functools import cached_property, reduce
from math import prod
from typing import Self

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
        return jnp.array((jnp.shape(self._penalty)[-1],))

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


def _diag_of_kron_of_diag_with_identities(in_diags: Sequence[jax.Array]) -> jax.Array:
    sizes = [v.shape[-1] for v in in_diags]
    diag = jnp.zeros(prod(sizes))

    for j in range(len(in_diags)):
        left_size = prod(sizes[:j])
        right_size = prod(sizes[(j + 1) :])
        d = in_diags[j]

        # First handle identities to the right: repeat
        d_rep = jnp.repeat(d, right_size)

        # Then handle identities to the left: tile
        d_tile = jnp.tile(d_rep, left_size)

        diag = diag + d_tile

    return diag


def _materialize_precision(penalties: Sequence[jax.Array]) -> jax.Array:
    """
    Build K(tau^2) = sum_{j=1}^p K_j / tau_j^2
    with K_j = I_{d1} ⊗ ... ⊗ I_{dj-1} ⊗ Ktilde_j ⊗ I_{dj+1} ⊗ ... ⊗ I_{dp}.

    Parameters
    ----------
    tau2 : array-like, shape (p,)
        Squared smoothing parameters (tau_1^2, ..., tau_p^2).
    K_tilde : sequence of arrays
        List/tuple of p matrices, K_tilde[j] has shape (d_j, d_j).
    dims : sequence of ints, optional
        d_j for each dimension. If None, inferred from K_tilde.

    Returns
    -------
    K : jnp.ndarray, shape (∏ d_j, ∏ d_j)
    """
    p = len(penalties)

    dims = [penalties[j].shape[-1] for j in range(p)]

    # Build K_1 / tau1^2 as initial value
    factors = [penalties[0] if i == 0 else jnp.eye(dims[i]) for i in range(p)]

    def kron_all(mats):
        """Kronecker product of a list of matrices."""
        return reduce(jnp.kron, mats)

    K = kron_all(factors)

    # Add remaining K_j / tau_j^2
    for j in range(1, p):
        factors = [penalties[j] if i == j else jnp.eye(dims[i]) for i in range(p)]
        Kj = kron_all(factors)
        K = K + Kj

    return K


def _compute_masks(
    penalties: Sequence[jax.Array],
    penalties_eigvalues: Sequence[jax.Array],
    eps: float = 1e-6,
) -> jax.Array:
    diag = _diag_of_kron_of_diag_with_identities

    B = penalties_eigvalues[0].shape[:-1]
    B_flat = int(jnp.prod(jnp.array(B))) if B else 1
    flat_evs = [ev.reshape(B_flat, ev.shape[-1]) for ev in penalties_eigvalues]

    diags = jax.vmap(diag)(flat_evs)  # (B_flat, N)
    K = _materialize_precision(penalties)  # (B, N, N)
    K = K.reshape((B_flat,) + K.shape[-2:])

    ranks = jax.vmap(jnp.linalg.matrix_rank)(K)  # (B_flat,)
    masks = (diags > eps).sum(-1)  # (B_flat,)

    if not jnp.allclose(masks, ranks):
        raise ValueError(
            f"Number of zero eigenvalues ({masks}) does not "
            f"correspond to penalty rank ({ranks}). Maybe a different value for "
            f"{eps=} can help."
        )
    mask = diags > eps

    return mask.reshape(B + (mask.shape[-1],))


def _apply_Ki_along_axis(B, K, axis, dims, total_size):
    """
    Apply K (Di x Di) along a given axis of B (shape dims),
    returning an array with the same shape as B.
    """
    # Move the axis we want to the front: (Di, rest...)
    B_perm = jnp.moveaxis(B, axis, 0)
    Di = dims[axis]
    rest = total_size // Di

    # Flatten everything except that axis: (Di, rest)
    B_flat = B_perm.reshape(Di, rest)

    # Matrix multiply: K @ B_flat  -> (Di, rest)
    C_flat = K @ B_flat

    # Restore original shape/order
    C_perm = C_flat.reshape(B_perm.shape)
    C = jnp.moveaxis(C_perm, 0, axis)  # back to shape = dims
    return C


def _kron_sum_quadratic(x: jax.Array, Ks: Sequence[jax.Array]) -> jax.Array:
    dims = [K.shape[0] for K in Ks]
    # Basic sanity checks (cheap, can remove if you like)
    for K, d in zip(Ks, dims):
        assert K.shape == (d, d)
    total_size = prod(dims)
    assert x.size == total_size

    # Reshape x into m-dimensional tensor
    B = x.reshape(dims)

    total = jnp.array(0.0, dtype=x.dtype)
    for axis, K in enumerate(Ks):
        C = _apply_Ki_along_axis(B, K, axis, dims, total_size)
        total = total + jnp.vdot(B, C)  # scalar

    return total


class StructuredPenaltyOperator:
    """
    - scales is an array with shape (B,K), where B is the batch shape and K is the
      number of penalties. Each scale parameter corresponds to one penalty.
    - penalties is a sequence of length K, containing arrays with shape (B, Di, Di).
      B is the batch shape.
      (Di, Di) is the block size of the individual penalty and can differ between
      elements of the penalties sequence.
      N = prod([p.shape[-1] for p in penalties]).
    - penalties_eigvalues is a sequence of length K, containing arrays of shape (B, Di).
    - penalties_eigvectors is a sequence of length K, containing arrays of shape
      (B, Di, Di).
    """

    def __init__(
        self,
        scales: jax.Array,
        penalties: Sequence[jax.Array],
        penalties_eigvalues: Sequence[jax.Array],
        masks: jax.Array | None = None,
        validate_args: bool = False,
        tol: float = 1e-6,
    ) -> None:
        self._scales = jnp.asarray(scales)
        self._penalties = tuple([jnp.asarray(p) for p in penalties])
        self._penalties_eigvalues = tuple(
            [jnp.asarray(ev) for ev in penalties_eigvalues]
        )

        if validate_args:
            self._validate_penalties()

        self._sizes = [K.shape[-1] for K in self._penalties]

        self._masks = masks
        self._tol = tol

    @classmethod
    def from_penalties(
        cls,
        scales: jax.Array,
        penalties: Sequence[jax.Array],
        eps: float = 1e-6,
        validate_args: bool = False,
    ) -> Self:
        evs = [jnp.linalg.eigh(K) for K in penalties]
        evals = [ev.eigenvalues for ev in evs]

        masks = _compute_masks(penalties=penalties, penalties_eigvalues=evals, eps=eps)

        return cls(
            scales=scales,
            penalties=penalties,
            penalties_eigvalues=evals,
            masks=masks,
            validate_args=validate_args,
        )

    @cached_property
    def variances(self) -> jax.Array:
        return jnp.square(self._scales)

    def materialize_precision(self) -> jax.Array:
        return self._materialize_precision(self.variances)

    def materialize_penalty(self) -> jax.Array:
        return self._materialize_precision(jnp.ones_like(self.variances))

    def _materialize_precision(self, variances: jax.Array) -> jax.Array:
        """This is inefficient, should be used for testing only."""
        p = len(self._penalties)

        dims = [self._penalties[j].shape[-1] for j in range(p)]

        def kron_all(mats):
            """Kronecker product of a list of matrices."""
            return reduce(jnp.kron, mats)

        def one_batch(variances, *penalties):
            # Build K_1 / tau1^2 as initial value
            factors = [penalties[0] if i == 0 else jnp.eye(dims[i]) for i in range(p)]

            K = kron_all(factors) / variances[0]

            # Add remaining K_j / tau_j^2
            for j in range(1, p):
                factors = [
                    penalties[j] if i == j else jnp.eye(dims[i]) for i in range(p)
                ]
                Kj = kron_all(factors)
                K = K + Kj / variances[j]

            return K

        batch_shape = variances.shape[:-1]
        K = variances.shape[-1]

        # flatten batch dims so we can vmap over a single leading dim
        B_flat = int(jnp.prod(jnp.array(batch_shape))) if batch_shape else 1
        tau2_flat = variances.reshape(B_flat, K)  # (B_flat, K)
        pens_flat = [p.reshape((B_flat,) + p.shape[-2:]) for p in self._penalties]

        big_K_fun = jax.vmap(one_batch, in_axes=(0,) + (0,) * K)
        big_K = big_K_fun(tau2_flat, *pens_flat)

        N = prod(dims)
        big_K = jnp.reshape(big_K, batch_shape + (N, N))
        return big_K

    def _sum_of_scaled_eigenvalues(
        self, variances: jax.Array, eigenvalues: Sequence[jax.Array]
    ) -> jax.Array:
        """
        Expects
        - variances (p,)
        - eigenvalues (p, Di)

        Returns (N,) where N = prod(Di)
        """
        diag = jnp.zeros(prod(self._sizes))

        for j in range(len(self._penalties)):
            left_size = prod(self._sizes[:j])
            right_size = prod(self._sizes[(j + 1) :])
            d = eigenvalues[j] / variances[j]

            # First handle identities to the right: repeat
            d_rep = jnp.repeat(d, right_size)

            # Then handle identities to the left: tile
            d_tile = jnp.tile(d_rep, left_size)

            diag = diag + d_tile

        return diag

    def log_pdet(self) -> jax.Array:
        variances = self.variances  # shape (B..., K)
        batch_shape = variances.shape[:-1]
        K = variances.shape[-1]

        # flatten batch dims so we can vmap over a single leading dim
        B_flat = int(jnp.prod(jnp.array(batch_shape))) if batch_shape else 1
        tau2_flat = variances.reshape(B_flat, K)  # (B_flat, K)

        # eigenvalues per penalty, flattened over batch
        eigvals_flat = [
            ev.reshape(B_flat, ev.shape[-1]) for ev in self._penalties_eigvalues
        ]  # list of K arrays (B_flat, Di)

        def _single_diag(variances, *eigenvalues):
            diag = self._sum_of_scaled_eigenvalues(variances, eigenvalues)
            return diag

        # vmap over flattened batch dimension
        diag_flat = jax.vmap(_single_diag, in_axes=(0,) + (0,) * K)(
            tau2_flat,
            *eigvals_flat,
        )
        diag = jnp.reshape(diag_flat, batch_shape + (diag_flat.shape[-1],))

        if self._masks is None:
            mask = diag > self._tol
        else:
            mask = self._masks

        logdet = jnp.log(jnp.where(mask, diag, 1.0)).sum(-1)

        return logdet

    def quad_form(self, x: jax.Array) -> jax.Array:
        variances = self.variances  # shape (B..., K)
        batch_shape = variances.shape[:-1]
        batch_shape_x = x.shape[:-1]
        if batch_shape_x and (batch_shape_x != batch_shape):
            raise ValueError(
                f"x has batch shape {batch_shape_x}, but batch size is {batch_shape}."
            )

        K = variances.shape[-1]
        N = x.shape[-1]

        # flatten batch dims so we can vmap over a single leading dim
        B_flat = int(jnp.prod(jnp.array(batch_shape))) if batch_shape else 1
        tau2_flat = variances.reshape(B_flat, K)  # (B_flat, K)
        if batch_shape_x:
            x_flat = x.reshape(B_flat, N)  # (B_flat, N)
            in_axis_x = 0
        else:
            x_flat = x
            in_axis_x = None

        pens_flat = [p.reshape((B_flat,) + p.shape[-2:]) for p in self._penalties]

        def kron_sum_quadratic(x, variances, *penalties):
            p = penalties
            v = variances
            scaled_penalties = [p[i] / v[i] for i in range(len(penalties))]
            return _kron_sum_quadratic(x, scaled_penalties)

        quad_form_vec = jax.vmap(
            kron_sum_quadratic, in_axes=(in_axis_x,) + (0,) + (0,) * K
        )
        quad_form_out = quad_form_vec(x_flat, tau2_flat, *pens_flat)

        quad_form_out = jnp.reshape(quad_form_out, batch_shape)
        return quad_form_out

    def _validate_penalties(self) -> None:
        # validate number of penalty matrices
        n_penalties1 = self._scales.shape[-1]
        n_penalties2 = len(self._penalties)
        n_penalties3 = len(self._penalties_eigvalues)

        if not len({n_penalties1, n_penalties2, n_penalties3}) == 1:
            msg1 = "Got inconsistent numbers of parameters. "
            msg2 = f"Number of scale parameters: {n_penalties1} "
            msg3 = f"Number of penalty matrices: {n_penalties2}. "
            msg4 = f"Number of eigenvalue vectors: {n_penalties3}. "
            raise ValueError(msg1 + msg2 + msg3 + msg4)


class MultivariateNormalStructured(tfd.Distribution):
    """
    - loc is an array with shape (B, N), where B is the batch shape and N is the
      event shape.
    - scales is an array with shape (B,K), where B is the batch shape and K is the
      number of penalties. Each scale parameter corresponds to one penalty.
    - penalties is a sequence of length K, containing arrays with shape (B, Di, Di).
      B is the batch shape.
      (Di, Di) is the block size of the individual penalty and can differ between
      elements of the penalties sequence.
      N = prod([p.shape[-1] for p in penalties]).
    - penalties_eigvalues is a sequence of length K, containing arrays of shape (B, Di).
    - penalties_eigvectors is a sequence of length K, containing arrays of shape
      (B, Di, Di).
    """

    def __init__(
        self,
        loc: Array,
        op: StructuredPenaltyOperator,
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "MultivariateNormalStructuredSingular",
        include_normalizing_constant: bool = True,
    ):
        parameters = dict(locals())

        self._loc = jnp.asarray(loc)
        self._op = op
        self._n = self._loc.shape[-1]
        self._include_normalizing_constant = include_normalizing_constant

        if validate_args:
            self._op._validate_penalties()
            self._validate_event_dim()

        super().__init__(
            dtype=self._loc.dtype,
            reparameterization_type=tfd.NOT_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name,
        )

    def _validate_event_dim(self) -> None:
        # validate sample size
        n_loc = self._loc.shape[-1]
        ndim_penalties = [p.shape[-1] for p in self._op._penalties]
        n_penalties = prod(ndim_penalties)

        if not n_loc == n_penalties:
            msg1 = "Got inconsistent event dimensions. "
            msg2 = f"Event dimension implied by loc: {n_loc}. "
            msg3 = f"Event dimension implied by penalties: {n_penalties}"
            raise ValueError(msg1 + msg2 + msg3)

    def _batch_shape(self):
        variances = self._op.variances  # shape (B..., K)
        batch_shape = tuple(variances.shape[:-1])
        return tf.TensorShape(batch_shape)

    def _batch_shape_tensor(self):
        variances = self._op.variances  # shape (B..., K)
        batch_shape = tuple(variances.shape[:-1])
        return jnp.array(batch_shape)

    def _event_shape(self):
        return tf.TensorShape((jnp.shape(self._loc)[-1],))

    def _event_shape_tensor(self):
        return jnp.array((jnp.shape(self._loc)[-1],), dtype=self._loc.dtype)

    def _log_prob(self, x: Array) -> jax.Array:
        x = jnp.asarray(x)
        x_centered = x - self._loc

        log_pdet = self._op.log_pdet()
        quad_form = self._op.quad_form(x_centered)

        # early returns, minimally more efficient
        if not self._include_normalizing_constant:
            return 0.5 * (log_pdet - quad_form)

        const = -(self._n / 2) * jnp.log(2 * jnp.pi)

        return 0.5 * (log_pdet - quad_form) + const

    @classmethod
    def from_penalties(
        cls,
        loc: Array,
        scales: Array,
        penalties: Sequence[Array],
        tol: float = 1e-6,
        precompute_masks: bool = True,
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        include_normalizing_constant: bool = True,
    ) -> Self:
        """
        This is expensive, because it computes eigenvalue decompositions of all
        penalty matrices. Should only be used when performance is irrelevant.
        """
        constructor = cls.get_locscale_constructor(
            penalties=penalties,
            tol=tol,
            precompute_masks=precompute_masks,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            include_normalizing_constant=include_normalizing_constant,
        )

        return constructor(loc, scales)

    @classmethod
    def get_locscale_constructor(
        cls,
        penalties: Sequence[Array],
        tol: float = 1e-6,
        precompute_masks: bool = True,
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        include_normalizing_constant: bool = True,
    ) -> Callable[[Array, Array], "MultivariateNormalStructured"]:
        penalties_ = [jnp.asarray(p) for p in penalties]
        evs = [jnp.linalg.eigh(K) for K in penalties]
        evals = [ev.eigenvalues for ev in evs]

        if precompute_masks:
            masks = _compute_masks(
                penalties=penalties_, penalties_eigvalues=evals, eps=tol
            )
        else:
            masks = None

        def construct_dist(loc: Array, scales: Array) -> "MultivariateNormalStructured":
            loc = jnp.asarray(loc)
            scales = jnp.asarray(scales)
            op = StructuredPenaltyOperator(
                scales=scales,
                penalties=penalties_,
                penalties_eigvalues=evals,
                masks=masks,
                tol=tol,
            )

            dist = cls(
                loc=loc,
                op=op,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                include_normalizing_constant=include_normalizing_constant,
            )

            return dist

        return construct_dist

    @cached_property
    def _sqrt_cov(self) -> Array:
        # TODO this is inefficient, because it does not make use of the sparsity
        # at the moment, at least it makes sampling possible. But we should update it.
        prec = self._op.materialize_precision()
        eigenvalues, evecs = jnp.linalg.eigh(prec)
        sqrt_eval = jnp.sqrt(1 / eigenvalues)

        if self._op._masks is None:
            raise ValueError("self._op_masks is None, but need pre-computed masks.")
        sqrt_eval = sqrt_eval.at[..., ~self._op._masks].set(0.0)

        event_shape = sqrt_eval.shape[-1]
        shape = sqrt_eval.shape + (event_shape,)

        r = tuple(range(event_shape))
        diags = jnp.zeros(shape).at[..., r, r].set(sqrt_eval)
        return evecs @ diags

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

        return centered_samples + loc
