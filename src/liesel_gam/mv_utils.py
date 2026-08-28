from __future__ import annotations

import jax
import jax.numpy as jnp
import liesel.model as lsl

ArrayLike = jax.typing.ArrayLike


def as_penalty_value(
    penalty: ArrayLike | lsl.Value,
    *,
    atol: float = 1e-6,
    name: str = "",
) -> lsl.Value:
    """Validate a cross-dimensional penalty and wrap it as a ``Value``."""
    if isinstance(penalty, lsl.Value):
        penalty_value = penalty
        penalty_array = jnp.asarray(penalty.value)
    else:
        penalty_array = jnp.asarray(penalty)
        penalty_value = lsl.Value(penalty_array, _name=name)

    if penalty_array.ndim != 2:
        raise ValueError(
            "A cross-dimensional penalty must be a matrix, "
            f"got shape {penalty_array.shape}."
        )
    if not penalty_array.shape[0] or not penalty_array.shape[1]:
        raise ValueError("A cross-dimensional penalty must not be empty.")
    if penalty_array.shape[0] != penalty_array.shape[1]:
        raise ValueError(
            "A cross-dimensional penalty must be square, "
            f"got shape {penalty_array.shape}."
        )
    if not bool(jnp.all(jnp.isfinite(penalty_array))):
        raise ValueError("A cross-dimensional penalty must contain finite values.")
    if not bool(jnp.allclose(penalty_array, penalty_array.T, atol=atol, rtol=0.0)):
        raise ValueError("A cross-dimensional penalty must be symmetric.")

    eigenvalues = jnp.linalg.eigvalsh(penalty_array)
    if bool(jnp.any(eigenvalues < -atol)):
        raise ValueError("A cross-dimensional penalty must be positive semidefinite.")

    penalty_value.value = penalty_array
    return penalty_value


def is_zero_penalty(penalty: lsl.Value, atol: float = 1e-8) -> bool:
    """Whether a penalty is numerically zero."""
    return bool(jnp.allclose(penalty.value, 0.0, atol=atol, rtol=0.0))


def scale_penalty_value(penalty: lsl.Value) -> lsl.Value:
    """Scale a nonzero penalty to unit infinity norm, in place."""
    if is_zero_penalty(penalty):
        return penalty

    norm = jnp.linalg.norm(penalty.value, ord=jnp.inf)
    penalty.value = penalty.value / norm
    return penalty


def as_reparam_value(
    reparam: ArrayLike | lsl.Value | None,
    *,
    latent_ndim: int,
    name: str = "",
) -> lsl.Value:
    """Validate a cross-dimensional reconstruction matrix."""
    if reparam is None:
        return lsl.Value(jnp.eye(latent_ndim), _name=name)
    if isinstance(reparam, lsl.Value):
        value = reparam
        array = jnp.asarray(reparam.value)
    else:
        array = jnp.asarray(reparam)
        value = lsl.Value(array, _name=name)

    if array.ndim != 2:
        raise ValueError(
            "A cross-dimensional reparameterization must be a matrix, "
            f"got shape {array.shape}."
        )
    if array.shape[1] != latent_ndim:
        raise ValueError(
            "The reparameterization matrix and penalty disagree: "
            f"got {array.shape[1]} columns for latent dimension {latent_ndim}."
        )
    if array.shape[0] < array.shape[1]:
        raise ValueError(
            "A cross-dimensional reparameterization cannot have fewer rows "
            "than columns."
        )
    if not bool(jnp.all(jnp.isfinite(array))):
        raise ValueError("A reparameterization matrix must contain finite values.")
    if int(jnp.linalg.matrix_rank(array)) != array.shape[1]:
        raise ValueError(
            "A cross-dimensional reparameterization must have full column rank."
        )

    value.value = array
    return value


def reconstruct_dimension(latent: ArrayLike, reparam: ArrayLike) -> jax.Array:
    """Map arrays with a latent trailing dimension back to the output dimension."""
    return jnp.einsum("...j,ij->...i", latent, reparam)
