from collections.abc import Sequence

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl


def _penalty_and_rank(
    coef: lsl.Var,
    penalty: jax.typing.ArrayLike | None,
    penalty_rank: jax.typing.ArrayLike | None,
) -> tuple[jax.Array | None, jax.Array | int]:
    """
    Find the penalty and its rank for the Gibbs update.

    Without an explicit penalty, both come from the coefficient's distribution node,
    so the update uses the same rank as the coefficient prior. The rank is computed
    with :func:`jax.numpy.linalg.matrix_rank` only when it is not available.
    """
    if penalty is not None:
        penalty_value = jnp.asarray(penalty)
        if penalty_rank is None:
            return penalty_value, jnp.linalg.matrix_rank(penalty_value)
        return penalty_value, jnp.asarray(penalty_rank)

    try:
        penalty_value = coef.dist_node["penalty"].value  # type: ignore
    except (KeyError, TypeError):
        # assuming identity penalty, but not materializing it here to be
        # memory-efficient
        return None, jnp.asarray(coef.value).shape[-1]

    if penalty_rank is None:
        try:
            penalty_rank = coef.dist_node["penalty_rank"].value  # type: ignore
        except (KeyError, TypeError):
            penalty_rank = jnp.linalg.matrix_rank(penalty_value)
    return penalty_value, jnp.asarray(penalty_rank)


def star_ig_gibbs(
    coef: lsl.Var,
    scale: lsl.Var,
    penalty: jax.typing.ArrayLike | None = None,
    penalty_rank: jax.typing.ArrayLike | None = None,
) -> gs.GibbsKernel:
    """
    The 'penalty' argument is used only in the case that no penalty can be retrieved
    from the coefficient's distribution node. The 'penalty_rank' argument defaults to
    the rank in the coefficient's distribution node, or to the numerical rank of
    'penalty'.
    """
    variance_var = scale.value_node[0]
    a_value = variance_var.dist_node["concentration"].value  # type: ignore
    b_value = variance_var.dist_node["scale"].value  # type: ignore

    penalty_value, rank_value = _penalty_and_rank(coef, penalty, penalty_rank)

    model = coef.model
    if model is None:
        raise ValueError("The model must be set in the coefficient variable.")

    name = variance_var.name

    def transition(prng_key, model_state):
        pos = model.extract_position([coef.name, name], model_state)

        coef_value = pos[coef.name]

        a_gibbs = jnp.squeeze(a_value + 0.5 * rank_value)
        if penalty_value is not None:
            b_gibbs = jnp.squeeze(
                b_value + 0.5 * (coef_value.T @ penalty_value @ coef_value)
            )
        else:
            b_gibbs = jnp.squeeze(b_value + 0.5 * (coef_value.T @ coef_value))

        draw = b_gibbs / jax.random.gamma(prng_key, a_gibbs)

        return {name: draw}

    return gs.GibbsKernel([name], transition)


def init_star_ig_gibbs(
    position_keys: Sequence[str],
    coef: lsl.Var,
    scale: lsl.Var,
    penalty: jax.typing.ArrayLike | None = None,
    penalty_rank: jax.typing.ArrayLike | None = None,
) -> gs.GibbsKernel:
    if len(position_keys) != 1:
        raise ValueError("The position keys must be a single key.")

    variance_var = scale.value_node[0]
    name = variance_var.name

    if position_keys[0] != name:
        raise ValueError(f"The position key must be {name}.")

    return star_ig_gibbs(coef, scale, penalty, penalty_rank)


def star_ig_gibbs_factored(
    scaled_coef: lsl.Var,
    latent_coef: lsl.Var,
    scale: lsl.Var,
    penalty: jax.typing.ArrayLike | None = None,
    penalty_rank: jax.typing.ArrayLike | None = None,
) -> gs.GibbsKernel:
    """
    The 'penalty' argument is used only in the case that no penalty can be retrieved
    from the coefficient's distribution node. The 'penalty_rank' argument defaults to
    the rank in the coefficient's distribution node, or to the numerical rank of
    'penalty'.
    """
    variance_var = scale.value_node[0]
    a_value = variance_var.dist_node["concentration"].value  # type: ignore
    b_value = variance_var.dist_node["scale"].value  # type: ignore

    penalty_value, rank_value = _penalty_and_rank(scaled_coef, penalty, penalty_rank)

    model = scaled_coef.model
    if model is None:
        raise ValueError("The model must be set in the coefficient variable.")

    name = variance_var.name

    def transition(prng_key, model_state):
        pos = model.extract_position([scaled_coef.name, name], model_state)

        coef_value = pos[scaled_coef.name]

        a_gibbs = jnp.squeeze(a_value + 0.5 * rank_value)
        if penalty_value is not None:
            b_gibbs = jnp.squeeze(
                b_value + 0.5 * (coef_value.T @ penalty_value @ coef_value)
            )
        else:
            b_gibbs = jnp.squeeze(b_value + 0.5 * (coef_value.T @ coef_value))

        draw = b_gibbs / jax.random.gamma(prng_key, a_gibbs)

        scaled_coef_value = coef_value.at[:rank_value].set(
            coef_value[:rank_value] / jnp.clip(jnp.sqrt(draw), 1e-10)
        )

        return {name: draw, latent_coef.name: scaled_coef_value}

    return gs.GibbsKernel([name], transition)


def init_star_ig_gibbs_factored(
    position_keys: Sequence[str],
    scaled_coef: lsl.Var,
    latent_coef: lsl.Var,
    scale: lsl.Var,
    penalty: jax.typing.ArrayLike | None = None,
    penalty_rank: jax.typing.ArrayLike | None = None,
) -> gs.GibbsKernel:
    if len(position_keys) != 1:
        raise ValueError("The position keys must be a single key.")

    variance_var = scale.value_node[0]
    name = variance_var.name

    if position_keys[0] != name:
        raise ValueError(f"The position key must be {name}.")

    return star_ig_gibbs_factored(
        scaled_coef, latent_coef, scale, penalty, penalty_rank
    )
