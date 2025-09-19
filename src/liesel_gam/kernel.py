from collections.abc import Sequence

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl


def star_ig_gibbs(
    coef: lsl.Var, scale: lsl.Var, penalty: jax.typing.ArrayLike | None = None
) -> gs.GibbsKernel:
    variance_var = scale.value_node[0]  # type: ignore
    a_value = variance_var.dist_node["concentration"].value  # type: ignore
    b_value = variance_var.dist_node["scale"].value  # type: ignore

    if coef.dist_node is None:
        penalty_value = jnp.asarray(penalty)
    else:
        penalty_value = coef.dist_node["penalty"].value  # type: ignore

    rank_value = jnp.linalg.matrix_rank(penalty_value)

    model = coef.model
    if model is None:
        raise ValueError("The model must be set in the coefficient variable.")

    name = variance_var.name

    def transition(prng_key, model_state):
        pos = model.extract_position([coef.name], model_state)

        coef_value = pos[coef.name]

        a_gibbs = jnp.squeeze(a_value + 0.5 * rank_value)
        b_gibbs = jnp.squeeze(b_value + 0.5 * (coef_value @ penalty_value @ coef_value))

        draw = b_gibbs / jax.random.gamma(prng_key, a_gibbs)

        return {name: draw}

    return gs.GibbsKernel([name], transition)


def init_star_ig_gibbs(
    position_keys: Sequence[str],
    coef: lsl.Var,
    scale: lsl.Var,
    penalty: jax.typing.ArrayLike | None = None,
) -> gs.GibbsKernel:
    if len(position_keys) != 1:
        raise ValueError("The position keys must be a single key.")

    variance_var = scale.value_node[0]  # type: ignore
    name = variance_var.name

    if position_keys[0] != name:
        raise ValueError(f"The position key must be {name}.")

    return star_ig_gibbs(coef, scale, penalty)  # type: ignore
