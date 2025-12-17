from __future__ import annotations

import copy
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import tensorflow_probability.substrates.jax.distributions as tfd

from .kernel import init_star_ig_gibbs, init_star_ig_gibbs_factored

InferenceTypes = Any
Array = jax.Array
ArrayLike = jax.typing.ArrayLike


class VarIGPrior(NamedTuple):
    concentration: float
    scale: float
    value: float = 1.0


def _append_name(name: str, append: str) -> str:
    if name == "":
        return ""
    else:
        return name + append


def _ensure_var_or_node(
    x: lsl.Var | lsl.Node | ArrayLike,
    name: str | None,
) -> lsl.Var | lsl.Node:
    """
    If x is an array, creates a new observed variable.
    """
    if isinstance(x, lsl.Var | lsl.Node):
        x_var = x
    else:
        name = name if name is not None else ""
        x_var = lsl.Var.new_obs(jnp.asarray(x), name=name)

    if name is not None and x_var.name != name:
        raise ValueError(f"{x_var.name=} and {name=} are incompatible.")

    return x_var


def _ensure_value(
    x: lsl.Var | lsl.Node | ArrayLike,
    name: str | None,
) -> lsl.Var | lsl.Node:
    """
    If x is an array, creates a new value node.
    """
    if isinstance(x, lsl.Var | lsl.Node):
        x_var = x
    else:
        name = name if name is not None else ""
        x_var = lsl.Value(jnp.asarray(x), _name=name)

    if name is not None and x_var.name != name:
        raise ValueError(f"{x_var.name=} and {name=} are incompatible.")

    return x_var


class UserVar(lsl.Var):
    @classmethod
    def new_calc(cls, *args, **kwargs) -> None:  # type: ignore
        raise NotImplementedError(
            f"This constructor is not implemented on {cls.__name__}."
        )

    @classmethod
    def new_obs(cls, *args, **kwargs) -> None:  # type: ignore
        raise NotImplementedError(
            f"This constructor is not implemented on {cls.__name__}."
        )

    @classmethod
    def new_param(cls, *args, **kwargs) -> None:  # type: ignore
        raise NotImplementedError(
            f"This constructor is not implemented on {cls.__name__}."
        )

    @classmethod
    def new_value(cls, *args, **kwargs) -> None:  # type: ignore
        raise NotImplementedError(
            f"This constructor is not implemented on {cls.__name__}."
        )


class ScaleIG(UserVar):
    """
    A variable with an Inverse Gamma prior on its square.

    The variance parameter (i.e. the squared scale) is flagged as a parameter.

    Parameters
    ----------
    value
        Initial value of the variable.
    concentration
        Concentration parameter of the inverse gamma distribution.\
        In some parameterizations, this parameter is called ``a``.
    scale
        Scale parameter of the inverse gamma distribution.\
        In some parameterizations, this parameter is called ``b``.
    name
        Name of the variable.
    inference
        Inference type.
    """

    def __init__(
        self,
        value: float | Array,
        concentration: float | lsl.Var | lsl.Node | ArrayLike,
        scale: float | lsl.Var | lsl.Node | ArrayLike,
        name: str = "",
        variance_name: str = "",
        inference: InferenceTypes = None,
    ):
        value = jnp.asarray(value)

        concentration_node = _ensure_value(
            concentration, name=_append_name(name, "_concentration")
        )
        scale_node = _ensure_value(scale, name=_append_name(name, "_scale"))

        prior = lsl.Dist(
            tfd.InverseGamma, concentration=concentration_node, scale=scale_node
        )

        variance_name = variance_name or _append_name(name, "_square")

        self._variance_param = lsl.Var.new_param(
            value**2, prior, inference=inference, name=variance_name
        )
        super().__init__(lsl.Calc(jnp.sqrt, self._variance_param), name=name)

    def setup_gibbs_inference(
        self, coef: lsl.Var, penalty: jax.typing.ArrayLike | None = None
    ) -> ScaleIG:
        if self.value.size != 1:
            raise ValueError(
                f"Gibbs sampler assumes scalar value, got size {self.value.size}."
            )
        init_gibbs = copy.copy(init_star_ig_gibbs)
        init_gibbs.__name__ = "StarVarianceGibbs"

        self._variance_param.inference = gs.MCMCSpec(
            init_star_ig_gibbs,
            kernel_kwargs={"coef": coef, "scale": self, "penalty": penalty},
        )
        return self

    def setup_gibbs_inference_factored(
        self,
        scaled_coef: lsl.Var,
        latent_coef: lsl.Var,
        penalty: jax.typing.ArrayLike | None = None,
    ) -> ScaleIG:
        if self.value.size != 1:
            raise ValueError(
                f"Gibbs sampler assumes scalar value, got size {self.value.size}."
            )
        init_gibbs = copy.copy(init_star_ig_gibbs_factored)
        init_gibbs.__name__ = "StarVarianceGibbs"

        self._variance_param.inference = gs.MCMCSpec(
            init_star_ig_gibbs_factored,
            kernel_kwargs={
                "scaled_coef": scaled_coef,
                "latent_coef": latent_coef,
                "scale": self,
                "penalty": penalty,
            },
        )
        return self
