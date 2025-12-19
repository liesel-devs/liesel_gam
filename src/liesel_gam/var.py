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
    """
    A :class:`liesel.model.Var`, adapted for subclassing.

    What differentiates this from the basic :class:`liesel.model.Var` is just that
    the alternative constructors

    - :meth:`liesel.model.Var.new_obs`
    - :meth:`liesel.model.Var.new_param`
    - :meth:`liesel.model.Var.new_calc`
    - :meth:`liesel.model.Var.new_value`

    are disabled to avoid potential errors when variables are subclassed and intended
    to be initialized directly.
    """

    @classmethod
    def new_calc(cls, *args, **kwargs) -> None:  # type: ignore
        """Disabled method."""
        raise NotImplementedError(
            f"This constructor is not implemented on {cls.__name__}."
        )

    @classmethod
    def new_obs(cls, *args, **kwargs) -> None:  # type: ignore
        """Disabled method."""
        raise NotImplementedError(
            f"This constructor is not implemented on {cls.__name__}."
        )

    @classmethod
    def new_param(cls, *args, **kwargs) -> None:  # type: ignore
        """Disabled method."""
        raise NotImplementedError(
            f"This constructor is not implemented on {cls.__name__}."
        )

    @classmethod
    def new_value(cls, *args, **kwargs) -> None:  # type: ignore
        """Disabled method."""
        raise NotImplementedError(
            f"This constructor is not implemented on {cls.__name__}."
        )


class ScaleIG(UserVar):
    r"""
    A variable with an Inverse Gamma prior on its square.

    The variance parameter (i.e. the squared scale) is flagged as a parameter.

    Parameters
    ----------
    value
        Initial value of the variable.
    concentration
        Concentration parameter of the inverse gamma distribution.\
        Often called ``a``.
    scale
        Scale parameter of the inverse gamma distribution.\
        Often called ``b``.
    name
        Name of the variable.

    Notes
    -----

    This class assumes that this variable represents the scale parameter
    :math:`\tau` in a structured additive term prior as described in
    :class:`.StrctTerm`.

    This class allows for easy setup of Gibbs sampling for :math:`\tau^2` via
    :meth:`.setup_gibbs_inference`. The Gibbs sampler is defined as follows.

    We have

    .. math::

        \tau^2 \sim \operatorname{InverseGamma}(a, b),

    where a is the init argument ``concentration`` and b is the init argument
    ``scale`` for :class:`.ScaleIG`. The value of this variable (ScaleIG) is
    :math:`\tau = \sqrt{\tau^2}`.

    In a structured additive term,
    the coefficient :math:`\boldsymbol{\beta} \in \mathbb{R}^J`
    receives a potentially rank-deficient multivariate normal prior

    .. math::

        p(\boldsymbol{\beta}) \propto \left(\frac{1}{\tau^2}\right)^{
        \operatorname{rk}(\mathbf{K})/2}
        \exp \left(
        - \frac{1}{\tau^2} \boldsymbol{\beta}^\top \mathbf{K} \boldsymbol{\beta}
        \right).

    The full conditional distribution for :math:`\tau^2` is then an inverse Gamma
    distribtion:

    .. math::

        \tau^2 | \cdot \sim \operatorname{InverseGamma}(\tilde{a}, \tilde{b})

    with parameters

    .. math::

        \tilde{a}  & = a + 0.5 \operatorname{rk}(\mathbf{K}) \\
        \tilde{b}  & = b + 0.5 \boldsymbol{\beta}^\top \mathbf{K} \boldsymbol{\beta}.

    The Gibbs sampler for :math:`\tau^2` repeatedly draws from this full conditional.

    References
    -----------

    Section 9.6.3 in 
    
    Fahrmeir, L., Kneib, T., Lang, S., & Marx, B. (2013). Regression—Models, methods 
    and applications. Springer. https://doi.org/10.1007/978-3-642-34333-9

    """

    def __init__(
        self,
        value: float | Array,
        concentration: float | lsl.Var | lsl.Node | ArrayLike,
        scale: float | lsl.Var | lsl.Node | ArrayLike,
        name: str = "",
        variance_name: str = "",
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

        self._variance_param = lsl.Var.new_param(value**2, prior, name=variance_name)
        super().__init__(lsl.Calc(jnp.sqrt, self._variance_param), name=name)

    def setup_gibbs_inference(
        self, coef: lsl.Var, penalty: jax.typing.ArrayLike | None = None
    ) -> ScaleIG:
        r"""
        Sets up a :class:`liesel.goose.GibbsKernel` for this variable, assuming
        that it is used as the variance parameter in a structured additive term.

        See the docs for the class :class:`.ScaleIG` for a description of the
        Gibbs sampler.

        .. note::
            Usually, this method does not have to be called manually, when you are
            working
            with :class:`.StrctTernm` objects or initializing terms using
            :class:`.TermBuilder`.

        Parameters
        ----------
        coef
            Coefficient variable.
        penalty
            Penalty matrix. If ``None``, the penalty is assumed to be the identity
            matrix of a dimension fitting the coefficient dimension.

        See Also
        --------
        .StrctTerm : Structured additive term class.

        """
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
