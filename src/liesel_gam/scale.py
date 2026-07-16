from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import liesel.model as lsl
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

InferenceTypes = Any


def _variance_name(scale_name: str) -> str:
    if not scale_name:
        return ""
    return f"{scale_name}^2"


def _scale_from_variance(
    value: float,
    prior: lsl.Dist,
    bijector: tfb.Bijector,
    name: str,
    inference: InferenceTypes,
) -> lsl.Var:
    scale_value = jnp.asarray(value)
    if not bool(jnp.all(scale_value > 0.0)):
        raise ValueError("The initial scale value must be positive.")

    variance = lsl.Var.new_param(
        jnp.square(scale_value),
        prior,
        name=_variance_name(name),
    )
    bijected_name = None if name else ""
    variance.biject(bijector, inference=inference, name=bijected_name)
    return lsl.Var.new_calc(jnp.sqrt, variance, name=name)


def scale_wb(
    value: float,
    scale: float,
    bijector: tfb.Bijector = tfb.Exp(),
    name: str = "{x}",
    inference: InferenceTypes = None,
) -> lsl.Var:
    r"""
    Create a scale variable with a Weibull prior on its square.

    This function constructs the scale :math:`\tau` as

    .. math::

        \tau^2 &\sim \operatorname{Weibull}(1/2, \lambda), \\
        \tau &= \sqrt{\tau^2},

    where :math:`\lambda` is the ``scale`` argument. The prior is placed on
    :math:`\tau^2`, rather than :math:`\tau`, following Klein and Kneib (2016).

    Parameters
    ----------
    value
        Initial value of the scale :math:`\tau`. The variance is initialized to
        ``value**2``. Must be positive.
    scale
        Scale parameter :math:`\lambda` of the Weibull distribution. For a
        scale-dependent prior, this value should be calibrated to the scale of the
        corresponding model term.
    bijector
        Bijector from the unconstrained parameterization to :math:`\tau^2`.
    name
        Name of the returned scale variable. The variance is named ``f"{name}^2"``.
        The default placeholder is filled by :class:`.TermBuilder` when the scale is
        used for a term.
    inference
        Inference specification for the bijected variance parameter.

    Returns
    -------
    Scale variable :math:`\tau`.

    References
    ----------
    Klein, N., & Kneib, T. (2016). Scale-dependent priors for variance parameters
    in structured additive distributional regression. *Bayesian Analysis*, 11(4),
    1071--1106. https://doi.org/10.1214/15-BA983
    """
    prior = lsl.Dist(
        tfd.Weibull,
        concentration=jnp.asarray(0.5),
        scale=jnp.asarray(scale),
    )
    return _scale_from_variance(value, prior, bijector, name, inference)


def scale_ig(
    value: float,
    concentration: float,
    scale: float,
    bijector: tfb.Bijector = tfb.Exp(),
    name: str = "{x}",
    inference: InferenceTypes = None,
) -> lsl.Var:
    r"""
    Create a scale variable with an inverse gamma prior on its square.

    This function constructs the scale :math:`\tau` as

    .. math::

        \tau^2 &\sim \operatorname{InverseGamma}(a, b), \\
        \tau &= \sqrt{\tau^2},

    where :math:`a` and :math:`b` are the ``concentration`` and ``scale``
    arguments, respectively.

    Parameters
    ----------
    value
        Initial value of the scale :math:`\tau`. The variance is initialized to
        ``value**2``. Must be positive.
    concentration
        Concentration parameter :math:`a` of the inverse gamma distribution.
    scale
        Scale parameter :math:`b` of the inverse gamma distribution.
    bijector
        Bijector from the unconstrained parameterization to :math:`\tau^2`.
    name
        Name of the returned scale variable. The variance is named ``f"{name}^2"``.
        The default placeholder is filled by :class:`.TermBuilder` when the scale is
        used for a term.
    inference
        Inference specification for the bijected variance parameter.

    Returns
    -------
    Scale variable :math:`\tau`.
    """
    prior = lsl.Dist(
        tfd.InverseGamma,
        concentration=jnp.asarray(concentration),
        scale=jnp.asarray(scale),
    )
    return _scale_from_variance(value, prior, bijector, name, inference)
