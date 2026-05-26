"""
IWLS proposal specifications for Gaussian additive predictors.

The helpers in this module construct :class:`liesel.goose.MCMCSpec` objects
that use :class:`liesel.goose.IWLSKernel` with a custom Cholesky factor for
structured additive terms.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
from jax import Array
from jax.typing import ArrayLike
from liesel.goose.types import ModelInterface, ModelState

from .predictor import AdditivePredictor
from .term import MRFTerm, RITerm, StrctLinTerm, StrctTerm

logger = logging.getLogger(__name__)

WorkingWeightsFn = Callable[[lsl.Model | ModelInterface, ModelState], ArrayLike]


def _raise_if_scale_factored(term: StrctTerm) -> None:
    """
    Raise an error if a term uses scale factorization.
    """
    if term.scale_is_factored:
        raise ValueError(
            "Gaussian IWLS proposal specs do not currently support scale-factored "
            f"terms, got {term}."
        )


def gaussian_loc_working_weights(
    model: lsl.Model | ModelInterface,
    model_state: ModelState,
    *,
    scale_name: str = "scale",
) -> Array:
    """
    Compute Gaussian location working weights.

    Parameters
    ----------
    model
        Liesel model or model interface used to extract the observation scale.
    model_state
        Current model state.
    scale_name
        Name of the model variable containing the Gaussian observation scale.

    Returns
    -------
    jax.Array
        Scalar or observation-wise weights ``1 / scale**2``.
    """
    pos = model.extract_position([scale_name], model_state)
    scale = pos[scale_name]
    eps = jnp.sqrt(jnp.finfo(jnp.asarray(scale).dtype).eps)
    return 1 / (jnp.clip(scale, min=eps) ** 2)


def gaussian_scale_working_weights(
    model: lsl.Model | ModelInterface,
    model_state: ModelState,
) -> Array:
    """
    Return the constant Gaussian scale working weight.

    Parameters
    ----------
    model
        Liesel model or model interface. The value is accepted for API symmetry
        with :func:`gaussian_loc_working_weights` and is not used.
    model_state
        Current model state. The value is accepted for API symmetry with
        :func:`gaussian_loc_working_weights` and is not used.

    Returns
    -------
    jax.Array
        Scalar weight with value ``2.0``.
    """
    return jnp.array(2.0)


def gaussian_iwls_spec_loc(
    term: StrctTerm,
    scale_name: str = "scale",
    **kwargs,
) -> gs.MCMCSpec:
    """
    Create an IWLS inference specification for a Gaussian location term.

    The returned :class:`liesel.goose.MCMCSpec` initializes an
    :class:`liesel.goose.IWLSKernel` whose proposal precision is based on the
    Gaussian location working weights ``1 / scale**2`` and on the term's
    structured prior penalty.

    Parameters
    ----------
    term
        Structured additive term whose coefficient variable should be sampled.
        Scale-factored terms are currently not supported.
    scale_name
        Name of the model variable containing the Gaussian observation scale.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`liesel.goose.IWLSKernel`. These values override the defaults
        ``da_tune_step_size=False`` and ``initial_step_size=1.0``.

    Returns
    -------
    liesel.goose.MCMCSpec
        Inference specification for ``term.coef``.
    """
    _raise_if_scale_factored(term)

    def init_iwls_kernel(position_keys, term):
        """
        Initialize the IWLS kernel after the term has been attached to a model.
        """
        _raise_if_scale_factored(term)
        chol_info = GaussianLocCholInfo(
            basis_name=term.basis.name,
            smooth_name=term.name,
            smooth_scale_name=term.scale.name,
            scale_name=scale_name,
            penalty=term.basis.penalty.value,
            model=term.model,
            n=term.value.shape[0],
        )

        kernel_kwargs = {
            "da_tune_step_size": False,
            "initial_step_size": 1.0,
            "chol_info_fn": chol_info.chol_info,
        }

        kernel_kwargs |= kwargs

        return gs.IWLSKernel(position_keys, **kernel_kwargs)

    spec = gs.MCMCSpec(
        kernel=init_iwls_kernel,
        kernel_kwargs={"term": term},
    )

    return spec


def gaussian_iwls_spec_scale(
    term: StrctTerm,
    **kwargs,
) -> gs.MCMCSpec:
    """
    Create an IWLS inference specification for a Gaussian scale term.

    The returned :class:`liesel.goose.MCMCSpec` initializes an
    :class:`liesel.goose.IWLSKernel` whose proposal precision uses the constant
    Gaussian scale working weight ``2`` and the term's structured prior penalty.

    Parameters
    ----------
    term
        Structured additive term whose coefficient variable should be sampled.
        Scale-factored terms are currently not supported.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`liesel.goose.IWLSKernel`. These values override the defaults
        ``da_tune_step_size=False`` and ``initial_step_size=1.0``.

    Returns
    -------
    liesel.goose.MCMCSpec
        Inference specification for ``term.coef``.
    """
    _raise_if_scale_factored(term)

    def init_iwls_kernel(position_keys, term):
        """
        Initialize the IWLS kernel after the term has been attached to a model.
        """
        _raise_if_scale_factored(term)
        chol_info = GaussianScaleCholInfo(
            basis_name=term.basis.name,
            smooth_name=term.name,
            smooth_scale_name=term.scale.name,
            penalty=term.basis.penalty.value,
            model=term.model,
            n=term.value.shape[0],
        )

        kernel_kwargs = {
            "da_tune_step_size": False,
            "initial_step_size": 1.0,
            "chol_info_fn": chol_info.chol_info,
        }

        kernel_kwargs |= kwargs

        return gs.IWLSKernel(position_keys, **kernel_kwargs)

    spec = gs.MCMCSpec(
        kernel=init_iwls_kernel,
        kernel_kwargs={"term": term},
    )

    return spec


def apply_gaussian_iwls_spec_loc(
    predictor: AdditivePredictor,
    scale_name: str = "scale",
    verbose: bool = False,
    **kwargs,
):
    """
    Assign Gaussian-location IWLS specifications to structured predictor terms.

    The function updates the ``inference`` attribute of each supported structured
    term's coefficient variable in-place. Unsupported terms are skipped.

    Parameters
    ----------
    predictor
        Additive predictor whose terms should be updated.
    scale_name
        Name of the model variable containing the Gaussian observation scale.
    verbose
        If ``True``, log whether each term is updated or skipped.
    **kwargs
        Additional keyword arguments forwarded to :func:`gaussian_iwls_spec_loc`.
    """
    for term in predictor.terms.values():
        if not isinstance(term, StrctTerm | RITerm | MRFTerm | StrctLinTerm):
            if verbose:
                logger.info(f"Skipping '{term.name}', inference left unchanged.")
            continue
        term.coef.inference = gaussian_iwls_spec_loc(
            term=term, scale_name=scale_name, **kwargs
        )
        if verbose:
            logger.info(f"Updating inference of '{term.name}' coefficient.")


def apply_gaussian_iwls_spec_scale(
    predictor: AdditivePredictor,
    verbose: bool = False,
    **kwargs,
):
    """
    Assign Gaussian-scale IWLS specifications to structured predictor terms.

    The function updates the ``inference`` attribute of each supported structured
    term's coefficient variable in-place. Unsupported terms are skipped.

    Parameters
    ----------
    predictor
        Additive predictor whose terms should be updated.
    verbose
        If ``True``, log whether each term is updated or skipped.
    **kwargs
        Additional keyword arguments forwarded to :func:`gaussian_iwls_spec_scale`.
    """
    for term in predictor.terms.values():
        if not isinstance(term, StrctTerm | RITerm | MRFTerm | StrctLinTerm):
            if verbose:
                logger.info(f"Skipping '{term.name}', inference left unchanged.")
            continue
        term.coef.inference = gaussian_iwls_spec_scale(term=term, **kwargs)
        if verbose:
            logger.info(f"Updating inference of '{term.name}' coefficient.")


@dataclass
class IWLSCholInfo:
    """
    Compute Cholesky factors for IWLS proposals from working weights.

    Parameters
    ----------
    basis_name
        Name of the basis matrix variable for the structured term.
    smooth_scale_name
        Name of the smoothing scale variable used in the coefficient prior.
    penalty
        Penalty matrix of the structured coefficient prior.
    model
        Liesel model or model interface used to extract variables from model
        states.
    working_weights_fn
        Function that computes scalar or observation-wise working weights from
        a model and model state.
    scale_factored
        Whether the corresponding term uses scale factorization. This is
        currently unsupported and raises an error if set to ``True``.
    """

    basis_name: str
    smooth_scale_name: str
    penalty: ArrayLike
    model: lsl.Model | ModelInterface
    working_weights_fn: WorkingWeightsFn = field(repr=False, compare=False)
    scale_factored: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        """
        Validate that the Cholesky helper can handle the term parameterization.
        """
        if self.scale_factored:
            raise ValueError(
                "Gaussian IWLS proposals do not currently support scale-factored "
                "chol-info objects."
            )

    def working_weights(self, model_state: ModelState) -> Array:
        """
        Compute the working weights for the current model state.

        Parameters
        ----------
        model_state
            Current model state.

        Returns
        -------
        jax.Array
            Scalar or observation-wise working weights.
        """
        return jnp.asarray(self.working_weights_fn(self.model, model_state))

    def precision(self, model_state: ModelState) -> Array:
        """
        Compute the IWLS proposal precision matrix.

        The precision is ``Z.T @ W @ Z + penalty / tau**2`` plus a small
        diagonal jitter, where ``Z`` is the basis matrix, ``W`` contains the
        working weights, and ``tau`` is the smoothing scale.

        Parameters
        ----------
        model_state
            Current model state.

        Returns
        -------
        jax.Array
            Positive-definite proposal precision matrix.
        """
        pos = self.model.extract_position(
            [self.basis_name, self.smooth_scale_name], model_state
        )
        Z = pos[self.basis_name]
        scale = pos[self.smooth_scale_name]

        # Weights: support scalar or vector without materializing a diagonal.
        w = jnp.asarray(self.working_weights(model_state), dtype=Z.dtype)
        ZW = Z * (w[:, None] if w.ndim == 1 else w)

        # Z^T W Z without constructing W.
        ZTWZ = Z.T @ ZW

        eps = jnp.sqrt(jnp.finfo(Z.dtype).eps)
        inv_scale2 = 1.0 / jnp.clip(scale, min=eps) ** 2

        P = ZTWZ + inv_scale2 * self.penalty
        return P + 1e-6 * jnp.mean(jnp.diag(P)) * jnp.eye(P.shape[0], P.shape[1])

    def chol_info(self, model_state: ModelState) -> Array:
        """
        Compute the lower Cholesky factor of the proposal precision matrix.
        """
        return jnp.linalg.cholesky(self.precision(model_state))


@dataclass(init=False)
class GaussianLocCholInfo(IWLSCholInfo):
    """
    Compute Cholesky factors for Gaussian location IWLS proposals.

    Parameters
    ----------
    basis_name
        Name of the basis matrix variable for the structured term.
    smooth_name
        Name of the structured term.
    smooth_scale_name
        Name of the smoothing scale variable used in the coefficient prior.
    scale_name
        Name of the Gaussian observation scale variable.
    penalty
        Penalty matrix of the structured coefficient prior.
    model
        Liesel model or model interface used to extract variables from model
        states.
    n
        Number of observations represented by the term.
    scale_factored
        Whether the corresponding term uses scale factorization. This is
        currently unsupported and raises an error if set to ``True``.
    """

    smooth_name: str
    scale_name: str
    n: int

    def __init__(
        self,
        basis_name: str,
        smooth_name: str,
        smooth_scale_name: str,
        scale_name: str,
        penalty: ArrayLike,
        model: lsl.Model | ModelInterface,
        n: int,
        scale_factored: bool = False,
    ) -> None:
        """
        Initialize the Cholesky helper.
        """
        super().__init__(
            basis_name=basis_name,
            smooth_scale_name=smooth_scale_name,
            penalty=penalty,
            model=model,
            working_weights_fn=partial(
                gaussian_loc_working_weights, scale_name=scale_name
            ),
            scale_factored=scale_factored,
        )
        self.smooth_name = smooth_name
        self.scale_name = scale_name
        self.n = n


@dataclass(init=False)
class GaussianScaleCholInfo(IWLSCholInfo):
    """
    Compute Cholesky factors for Gaussian scale IWLS proposals.

    Parameters
    ----------
    basis_name
        Name of the basis matrix variable for the structured term.
    smooth_name
        Name of the structured term.
    smooth_scale_name
        Name of the smoothing scale variable used in the coefficient prior.
    penalty
        Penalty matrix of the structured coefficient prior.
    model
        Liesel model or model interface used to extract variables from model
        states.
    n
        Number of observations represented by the term.
    scale_factored
        Whether the corresponding term uses scale factorization. This is
        currently unsupported and raises an error if set to ``True``.
    """

    smooth_name: str
    n: int

    def __init__(
        self,
        basis_name: str,
        smooth_name: str,
        smooth_scale_name: str,
        penalty: ArrayLike,
        model: lsl.Model | ModelInterface,
        n: int,
        scale_factored: bool = False,
    ) -> None:
        """
        Initialize the Cholesky helper.
        """
        super().__init__(
            basis_name=basis_name,
            smooth_scale_name=smooth_scale_name,
            penalty=penalty,
            model=model,
            working_weights_fn=gaussian_scale_working_weights,
            scale_factored=scale_factored,
        )
        self.smooth_name = smooth_name
        self.n = n
