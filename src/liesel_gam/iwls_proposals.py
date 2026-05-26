"""
IWLS proposal specifications for additive predictors.

The helpers in this module construct :class:`liesel.goose.MCMCSpec` objects
that use :class:`liesel.goose.IWLSKernel` with a custom Cholesky factor for
structured additive terms.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Self

import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
from jax import Array, grad
from jax.flatten_util import ravel_pytree
from jax.typing import ArrayLike
from liesel.goose.types import ModelInterface, ModelState, Position

from .predictor import AdditivePredictor
from .term import MRFTerm, RITerm, StrctLinTerm, StrctTerm

logger = logging.getLogger(__name__)

WorkingWeightsFn = Callable[[lsl.Model | ModelInterface, ModelState], ArrayLike]
IWLSProposalTerm = StrctTerm | RITerm | MRFTerm | StrctLinTerm
_CONSTANT_WORKING_WEIGHTS_ATTR = "_liesel_gam_constant_working_weights"


def _mark_constant_working_weights(
    working_weights_fn: WorkingWeightsFn,
) -> WorkingWeightsFn:
    """
    Mark a working-weights function as coming from IWLSWeights.constant.
    """
    setattr(working_weights_fn, _CONSTANT_WORKING_WEIGHTS_ATTR, True)
    return working_weights_fn


def _uses_constant_working_weights(working_weights_fn: WorkingWeightsFn) -> bool:
    """
    Check whether a working-weights function comes from IWLSWeights.constant.
    """
    return bool(getattr(working_weights_fn, _CONSTANT_WORKING_WEIGHTS_ATTR, False))


def _raise_if_scale_factored(term: IWLSProposalTerm) -> None:
    """
    Raise an error if a term uses scale factorization.
    """
    if term.scale_is_factored:
        raise ValueError(
            "IWLS proposal specs do not currently support scale-factored terms, "
            f"got {term}."
        )


class IWLSWeights:
    """
    Working-weight factories for IWLS proposals.

    The static methods return functions with the signature expected by
    :class:`.IWLSProposal`.
    """

    @staticmethod
    def constant(value: ArrayLike = 1.0) -> WorkingWeightsFn:
        """
        Return constant working weights.

        Parameters
        ----------
        value
            Scalar or observation-wise constant weights. The default is ``1.0``.

        Returns
        -------
        callable
            Function that ignores the model state and returns ``value`` as a JAX
            array.
        """

        def working_weights(
            model: lsl.Model | ModelInterface,
            model_state: ModelState,
        ) -> Array:
            return jnp.asarray(value)

        return _mark_constant_working_weights(working_weights)

    @staticmethod
    def score_squared(
        eta_name: str,
        *,
        min_weight: float = 1e-6,
        max_weight: float | None = None,
    ) -> WorkingWeightsFn:
        """
        Return deterministic autodiff weights based on squared scores.

        The weights are computed by differentiating the model log probability
        with respect to the named linear predictor and squaring the resulting
        score. This gives a positive Fisher-like proposal geometry without
        materializing a Hessian.

        Parameters
        ----------
        eta_name
            Name of the model variable containing the linear predictor.
        min_weight
            Lower clipping bound applied to the squared scores.
        max_weight
            Optional upper clipping bound applied to the squared scores.

        Returns
        -------
        callable
            Function that computes scalar or observation-wise weights from a model
            and model state.
        """

        def working_weights(
            model: lsl.Model | ModelInterface,
            model_state: ModelState,
        ) -> Array:
            pos = model.extract_position([eta_name], model_state)
            eta = pos[eta_name]
            flat_eta, unravel_fn = ravel_pytree(eta)

            def flat_log_prob_fn(flat_eta: Array) -> Array:
                eta_position = Position({eta_name: unravel_fn(flat_eta)})
                updated_state = model.update_state(eta_position, model_state)
                return jnp.asarray(model.log_prob(updated_state))

            flat_score = grad(flat_log_prob_fn)(flat_eta)
            score = unravel_fn(flat_score)
            weights = jnp.square(score)

            if max_weight is None:
                return jnp.clip(weights, min=min_weight)

            return jnp.clip(weights, min=min_weight, max=max_weight)

        return working_weights

    @staticmethod
    def gaussian_loc(scale_name: str = "scale") -> WorkingWeightsFn:
        """
        Return working weights for Gaussian location terms.

        Parameters
        ----------
        scale_name
            Name of the model variable containing the Gaussian observation scale.

        Returns
        -------
        callable
            Function that computes scalar or observation-wise weights
            ``1 / scale**2`` from a model and model state.
        """

        def working_weights(
            model: lsl.Model | ModelInterface,
            model_state: ModelState,
        ) -> Array:
            pos = model.extract_position([scale_name], model_state)
            scale = pos[scale_name]
            eps = jnp.sqrt(jnp.finfo(jnp.asarray(scale).dtype).eps)
            return 1 / (jnp.clip(scale, min=eps) ** 2)

        return working_weights

    @staticmethod
    def gaussian_scale() -> WorkingWeightsFn:
        """
        Return working weights for Gaussian scale terms.

        Returns
        -------
        callable
            Function that returns the constant scalar weight ``2.0``.
        """

        def working_weights(
            model: lsl.Model | ModelInterface,
            model_state: ModelState,
        ) -> Array:
            return jnp.array(2.0)

        return working_weights


class GaussianIWLSWeights(IWLSWeights):
    """
    Backward-compatible Gaussian working-weight factories.

    Prefer :class:`.IWLSWeights` for new code.
    """

    loc = staticmethod(IWLSWeights.gaussian_loc)
    scale = staticmethod(IWLSWeights.gaussian_scale)


def iwls_spec(
    term: IWLSProposalTerm,
    working_weights_fn: WorkingWeightsFn | None = None,
    **kwargs,
) -> gs.MCMCSpec:
    """
    Create an IWLS inference specification for a structured additive term.

    The returned :class:`liesel.goose.MCMCSpec` initializes an
    :class:`liesel.goose.IWLSKernel` whose proposal precision is based on the
    supplied working weights and on the term's structured prior penalty. If no
    working weights are supplied, constant unit weights are used.

    Parameters
    ----------
    term
        Structured additive term whose coefficient variable should be sampled.
        Scale-factored terms are currently not supported.
    working_weights_fn
        Function that computes scalar or observation-wise working weights from a
        model and model state. Defaults to :meth:`IWLSWeights.constant`.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`liesel.goose.IWLSKernel`. The step-size defaults depend on the
        working weights: weights created by :meth:`IWLSWeights.constant` inherit
        the defaults of :class:`liesel.goose.IWLSKernel`; other weights use
        ``da_tune_step_size=False`` and ``initial_step_size=1.0``. All of these
        values can be overridden.

    Returns
    -------
    liesel.goose.MCMCSpec
        Inference specification for ``term.coef``.
    """
    _raise_if_scale_factored(term)
    working_weights_fn = working_weights_fn or IWLSWeights.constant()

    def init_iwls_kernel(position_keys, term):
        """
        Initialize the IWLS kernel after the term has been attached to a model.
        """
        _raise_if_scale_factored(term)
        proposal = IWLSProposal.from_term(term, working_weights_fn)

        return proposal.kernel_factory()(position_keys, **kwargs)

    spec = gs.MCMCSpec(
        kernel=init_iwls_kernel,
        kernel_kwargs={"term": term},
    )

    return spec


def apply_iwls_spec(
    predictor: AdditivePredictor,
    working_weights_fn: WorkingWeightsFn | None = None,
    verbose: bool = False,
    **kwargs,
):
    """
    Assign IWLS specifications to structured predictor terms.

    The function updates the ``inference`` attribute of each supported structured
    term's coefficient variable in-place. Unsupported terms are skipped.

    Parameters
    ----------
    predictor
        Additive predictor whose terms should be updated.
    working_weights_fn
        Function that computes scalar or observation-wise working weights from a
        model and model state. Defaults to :meth:`IWLSWeights.constant`.
    verbose
        If ``True``, log whether each term is updated or skipped.
    **kwargs
        Additional keyword arguments forwarded to :func:`iwls_spec`.
    """
    for term in predictor.terms.values():
        if not isinstance(term, StrctTerm | RITerm | MRFTerm | StrctLinTerm):
            if verbose:
                logger.info(f"Skipping '{term.name}', inference left unchanged.")
            continue
        term.coef.inference = iwls_spec(
            term=term,
            working_weights_fn=working_weights_fn,
            **kwargs,
        )
        if verbose:
            logger.info(f"Updating inference of '{term.name}' coefficient.")


def gaussian_iwls_spec_loc(
    term: IWLSProposalTerm,
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
        proposal = GaussianLocIWLSProposal.from_term(term, scale_name=scale_name)

        return proposal.kernel_factory()(position_keys, **kwargs)

    spec = gs.MCMCSpec(
        kernel=init_iwls_kernel,
        kernel_kwargs={"term": term},
    )

    return spec


def gaussian_iwls_spec_scale(
    term: IWLSProposalTerm,
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
        proposal = GaussianScaleIWLSProposal.from_term(term)

        return proposal.kernel_factory()(position_keys, **kwargs)

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
class IWLSProposal:
    """
    Compute IWLS proposal geometry from working weights.

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

    @classmethod
    def from_term(
        cls,
        term: IWLSProposalTerm,
        working_weights_fn: WorkingWeightsFn,
    ) -> Self:
        """
        Construct an IWLS proposal from a structured term.

        Parameters
        ----------
        term
            Structured term whose basis, smoothing scale, penalty, and model
            should define the proposal geometry.
        working_weights_fn
            Function that computes scalar or observation-wise working weights
            from a model and model state.

        Returns
        -------
        IWLSProposal
            Proposal object using geometry extracted from ``term``.
        """
        model = term.model
        if model is None:
            raise ValueError(f"The term {term} must be attached to a model.")

        if term.scale is None:
            raise ValueError(f"The term {term} must have a smoothing scale.")

        if term.basis.penalty is None:
            raise ValueError(f"The term {term} must have a penalty matrix.")

        basis_name = term.basis.name
        if not basis_name:
            raise ValueError(f"The basis of term {term} must be named.")

        smooth_scale_name = term.scale.name
        if not smooth_scale_name:
            raise ValueError(f"The smoothing scale of term {term} must be named.")

        return cls(
            basis_name=basis_name,
            smooth_scale_name=smooth_scale_name,
            penalty=term.basis.penalty.value,
            model=model,
            working_weights_fn=working_weights_fn,
            scale_factored=term.scale_is_factored,
        )

    def kernel_factory(self) -> Callable[..., gs.IWLSKernel]:
        """
        Return an IWLS kernel factory using this proposal object.

        Returns
        -------
        callable
            Function that takes ``position_keys`` and keyword arguments for
            :class:`liesel.goose.IWLSKernel`, and returns an IWLS kernel using
            this proposal object's :meth:`chol_info` method. For constant
            working weights, the returned factory inherits the step-size defaults
            of :class:`liesel.goose.IWLSKernel`; otherwise, it defaults to an
            untuned step size of ``1.0``.
        """

        def init_iwls_kernel(position_keys, **kwargs) -> gs.IWLSKernel:
            kernel_kwargs: dict[str, Any] = {"chol_info_fn": self.chol_info}

            if not _uses_constant_working_weights(self.working_weights_fn):
                kernel_kwargs["da_tune_step_size"] = False
                kernel_kwargs["initial_step_size"] = 1.0

            tune_step_size = kwargs.get("da_tune_step_size", False)
            if tune_step_size and "initial_step_size" not in kwargs:
                kernel_kwargs.pop("initial_step_size", None)

            kernel_kwargs |= kwargs

            return gs.IWLSKernel(position_keys, **kernel_kwargs)

        return init_iwls_kernel

    def __post_init__(self) -> None:
        """
        Validate that the Cholesky helper can handle the term parameterization.
        """
        if self.scale_factored:
            raise ValueError(
                "IWLS proposals do not currently support scale-factored proposal "
                "objects."
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
class GaussianLocIWLSProposal(IWLSProposal):
    """
    Compute IWLS proposals for Gaussian location terms.

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

    @classmethod
    def from_term(  # type: ignore[override]
        cls,
        term: IWLSProposalTerm,
        *,
        scale_name: str,
    ) -> Self:
        """
        Construct a Gaussian location IWLS proposal from a structured term.
        """
        working_weights_fn = IWLSWeights.gaussian_loc(scale_name=scale_name)
        base = IWLSProposal.from_term(term, working_weights_fn)
        return cls(
            basis_name=base.basis_name,
            smooth_name=term.name,
            smooth_scale_name=base.smooth_scale_name,
            scale_name=scale_name,
            penalty=base.penalty,
            model=base.model,
            n=term.value.shape[0],
            scale_factored=base.scale_factored,
        )

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
        working_weights_fn = IWLSWeights.gaussian_loc(scale_name=scale_name)
        super().__init__(
            basis_name=basis_name,
            smooth_scale_name=smooth_scale_name,
            penalty=penalty,
            model=model,
            working_weights_fn=working_weights_fn,
            scale_factored=scale_factored,
        )
        self.smooth_name = smooth_name
        self.scale_name = scale_name
        self.n = n


@dataclass(init=False)
class GaussianScaleIWLSProposal(IWLSProposal):
    """
    Compute IWLS proposals for Gaussian scale terms.

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

    @classmethod
    def from_term(  # type: ignore[override]
        cls,
        term: IWLSProposalTerm,
    ) -> Self:
        """
        Construct a Gaussian scale IWLS proposal from a structured term.
        """
        working_weights_fn = IWLSWeights.gaussian_scale()
        base = IWLSProposal.from_term(term, working_weights_fn)
        return cls(
            basis_name=base.basis_name,
            smooth_name=term.name,
            smooth_scale_name=base.smooth_scale_name,
            penalty=base.penalty,
            model=base.model,
            n=term.value.shape[0],
            scale_factored=base.scale_factored,
        )

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
        working_weights_fn = IWLSWeights.gaussian_scale()
        super().__init__(
            basis_name=basis_name,
            smooth_scale_name=smooth_scale_name,
            penalty=penalty,
            model=model,
            working_weights_fn=working_weights_fn,
            scale_factored=scale_factored,
        )
        self.smooth_name = smooth_name
        self.n = n


# Backward-compatible aliases for the original Cholesky-focused names.
GaussianIWLS = GaussianIWLSWeights
IWLS = IWLSWeights
IWLSCholInfo = IWLSProposal
GaussianLocCholInfo = GaussianLocIWLSProposal
GaussianScaleCholInfo = GaussianScaleIWLSProposal
