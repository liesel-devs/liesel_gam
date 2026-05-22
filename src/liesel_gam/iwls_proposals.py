import logging
from dataclasses import dataclass

import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
from jax import Array
from jax.typing import ArrayLike
from liesel.goose.types import ModelInterface, ModelState

from .predictor import AdditivePredictor
from .term import MRFTerm, RITerm, StrctLinTerm, StrctTerm

logger = logging.getLogger(__name__)


def _raise_if_scale_factored(term: StrctTerm) -> None:
    if term.scale_is_factored:
        raise ValueError(
            "Gaussian IWLS proposal specs do not currently support scale-factored "
            f"terms, got {term}."
        )


def gaussian_iwls_spec_loc(
    term: StrctTerm,
    scale_name: str = "scale",
    **kwargs,
) -> gs.MCMCSpec:
    _raise_if_scale_factored(term)

    def init_iwls_kernel(position_keys, term):
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
    _raise_if_scale_factored(term)

    def init_iwls_kernel(position_keys, term):
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
    for term in predictor.terms.values():
        if not isinstance(term, StrctTerm | RITerm | MRFTerm | StrctLinTerm):
            if verbose:
                logger.info(f"Skipping '{term.name}', inference left unchanged.")
            continue
        term.coef.inference = gaussian_iwls_spec_scale(term=term, **kwargs)
        if verbose:
            logger.info(f"Updating inference of '{term.name}' coefficient.")


@dataclass
class GaussianLocCholInfo:
    basis_name: str
    smooth_name: str
    smooth_scale_name: str
    scale_name: str
    penalty: ArrayLike
    model: lsl.Model | ModelInterface
    n: int
    scale_factored: bool = False

    def __post_init__(self) -> None:
        if self.scale_factored:
            raise ValueError(
                "Gaussian IWLS proposals do not currently support scale-factored "
                "chol-info objects."
            )

    def working_weights(self, model_state: ModelState) -> Array:
        pos = self.model.extract_position([self.scale_name], model_state)
        scale = pos[self.scale_name]
        eps = jnp.sqrt(jnp.finfo(jnp.asarray(scale).dtype).eps)
        return 1 / (jnp.clip(scale, min=eps) ** 2)

    def precision(self, model_state: ModelState) -> Array:
        pos = self.model.extract_position(
            [self.basis_name, self.smooth_scale_name], model_state
        )
        Z = pos[self.basis_name]
        scale = pos[self.smooth_scale_name]

        # Weights: support scalar or vector without materializing a diagonal
        w = jnp.asarray(self.working_weights(model_state), dtype=Z.dtype)
        # if scalar, broadcasts; if vector, row-weights
        ZW = Z * (w[:, None] if w.ndim == 1 else w)

        # Z^T W Z without constructing W
        ZTWZ = Z.T @ ZW

        eps = jnp.sqrt(jnp.finfo(Z.dtype).eps)  # small but not too small
        inv_scale2 = 1.0 / jnp.clip(scale, min=eps) ** 2

        P = ZTWZ + inv_scale2 * self.penalty
        return P + 1e-6 * jnp.mean(jnp.diag(P)) * jnp.eye(P.shape[0], P.shape[1])

    def chol_info(self, model_state: ModelState) -> Array:
        return jnp.linalg.cholesky(self.precision(model_state))


@dataclass
class GaussianScaleCholInfo:
    basis_name: str
    smooth_name: str
    smooth_scale_name: str
    penalty: ArrayLike
    model: lsl.Model | ModelInterface
    n: int
    scale_factored: bool = False

    def __post_init__(self) -> None:
        if self.scale_factored:
            raise ValueError(
                "Gaussian IWLS proposals do not currently support scale-factored "
                "chol-info objects."
            )

    def working_weights(self, model_state: ModelState) -> Array:
        return jnp.array(2.0)

    def precision(self, model_state: ModelState) -> Array:
        pos = self.model.extract_position(
            [self.basis_name, self.smooth_scale_name], model_state
        )
        Z = pos[self.basis_name]
        scale = pos[self.smooth_scale_name]

        # Weights: support scalar or vector without materializing a diagonal
        w = jnp.asarray(self.working_weights(model_state), dtype=Z.dtype)
        ZW = Z * (
            w[:, None] if w.ndim == 1 else w
        )  # if scalar, broadcasts; if vector, row-weights

        # Z^T W Z without constructing W
        ZTWZ = Z.T @ ZW

        eps = jnp.sqrt(jnp.finfo(Z.dtype).eps)  # small but not too small
        inv_scale2 = 1.0 / jnp.clip(scale, min=eps) ** 2

        P = ZTWZ + inv_scale2 * self.penalty
        return P + 1e-6 * jnp.mean(jnp.diag(P)) * jnp.eye(P.shape[0], P.shape[1])

    def chol_info(self, model_state: ModelState) -> Array:
        return jnp.linalg.cholesky(self.precision(model_state))
