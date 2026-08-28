from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, Self, cast

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import numpy as np

from .constraint import LinearConstraintEVD
from .mv_utils import (
    as_penalty_value,
    as_reparam_value,
    is_zero_penalty,
    reconstruct_dimension,
    scale_penalty_value,
)
from .term import mvn_structured_prior
from .var import ScaleIG, UserVar, VarIGPrior, _append_name

Array = jax.Array
ArrayLike = jax.typing.ArrayLike
Integer = int | np.integer[Any]

term_types = lsl.Var


class MultivariateContribution(UserVar):
    """A contribution to a multivariate additive predictor.

    The contribution is represented in latent cross-dimensional coordinates and
    reconstructed on the predictor's output dimensions.

    Parameters
    ----------
    latent
        Variable or node whose trailing axis contains the latent cross-dimensional
        coordinates.
    dimension_reparam
        Matrix mapping latent coordinates to the reconstructed output dimensions.
    dimension_penalty
        Cross-dimensional penalty associated with ``latent``.
    name
        Contribution variable name.

    Examples
    --------
    >>> latent = lsl.Var.new_value(jnp.array([[1.0, -1.0]]))
    >>> contribution = MultivariateContribution(
    ...     latent,
    ...     dimension_reparam=lsl.Value(jnp.eye(2)),
    ...     dimension_penalty=lsl.Value(jnp.eye(2)),
    ...     name="effect",
    ... )
    >>> contribution.value.tolist()
    [[1.0, -1.0]]
    """

    def __init__(
        self,
        latent: lsl.Var | lsl.Node,
        dimension_reparam: lsl.Value,
        dimension_penalty: lsl.Value,
        name: str = "",
    ) -> None:
        latent_shape = jnp.shape(latent.value)
        latent_ndim = dimension_reparam.value.shape[1]
        if not latent_shape or latent_shape[-1] != latent_ndim:
            raise ValueError(
                f"Latent value must have trailing dimension {latent_ndim}, "
                f"got shape {latent_shape}."
            )

        self.latent = latent
        self.dimension_reparam = dimension_reparam
        self.dimension_penalty = dimension_penalty

        calc = lsl.Calc(
            reconstruct_dimension,
            latent=latent,
            reparam=dimension_reparam,
        )
        super().__init__(calc, name=name)
        self.update()

    @property
    def ndim(self) -> int:
        """Number of dimensions in the reconstructed contribution.

        Examples
        --------
        >>> contribution = MultivariateContribution(
        ...     lsl.Var.new_value(jnp.zeros((3, 2))),
        ...     dimension_reparam=lsl.Value(jnp.ones((4, 2))),
        ...     dimension_penalty=lsl.Value(jnp.eye(2)),
        ... )
        >>> contribution.ndim
        4
        """
        return int(self.dimension_reparam.value.shape[0])

    @property
    def latent_ndim(self) -> int:
        """Number of unconstrained cross-dimensional coordinates.

        Examples
        --------
        >>> contribution = MultivariateContribution(
        ...     lsl.Var.new_value(jnp.zeros((3, 2))),
        ...     dimension_reparam=lsl.Value(jnp.ones((4, 2))),
        ...     dimension_penalty=lsl.Value(jnp.eye(2)),
        ... )
        >>> contribution.latent_ndim
        2
        """
        return int(self.dimension_reparam.value.shape[1])


class MultivariateIntercept(MultivariateContribution):
    """Basis-free, cross-dimensionally structured intercept term.

    Parameters
    ----------
    dimension_penalty
        Cross-dimensional penalty for the latent intercept coefficients.
    dimension_reparam
        Matrix mapping latent coefficients to the output dimensions.
    scale
        Scalar scale for the structured coefficient prior. It may be ``None`` only
        when ``dimension_penalty`` is zero.
    name
        Intercept variable name.
    inference
        Inference specification for the coefficient variable.
    coef_name
        Coefficient variable name. By default it is derived from ``name``.

    Examples
    --------
    >>> intercept = MultivariateIntercept(
    ...     dimension_penalty=lsl.Value(jnp.eye(2)),
    ...     dimension_reparam=lsl.Value(jnp.eye(2)),
    ...     scale=lsl.Var.new_value(1.0),
    ...     name="intercept",
    ... )
    >>> intercept.value.shape
    (2,)
    """

    def __init__(
        self,
        dimension_penalty: lsl.Value,
        dimension_reparam: lsl.Value,
        scale: ScaleIG | lsl.Var | None,
        name: str = "",
        inference: Any = None,
        coef_name: str | None = None,
    ) -> None:
        self.scale = scale
        self.coef_name = _append_name(name, "_coef") if coef_name is None else coef_name

        prior = self._make_prior(dimension_penalty)
        coef = lsl.Var.new_param(
            jnp.zeros(dimension_penalty.value.shape[-1]),
            distribution=prior,
            inference=inference,
            name=self.coef_name,
        )
        self.coef = coef

        super().__init__(
            latent=coef,
            dimension_reparam=dimension_reparam,
            dimension_penalty=dimension_penalty,
            name=name,
        )
        self._setup_scale_inference()

    def _make_prior(self, penalty: lsl.Value) -> lsl.Dist | None:
        if is_zero_penalty(penalty):
            return None
        if self.scale is None:
            raise ValueError("A nonzero intercept penalty requires a scale parameter.")
        return mvn_structured_prior(self.scale, penalty)

    def _setup_scale_inference(self) -> None:
        if hasattr(self.scale, "setup_gibbs_inference"):
            self.scale.setup_gibbs_inference(  # type: ignore
                self.coef, penalty=self.dimension_penalty.value
            )

    def _apply_constraint(self, reparam: jnp.ndarray) -> None:
        """Apply a predictor-owned cross-dimensional constraint in place."""
        old_coef = jnp.asarray(self.coef.value)
        old_penalty = jnp.asarray(self.dimension_penalty.value)
        old_reparam = jnp.asarray(self.dimension_reparam.value)

        self.dimension_penalty.value = reparam.T @ old_penalty @ reparam
        self.dimension_reparam.value = old_reparam @ reparam
        self.coef.value = reparam.T @ old_coef
        self.coef.dist_node = self._make_prior(self.dimension_penalty)
        self.coef.update()
        self.update()
        self._setup_scale_inference()

    @property
    def input_obs(self) -> dict[str, lsl.Var]:
        """The empty mapping, because an intercept has no covariate inputs.

        Examples
        --------
        >>> intercept = MultivariateIntercept(
        ...     lsl.Value(jnp.eye(2)),
        ...     lsl.Value(jnp.eye(2)),
        ...     lsl.Var.new_value(1.0),
        ... )
        >>> intercept.input_obs
        {}
        """
        return {}


def _init_intercept_scale(
    scale: ScaleIG | lsl.Var | float | Literal["default"] | VarIGPrior | None,
    *,
    name: str,
) -> ScaleIG | lsl.Var | None:
    if isinstance(scale, str):
        if scale != "default":
            raise ValueError(f"Unknown intercept scale option: {scale!r}.")
        scale = VarIGPrior(1.0, 0.005)

    if isinstance(scale, VarIGPrior):
        return ScaleIG(
            value=scale.value,
            concentration=scale.concentration,
            scale=scale.scale,
            name=name,
            variance_name=f"{name}^2" if name else "",
        )
    if isinstance(scale, float):
        return lsl.Var.new_value(jnp.asarray(scale), name=name)
    if isinstance(scale, ScaleIG | lsl.Var):
        if jnp.asarray(scale.value).size != 1:
            raise ValueError(
                "A cross-dimensional intercept scale must be scalar, "
                f"got size {jnp.asarray(scale.value).size}."
            )
        return scale
    if scale is None:
        return None
    raise TypeError(f"Unexpected intercept scale type: {type(scale)}")


class MVAdditivePredictor(UserVar):
    """Additive predictor with shared cross-dimensional structure.

    The predictor infers its output dimension from ``dimension_penalty``. Terms are
    validated on their latent cross-dimensional scale and summed through their
    reconstructed full-dimensional views. By default, the penalty is divided by its
    infinity norm.

    Call :meth:`constrain` before constructing or adding terms. A linked
    :class:`.MVTermBuilder` then reuses the projected penalty and reconstruction
    matrix for every term.

    Parameters
    ----------
    name
        Predictor variable name.
    dimension_penalty
        Square, symmetric, positive-semidefinite cross-dimensional penalty.
    inv_link
        Optional inverse link, applied after reconstruction to the full dimension.
    intercept
        Whether to create a basis-free structured intercept, or a custom variable.
    intercept_scale
        Scale for the automatic structured intercept.
    intercept_inference
        Inference specification for the automatic intercept coefficient.
    intercept_name
        Format string for the automatic intercept name. It must accept the
        ``subscript`` field.
    scale_penalty
        Whether to scale the cross-dimensional penalty to unit infinity norm.

    Examples
    --------
    >>> import liesel_gam as gam
    >>> predictor = gam.MVAdditivePredictor.from_random_walk("delta", ndim=4)
    >>> predictor.constrain("sumzero_coef")
    MVAdditivePredictor(self.name='delta', 0 terms)
    >>> predictor.ndim, predictor.latent_ndim
    (4, 3)
    """

    def __init__(
        self,
        name: str,
        dimension_penalty: ArrayLike | lsl.Value,
        inv_link: Callable[[Array], Array] | None = None,
        intercept: bool | lsl.Var = True,
        intercept_scale: ScaleIG
        | lsl.Var
        | float
        | Literal["default"]
        | VarIGPrior
        | None = "default",
        intercept_inference: Any = gs.MCMCSpec(gs.IWLSKernel.untuned),
        intercept_name: str = "$\\beta{subscript}$",
        scale_penalty: bool = True,
    ) -> None:
        penalty_name = _append_name(name, "_dimension_penalty")
        reparam_name = _append_name(name, "_dimension_reparam")
        self._penalty = as_penalty_value(dimension_penalty, name=penalty_name)
        if scale_penalty:
            scale_penalty_value(self._penalty)

        self._ndim = int(self._penalty.value.shape[-1])
        self._dimension_reparam = as_reparam_value(
            None, latent_ndim=self._ndim, name=reparam_name
        )
        self._constraint: str | None = None
        self._reparam_matrix: Any | None = None
        self._structure_locked = False
        self.terms: dict[str, lsl.Var] = {}
        self.latent_terms: dict[str, lsl.Var | lsl.Node] = {}

        if inv_link is None:

            def inv_link(x):
                return x

        def _sum(*args, intercept):
            return inv_link(sum(args) + 0.0 + intercept)

        name_cleaned = name.replace("$", "")
        automatic_intercept_name = intercept_name.format(
            subscript="_{0," + name_cleaned + "}"
        )

        if intercept and not isinstance(intercept, lsl.Var):
            if is_zero_penalty(self._penalty):
                if not (
                    intercept_scale is None
                    or (
                        isinstance(intercept_scale, str)
                        and intercept_scale == "default"
                    )
                ):
                    raise ValueError(
                        "An intercept scale is not identified when the "
                        "cross-dimensional penalty is zero."
                    )
                scale = None
            else:
                scale = _init_intercept_scale(
                    intercept_scale,
                    name=f"$\\psi_{{{automatic_intercept_name.replace('$', '')}}}$",
                )

            intercept_: lsl.Var = MultivariateIntercept(
                dimension_penalty=self._penalty,
                dimension_reparam=self._dimension_reparam,
                scale=scale,
                name=automatic_intercept_name,
                inference=intercept_inference,
            )
            self._intercept_kind = "automatic"
        elif isinstance(intercept, lsl.Var):
            intercept_ = intercept
            latent = self._get_latent(intercept)
            self._validate_latent_dimension(latent, what="intercept")
            self._validate_full_dimension(intercept, what="intercept")
            self._intercept_kind = "custom"
        else:
            intercept_ = lsl.Var.new_value(
                jnp.zeros(self.ndim),
                name=_append_name(name, "_zero_intercept"),
            )
            self._intercept_kind = "none"

        self._intercept = intercept_
        super().__init__(
            lsl.Calc(
                _sum,
                intercept=intercept_,
            ),
            name=name,
        )
        self.update()

    @staticmethod
    def _get_latent(term: lsl.Var) -> lsl.Var | lsl.Node:
        latent = getattr(term, "latent", term)
        if not isinstance(latent, lsl.Var | lsl.Node):
            raise TypeError(f"Invalid latent representation on {term}.")
        return latent

    def _validate_latent_dimension(
        self, latent: lsl.Var | lsl.Node, *, what: str
    ) -> None:
        shape = jnp.shape(latent.value)
        if not shape or shape[-1] != self.latent_ndim:
            raise ValueError(
                f"{what.capitalize()} must have trailing dimension "
                f"{self.latent_ndim}, got shape {shape}."
            )

    def _validate_full_dimension(self, term: lsl.Var, *, what: str) -> None:
        shape = jnp.shape(term.value)
        if not shape or shape[-1] != self.ndim:
            raise ValueError(
                f"{what.capitalize()} must have trailing full dimension "
                f"{self.ndim}, got shape {shape}."
            )

    def _validate_term_structure(self, term: lsl.Var) -> None:
        penalty = getattr(term, "dimension_penalty", None)
        if penalty is not None:
            penalty_array = penalty.value if isinstance(penalty, lsl.Value) else penalty
            if not jnp.allclose(penalty_array, self.penalty.value):
                raise ValueError(
                    f"{term} was built with a different cross-dimensional penalty."
                )

        reparam = getattr(term, "dimension_reparam", None)
        if reparam is not None:
            reparam_array = reparam.value if isinstance(reparam, lsl.Value) else reparam
            if not jnp.allclose(reparam_array, self.dimension_reparam.value):
                raise ValueError(
                    f"{term} was built with a different cross-dimensional constraint."
                )

    @property
    def penalty(self) -> lsl.Value:
        """The current, potentially projected cross-dimensional penalty.

        Examples
        --------
        >>> predictor = MVAdditivePredictor.from_identity("delta", 2)
        >>> predictor.penalty.value.tolist()
        [[1.0, 0.0], [0.0, 1.0]]
        """
        return self._penalty

    @property
    def dimension_reparam(self) -> lsl.Value:
        """Shared matrix mapping latent contributions to the output dimension.

        Examples
        --------
        >>> predictor = MVAdditivePredictor.from_identity("delta", 2)
        >>> predictor.dimension_reparam.value.tolist()
        [[1.0, 0.0], [0.0, 1.0]]
        """
        return self._dimension_reparam

    @property
    def ndim(self) -> int:
        """Original output dimension of the predictor.

        Examples
        --------
        >>> MVAdditivePredictor.from_identity("delta", 3).ndim
        3
        """
        return self._ndim

    @property
    def latent_ndim(self) -> int:
        """Current number of unconstrained cross-dimensional coordinates.

        Examples
        --------
        >>> predictor = MVAdditivePredictor.from_identity("delta", 3)
        >>> predictor.constrain("sumzero_coef").latent_ndim
        2
        """
        return int(self.penalty.value.shape[-1])

    @property
    def constraint(self) -> str | None:
        """Applied cross-dimensional constraint type, if any.

        Examples
        --------
        >>> predictor = MVAdditivePredictor.from_identity("delta", 3)
        >>> predictor.constrain("sumzero_coef").constraint
        'sumzero_coef'
        """
        return self._constraint

    @property
    def reparam_matrix(self) -> Array | None:
        """Accumulated reconstruction matrix created by :meth:`constrain`.

        Examples
        --------
        >>> predictor = MVAdditivePredictor.from_identity("delta", 3)
        >>> predictor.constrain("sumzero_coef").reparam_matrix.shape
        (3, 2)
        """
        return self._reparam_matrix

    @property
    def intercept(self) -> lsl.Var:
        """This predictor's full-dimensional intercept object.

        Assigning this property replaces the intercept with a compatible variable.

        Examples
        --------
        >>> predictor = MVAdditivePredictor.from_identity("delta", 2, intercept=False)
        >>> replacement = lsl.Var.new_value(jnp.ones(2), name="offset")
        >>> predictor.intercept = replacement
        >>> predictor.intercept.name
        'offset'
        """
        return self._intercept

    @intercept.setter
    def intercept(self, value: lsl.Var) -> None:
        """Replace the predictor intercept.

        Parameters
        ----------
        value
            Compatible full-dimensional intercept variable.

        Examples
        --------
        >>> predictor = MVAdditivePredictor.from_identity("delta", 2, intercept=False)
        >>> predictor.intercept = lsl.Var.new_value(jnp.ones(2), name="offset")
        >>> predictor.intercept.name
        'offset'
        """
        if not isinstance(value, lsl.Var):
            raise TypeError(f"Expected a liesel Var, got {type(value)}.")
        latent = self._get_latent(value)
        self._validate_latent_dimension(latent, what="intercept")
        self._validate_full_dimension(value, what="intercept")
        self._validate_term_structure(value)
        self._intercept = value
        self._intercept_kind = "custom"
        self.value_node["intercept"] = value
        self.update()

    def scale_penalty(self) -> Self:
        """Scale the cross-dimensional penalty to unit infinity norm.

        This operation must be performed before constructing or adding terms.

        Examples
        --------
        >>> predictor = MVAdditivePredictor(
        ...     "delta", 2.0 * jnp.eye(2), scale_penalty=False
        ... )
        >>> predictor.scale_penalty()
        MVAdditivePredictor(self.name='delta', 0 terms)
        >>> float(jnp.linalg.norm(predictor.penalty.value, ord=jnp.inf))
        1.0
        """
        if self.terms or self._structure_locked:
            raise RuntimeError(
                "The cross-dimensional penalty cannot be scaled after terms are "
                "constructed or added."
            )
        scale_penalty_value(self.penalty)
        return self

    def constrain(
        self,
        constraint: ArrayLike | str,
    ) -> Self:
        """Apply a shared linear constraint across predictor dimensions.

        Parameters
        ----------
        constraint
            ``"sumzero_coef"`` or a full-row-rank constraint matrix whose columns
            match the current latent dimension.

        Examples
        --------
        >>> predictor = MVAdditivePredictor.from_identity("delta", 3)
        >>> predictor.constrain("sumzero_coef")
        MVAdditivePredictor(self.name='delta', 0 terms)
        >>> predictor.dimension_reparam.value.shape
        (3, 2)
        """
        if self.terms or self._structure_locked:
            raise RuntimeError(
                "Cross-dimensional constraints must be applied before terms are "
                "constructed or added."
            )
        if self.constraint is not None:
            raise ValueError(
                f"A '{self.constraint}' constraint has already been applied."
            )
        if self._intercept_kind == "custom":
            raise RuntimeError(
                "Apply the predictor constraint before assigning a custom intercept."
            )

        if isinstance(constraint, str):
            if constraint != "sumzero_coef":
                raise ValueError(f"Unknown constraint type: {constraint!r}.")
            if self.latent_ndim < 2:
                raise ValueError(
                    "A sum-to-zero constraint requires at least two dimensions."
                )
            reparam = LinearConstraintEVD.sumzero_coef(self.latent_ndim)
            constraint_type = constraint
        else:
            matrix = jnp.asarray(constraint)
            if matrix.ndim != 2:
                raise ValueError("A custom constraint must be a matrix.")
            if matrix.shape[1] != self.latent_ndim:
                raise ValueError(
                    "Constraint and predictor dimensions disagree: "
                    f"got {matrix.shape[1]} columns for {self.latent_ndim} dimensions."
                )
            if not 0 < matrix.shape[0] < matrix.shape[1]:
                raise ValueError(
                    "A custom constraint must have between one and D-1 rows."
                )
            if int(jnp.linalg.matrix_rank(matrix)) != matrix.shape[0]:
                raise ValueError("A custom constraint must have full row rank.")
            reparam = LinearConstraintEVD.general(matrix)
            constraint_type = "custom"

        if self._intercept_kind == "automatic":
            assert isinstance(self.intercept, MultivariateIntercept)
            self.intercept._apply_constraint(reparam)
        else:
            old_penalty = jnp.asarray(self.penalty.value)
            old_reparam = jnp.asarray(self.dimension_reparam.value)
            self.penalty.value = reparam.T @ old_penalty @ reparam
            self.dimension_reparam.value = old_reparam @ reparam
            self._intercept.value = jnp.zeros(self.ndim)

        self._constraint = constraint_type
        self._reparam_matrix = jnp.asarray(self.dimension_reparam.value)
        self.update()
        return self

    def _lock_structure(self) -> None:
        """Prevent penalty transformations after a linked term is constructed."""
        self._structure_locked = True

    def update(self) -> Self:
        """Update the predictor and return it.

        Examples
        --------
        >>> predictor = MVAdditivePredictor.from_identity("delta", 2)
        >>> predictor.update()
        MVAdditivePredictor(self.name='delta', 0 terms)
        """
        return cast(Self, super().update())

    def __iadd__(self, other: term_types | Sequence[term_types]) -> Self:
        if isinstance(other, term_types):
            self.append(other)
        else:
            self.extend(other)
        return self

    def append(self, term: term_types) -> None:
        """Append one compatible multivariate term.

        Parameters
        ----------
        term
            Named term with matching latent dimension, reconstructed dimension,
            cross-dimensional penalty, and reparameterization.

        Examples
        --------
        >>> predictor = MVAdditivePredictor.from_identity("delta", 2, intercept=False)
        >>> predictor.append(lsl.Var.new_value(jnp.ones((3, 2)), name="linear"))
        >>> list(predictor.terms)
        ['linear']
        """
        if not isinstance(term, term_types):
            raise TypeError(f"{term} is of unsupported type {type(term)}.")
        if term.name in self.terms:
            raise RuntimeError(f"{self} already contains a term of name {term.name}.")

        latent = self._get_latent(term)
        self._validate_latent_dimension(latent, what="term")
        self._validate_full_dimension(term, what="term")
        self._validate_term_structure(term)
        self.value_node.add_inputs(term)
        self.terms[term.name] = term
        self.latent_terms[term.name] = latent
        self.update()

    def extend(self, terms: Sequence[term_types]) -> None:
        """Append several compatible multivariate terms.

        Parameters
        ----------
        terms
            Sequence of terms accepted by :meth:`append`.

        Examples
        --------
        >>> predictor = MVAdditivePredictor.from_identity("delta", 2, intercept=False)
        >>> terms = [
        ...     lsl.Var.new_value(jnp.ones((3, 2)), name="first"),
        ...     lsl.Var.new_value(jnp.ones((3, 2)), name="second"),
        ... ]
        >>> predictor.extend(terms)
        >>> list(predictor.terms)
        ['first', 'second']
        """
        for term in terms:
            self.append(term)

    def __getitem__(self, name: str) -> lsl.Var:
        return self.terms[name]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name=}, {len(self.terms)} terms)"

    @staticmethod
    def _validate_ndim(ndim: Integer) -> int:
        if not isinstance(ndim, (int, np.integer)) or ndim < 1:
            raise ValueError(f"ndim must be a positive integer, got {ndim!r}.")
        return int(ndim)

    @classmethod
    def from_random_walk(
        cls,
        name: str,
        ndim: Integer,
        order: int = 1,
        **kwargs: Any,
    ) -> Self:
        """Construct a predictor with a random-walk cross penalty.

        Parameters
        ----------
        name
            Predictor variable name.
        ndim
            Positive number of output dimensions.
        order
            Difference order, between one and ``ndim - 1``.
        **kwargs
            Additional arguments forwarded to :class:`MVAdditivePredictor`.

        Examples
        --------
        >>> predictor = MVAdditivePredictor.from_random_walk("delta", 4)
        >>> predictor.penalty.value.shape
        (4, 4)
        """
        ndim_int = cls._validate_ndim(ndim)
        if not isinstance(order, int) or not 0 < order < ndim_int:
            raise ValueError(
                f"order must be an integer between 1 and ndim-1, got {order!r}."
            )
        differences = jnp.diff(jnp.eye(ndim_int), n=order, axis=0)
        penalty = differences.T @ differences
        return cls(name=name, dimension_penalty=penalty, **kwargs)

    @classmethod
    def from_identity(cls, name: str, ndim: Integer, **kwargs: Any) -> Self:
        """Construct a predictor with an identity cross penalty.

        Parameters
        ----------
        name
            Predictor variable name.
        ndim
            Positive number of output dimensions.
        **kwargs
            Additional arguments forwarded to :class:`MVAdditivePredictor`.

        Examples
        --------
        >>> predictor = MVAdditivePredictor.from_identity("delta", 2)
        >>> predictor.penalty.value.tolist()
        [[1.0, 0.0], [0.0, 1.0]]
        """
        ndim_int = cls._validate_ndim(ndim)
        return cls(name=name, dimension_penalty=jnp.eye(ndim_int), **kwargs)

    @classmethod
    def from_no_penalty(cls, name: str, ndim: Integer, **kwargs: Any) -> Self:
        """Construct a predictor with no cross-dimensional penalty.

        Parameters
        ----------
        name
            Predictor variable name.
        ndim
            Positive number of output dimensions.
        **kwargs
            Additional arguments forwarded to :class:`MVAdditivePredictor`.

        Examples
        --------
        >>> predictor = MVAdditivePredictor.from_no_penalty("delta", 2)
        >>> predictor.penalty.value.tolist()
        [[0.0, 0.0], [0.0, 0.0]]
        """
        ndim_int = cls._validate_ndim(ndim)
        return cls(
            name=name,
            dimension_penalty=jnp.zeros((ndim_int, ndim_int)),
            **kwargs,
        )
