from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from typing import Any, Literal, Self

import jax
import jax.numpy as jnp
import liesel.model as lsl
import tensorflow_probability.substrates.jax.distributions as tfd
from formulaic import ModelSpec

from liesel_gam.category_mapping import CategoryMapping

from .basis import Basis, is_diagonal
from .dist import MultivariateNormalSingular, MultivariateNormalStructured
from .var import ScaleIG, UserVar, VarIGPrior, _append_name

InferenceTypes = Any
Array = jax.Array
ArrayLike = jax.typing.ArrayLike


def mvn_diag_prior(scale: lsl.Var) -> lsl.Dist:
    return lsl.Dist(tfd.Normal, loc=0.0, scale=scale)


def mvn_structured_prior(scale: lsl.Var, penalty: lsl.Var | lsl.Value) -> lsl.Dist:
    if isinstance(penalty, lsl.Var) and not penalty.strong:
        raise NotImplementedError(
            "Varying penalties are currently not supported by this function."
        )
    prior = lsl.Dist(
        MultivariateNormalSingular,
        loc=0.0,
        scale=scale,
        penalty=penalty,
        penalty_rank=jnp.linalg.matrix_rank(penalty.value),
    )
    return prior


def term_prior(
    scale: lsl.Var | None,
    penalty: lsl.Var | lsl.Value | None,
) -> lsl.Dist | None:
    """
    Returns
    - None if scale=None
    - A simple Normal prior with loc=0.0 and scale=scale if penalty=None
    - A potentially rank-deficient structured multivariate normal prior otherwise
    """
    if scale is None:
        if penalty is not None:
            raise ValueError(f"If {scale=}, then penalty must also be None.")
        return None

    if penalty is None:
        return mvn_diag_prior(scale)

    return mvn_structured_prior(scale, penalty)


def _init_scale_ig(
    x: ScaleIG | VarIGPrior | lsl.Var | ArrayLike | None,
    validate_scalar: bool = False,
) -> ScaleIG | lsl.Var | None:
    if isinstance(x, VarIGPrior):
        concentration = jnp.asarray(x.concentration)
        scale_ = jnp.asarray(x.scale)

        if validate_scalar:
            if not concentration.size == 1:
                raise ValueError(
                    "Expected scalar hyperparameter 'concentration', "
                    f"got size {concentration.size}"
                )

            if not scale_.size == 1:
                raise ValueError(
                    f"Expected scalar hyperparameter 'scale', got size {scale_.size}"
                )

        scale_var: ScaleIG | lsl.Var | None = ScaleIG(
            value=jnp.sqrt(jnp.array(x.value)),
            concentration=concentration,
            scale=scale_,
        )
    elif isinstance(x, ScaleIG | lsl.Var):
        if isinstance(x, ScaleIG):
            if x._variance_param.strong:
                x._variance_param.value = jnp.asarray(x._variance_param.value)
                x.update()
        elif x.strong:
            try:
                x.value = jnp.asarray(x.value)
            except Exception as e:
                raise TypeError(
                    f"Unexpected type for scale value: {type(x.value)}"
                ) from e

        scale_var = x
        if validate_scalar:
            size = jnp.asarray(scale_var.value).size
            if not size == 1:
                raise ValueError(f"Expected scalar scale, got size {size}")
    elif x is None:
        scale_var = x
    else:
        try:
            scale_var = lsl.Var.new_value(jnp.asarray(x))
        except Exception as e:
            raise TypeError(f"Unexpected type for scale: {type(x)}") from e
        if validate_scalar:
            size = scale_var.value.size
            if not size == 1:
                raise ValueError(f"Expected scalar scale, got size {size}")

    return scale_var


def _validate_scalar_or_p_scale(scale_value: Array, p):
    is_scalar = scale_value.size == 1
    is_p = scale_value.size == p
    if not (is_scalar or is_p):
        raise ValueError(
            f"Expected scale to have size 1 or {p}, got size {scale_value.size}"
        )


class StrctTerm(UserVar):
    """
    General structured additive term.

    A structured additive term represents a smooth or structured effect in a
    generalized additive model. The term wraps a design/basis matrix together
    with a prior/penalty and a set of coefficients. The object exposes the
    coefficient variable and evaluates the term as the matrix-vector product
    of the basis and the coefficients.
    The term evaluates to ``basis @ coef``.

    Parameters
    ----------
    basis
        A :class:`.Basis` instance that produces the design matrix for the \
        term. The basis must evaluate to a 2-D array with shape ``(n_obs, n_bases)``.
    penalty
        Penalty matrix or a variable/value wrapping the penalty \
        used to construct the multivariate normal prior for the coefficients.
    scale
        Scale parameter for the prior on the coefficients. This \
        is typically either a scalar or a per-coefficient scale variable.
    name
        Human-readable name for the term. Used for labelling variables and \
        building sensible default names for internal nodes.
    inference
        :class:`liesel.goose.MCMCSpec` inference specification forwarded to coefficient\
        creation.
    coef_name
        Name for the coefficient variable. If ``None``, a default name based \
        on ``name`` will be used.
    _update_on_init
        If ``True`` (default) the internal calculation/graph nodes are \
        evaluated during initialization. Set to ``False`` to delay \
        initial evaluation.

    Raises
    ------
    ValueError
        If ``basis.value`` does not have two dimensions.

    Attributes
    ----------
    scale
        The scale variable used by the prior on the coefficients.
    nbases
        Number of basis functions (number of columns in the basis matrix).
    basis
        The basis object provided to the constructor.
    coef
        The coefficient variable created for this term. It holds the prior
        (multivariate normal singular) and is used in the evaluation of the
        term.
    scale_is_factored
        Whether the term has been reparameterized to the non-centered form.

    """

    def __init__(
        self,
        basis: Basis,
        penalty: lsl.Var | lsl.Value | ArrayLike | None,
        scale: ScaleIG | VarIGPrior | lsl.Var | ArrayLike | None,
        name: str = "",
        inference: InferenceTypes = None,
        coef_name: str | None = None,
        _update_on_init: bool = True,
        validate_scalar_scale: bool = True,
    ):
        scale = _init_scale_ig(scale, validate_scalar=validate_scalar_scale)
        coef_name = _append_name(name, "_coef") if coef_name is None else coef_name

        self.basis = basis

        if isinstance(penalty, lsl.Var | lsl.Value):
            nparam = jnp.shape(penalty.value)[-1]
            self._penalty: lsl.Var | lsl.Value | None = penalty
        elif penalty is not None:
            nparam = jnp.shape(penalty)[-1]
            self._penalty = lsl.Value(jnp.asarray(penalty))
        else:
            nparam = self.nbases
            self._penalty = None

        prior = term_prior(scale, self._penalty)

        if scale is not None:
            _validate_scalar_or_p_scale(scale.value, nparam)
        self.coef = lsl.Var.new_param(
            jnp.zeros(nparam), prior, inference=inference, name=coef_name
        )
        calc = lsl.Calc(
            lambda basis, coef: jnp.dot(basis, coef),
            basis=basis,
            coef=self.coef,
            _update_on_init=_update_on_init,
        )
        self._scale = scale

        super().__init__(calc, name=name)
        if _update_on_init:
            self.coef.update()

        self.scale_is_factored = False

        if hasattr(self.scale, "setup_gibbs_inference"):
            try:
                self.scale.setup_gibbs_inference(self.coef)  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Failed to setup Gibbs kernel for {self}") from e

    @property
    def nbases(self) -> int:
        return jnp.shape(self.basis.value)[-1]

    @property
    def scale(self) -> lsl.Var | lsl.Node | None:
        return self._scale

    def _validate_scale_for_factoring(self):
        if self.scale is None:
            raise ValueError(
                f"Scale factorization of {self} fails, because {self.scale=}."
            )
        if self.scale.value.size > 1:
            raise ValueError(
                f"Scale factorization of {self} fails, "
                f"because scale must be scalar, but got {self.scale.value.size=}."
            )

    def _validate_penalty_for_factoring(self, atol: float = 1e-5) -> Array:
        if self._penalty is None:
            return jnp.array(self.coef.value.shape[-1])

        pen_rank = jnp.linalg.matrix_rank(self._penalty.value)

        if pen_rank == self._penalty.value.shape[-1]:
            # full-rank penalty always works
            return pen_rank

        if not is_diagonal(self._penalty.value, atol):
            # rank-deficient penalty must be diagonal
            raise ValueError(
                "With rank deficient penalties, factoring out the scale is "
                "only supported when using diagonalized penalties. "
                "This is "
                "because the scale is only applied to the penalized part, "
                "and we cannot reliably distinguish the penalized and "
                "unpenalized parts without diagonalization."
            )

        unpenalized_parts = self._penalty.value[pen_rank:, pen_rank:]
        zeros = jnp.zeros_like(unpenalized_parts)
        if not jnp.allclose(unpenalized_parts, zeros, atol=atol):
            # rank-deficient part must be the last rows/columns of the penalty
            raise ValueError(
                "With rank deficient penalties, factoring out the scale is "
                "only supported when using diagonalized penalties. "
                "The null space of the penalty must be organized in the "
                "last R rows/columns, i.e. these must be all zero. "
                "R refers to the rank of the penalty, in your "
                f"case: {pen_rank}. "
                "Your penalty seems to be diagonal, but not have these "
                "zero-row/columns."
                "This is important"
                "because the scale is only applied to the penalized part, "
                "and we cannot reliably distinguish the penalized and "
                "unpenalized parts without this structure."
            )

        return pen_rank

    def factor_scale(self, atol: float = 1e-5) -> Self:
        """
        Turns this term into a partially standardized form, which means the prior for
        the coefficient will be turned from ``coef ~ N(0, scale^2 * inv(penalty))`` into
        ``latent_coef ~ N(0, inv(penalty)); coef = scale * latent_coef``.
        """

        self._validate_scale_for_factoring()
        pen_rank = self._validate_penalty_for_factoring(atol)

        if self.scale_is_factored:
            return self

        assert self.coef.dist_node is not None

        self.coef.dist_node["scale"] = lsl.Value(jnp.array(1.0))

        assert self.scale is not None  # checked in validation method above
        if self.scale.name and self.coef.name:
            scaled_name = self.scale.name + "*" + self.coef.name
        else:
            scaled_name = _append_name(self.coef.name, "_scaled")

        def scale_coef(scale, coef):
            coef = coef.at[:pen_rank].set(coef[:pen_rank] * scale)
            return coef

        scaled_coef = lsl.Var.new_calc(
            scale_coef,
            self.scale,
            self.coef,
            name=scaled_name,
        )

        self.value_node["coef"] = scaled_coef
        self.coef.update()
        self.update()
        self.scale_is_factored = True

        if hasattr(self.scale, "setup_gibbs_inference_factored"):
            try:
                pen = self._penalty.value if self._penalty is not None else None
                self.scale.update()
                self.scale.setup_gibbs_inference_factored(
                    scaled_coef, self.coef, penalty=pen
                )  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Failed to setup Gibbs kernel for {self}") from e

        return self

    @classmethod
    def f(
        cls,
        basis: Basis,
        fname: str = "f",
        scale: ScaleIG | lsl.Var | ArrayLike | VarIGPrior | None = None,
        inference: InferenceTypes = None,
        coef_name: str | None = None,
        factor_scale: bool = False,
    ) -> Self:
        """
        Construct a smooth term from a :class:`.Basis`.

        This convenience constructor builds a named ``term`` using the
        provided basis. The penalty matrix is taken from ``basis.penalty`` and
        a coefficient variable with an appropriate multivariate-normal prior
        is created. The returned term evaluates to ``basis @ coef``.

        Parameters
        ----------
        basis
            Basis object that provides the design matrix and penalty for the \
            smooth term. The basis must have an associated input variable with \
            a meaningful name (used to compose the term name).
        fname
            Function-name prefix used when constructing the term name. Default \
            is ``'f'`` which results in names like ``f(x)`` when the basis \
            input is named ``x``.
        scale
            Scale parameter passed to the coefficient prior.
        inference
            Inference specification forwarded to the coefficient variable \
            creation, a :class:`liesel.goose.MCMCSpec`.
        factor_scale
            If ``True``, the term is reparameterized by factoring out the scale \
            form via :meth:`.factor_scale` before being returned.
        coef_name
            Coefficient name. The default coefficient name is a LaTeX-like string \
            ``"$\\beta_{f(x)}$"`` to improve readability in printed summaries.

        Returns
        -------
        A :class:`.Term` instance configured with the given basis and prior settings.
        """
        if not basis.x.name:
            raise ValueError("basis.x must be named.")

        if not basis.name:
            raise ValueError("basis must be named.")

        if not isinstance(fname, str):
            raise TypeError(f"Expected type str, got {type(fname)}.")

        name = f"{fname}({basis.x.name})"
        coef_name = coef_name or "$\\beta_{" + f"{name}" + "}$"

        term = cls(
            basis=basis,
            penalty=basis.penalty if scale is not None else None,
            scale=scale,
            inference=inference,
            coef_name=coef_name,
            name=name,
            validate_scalar_scale=not factor_scale,
        )

        if factor_scale:
            term.factor_scale()

        return term

    @classmethod
    def new_ig(
        cls,
        basis: Basis,
        penalty: lsl.Var | lsl.Value | Array | None,
        name: str,
        ig_concentration: float = 1.0,
        ig_scale: float = 0.005,
        inference: InferenceTypes = None,
        scale_value: float = 100.0,
        scale_name: str | None = None,
        coef_name: str | None = None,
        factor_scale: bool = False,
    ) -> StrctTerm:
        """
        Construct a smooth term with an inverse-gamma prior on the variance.

        This convenience constructor creates a term similar to :meth:`.f` but
        sets up an explicit variance parameter with an Inverse-Gamma prior.
        A scale variable is set up by taking the square-root, and the
        coefficient prior uses the derived ``scale`` together with the basis
        penalty. By default a Gibbs-style initialization is attached to the
        variance inference via an internal kernel; an optional jitter
        distribution can be provided for MCMC initialization.

        Parameters
        ----------
        basis
            Basis object providing the design matrix and penalty.
        name
            Term name.
        penalty
            Penalty matrix or a variable/value wrapping the penalty \
            used to construct the multivariate normal prior for the coefficients.
        ig_concentration
            Concentration (shape) parameter of the Inverse-Gamma prior for the \
            variance.
        ig_scale
            Scale parameter of the Inverse-Gamma prior for the variance.
        inference
            Inference specification forwarded to the coefficient variable \
            creation, a :class:`liesel.goose.MCMCSpec`.
        variance_value
            Initial value for the variance parameter.
        variance_name
            Variance parameter name. The default is a LaTeX-like representation \
            ``"$\\tau^2_{...}$"`` for readability in summaries.
        coef_name
            Coefficient name. The default coefficient name is a LaTeX-like string \
            ``"$\\beta_{f(x)}$"`` to improve readability in printed summaries.
        factor_scale
            If ``True``, reparameterize the term to non-centered form \
            (see :meth:`.factor_scale`).

        Returns
        -------
        A :class:`.Term` instance configured with an inverse-gamma prior on
        the variance and an appropriate inference specification for
        variance updates.

        """
        coef_name = coef_name or "$\\beta_{" + f"{name}" + "}$"
        scale_name = scale_name or "$\\tau$"
        scale = ScaleIG(
            jnp.asarray(scale_value),
            concentration=ig_concentration,
            scale=ig_scale,
            name=scale_name,
        )

        term = cls(
            basis=basis,
            scale=scale,
            penalty=penalty,
            inference=inference,
            name=name,
            coef_name=coef_name,
        )

        if factor_scale:
            term.factor_scale()

        return term

    def _assert_penalty_is_basis_penalty(self):
        if self._penalty is None:
            raise ValueError(
                f"Penalty of {self} is None."
                " This functionality is only available if the term is initialized with "
                "the same penalty object as its basis."
            )
        if self._penalty is not self.basis.penalty:
            raise ValueError(
                f"Different penalty objects found on {self} and its basis {self.basis}."
                " This functionality is only available if the term is initialized with "
                "the same penalty object as its basis."
            )

    def diagonalize_penalty(self, atol: float = 1e-6) -> Self:
        """
        Diagonalize the penalty via an eigenvalue decomposition.

        This method computes a transformation that diagonalizes
        the penalty matrix and updates the internal basis function such that
        subsequent evaluations use the accordingly transformed basis. The penalty is
        updated to the diagonalized version.

        Returns
        -------
        The modified term instance (self).
        """
        self._assert_penalty_is_basis_penalty()
        self.basis.diagonalize_penalty(atol)
        return self

    def scale_penalty(self) -> Self:
        """
        Scale the penalty matrix by its infinite norm.

        The penalty matrix is divided by its infinity norm (max absolute row
        sum) so that its values are numerically well-conditioned for
        downstream use. The updated penalty replaces the previous one.

        Returns
        -------
        The modified term instance (self).
        """
        self._assert_penalty_is_basis_penalty()
        self.basis.scale_penalty()
        return self

    def constrain(
        self,
        constraint: ArrayLike
        | Literal["sumzero_term", "sumzero_coef", "constant_and_linear"],
    ) -> Self:
        """
        Apply a linear constraint to the term's basis and corresponding penalty.

        Parameters
        ----------
        constraint
            Type of constraint or custom linear constraint matrix to apply. \
            If an array is supplied, the constraint will be \
            ``A @ coef == 0``, where ``A`` is the supplied constraint matrix.

        Returns
        -------
        The modified term instance (self).
        """
        self._assert_penalty_is_basis_penalty()
        self.basis.constrain(constraint)
        self.coef.value = jnp.zeros(self.nbases)
        return self


SmoothTerm = StrctTerm


class MRFTerm(StrctTerm):
    _neighbors = None
    _polygons = None
    _ordered_labels = None
    _labels = None
    _mapping = None

    @property
    def neighbors(self) -> dict[str, list[str]] | None:
        return self._neighbors

    @neighbors.setter
    def neighbors(self, value: dict[str, list[str]] | None) -> None:
        self._neighbors = value

    @property
    def polygons(self) -> dict[str, ArrayLike] | None:
        return self._polygons

    @polygons.setter
    def polygons(self, value: dict[str, ArrayLike] | None) -> None:
        self._polygons = value

    @property
    def labels(self) -> list[str] | None:
        """Region labels."""
        return self._labels

    @labels.setter
    def labels(self, value: list[str]) -> None:
        self._labels = value

    @property
    def mapping(self) -> CategoryMapping:
        if self._mapping is None:
            raise ValueError("No mapping defined.")
        return self._mapping

    @mapping.setter
    def mapping(self, value: CategoryMapping) -> None:
        self._mapping = value

    @property
    def ordered_labels(self) -> list[str] | None:
        """Ordered labels, if they are available."""
        return self._ordered_labels

    @ordered_labels.setter
    def ordered_labels(self, value: list[str]) -> None:
        self._ordered_labels = value


class IndexingTerm(StrctTerm):
    def __init__(
        self,
        basis: Basis,
        penalty: lsl.Var | lsl.Value | Array | None,
        scale: ScaleIG | VarIGPrior | lsl.Var | ArrayLike | None,
        name: str = "",
        inference: InferenceTypes = None,
        coef_name: str | None = None,
        _update_on_init: bool = True,
        validate_scalar_scale: bool = True,
    ):
        if not basis.value.ndim == 1:
            raise ValueError(f"IndexingTerm requires 1d basis, got {basis.value.ndim=}")

        if not jnp.issubdtype(jnp.dtype(basis.value), jnp.integer):
            raise TypeError(
                f"IndexingTerm requires integer basis, got {jnp.dtype(basis.value)=}."
            )

        super().__init__(
            basis=basis,
            penalty=penalty,
            scale=scale,
            name=name,
            inference=inference,
            coef_name=coef_name,
            _update_on_init=False,
            validate_scalar_scale=validate_scalar_scale,
        )

        # mypy warns that self.value_node might be a lsl.Node, which does not have the
        # attribute "function".
        # But we can assume safely that self.value_node is a lsl.Calc, which does have
        # one.
        assert isinstance(self.value_node, lsl.Calc)
        self.value_node.function = lambda basis, coef: jnp.take(coef, basis)
        if _update_on_init:
            self.coef.update()
            self.update()

    @property
    def nbases(self) -> int:
        return self.nclusters

    @property
    def nclusters(self) -> int:
        nclusters = jnp.unique(self.basis.value).size
        return int(nclusters)

    def init_full_basis(self) -> Basis:
        full_basis = Basis(
            self.basis.x, basis_fn=jax.nn.one_hot, num_classes=self.nclusters, name=""
        )
        return full_basis


class RITerm(IndexingTerm):
    _labels = None
    _mapping = None

    @property
    def nclusters(self) -> int:
        try:
            nclusters = len(self.mapping.labels_to_integers_map)
        except ValueError:
            nclusters = jnp.unique(self.basis.value).size

        return int(nclusters)

    def init_full_basis(self) -> Basis:
        full_basis = Basis(
            self.basis.x, basis_fn=jax.nn.one_hot, num_classes=self.nclusters, name=""
        )
        return full_basis

    @property
    def labels(self) -> list[str]:
        if self._labels is None:
            raise ValueError("No labels defined.")
        return self._labels

    @labels.setter
    def labels(self, value: list[str]) -> None:
        if not len(value) == self.nclusters:
            raise ValueError(f"Expected {self.nclusters} labels, got {len(value)}.")
        self._labels = value

    @property
    def mapping(self) -> CategoryMapping:
        if self._mapping is None:
            raise ValueError("No mapping defined.")
        return self._mapping

    @mapping.setter
    def mapping(self, value: CategoryMapping) -> None:
        self._mapping = value


class BasisDot(UserVar):
    def __init__(
        self,
        basis: Basis,
        prior: lsl.Dist | None = None,
        name: str = "",
        inference: InferenceTypes = None,
        coef_name: str | None = None,
        _update_on_init: bool = True,
    ):
        self.basis = basis
        self.nbases = self.basis.nbases
        coef_name = _append_name(name, "_coef") if coef_name is None else coef_name

        self.coef = lsl.Var.new_param(
            jnp.zeros(self.basis.nbases), prior, inference=inference, name=coef_name
        )
        calc = lsl.Calc(
            lambda basis, coef: jnp.dot(basis, coef),
            basis=self.basis,
            coef=self.coef,
            _update_on_init=_update_on_init,
        )

        super().__init__(calc, name=name)


class Intercept(UserVar):
    def __init__(
        self,
        name: str,
        value: ArrayLike | float = 0.0,
        distribution: lsl.Dist | None = None,
        inference: InferenceTypes = None,
    ) -> None:
        super().__init__(
            value=jnp.asarray(value),
            distribution=distribution,
            name=name,
            inference=inference,
        )
        self.parameter = True


class LinMixin:
    _model_spec: ModelSpec | None = None
    _mappings: dict[str, CategoryMapping] | None = None
    _column_names: list[str] | None = None

    @property
    def model_spec(self) -> ModelSpec:
        if self._model_spec is None:
            raise ValueError("No model spec defined.")
        return self._model_spec

    @model_spec.setter
    def model_spec(self, value: ModelSpec):
        if not isinstance(value, ModelSpec):
            raise TypeError(
                f"Replacement must be of type {ModelSpec}, got {type(value)}."
            )
        self._model_spec = value

    @property
    def mappings(self) -> dict[str, CategoryMapping]:
        if self._mappings is None:
            raise ValueError("No mappings defined.")
        return self._mappings

    @mappings.setter
    def mappings(self, value: dict[str, CategoryMapping]):
        if not isinstance(value, dict):
            raise TypeError(f"Replacement must be of type dict, got {type(value)}.")

        for val in value.values():
            if not isinstance(val, CategoryMapping):
                raise TypeError(
                    f"The values in the replacement must be of type {CategoryMapping}, "
                    f"got {type(val)}."
                )
        self._mappings = value

    @property
    def column_names(self) -> list[str]:
        if self._column_names is None:
            raise ValueError("No column names defined.")
        return self._column_names

    @column_names.setter
    def column_names(self, value: Sequence[str]):
        if not isinstance(value, Sequence):
            raise TypeError(f"Replacement must be a sequence, got {type(value)}.")

        if isinstance(value, str):
            raise TypeError("Replacement type cannot be string.")

        for val in value:
            if not isinstance(val, str):
                raise TypeError(
                    f"The values in the replacement must be of type str, "
                    f"got {type(val)}."
                )
        self._column_names = list(value)


class LinTerm(BasisDot, LinMixin):
    pass


class StrctLinTerm(StrctTerm, LinMixin):
    pass


class StrctTensorProdTerm(UserVar):
    """
    General anisotropic structured additive tensor product term.

    A bivariate tensor product can have:

    1. One scale parameter (when using ita)
    2. Two scale parameters (when using include_main_effects)
    3. Three scale parameters (when using common_scale and include_main_effects,
        or adding main effects separately)
    4. Four scale parameters (when adding main effects separately)

    Option four is the most flexible one, since it also allows you to use different
    basis dimensions for the main effects and the interaction.
    """

    def __init__(
        self,
        *marginals: StrctTerm | IndexingTerm | RITerm | MRFTerm,
        common_scale: ScaleIG | lsl.Var | ArrayLike | VarIGPrior | None = None,
        name: str = "",
        inference: InferenceTypes = None,
        coef_name: str | None = None,
        basis_name: str | None = None,
        include_main_effects: bool = False,
        _update_on_init: bool = True,
    ):
        self._validate_marginals(marginals)
        coef_name = _append_name(name, "_coef") if coef_name is None else coef_name
        bases = self._get_bases(marginals)
        penalties = self._get_penalties(bases)

        if common_scale is None:
            scales = [t.scale for t in marginals]
        else:
            scales = [_init_scale_ig(common_scale) for _ in bases]

        _rowwise_kron = jax.vmap(jnp.kron)

        def rowwise_kron(*bases):
            return reduce(_rowwise_kron, bases)

        if basis_name is None:
            basis_name = "B(" + ",".join(list(self._input_obs(bases))) + ")"

        assert basis_name is not None
        basis = lsl.Var.new_calc(rowwise_kron, *bases, name=basis_name)
        nbases = jnp.shape(basis.value)[-1]

        mvnds = MultivariateNormalStructured.get_locscale_constructor(
            penalties=penalties
        )

        scales_var = lsl.Calc(lambda *x: jnp.stack(x, axis=-1), *scales)

        prior = lsl.Dist(distribution=mvnds, loc=jnp.zeros(nbases), scales=scales_var)

        coef = lsl.Var.new_param(
            jnp.zeros(nbases),
            distribution=prior,
            inference=inference,
            name=coef_name,
        )

        self.basis = basis
        self.marginals = marginals
        self.bases = bases
        self.penalties = penalties
        self.scales = scales

        self.nbases = nbases
        self.basis = basis
        self.coef = coef
        self.scale = scales_var
        self.include_main_effects = include_main_effects

        if include_main_effects:
            calc = lsl.Calc(
                lambda *marginals, basis, coef: sum(marginals) + jnp.dot(basis, coef),
                *marginals,
                basis=basis,
                coef=self.coef,
                _update_on_init=_update_on_init,
            )
        else:
            calc = lsl.Calc(
                lambda basis, coef: jnp.dot(basis, coef),
                basis=basis,
                coef=self.coef,
                _update_on_init=_update_on_init,
            )

        super().__init__(calc, name=name)
        if _update_on_init:
            self.coef.update()

    @staticmethod
    def _get_bases(
        marginals: Sequence[StrctTerm | RITerm | MRFTerm | IndexingTerm],
    ) -> list[Basis]:
        bases = []
        for t in marginals:
            if hasattr(t, "init_full_basis"):
                bases.append(t.init_full_basis())
            else:
                bases.append(t.basis)
        return bases

    @staticmethod
    def _get_penalties(bases: Sequence[Basis]) -> list[Array]:
        penalties = []
        for b in bases:
            if b.penalty is None:
                raise TypeError(
                    f"All bases must have a penalty matrix, got 'None' for {b}."
                )
            penalties.append(b.penalty.value)
        return penalties

    @staticmethod
    def _validate_marginals(marginals: Sequence[StrctTerm]):
        for t in marginals:
            if t.scale is None:
                raise ValueError(f"Invalid scale for {t}: {t.scale}")

    @property
    def input_obs(self) -> dict[str, lsl.Var]:
        """
        A dictionary of strong input input variables.
        """
        return self._input_obs(self.bases)

    @staticmethod
    def _input_obs(bases: Sequence[Basis]) -> dict[str, lsl.Var]:
        # this method includes assumptions about how the individual bases are
        # structured: Basis.x can be a strong observed variable directly, or a
        # calculator variable that depends on strong observed variables.
        # If these assumptions are violated, this method may produce unexpected results.
        # The bases created by BasisBuilder fit theses assumptions.
        _input_x = {}
        for b in bases:
            if isinstance(b.x, lsl.Var):
                if b.x.strong and b.x.observed:
                    # case: ordinary univariate marginal basis, like ps
                    if not b.x.name:
                        raise ValueError(f"{b}.x is unnamed.")
                    _input_x[b.x.name] = b.x
                elif b.x.weak:
                    # currently, I don't expect this case to be present
                    # but it would make sense
                    for xi in b.x.all_input_vars():
                        if xi.observed:
                            if not xi.name:
                                raise ValueError(f"Observed name not found for {b}")
                            _input_x[xi.name] = xi

            else:
                # case: potentially multivariate marginal, possibly thin plate,
                # where basis.x is a calculator that collects the strong inputs.
                for xj in b.x.all_input_nodes():
                    if xj.var is not None:
                        if xj.var.observed:
                            if not xj.var.name:
                                raise ValueError(f"Observed name not found for {b}")
                            _input_x[xj.var.name] = xj.var

        return _input_x

    @classmethod
    def f(
        cls,
        *marginals: StrctTerm,
        common_scale: ScaleIG | lsl.Var | ArrayLike | VarIGPrior | None = None,
        fname: str = "ta",
        inference: InferenceTypes = None,
        _update_on_init: bool = True,
    ) -> Self:
        xnames = list(cls._input_obs(cls._get_bases(marginals)))
        name = fname + "(" + ",".join(xnames) + ")"

        coef_name = "$\\beta_{" + name + "}$"

        term = cls(
            *marginals,
            common_scale=common_scale,
            inference=inference,
            coef_name=coef_name,
            name=name,
            basis_name=None,
            _update_on_init=_update_on_init,
        )

        return term
