from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, NamedTuple, Self

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import tensorflow_probability.substrates.jax.distributions as tfd
from formulaic import ModelSpec

from liesel_gam.builder.category_mapping import CategoryMapping

from .constraint import LinearConstraintEVD, penalty_to_unit_design
from .dist import MultivariateNormalSingular
from .kernel import init_star_ig_gibbs

InferenceTypes = Any
Array = jax.Array
ArrayLike = jax.typing.ArrayLike


class VarIGPrior(NamedTuple):
    concentration: float
    scale: float


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


def mvn_diag_prior(scale: lsl.Var) -> lsl.Dist:
    return lsl.Dist(tfd.Normal, loc=0.0, scale=scale)


def mvn_structured_prior(scale: lsl.Var, penalty: lsl.Var | lsl.Value) -> lsl.Dist:
    if isinstance(penalty, lsl.Var) and not penalty.strong:
        raise NotImplementedError(
            "Varying penalties or currently not supported by this function."
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
    scale: lsl.Var | Array | None,
    penalty: lsl.Var | lsl.Value | Array | None,
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

    if not isinstance(scale, lsl.Var | lsl.Value):
        scale = lsl.Var(scale)

    if penalty is None:
        return mvn_diag_prior(scale)

    if not isinstance(penalty, lsl.Var | lsl.Value):
        penalty = lsl.Value(penalty)

    return mvn_structured_prior(scale, penalty)


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
        if value.size != 1:
            raise ValueError(
                f"Expected scalar value for ScaleIG, got size {value.size}."
            )

        concentration_node = _ensure_value(
            concentration, name=_append_name(name, "_concentration")
        )
        scale_node = _ensure_value(scale, name=_append_name(name, "_scale"))

        prior = lsl.Dist(
            tfd.InverseGamma, concentration=concentration_node, scale=scale_node
        )

        variance_name = variance_name or _append_name(name, "_square")

        self._variance_param = lsl.Var.new_param(
            value, prior, inference=inference, name=variance_name
        )
        super().__init__(lsl.Calc(jnp.sqrt, self._variance_param), name=name)

    def setup_gibbs_inference(
        self, coef: lsl.Var, penalty: jax.typing.ArrayLike | None = None
    ) -> ScaleIG:
        self._variance_param.inference = gs.MCMCSpec(
            init_star_ig_gibbs,
            kernel_kwargs={"coef": coef, "scale": self, "penalty": penalty},
        )
        return self


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
            value=jnp.array(1.0),
            concentration=concentration,
            scale=scale_,
        )
    elif isinstance(x, ScaleIG | lsl.Var):
        if isinstance(x, ScaleIG):
            if x._variance_param.strong:
                x._variance_param.value = jnp.asarray(x._variance_param.value)
                x.update()
        elif x.strong:
            x.value = jnp.asarray(x.value)

        scale_var = x
        if validate_scalar:
            size = jnp.asarray(scale_var.value).size
            if not size == 1:
                raise ValueError(f"Expected scalar scale, got size {size}")
    elif x is not None:
        scale_var = lsl.Var.new_value(jnp.asarray(x))
        if validate_scalar:
            size = scale_var.value.size
            if not size == 1:
                raise ValueError(f"Expected scalar scale, got size {size}")
    elif x is None:
        scale_var = x
    else:
        raise TypeError(f"Unexpected type for scale: {type(x)}")

    return scale_var


def _validate_scalar_or_p_scale(scale_value: Array, p):
    try:
        is_scalar = scale_value.size == 1
    except AttributeError:
        raise TypeError(
            f"Expected scale value to be an array, got type {type(scale_value)}"
        )
    is_p = scale_value.size == p
    if not (is_scalar or is_p):
        raise ValueError(
            f"Expected scale to have size 1 or {p}, got size {scale_value.size}"
        )


class Term(UserVar):
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
    is_noncentered
        Whether the term has been reparameterized to the non-centered form.

    """

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
        scale = _init_scale_ig(scale, validate_scalar=validate_scalar_scale)

        coef_name = _append_name(name, "_coef") if coef_name is None else coef_name

        prior = term_prior(scale, penalty)

        self.basis = basis

        if isinstance(penalty, lsl.Var | lsl.Value):
            nparam = jnp.shape(penalty.value)[-1]
        elif penalty is not None:
            nparam = jnp.shape(penalty)[-1]
        else:
            nparam = self.nbases

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

        self.is_noncentered = False

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

    def reparam_noncentered(self) -> Self:
        """
        Turns this term into noncentered form, which means the prior for
        the coefficient will be turned from ``coef ~ N(0, scale^2 * inv(penalty))`` into
        ``latent_coef ~ N(0, inv(penalty)); coef = scale * latent_coef``.
        This can sometimes be helpful when sampling with the No-U-Turn Sampler.
        """
        if self.scale is None:
            raise ValueError(
                f"Noncentering reparameterization of {self} fails, "
                f"because {self.scale=}."
            )
        if self.is_noncentered:
            return self

        assert self.coef.dist_node is not None

        self.coef.dist_node["scale"] = lsl.Value(jnp.array(1.0))

        if self.scale.name and self.coef.name:
            scaled_name = self.scale.name + "*" + self.coef.name
        else:
            scaled_name = _append_name(self.coef.name, "_scaled")

        scaled_coef = lsl.Var.new_calc(
            lambda scale, coef: scale * coef,
            self.scale,
            self.coef,
            name=scaled_name,
        )

        self.value_node["coef"] = scaled_coef
        self.coef.update()
        self.update()
        self.is_noncentered = True

        if hasattr(self.scale, "setup_gibbs_inference"):
            try:
                pen = self.coef.dist_node["penalty"].value
                self.scale.setup_gibbs_inference(scaled_coef, penalty=pen)  # type: ignore
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
        noncentered: bool = False,
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
        noncentered
            If ``True``, the term is reparameterized to the non-centered \
            form via :meth:`.reparam_noncentered` before being returned.
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

        name = f"{fname}({basis.x.name})"
        coef_name = coef_name or "$\\beta_{" + f"{name}" + "}$"

        term = cls(
            basis=basis,
            penalty=basis.penalty if scale is not None else None,
            scale=scale,
            inference=inference,
            coef_name=coef_name,
            name=name,
            validate_scalar_scale=not noncentered,
        )

        if noncentered:
            term.reparam_noncentered()

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
        noncentered: bool = False,
    ) -> Term:
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
        noncentered
            If ``True``, reparameterize the term to non-centered form \
            (see :meth:`.reparam_noncentered`).

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

        if noncentered:
            term.reparam_noncentered()

        return term

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
        self.basis.constrain(constraint)
        self.coef.value = jnp.zeros(self.nbases)
        return self


SmoothTerm = Term


class MRFTerm(Term):
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
        return self._ordered_labels

    @ordered_labels.setter
    def ordered_labels(self, value: list[str]) -> None:
        self._ordered_labels = value


class IndexingTerm(Term):
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
        self.value_node.function = lambda basis, coef: jnp.take(coef, basis)  # type: ignore
        if _update_on_init:
            self.coef.update()
            self.update()


class RITerm(IndexingTerm):
    _labels = None
    _mapping = None

    @property
    def labels(self) -> list[str]:
        if self._labels is None:
            raise ValueError("No labels defined.")
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


class BasisDot(UserVar):
    def __init__(
        self,
        basis: Basis,
        prior: lsl.Dist | None = None,
        name: str = "",
        inference: InferenceTypes = None,
        coef_name: str = "",
        _update_on_init: bool = True,
    ):
        self.basis = basis
        self.nbases = self.basis.nbases
        coef_name = _append_name(name, "_coef")

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


def make_callback(function, output_shape, dtype, m: int = 0):
    if len(output_shape):
        k = output_shape[-1]

    def fn(x, **basis_kwargs):
        n = jnp.shape(jnp.atleast_1d(x))[0]
        if len(output_shape) == 2:
            shape = (n - m, k)
        elif len(output_shape) == 1:
            shape = (n - m,)
        elif not len(output_shape):
            shape = ()
        else:
            raise RuntimeError(
                "Return shape of 'basis_fn(value)' must"
                f" have <= 2 dimensions, got {output_shape}"
            )
        result_shape = jax.ShapeDtypeStruct(shape, dtype)
        result = jax.pure_callback(
            function, result_shape, x, vmap_method="sequential", **basis_kwargs
        )
        return result

    return fn


def is_diagonal(M, atol=1e-12):
    # mask for off-diagonal elements
    off_diag_mask = ~jnp.eye(M.shape[-1], dtype=bool)
    off_diag_values = M[off_diag_mask]
    return jnp.all(jnp.abs(off_diag_values) < atol)


class Basis(UserVar):
    """
    General basis for a structured additive term.

    The ``Basis`` class wraps either a provided observation variable or a raw
    array and a basis-generation function. It constructs an internal
    calculation node that produces the basis (design) matrix used by
    smooth terms. The basis function may be executed via a
    callback that does not need to be jax-compatible (the default, potentially slow)
    with a jax-compatible function that is included in just-in-time-compilation
    (when ``use_callback=False``).

    Parameters
    ----------
    value
        If a :class:`liesel.model.Var` or node is provided it is used as \
        the input variable for the basis. Otherwise a raw array-like \
        object may be supplied together with ``xname`` to create an \
        observed variable internally.
    basis_fn
        Function mapping the input variable's values to a basis matrix or \
        vector. It must accept the input array and any ``basis_kwargs`` \
        and return an array of shape ``(n_obs, n_bases)`` (or a scalar/1-d \
        array for simpler bases). By default this is the identity \
        function (``lambda x: x``).
    name
        Optional name for the basis object. If omitted, a sensible name \
        is constructed from the input variable's name (``B(<xname>)``).
    xname
        Required when ``value`` is a raw array: provides a name for the \
        observation variable that will be created.
    use_callback
        If ``True`` (default) the basis_fn is wrapped in a JAX \
        ``pure_callback`` via :func:`make_callback` to allow arbitrary \
        Python basis functions while preserving JAX tracing. If ``False`` \
        the function is used directly and must be jittable via JAX.
    cache_basis
        If ``True`` the computed basis is cached in a persistent \
        calculation node (``lsl.Calc``), which avoids re-computation \
        when not required, but uses memory. If ``False`` a transient \
        calculation node (``lsl.TransientCalc``) is used and the basis \
        will be recomputed with each evaluation of ``Basis.value``, \
        but not stored in memory.
    penalty
        Penalty matrix associated with the basis. If omitted, \
        a default identity penalty is created based on the number \
        of basis functions.
    **basis_kwargs
        Additional keyword arguments forwarded to ``basis_fn``.

    Raises
    ------
    ValueError
        If ``value`` is an array and ``xname`` is not provided, or if
        the created input variable has no name.

    Notes
    -----
    The basis is evaluated once during initialization (via
    ``self.update()``) to determine its shape and dtype. The internal
    callback wrapper inspects the return shape to build a compatible
    JAX ShapeDtypeStruct for the pure callback.

    Attributes
    ----------
    role
        The role assigned to this variable.
    observed
        Whether the basis is derived from an observed variable (always \
        ``True`` for bases created from input data).
    x
        The input variable (observations) used to construct the basis.
    nbases
        Number of basis functions (number of columns in the basis matrix).
    penalty
        Penalty matrix (wrapped as a :class:`liesel.model.Value`) associated \
        with the basis.

    Examples
    --------
    Identity basis from a named variable::

        import liesel.model as lsl
        import jax.numpy as jnp
        xvar = lsl.Var.new_obs(jnp.array([1.,2.,3.]), name='x')
        b = Basis(value=xvar)
    """

    def __init__(
        self,
        value: lsl.Var | lsl.Node | ArrayLike,
        basis_fn: Callable[[Array], Array] | Callable[..., Array] = lambda x: x,
        name: str | None = None,
        xname: str | None = None,
        use_callback: bool = True,
        cache_basis: bool = True,
        penalty: ArrayLike | lsl.Value | None = None,
        **basis_kwargs,
    ) -> None:
        self._validate_xname(value, xname)
        value_var = _ensure_var_or_node(value, xname)

        if use_callback:
            value_ar = jnp.asarray(value_var.value)
            basis_kwargs_arr = {}
            for key, val in basis_kwargs.items():
                if isinstance(val, lsl.Var | lsl.Node):
                    basis_kwargs_arr[key] = val.value
                else:
                    basis_kwargs_arr[key] = val
            basis_ar = basis_fn(value_ar, **basis_kwargs_arr)
            dtype = basis_ar.dtype
            input_shape = jnp.shape(basis_ar)

            # This is special-case handling for compatibility with
            # basis functions that remove cases. For example, if you have a formulaic
            # formula "x + lag(x)", then the resulting basis will have one case less
            # than the original x, because the first case is dropped.
            if value_ar.shape:
                p = value_ar.shape[0] if value_ar.shape else 0
                k = input_shape[0] if input_shape else 0
                m = p - k
            else:
                m = 0

            fn = make_callback(basis_fn, input_shape, dtype, m)
        else:
            fn = basis_fn

        name_ = self._basis_name(value_var, name)

        if cache_basis:
            calc = lsl.Calc(
                fn, value_var, **basis_kwargs, _name=_append_name(name_, "_calc")
            )
        else:
            calc = lsl.TransientCalc(
                fn, value_var, **basis_kwargs, _name=_append_name(name_, "_calc")
            )

        super().__init__(calc, name=name_)
        self.update()
        self.observed = True

        self.x: lsl.Var | lsl.Node = value_var
        basis_shape = jnp.shape(self.value)
        if len(basis_shape) >= 1:
            self.nbases: int = basis_shape[-1]
        else:
            self.nbases = 1  # scalar case

        if isinstance(penalty, lsl.Value):
            penalty_var = penalty
        elif penalty is None:
            penalty_arr = jnp.eye(self.nbases)
            penalty_var = lsl.Value(penalty_arr)
        else:
            penalty_arr = jnp.asarray(penalty)
            penalty_var = lsl.Value(penalty_arr)

        self._penalty = penalty_var

        self._constraint: str | None = None
        self._reparam_matrix: Array | None = None

    @property
    def constraint(self) -> str | None:
        return self._constraint

    @property
    def reparam_matrix(self) -> Array | None:
        return self._reparam_matrix

    def _validate_xname(self, value: lsl.Var | lsl.Node | ArrayLike, xname: str | None):
        if isinstance(value, lsl.Var | lsl.Node) and xname is not None:
            raise ValueError(
                "When supplying a variable or node to `value`, `xname` must not be "
                "used. Name the variable instead."
            )

    def _basis_name(self, value: lsl.Var | lsl.Node | ArrayLike, name: str | None):
        if name is not None and name != "":
            return name

        if isinstance(value, lsl.Var | lsl.Node) and value.name == "":
            return ""

        if hasattr(value, "name"):
            return f"B({value.name})"
        return ""

    @property
    def penalty(self) -> lsl.Value:
        """
        Return the penalty matrix wrapped as a :class:`liesel.model.Value`.

        Returns
        -------
        lsl.Value
            Value wrapper holding the penalty (precision) matrix for this
            basis.
        """
        return self._penalty

    def update_penalty(self, value: ArrayLike | lsl.Value):
        """
        Update the penalty matrix for this basis.

        Parameters
        ----------
        value
            New penalty matrix or an already-wrapped :class:`liesel.model.Value`.
        """
        if isinstance(value, lsl.Value):
            self._penalty.value = value.value
        else:
            penalty_arr = jnp.asarray(value)
            self._penalty.value = penalty_arr

    @classmethod
    def new_linear(
        cls,
        value: lsl.Var | lsl.Node | Array,
        name: str | None = None,
        xname: str | None = None,
        add_intercept: bool = False,
    ):
        """
        Create a linear basis (design matrix) from input values.

        Parameters
        ----------
        value
            Input variable or raw array used to construct the design matrix.
        name
            Optional name for the basis.
        xname
            Name for the observation variable when ``value`` is \
            a raw array.
        add_intercept
            If ``True``, adds an intercept column of ones as the first \
            column of the design matrix.

        Returns
        -------
        A :class:`.Basis` instance that produces a (n_obs, n_features)
        design matrix.
        """

        def as_matrix(x):
            x = jnp.atleast_1d(x)
            if len(jnp.shape(x)) == 1:
                x = jnp.expand_dims(x, -1)
            if add_intercept:
                ones = jnp.ones(x.shape[0])
                x = jnp.c_[ones, x]
            return x

        basis = cls(
            value=value,
            basis_fn=as_matrix,
            name=name,
            xname=xname,
            use_callback=False,
            cache_basis=False,
        )

        return basis

    def diagonalize_penalty(self, atol: float = 1e-6) -> Self:
        """
        Diagonalize the penalty via an eigenvalue decomposition.

        This method computes a transformation that diagonalizes
        the penalty matrix and updates the internal basis function such that
        subsequent evaluations use the accordingly transformed basis. The penalty is
        updated to the diagonalized version.

        Returns
        -------
        The modified basis instance (self).
        """
        assert isinstance(self.value_node, lsl.Calc)
        basis_fn = self.value_node.function

        K = self.penalty.value
        if is_diagonal(K, atol=atol):
            return self

        Z = penalty_to_unit_design(K)

        def reparam_basis(*args, **kwargs):
            return basis_fn(*args, **kwargs) @ Z

        self.value_node.function = reparam_basis
        self.update()
        penalty = Z.T @ K @ Z
        self.update_penalty(penalty)

        return self

    def scale_penalty(self) -> Self:
        """
        Scale the penalty matrix by its infinite norm.

        The penalty matrix is divided by its infinity norm (max absolute row
        sum) so that its values are numerically well-conditioned for
        downstream use. The updated penalty replaces the previous one.

        Returns
        -------
        The modified basis instance (self).
        """
        K = self.penalty.value
        scale = jnp.linalg.norm(K, ord=jnp.inf)
        penalty = K / scale
        self.update_penalty(penalty)
        return self

    def _apply_constraint(self, Z: Array) -> Self:
        """
        Apply a linear reparameterisation to the basis using matrix Z.

        This internal helper multiplies the basis functions by ``Z`` (i.e.
        right-multiplies the design matrix) and updates the penalty to
        reflect the change of basis: ``K_new = Z.T @ K @ Z``.

        Parameters
        ----------
        Z
            Transformation matrix applied to the basis functions.

        Returns
        -------
        The modified basis instance (self).
        """

        assert isinstance(self.value_node, lsl.Calc)
        basis_fn = self.value_node.function

        K = self.penalty.value

        def reparam_basis(*args, **kwargs):
            return basis_fn(*args, **kwargs) @ Z

        self.value_node.function = reparam_basis
        self.update()
        penalty = Z.T @ K @ Z
        self.update_penalty(penalty)
        return self

    def constrain(
        self,
        constraint: ArrayLike
        | Literal["sumzero_term", "sumzero_coef", "constant_and_linear"],
    ) -> Self:
        """
        Apply a linear constraint to the basis and corresponding penalty.

        Parameters
        ----------
        constraint
            Type of constraint or custom linear constraint matrix to apply.
            If an array is supplied, the constraint will be \
            ``A @ coef == 0``, where ``A`` is the supplied constraint matrix.

        Returns
        -------
        The modified basis instance (self).
        """
        if not self.value.ndim == 2:
            raise ValueError(
                "Constraints can only be applied to matrix-valued bases. "
                f"{self} has shape {self.value.shape}"
            )

        if self.constraint is not None:
            raise ValueError(
                f"A '{self.constraint}' constraint has already been applied."
            )

        if isinstance(constraint, str):
            type_: str = constraint
        else:
            constraint_matrix = jnp.asarray(constraint)
            type_ = "custom"

        match type_:
            case "sumzero_coef":
                Z = LinearConstraintEVD.sumzero_coef(self.nbases)
            case "sumzero_term":
                Z = LinearConstraintEVD.sumzero_term(self.value)
            case "constant_and_linear":
                Z = LinearConstraintEVD.constant_and_linear(self.x.value, self.value)
            case "custom":
                Z = LinearConstraintEVD.general(constraint_matrix)

        self._apply_constraint(Z)
        self._constraint = type_
        self._reparam_matrix = Z

        return self


class MRFSpec(NamedTuple):
    mapping: CategoryMapping
    nb: dict[str, list[str]] | None
    ordered_labels: list[str] | None


class MRFBasis(Basis):
    _mrf_spec: MRFSpec | None = None

    @property
    def mrf_spec(self) -> MRFSpec:
        if self._mrf_spec is None:
            raise ValueError("No MRF spec defined.")
        return self._mrf_spec

    @mrf_spec.setter
    def mrf_spec(self, value: MRFSpec):
        if not isinstance(value, MRFSpec):
            raise TypeError(
                f"Replacement must be of type {MRFSpec}, got {type(value)}."
            )
        self._mrf_spec = value


class LinBasis(Basis):
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
            raise ValueError("No model spec defined.")
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
            raise ValueError("No model spec defined.")
        return self._column_names

    @column_names.setter
    def column_names(self, value: Sequence[str]):
        if not isinstance(value, Sequence):
            raise TypeError(f"Replacement must be a sequence, got {type(value)}.")

        for val in value:
            if not isinstance(val, str):
                raise TypeError(
                    f"The values in the replacement must be of type str, "
                    f"got {type(val)}."
                )
        self._column_names = list(value)


class LinTerm(BasisDot):
    _model_spec: ModelSpec | None = None
    _mappings: dict[str, CategoryMapping] | None = None
    _column_names: list[str] | None = None

    @property
    def model_spec(self) -> ModelSpec | None:
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
            raise ValueError("No model spec defined.")
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
            raise ValueError("No model spec defined.")
        return self._column_names

    @column_names.setter
    def column_names(self, value: Sequence[str]):
        if not isinstance(value, Sequence):
            raise TypeError(f"Replacement must be a sequence, got {type(value)}.")

        for val in value:
            if not isinstance(val, str):
                raise TypeError(
                    f"The values in the replacement must be of type str, "
                    f"got {type(val)}."
                )
        self._column_names = list(value)
