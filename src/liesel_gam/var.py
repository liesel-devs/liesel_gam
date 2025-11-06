from __future__ import annotations

from collections.abc import Callable
from typing import Any, Self

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.typing import ArrayLike

from .dist import MultivariateNormalSingular
from .kernel import init_star_ig_gibbs
from .roles import Roles

InferenceTypes = Any
Array = Any


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
    scale: lsl.Var | Array | None, penalty: lsl.Var | lsl.Value | Array | None
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
        scale: lsl.Var | Array | None,
        name: str,
        inference: InferenceTypes = None,
        coef_name: str | None = None,
        _update_on_init: bool = True,
    ):
        coef_name = f"{name}_coef" if coef_name is None else coef_name

        if not jnp.asarray(basis.value).ndim == 2:
            raise ValueError(f"basis must have 2 dimensions, got {basis.value.ndim}.")

        nbases = jnp.shape(basis.value)[-1]

        prior = term_prior(scale, penalty)

        self.nbases = nbases
        self.basis = basis
        self.coef = lsl.Var.new_param(
            jnp.zeros(nbases), prior, inference=inference, name=coef_name
        )
        calc = lsl.Calc(
            lambda basis, coef: jnp.dot(basis, coef),
            basis=basis,
            coef=self.coef,
            _update_on_init=_update_on_init,
        )

        super().__init__(calc, name=name)
        if _update_on_init:
            self.coef.update()
        self.coef.role = Roles.coef_smooth
        self.role = Roles.term_smooth

        self.is_noncentered = False

    @property
    def scale(self) -> lsl.Var | lsl.Node | None:
        prior = self.coef.dist_node
        return prior["scale"] if prior is not None else None

    @scale.setter
    def scale(self, value: lsl.Var):
        prior = self.coef.dist_node
        if prior is None:
            raise ValueError(f"{self.coef.dist_node=}, so scale cannot be set.")

        prior["scale"] = value

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

        self.coef.dist_node["scale"] = lsl.Value(1.0)

        def scaled_dot(basis, coef, scale):
            return jnp.dot(basis, scale * coef)

        self.value_node = lsl.Calc(scaled_dot, self.basis, self.coef, self.scale)
        self.coef.update()
        self.update()
        self.is_noncentered = True
        return self

    @classmethod
    def f(
        cls,
        basis: Basis,
        fname: str = "f",
        scale: lsl.Var | Array | None = None,
        inference: InferenceTypes = None,
        coef_name: str | None = None,
        noncentered: bool = False,
    ) -> Term:
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
            Scale parameter passed to the coefficient prior. \
            Defaults to ``1000.0`` for a weakly-informative prior.
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
        name = f"{fname}({basis.x.name})"

        coef_name = coef_name or "$\\beta_{" + f"{name}" + "}$"

        term = cls(
            basis=basis,
            penalty=basis.penalty,
            scale=scale,
            inference=inference,
            coef_name=coef_name,
            name=name,
        )

        if noncentered:
            term.reparam_noncentered()

        return term

    @classmethod
    def f_ig(
        cls,
        basis: Basis,
        fname: str = "f",
        ig_concentration: float = 1.0,
        ig_scale: float = 0.005,
        inference: InferenceTypes = None,
        variance_value: float = 100.0,
        variance_name: str | None = None,
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
        fname
            Prefix used to build the term name (default: ``'f'``).
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
        name = f"{fname}({basis.x.name})"
        coef_name = coef_name or "$\\beta_{" + f"{name}" + "}$"
        variance_name = variance_name or "$\\tau^2_{" + f"{name}" + "}$"

        variance = lsl.Var.new_param(
            value=variance_value,
            distribution=lsl.Dist(
                tfd.InverseGamma,
                concentration=ig_concentration,
                scale=ig_scale,
            ),
            name=variance_name,
        )
        variance.role = Roles.variance_smooth

        scale = lsl.Var.new_calc(jnp.sqrt, variance, name="$\\tau_{" + f"{name}" + "}$")
        scale.role = Roles.scale_smooth

        term = cls(
            basis=basis,
            scale=scale,
            penalty=basis.penalty,
            inference=inference,
            name=name,
            coef_name=coef_name,
        )

        variance.inference = gs.MCMCSpec(
            init_star_ig_gibbs,
            kernel_kwargs={"coef": term.coef, "scale": scale},
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
        variance_value: float = 100.0,
        variance_name: str | None = None,
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
        variance_name = variance_name or "$\\tau^2_{" + f"{name}" + "}$"

        variance = lsl.Var.new_param(
            value=variance_value,
            distribution=lsl.Dist(
                tfd.InverseGamma,
                concentration=ig_concentration,
                scale=ig_scale,
            ),
            name=variance_name,
        )
        variance.role = Roles.variance_smooth

        scale = lsl.Var.new_calc(jnp.sqrt, variance, name="$\\tau_{" + f"{name}" + "}$")
        scale.role = Roles.scale_smooth

        term = cls(
            basis=basis,
            scale=scale,
            penalty=penalty,
            inference=inference,
            name=name,
            coef_name=coef_name,
        )

        variance.inference = gs.MCMCSpec(
            init_star_ig_gibbs,
            kernel_kwargs={"coef": term.coef, "scale": scale},
        )

        if noncentered:
            term.reparam_noncentered()

        return term


SmoothTerm = Term


class LinearTerm(Term):
    """Kept for backwards-compatibility of the interface."""

    def __init__(
        self,
        x: lsl.Var | Array,
        name: str,
        distribution: lsl.Dist | None = None,
        inference: InferenceTypes = None,
        add_intercept: bool = False,
        coef_name: str | None = None,
        basis_name: str | None = None,
    ):
        if not isinstance(x, lsl.Var):
            x = lsl.Var.new_obs(x, name=f"{name}_input")

        if not x.name:
            # to ensure sensible basis name
            raise ValueError(f"{x=} must be named.")

        coef_name = coef_name or f"{name}_coef"
        basis_name = basis_name or f"B({name})"
        basis = Basis.new_linear(value=x, name=basis_name, add_intercept=add_intercept)

        nbases = jnp.shape(basis.value)[-1]
        penalty = jnp.eye(nbases)
        # just a temporary variable to satisfy the api of Term
        scale = lsl.Var(1.0, name=f"_{name}_scale_tmp")

        super().__init__(
            basis=basis,
            penalty=penalty,
            scale=scale,
            name=name,
            inference=inference,
            coef_name=coef_name,
        )
        self.coef.dist_node = distribution
        self.coef.role = Roles.coef_linear
        self.role = Roles.term_linear


class LinearTerm2(UserVar):
    def __init__(
        self,
        value: lsl.Var | Array,
        name: str,
        prior: lsl.Dist | None = None,
        inference: InferenceTypes = None,
        add_intercept: bool = False,
        coef_name: str | None = None,
        basis_name: str | None = None,
        _update_on_init: bool = True,
    ):
        if not isinstance(value, lsl.Var):
            x: lsl.Var = lsl.Var.new_obs(value, name=f"{name}_input")
        else:
            x = value

        if not x.name:
            # to ensure sensible basis name
            raise ValueError(f"{value=} must be named.")

        coef_name = coef_name or f"{name}_coef"
        basis_name = basis_name or f"B({name})"
        self.basis = Basis.new_linear(
            value=x, name=basis_name, add_intercept=add_intercept
        )

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

        self.coef.role = Roles.coef_linear
        self.role = Roles.term_linear


class Intercept(UserVar):
    def __init__(
        self,
        name: str,
        value: Array | float = 0.0,
        distribution: lsl.Dist | None = None,
        inference: InferenceTypes = None,
    ) -> None:
        super().__init__(
            value=value, distribution=distribution, name=name, inference=inference
        )
        self.parameter = True
        self.role = Roles.intercept


def make_callback(function, output_shape, dtype):
    if len(output_shape):
        k = output_shape[-1]

    def fn(x, **basis_kwargs):
        n = jnp.shape(jnp.atleast_1d(x))[0]
        if len(output_shape) == 2:
            shape = (n, k)
        elif len(output_shape) == 1:
            shape = (n,)
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


def _append_name(name: str, append: str) -> str:
    if name == "":
        return ""
    else:
        return name + append


def _ensure_var_or_node(
    x: lsl.Var | lsl.Node | ArrayLike, name: str | None
) -> lsl.Var | lsl.Node:
    if isinstance(x, lsl.Var | lsl.Node):
        x_var = x
    else:
        name = name if name is not None else ""
        x_var = lsl.Var.new_obs(jnp.asarray(x), name=name)

    if name is not None and x_var.name != name:
        raise ValueError(f"{x_var.name=} and {name=} are incompatible.")

    return x_var


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
            for k, v in basis_kwargs.items():
                if isinstance(v, lsl.Var | lsl.Node):
                    basis_kwargs_arr[k] = v.value
                else:
                    basis_kwargs_arr[k] = v
            basis_ar = basis_fn(value_ar, **basis_kwargs_arr)
            dtype = basis_ar.dtype
            input_shape = jnp.shape(basis_ar)
            fn = make_callback(basis_fn, input_shape, dtype)
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
        self.role = Roles.basis
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
            self._penalty = lsl.Value(penalty_arr)

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
