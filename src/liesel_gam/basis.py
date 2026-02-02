from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, NamedTuple, Self

import jax
import jax.numpy as jnp
import liesel.model as lsl
from formulaic import ModelSpec

from liesel_gam.category_mapping import CategoryMapping

from .constraint import LinearConstraintEVD, penalty_to_unit_design
from .var import UserVar, _append_name, _ensure_var_or_node

InferenceTypes = Any
Array = jax.Array
ArrayLike = jax.typing.ArrayLike


def make_callback(function, output_shape, dtype, m: int = 0):
    k = output_shape[-1] if len(output_shape) else None

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

        sig = jax.ShapeDtypeStruct(shape, dtype)

        # ordered=True enforces sequencing of callback executions
        return jax.experimental.io_callback(
            function,
            sig,
            x,
            ordered=True,
            **basis_kwargs,
        )

    return fn


def is_diagonal(M, atol=1e-12):
    # mask for off-diagonal elements
    off_diag_mask = ~jnp.eye(M.shape[-1], dtype=bool)
    off_diag_values = M[off_diag_mask]
    return jnp.all(jnp.abs(off_diag_values) < atol)


class Basis(UserVar):
    """
    General basis for a structured additive term.

    The ``Basis`` class wraps an observation variable (or an array) and a
    basis-generation function. It constructs an internal calculation node that produces
    the basis (design) matrix by computing ``basis_fn(value)``. The basis function may
    be executed via a callback, in which case it does not need to be jax-compatible.
    This is the default, but it is potentially very slow, if the value of the basis
    needs to be recomputed during estimation. We recommend it only for bases that remain
    static during estimation.

    Parameters
    ----------
    value
        If a :class:`liesel.model.Var` or node is provided it is used as the input
        variable for the basis. Otherwise a raw array-like object may be supplied
        together with ``xname`` to create an observed variable internally.
    basis_fn
        Function mapping the input variable's values to a basis matrix or vector. It
        must accept the input array and any ``basis_kwargs`` and return an array of
        shape ``(n_obs, n_bases)`` (or a scalar/1-d array for simpler bases). By default
        this is the identity function (``lambda x: x``).
    name
        Optional name for the basis object. If omitted, a sensible name is constructed
        from the input variable's name (``B(<xname>)``).
    xname
        Required when ``value`` is a raw array: provides a name for the observation
        variable that will be created.
    use_callback
        If ``True`` (default) the basis_fn is wrapped in a JAX ``pure_callback`` via
        :func:`make_callback` to allow arbitrary Python basis functions while preserving
        JAX tracing. If ``False`` the function is used directly and must be jittable via
        JAX.
    cache_basis
        If ``True`` the computed basis is cached in a persistent calculation node
        (``lsl.Calc``), which avoids re-computation when not required, but uses memory.
        If ``False`` a transient calculation node (``lsl.TransientCalc``) is used and
        the basis will be recomputed with each evaluation of ``Basis.value``, but not
        stored in memory.
    penalty
        Penalty matrix associated with the basis. If ``"identity"``, a default identity
        penalty is created based on the number of basis functions. If *None*, an
        identity penalty is assumed, but not materialized, which saves memory but must
        be handled explicitly later, if downstream functionality relies on an explicit
        penalty matrix.
    **basis_kwargs
        Additional keyword arguments forwarded to ``basis_fn``.


    See Also
    ---------

    .TermBuilder : Initializes structured additive terms.
    .BasisBuilder : Initializesstructured additive terms.
    .StrctTerm : A general structured additive term.

    Notes
    -----
    The basis is evaluated once during initialization (via ``self.update()``) to
    determine its shape and dtype. The internal callback wrapper inspects the return
    shape to build a compatible JAX ShapeDtypeStruct for the pure callback.

    Examples
    --------
    Implementing a B-spline basis manually:

    >>> from liesel.contrib.splines import (
    ...     basis_matrix,
    ...     equidistant_knots,
    ...     pspline_penalty,
    ... )
    >>> import liesel_gam as gam
    >>> df = gam.demo_data(n=100)

    >>> knots = equidistant_knots(df["x_nonlin"].to_numpy(), n_param=20)
    >>> pen = pspline_penalty(d=20)

    >>> def bspline_basis(x):
    ...     return basis_matrix(x, knots=knots)

    >>> gam.Basis(
    ...     value=df["x_nonlin"].to_numpy(),
    ...     basis_fn=bspline_basis,
    ...     xname="x",
    ...     penalty=pen,
    ... )
    Basis(name="B(x)")

    Implementing a fixed basis matrix (without using the basis function). This is not
    recommended, because it means you cannot simply supply new covariate values to
    :meth:`liesel.model.Model.predict` for evaluating the basis matrix for predictions.

    >>> from liesel.contrib.splines import equidistant_knots, basis_matrix
    >>> import liesel_gam as gam
    >>> df = gam.demo_data(n=100)

    >>> knots = equidistant_knots(df["x_nonlin"].to_numpy(), n_param=20)
    >>> def bspline_basis(x):
    ...     return basis_matrix(x, knots=knots)

    >>> x = df["x_nonlin"].to_numpy()
    >>> gam.Basis(value=bspline_basis(x), name="B(x)")
    Basis(name="B(x)")
    """

    def __init__(
        self,
        value: lsl.Var | lsl.Node | ArrayLike,
        basis_fn: Callable[[Array], Array] | Callable[..., Array] = lambda x: x,
        name: str | None = None,
        xname: str | None = None,
        use_callback: bool = True,
        cache_basis: bool = True,
        penalty: ArrayLike | lsl.Value | Literal["identity"] | None = "identity",
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

        if isinstance(penalty, lsl.Value):
            penalty_var = penalty
        elif isinstance(penalty, str) and penalty == "identity":
            penalty_arr = jnp.eye(self.nbases)
            penalty_var = lsl.Value(penalty_arr)
        elif penalty is None:
            penalty_var = None
        else:
            penalty_arr = jnp.asarray(penalty)
            penalty_var = lsl.Value(penalty_arr)

        self._penalty = penalty_var

        self._constraint: str | None = None
        self._reparam_matrix: Array | None = None

    @property
    def nbases(self) -> int:
        """Number of basis functions (number of columns in the basis matrix)."""
        basis_shape = jnp.shape(self.value)
        if len(basis_shape) >= 1:
            nbases: int = basis_shape[-1]
        else:
            nbases = 1  # scalar case

        return nbases

    @property
    def x(self) -> lsl.Var | lsl.Node:
        """The input variable (observations) used to construct the basis."""
        return self.value_node[0]

    @property
    def constraint(self) -> str | None:
        """
        The type of constraint applied to this basis and penalty (if any).

        See :meth:`.Basis.constrain` for details.
        """
        return self._constraint

    @property
    def reparam_matrix(self) -> Array | None:
        """
        Reparameterization matrix used for constraint of this basis and penalty (if
        any).

        See :meth:`.Basis.constrain` for details.
        """
        return self._reparam_matrix

    def _validate_xname(self, value: lsl.Var | lsl.Node | ArrayLike, xname: str | None):
        if isinstance(value, lsl.Var | lsl.Node) and xname is not None:
            raise ValueError(
                "When supplying a variable or node to `value`, `xname` must not be "
                "used. Name the variable instead."
            )

    def _basis_name(self, value: lsl.Var | lsl.Node, name: str | None):
        if name is not None:
            return name

        if value.name == "":
            return ""

        return f"B({value.name})"

    @property
    def penalty(self) -> lsl.Value | None:
        """
        Penalty matrix, wrapped as a :class:`liesel.model.Value` (if any).
        """
        return self._penalty

    def _validate_penalty_shape(self, pen: ArrayLike | lsl.Value) -> lsl.Value:
        if isinstance(pen, lsl.Value):
            pen_arr = jnp.asarray(pen.value)
            pen_val = pen
            pen_val.value = pen_arr
        else:
            pen_arr = jnp.asarray(pen)
            pen_val = lsl.Value(pen_arr)

        if not pen_arr.shape[-1] == self.nbases:
            raise ValueError(
                f"Basis has {self.nbases} columns, replacement penalty has "
                f"{pen_arr.shape[-1]}"
            )
        return pen_val

    def update_penalty(self, value: ArrayLike | lsl.Value) -> None:
        """
        Updates the penalty matrix for this basis.

        If :attr:`.Basis.penalty` is not None, this method will only update the
        value of the penalty node, not the whole object. Even if the argument to
        this method is a node.

        Parameters
        ----------
        value
            New penalty matrix or a :class:`liesel.model.Value` wrapping a penalty
            matrix.
        """
        if self._penalty is None:
            self._penalty = self._validate_penalty_shape(value)
        else:
            self._penalty.value = self._validate_penalty_shape(value).value

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

        This method computes a transformation that diagonalizes the penalty matrix and
        updates the internal basis function such that subsequent evaluations use the
        accordingly transformed basis. The penalty is updated to the diagonalized
        version.

        Parameters
        ----------
        atol
            Absolute tolerance used in testing whether the existing penalty is already
            diagonal. If that is the case, the basis instance is returned without any
            further changes.

        Returns
        -------
        The modified basis instance (self).

        Notes
        -----

        Penalty diagonalization works via an eigenvalue decomposition of the penalty
        matrix. Let the eigenvalue decomposition of the :math:`d \\times d` penalty
        matrix :math:`\\mathbf{K}` be given by

        .. math::

            \\mathbf{K} = \\mathbf{U} \\boldsymbol{\\Lambda} \\mathbf{U}^\\top,

        where :math:`\\boldsymbol{\\Lambda} = \\operatorname{diag}(\\lambda_1, \\dots,
        \\lambda_r)` contains the eigenvalues of :math:`\\mathbf{K}` in decreasing order
        and :math:`\\mathbf{U}` the corresponding eigenvectors. Let :math:`r` denote the
        rank of :math:`\\mathbf{K}`.

        The function obtains a reparameterization matrix :math:`\\mathbf{Z}` as

        .. math::

            \\mathbf{Z} = \\mathbf{U} \\boldsymbol{\\Lambda}^{-1/2},

        where :math:`\\boldsymbol{\\Lambda}^{-1/2} =
        \\operatorname{diag}(\\lambda_1^{-1/2}, \\dots, \\lambda_r^{-1/2},
        \\mathbf{0}_{d-r}^\\top)`. The element :math:`\\mathbf{0}_{d-r}^\\top` is a
        zero-vector of length :math:`d-r`, corresponding to the zero eigenvalues of the
        penalty matrix.

        The basis matrix :math:`\\mathbf{B}` is then updated as :math:`\\mathbf{B}_Z =
        \\mathbf{B} \\mathbf{Z}`, and the penalty matrix is updated to the :math:`d
        \\times d` identity matrix.

        The basis function is likewise updated to evaluate to the reparamterized basis
        matrix during prediction.

        References
        ----------

        Kneib, T., Klein, N., Lang, S., & Umlauf, N. (2019). Modular regression—A Lego
        system for building structured additive distributional regression models with
        tensor product interactions. TEST, 28(1), 1–39.
        https://doi.org/10.1007/s11749-019-00631-z

        """
        if self.penalty is None:
            raise TypeError("Basis.penalty is None, cannot apply transformation.")
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
        penalty = jnp.eye(Z.shape[-1])  # practically equal to: penalty = Z.T @ K @ Z
        self.update_penalty(penalty)

        return self

    def scale_penalty(self) -> Self:
        """
        Scale the penalty matrix by its infinite norm.

        The penalty matrix is divided by its infinity norm (max absolute row sum) so
        that its values are numerically well-conditioned for downstream use. The updated
        penalty replaces the previous one.

        Returns
        -------
        The modified basis instance (self).
        """
        if self.penalty is None:
            raise TypeError("Basis.penalty is None, cannot apply transformation.")
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
        if self.penalty is None:
            raise TypeError("Basis.penalty is None, cannot apply transformation.")

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
        r"""
        Apply a linear constraint to the basis and corresponding penalty.

        When a constraint is applied, the type of constraint is saved to
        :attr:`.Basis.constraint`, and the reparamterization matrix is saved to
        :attr:`.Basis.reparam_matrix`.

        Parameters
        ----------
        constraint
            Type of constraint or custom linear constraint matrix to apply. If an array
            is supplied, the constraint will be ``A @ coef == 0``, where ``A`` is the
            supplied array (the constraint matrix).


        Returns
        -------
        The modified basis instance (self).

        Notes
        -----

        This method implements the procedure detailed by Kneib et al. (2019). For the
        following exposition, which is quoted almost verbatim from Kneib et al. (2019),
        assume that this basis is used to evaluate a function

        .. math::
            s(\mathbf{x}_i) = \sum_{j=1}^J B_j(\mathbf{x}_i) \beta_j
            = \mathbf{b}(\mathbf{x}_i)^\top \boldsymbol{\beta},

        where

        - :math:`i=1, \dots, N` is the observation index,
        - :math:`\mathbf{x}_i^\top = [x_{i,1}, \dots, x_{i,M}]` are covariate
          observations, where :math:`M` denotes the number of covariates,
        - :math:`\mathbf{b}^\top = [B_1(\mathbf{x}), \dots, B_J(\mathbf{x})]`
          are a set of basis function evaluations, and
        - :math:`\boldsymbol{\beta}^\top = [\beta_1, \dots, \beta_J]`
          are the corresponding coefficients.

        The basis matrix for such a term is

        .. math::

            \mathbf{B} = \begin{bmatrix}
            \mathbf{b}(\mathbf{x}_1)^\top \\
            \vdots \\
            \mathbf{b}(\mathbf{x}_N)^\top
            \end{bmatrix},

        and the term can be written in matrix form as

        .. math::

            \mathbf{s} = \mathbf{B} \boldsymbol{\beta},

        where :math:`\mathbf{B}` is the basis matrix of dimension :math:`N
        \times J`. We consider :math:`\boldsymbol{\beta} \in \mathbb{R}^J` to be subject to linear constraints of the
        form

        .. math::

            \mathbf{A} \boldsymbol{\beta} = \mathbf{0}.

        :math:`\mathbf{A}` is an :math:`A \times J` constraint matrix. To explicitly
        remove the constrained component, we construct a complementary matrix
        :math:`\bar{\mathbf{A}} \in \mathbb{R}^{(J-A) \times J}` such that

        .. math::

            \bar{\mathbf{A}} \mathbf{A}^\top = \mathbf{0},

        and the stacked matrix :math:`[\mathbf{A}^\top,
        \bar{\mathbf{A}}^\top]^\top` is of full rank. One possible construction of
        :math:`\bar{\mathbf{A}}` is based on the eigenvalue decomposition of
        :math:`\mathbf{A}^\top \mathbf{A}`, using the eigenvectors corresponding to
        zero eigenvalues. This is the construction of :math:`\bar{\mathbf{A}}` used in
        this method. Under the full-rank assumption, the inverse of the composed matrix
        exists and can be written as

        .. math::

            \begin{bmatrix}
            \mathbf{A} \\
            \bar{\mathbf{A}}
            \end{bmatrix}^{-1}
            =
            \begin{bmatrix}
            \mathbf{C}, \bar{\mathbf{C}} \end{bmatrix},

        where :math:`\mathbf{C} \in \mathbb{R}^{J \times A}` and
        :math:`\bar{\mathbf{C}} \in \mathbb{R}^{J \times (J-A)}`. This yields the
        reparameterisation

        .. math::

            \boldsymbol{\beta} = \mathbf{C} \boldsymbol{\alpha} +
            \bar{\mathbf{C}} \boldsymbol{\gamma},

        where :math:`\boldsymbol{\alpha} = \mathbf{A} \boldsymbol{\beta} =
        \mathbf{0}` vanishes due to the constraint and :math:`\boldsymbol{\gamma} =
        \bar{\mathbf{A}} \boldsymbol{\beta}` represents the remaining unconstrained
        coefficients. Applying this reparameterisation to the functional effect gives
        :math:`\bar{\mathbf{s}} = \bar{\mathbf{B}} \boldsymbol{\alpha}`,
        where the basis matrix is reparameterized as

        .. math::

            \bar{\mathbf{B}} = \mathbf{B} \bar{\mathbf{C}}.

        Accordingly, the original penalty matrix :math:`\mathbf{K}` is reparamterized
        as

        .. math::

            \bar{\mathbf{K}} = \bar{\mathbf{C}}^\top \mathbf{K} \bar{\mathbf{C}}.


        .. rubric:: Default constraint options

        The default options correspond to the following constraint matrices:

        - ``"sumzero_term"``: :math:`\mathbf{A} = \mathbf{1}^\top \mathbf{B}`,
          where :math:`\mathbf{B}` is the basis matrix. This is the preferred option
          for a sum to zero constraint, because it centers the evaluated term.

        - ``"sumzero_coef"``: :math:`\mathbf{A} = \mathbf{1}^\top`.
          This is an alternative sum to zero constraint, focusing only on ensuring
          that the coefficients sum to zero.

        - ``"constant_and_linear"``:
          :math:`\mathbf{A}=(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top
          \mathbf{B}`,
          where :math:`\mathbf{X} = [\mathbf{1}, \mathbf{x}]` is a design matrix
          built with the covariate observations :math:`\mathbf{x}` used in this
          basis. This constraint removes both a constant (like ``"sumzero_term"``) and
          a linear trend from the term modeled with this basis.



        References
        ----------

        Kneib, T., Klein, N., Lang, S., & Umlauf, N. (2019). Modular regression—A Lego
        system for building structured additive distributional regression models with
        tensor product interactions. TEST, 28(1), 1–39.
        https://doi.org/10.1007/s11749-019-00631-z
        """  # noqa: E501
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


class MRFBasis(Basis):
    """
    Dedicated basis object for Markov random fields.

    See :class:`.Basis` for general usage information. This class additionally offers
    information about the Markov random field setup in :attr:`.mrf_spec`.
    """

    _mrf_spec: MRFSpec | None = None

    @property
    def mrf_spec(self) -> MRFSpec:
        """
        A named tuple, containing information about the Markov random field setup.

        The :class:`.MRFSpec` has the attributes ``nb`` (neighborhood structure),
        ``mapping`` (label-integer map for the region labels), and ``ordered_labels``
        (ordered labels, such that the order correspond to the columns of the basis
        matrix.)
        """
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
    """
    Dedicated basis object for linear effects.

    See :class:`.Basis` for general usage information. This class additionally offers

    - :attr:`.model_spec`: The model spec used internally by ``formulaic`` to set up
      the basis matrix.
    - :attr:`.mappings`: A dictionary of label-integer mappings for all categorical
      variables in this basis.
    - :attr:`.column_names`: List of column names for this basis.
    """

    _model_spec: ModelSpec | None = None
    _mappings: dict[str, CategoryMapping] | None = None
    _column_names: list[str] | None = None

    @property
    def model_spec(self) -> ModelSpec:
        """
        The model spec used internally by ``formulaic`` to set up the basis matrix.
        """
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
        """
        A dictionary of label-integer mappings for all categorical variables in this
        basis.
        """
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
        """List of column names for this basis."""
        if not isinstance(value, Sequence):
            raise TypeError(f"Replacement must be a sequence, got {type(value)}.")

        if isinstance(value, str):
            raise TypeError("Replacement type cannot be string.")

        if not len(value) == self.value.shape[-1]:
            raise ValueError(
                f"Expected {self.value.shape[-1]} column names, got {len(value)}"
            )

        for val in value:
            if not isinstance(val, str):
                raise TypeError(
                    f"The values in the replacement must be of type str, "
                    f"got {type(val)}."
                )
        self._column_names = list(value)


class MRFSpec(NamedTuple):
    """
    A named tuple, containing information about the Markov random field setup.

    The :class:`.MRFSpec` has the attributes ``nb`` (neighborhood structure),
    ``mapping`` (label-integer map for the region labels), and ``ordered_labels``
    (ordered labels, such that the order correspond to the columns of the basis
    matrix.)
    """

    mapping: CategoryMapping
    nb: dict[str, list[str]] | None
    ordered_labels: list[str] | None
    polys: dict[str, ArrayLike] | None
