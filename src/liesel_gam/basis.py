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
        self.observed = True

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
        basis_shape = jnp.shape(self.value)
        if len(basis_shape) >= 1:
            nbases: int = basis_shape[-1]
        else:
            nbases = 1  # scalar case

        return nbases

    @property
    def x(self) -> lsl.Var | lsl.Node:
        return self.value_node[0]

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

    def _basis_name(self, value: lsl.Var | lsl.Node, name: str | None):
        if name is not None:
            return name

        if value.name == "":
            return ""

        return f"B({value.name})"

    @property
    def penalty(self) -> lsl.Value | None:
        """
        Return the penalty matrix wrapped as a :class:`liesel.model.Value`.

        Returns
        -------
        lsl.Value
            Value wrapper holding the penalty (precision) matrix for this
            basis.
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

    def update_penalty(self, value: ArrayLike | lsl.Value):
        """
        Update the penalty matrix for this basis.

        Parameters
        ----------
        value
            New penalty matrix or an already-wrapped :class:`liesel.model.Value`.
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

        This method computes a transformation that diagonalizes
        the penalty matrix and updates the internal basis function such that
        subsequent evaluations use the accordingly transformed basis. The penalty is
        updated to the diagonalized version.

        Returns
        -------
        The modified basis instance (self).
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

        The penalty matrix is divided by its infinity norm (max absolute row
        sum) so that its values are numerically well-conditioned for
        downstream use. The updated penalty replaces the previous one.

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
    mapping: CategoryMapping
    nb: dict[str, list[str]] | None
    ordered_labels: list[str] | None
