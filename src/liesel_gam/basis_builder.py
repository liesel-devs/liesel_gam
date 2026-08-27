from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, cast, get_args

import formulaic as fo
import jax
import jax.numpy as jnp
import liesel.model as lsl
import numpy as np
import pandas as pd
import smoothcon

from .basis import ApproximationSpec, Basis, LinBasis, MRFBasis, MRFSpec
from .names import NameManager
from .registry import CategoryMapping, PandasRegistry

InferenceTypes = Any

Array = jax.Array
ArrayLike = jax.typing.ArrayLike

BasisTypes = Literal["tp", "ts", "cr", "cs", "cc", "bs", "ps", "cp", "gp"]


logger = logging.getLogger(__name__)


def _validate_bs(bs):
    if isinstance(bs, str):
        bs = [bs]
    allowed = get_args(BasisTypes)
    for bs_str in bs:
        if bs_str not in allowed:
            raise ValueError(f"Allowed values for 'bs' are: {allowed}; got {bs=}.")


def _validate_formula(formula: str) -> None:
    if "~" in formula:
        raise ValueError("'~' in formulas is not supported.")

    terms = ["".join(x.split()) for x in formula.split("+")]
    for term in terms:
        if term == "1":
            raise ValueError(
                "Using '1 +' is not supported. To add an intercept, use the "
                "argument 'include_intercept'."
            )
        if term == "0" or term == "-1":
            raise ValueError(
                "Using '0 +' or '-1' is not supported. Intercepts are not included "
                "by default and can be added manually with the argument "
                "'include_intercept'."
            )


def _validate_penalty_order(penalty_order: int):
    if not isinstance(penalty_order, int):
        raise TypeError(
            f"'penalty_order' must be int or None, got {type(penalty_order)}"
        )
    if not penalty_order > 0:
        raise ValueError(f"'penalty_order' must be >0, got {penalty_order}")


class BasisBuilder:
    """
    Initializes :class:`.Basis` objects from data in a :class:`.PandasRegistry`.

    Parameters
    ----------
    registry
        A pandas registry, giving access to the data.
    names
        A name manager for creating unique names.
    approximation
        Default approximation policy for eligible univariate continuous bases.
        ``False`` keeps exact evaluation, ``True`` uses
        :class:`.ApproximationSpec` defaults, and an ``ApproximationSpec`` supplies
        shared tolerances and the grid-size guard. Builder-level specifications
        cannot define bounds because bounds belong to individual covariates.

    See Also
    --------

    .TermBuilder : Initializes structured additive terms.
    .Basis : Basic basis class.
    .LinBasis : Specialized basis for linear effects.
    .MRFBasis : Specialized basis for Gaussian Markov random fields.

    Notes
    -----
    Eligible methods accept a per-call ``approximation`` override. ``None``
    inherits the builder policy. Multivariate, linear, categorical, random-effect,
    and MRF bases remain exact.

    Examples
    --------
    >>> import liesel_gam as gam
    >>> df = gam.demo_data(n=100)
    >>> registry = gam.PandasRegistry(df)
    >>> bb = gam.BasisBuilder(registry)
    >>> bb.ps("x_nonlin", k=20)
    Basis(name="B(x_nonlin)")
    """

    def __init__(
        self,
        registry: PandasRegistry,
        names: NameManager | None = None,
        approximation: bool | ApproximationSpec = False,
    ) -> None:
        if not isinstance(approximation, bool | ApproximationSpec):
            raise TypeError("approximation must be a bool or ApproximationSpec.")
        if (
            isinstance(approximation, ApproximationSpec)
            and approximation.bounds is not None
        ):
            raise ValueError(
                "Builder-level approximation bounds must be None; "
                "set bounds on an individual basis."
            )
        self.registry = registry
        self.mappings: dict[str, CategoryMapping] = {}
        self.names = NameManager() if names is None else names
        self.approximation = approximation

    def __repr__(self) -> str:
        return f"{type(self).__name__}(data_shape={self.registry.data.shape})"

    @property
    def data(self) -> pd.DataFrame:
        """The dataframe wrapped by this builder's registry."""
        return self.registry.data

    def basis(
        self,
        *x: str | lsl.Var,
        basis_fn: Callable[[Array], Array],
        use_callback: bool = True,
        cache_basis: bool = True,
        penalty: ArrayLike | lsl.Value | None = None,
        basis_name: str = "B",
        approximation: bool | ApproximationSpec | None = None,
        row_wise: bool | None = None,
    ) -> Basis:
        """
        Initializes a general basis given a basis function.

        Parameters
        ----------
        *x
            Names of input variables.
        basis_fn
            Basis function. Must take a 2d-array as input and return a 2d array.
        use_callback
            If *True*, the basis function is evaluated using a Python callback,
            which means that it does not have to be jit-compatible via JAX. This also
            means that the basis must remain constant throughout estimation.
            Passed on to :class:`.Basis`.
        cache_basis
            If ``True`` the computed basis is cached in a persistent
            calculation node (``lsl.Calc``), which avoids re-computation
            when not required. Passed on to :class:`.Basis`.
        penalty
            Penalty matrix associated with the basis.
            Passed on to :class:`.Basis`.
        basis_name
            Function-name for the basis matrix. If ``"B"``, and the basis is a function
            of the variable ``"x"``, the full name of the :class:`.Basis` object will
            be ``"B(x)"``. Names are made unique by appending a counter if necessary.
        approximation
            ``None`` inherits the builder policy, ``False`` keeps exact evaluation,
            ``True`` uses default approximation settings, and an
            :class:`.ApproximationSpec` supplies custom settings. Approximation
            requires exactly one scalar covariate.
        row_wise
            Whether each output row depends only on the corresponding input row.
            Passed on to :class:`.Basis`.

        Examples
        --------

        .. rubric:: Manually specified B-Spline basis

        >>> from liesel.contrib.splines import basis_matrix, equidistant_knots
        >>> from liesel.contrib.splines import pspline_penalty
        >>> import liesel_gam as gam

        >>> df = gam.demo_data(n=100)
        >>> registry = gam.PandasRegistry(df)
        >>> bb = gam.BasisBuilder(registry)

        >>> knots = equidistant_knots(df["x_nonlin"].to_numpy(), n_param=20)
        >>> pen = pspline_penalty(d=20)

        The basis function should always expect a matrix-valued array as an input.

        >>> def bspline_basis(x_mat):
        ...     # x_mat is shape (n, 1)
        ...     x_vec = x_mat.squeeze()  # shape (n,)
        ...     return basis_matrix(x_vec, knots=knots)

        >>> bb.basis("x_nonlin", basis_fn=bspline_basis, penalty=pen)
        Basis(name="B(x_nonlin)")

        .. rubric:: Manually specified linear basis

        This is a minimal example for how a basis as a function of multiple variables
        works.

        >>> import jax.numpy as jnp
        >>> import liesel_gam as gam
        >>> df = gam.demo_data(n=100)
        >>> registry = gam.PandasRegistry(df)
        >>> bb = gam.BasisBuilder(registry)

        >>> def linear_basis(x_mat):
        ...     # x_mat is shape (n, 2)
        ...     basis_mat = jnp.column_stack((jnp.ones(df.shape[0]), x_mat))
        ...     return basis_mat

        >>> basis = bb.basis("x_nonlin", "x_lin", basis_fn=linear_basis)
        >>> basis
        Basis(name="B(x_nonlin,x_lin)")

        >>> basis.value.shape
        (100, 3)
        """
        if isinstance(penalty, lsl.Value):
            penalty.value = jnp.asarray(penalty.value)
        elif penalty is not None:
            penalty = jnp.asarray(penalty)

        x_vars = []
        x_names = []
        for x_name in x:
            x_var = self._get_var_and_value(x_name)[0]
            x_names.append(x_var.name)
            x_vars.append(x_var)

        Xname = self.registry.prefix + ",".join(x_names)

        Xvar = lsl.TransientCalc(
            lambda *x: jnp.column_stack(x),
            *x_vars,
            _name=self.names.create(f"[{Xname}]"),
        )

        basis = Basis(
            value=Xvar,
            basis_fn=basis_fn,
            name=self.names.create(basis_name + "(" + Xname + ")"),
            use_callback=use_callback,
            cache_basis=cache_basis,
            penalty=penalty,
            row_wise=row_wise,
        )
        basis._input_name = Xname

        return self._maybe_approximate(
            basis,
            approximation,
            eligible=len(x) == 1,
        )

    def _maybe_approximate(
        self,
        basis: Basis,
        approximation: bool | ApproximationSpec | None,
        *,
        eligible: bool = True,
    ) -> Basis:
        if not eligible:
            if approximation is not None and approximation is not False:
                raise ValueError(
                    "Basis approximation requires exactly one scalar covariate."
                )
            return basis

        setting = self.approximation if approximation is None else approximation
        if setting is False:
            return basis
        if setting is True:
            return basis.approximate()
        if isinstance(setting, ApproximationSpec):
            return basis.approximate(setting)
        raise TypeError("approximation must be None, a bool, or ApproximationSpec.")

    def _get_var_and_value(self, x: str | lsl.Var) -> tuple[lsl.Var, jax.Array]:
        if isinstance(x, str):
            x_array = jnp.asarray(self.registry.data[x].to_numpy())
            x_var = self.registry.get_numeric_obs(x)

        elif isinstance(x, lsl.Var):
            if not x.name:
                raise ValueError("If you supply a variable for 'x', it must be named.")
            x_array = jnp.asarray(x.value)
            x_var = x
        else:
            raise TypeError(f"Type {type(x)} not supported for 'x'.")

        return x_var, x_array

    def _native_basis(
        self,
        value: lsl.Var | lsl.Node,
        *,
        xname: str,
        smooth: smoothcon.Smooth,
        absorb_cons: bool,
        diagonal_penalty: bool,
        scale_penalty: bool,
        basis_name: str,
        skip_constraint: bool = False,
        approximation: bool | ApproximationSpec | None = None,
        approximation_eligible: bool = True,
        input_name: str | None = None,
    ) -> Basis:
        """Wrap and transform a raw native smooth in the standard order."""
        basis = Basis(
            value,
            name=self.names.create(basis_name + "(" + xname + ")"),
            basis_fn=smooth.basis,
            penalty=smooth.penalty,
            use_callback=False,
            cache_basis=True,
            row_wise=True,
        )
        basis._input_name = input_name or xname
        # Native constructors know the mathematical rank before float32
        # roundoff. Preserve it for the mixed-model reparameterization.
        basis._penalty_rank = smooth.rank
        if scale_penalty:
            basis.scale_penalty()
        if absorb_cons and not skip_constraint:
            basis.constrain("sumzero_term")
        if diagonal_penalty:
            basis.diagonalize_penalty()
        return self._maybe_approximate(
            basis,
            approximation,
            eligible=approximation_eligible,
        )

    def _get_matrix(
        self, *x: str | lsl.Var, cache: bool = False
    ) -> lsl.Calc | lsl.TransientCalc:
        """Get a calculation node that column-stacks named or supplied variables.

        All inputs must be either names of numeric registry variables or named
        ``lsl.Var`` objects. Registry-backed matrices are cached by the registry;
        matrices from supplied variables are created directly.
        """
        all_str = all(isinstance(x_, str) for x_ in x)
        all_var = all(isinstance(x_, lsl.Var) for x_ in x)

        if all_str:
            names = cast(tuple[str, ...], x)
            calc: lsl.Calc | lsl.TransientCalc = self.registry.get_many_numeric_obs(
                *names, cache=cache
            )
            return calc

        if not all_var:
            raise ValueError(
                f"Must supply either only variables or only names, got {x}."
            )

        vars_ = cast(tuple[lsl.Var, ...], x)
        xname = ",".join([v.name for v in vars_])
        xname = self.names.create(f"[{xname}]")

        if not cache:
            calc = lsl.TransientCalc(
                lambda *args: jnp.vstack(args).T,
                *vars_,
                _name=xname,
            )
        else:
            calc = lsl.Calc(
                lambda *args: jnp.vstack(args).T,
                *vars_,
                _name=xname,
            )

        return calc

    def ps(
        self,
        x: str | lsl.Var,
        *,
        k: int,
        basis_degree: int = 3,
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
        approximation: bool | ApproximationSpec | None = None,
    ) -> Basis:
        """
        B-spline basis with a discrete (P-spline) penalty matrix.

        Parameters
        ----------
        x
            Name of input variable.
        k
            Number of (unconstrained) bases.
        basis_degree
            Degree of the polynomials used in the B-spline basis function. Default is 3
            for cubic B-splines.
        penalty_order
            Order of the penalty.
        knots
            Knots used to set up the basis. If ``None`` (default), a set of equidistant
            knots will be set up automatically, with the domain boundaries inferred from
            the minimum and maximum of the observed values. The number of knots must be
            ``k + basis_degree + 1``, and for the observed data, it must be true that
            ``knots[basis_degree] < min(x)`` and ``max(x) < knots[-basis_degree]``.
        absorb_cons
            Whether the default identification constraint should be applied by
            reparameterization and absorbing the reparameterization matrix into the
            basis and penalty matrices for computational efficiency. If ``False``, the
            basis is unconstrained, if ``True`` it receives a sum to zero constrained.
            Also see :meth:`.Basis.constrain`.
        diagonal_penalty
            Whether the penalty matrix associated with this term should be
            reparameterized into a diagonal matrix. In this case, the basis matrix is
            reparameterized accordingly. This can be beneficial for posterior geometry,
            which is why it is the default. Also see :meth:`.Basis.diagonalize_penalty`.
        scale_penalty
            Whether to use design-aware penalty scaling. Also see
            :meth:`.Basis.scale_penalty`.
        basis_name
            Function-name for the basis matrix. If ``"B"``, and the basis is a function
            of the variable ``"x"``, the full name of the :class:`.Basis` object will be
            ``"B(x)"``. Names are made unique by appending a counter if necessary.

        Notes
        -----

        This native JAX basis uses ``use_callback=False`` and ``cache_basis=True``.
        See :class:`.Basis` for details.

        The basis and penalty are constructed natively in JAX.

        References
        ----------
        - Lang, S., & Brezger, A. (2004). Bayesian P-splines. Journal of Computational
          and Graphical Statistics, 13(1), 183–212.
          https://doi.org/10.1198/1061860043010
        - Wood, S.N. (2017) Generalized Additive Models: An Introduction with R (2nd
          edition). Chapman and Hall/CRC.
        - R package mgcv https://cran.r-project.org/web/packages/mgcv/index.html

        Examples
        --------
        >>> import liesel_gam as gam
        >>> df = gam.demo_data(n=100)
        >>> registry = gam.PandasRegistry(df)
        >>> bb = gam.BasisBuilder(registry)
        >>> bb.ps("x_nonlin", k=20)
        Basis(name="B(x_nonlin)")

        The default is a constrained basis:

        >>> bb.ps("x_nonlin", k=20).value.shape
        (100, 19)

        The constraint can be turned off by passing ``absorb_cons=False``:

        >>> bb.ps("x_nonlin", k=20, absorb_cons=False).value.shape
        (100, 20)

        """
        _validate_penalty_order(penalty_order)
        x_var, x_array = self._get_var_and_value(x)
        smooth = smoothcon.pspline(
            x_array,
            k=k,
            degree=basis_degree,
            penalty_order=penalty_order,
            knots=knots,
        )
        return self._native_basis(
            x_var,
            xname=x_var.name,
            smooth=smooth,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name=basis_name,
            approximation=approximation,
        )

    def cr(
        self,
        x: str | lsl.Var,
        *,
        k: int,
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
        approximation: bool | ApproximationSpec | None = None,
    ) -> Basis:
        """
        Cubic regression spline basis and penalty matrix.

        Parameters
        ----------
        x
            Name of input variable.
        k
            Number of (unconstrained) bases.
        penalty_order
            Order of the penalty.
        knots
            Knots used to set up the basis. If ``None`` (default), a set of equidistant
            knots will be set up automatically, with the domain boundaries inferred from
            the minimum and maximum of the observed values.
        absorb_cons
            Whether the default identification constraint should be applied by
            reparameterization and absorbing the reparameterization matrix into the
            basis and penalty matrices for computational efficiency. If ``False``, the
            basis is unconstrained, if ``True`` it receives a sum to zero constrained.
            Also see :meth:`.Basis.constrain`.
        diagonal_penalty
            Whether the penalty matrix associated with this term should be
            reparameterized into a diagonal matrix. In this case, the basis matrix is
            reparameterized accordingly. This can be beneficial for posterior geometry,
            which is why it is the default. Also see :meth:`.Basis.diagonalize_penalty`.
        scale_penalty
            Whether to use design-aware penalty scaling. Also see
            :meth:`.Basis.scale_penalty`.
        basis_name
            Function-name for the basis matrix. If ``"B"``, and the basis is a function
            of the variable ``"x"``, the full name of the :class:`.Basis` object will be
            ``"B(x)"``. Names are made unique by appending a counter if necessary.

        See Also
        --------

        .cs : Cubic regression splines with additinal shrinkage on the null space.

        Notes
        -----

        This native JAX basis uses ``use_callback=False`` and ``cache_basis=True``.
        See :class:`.Basis` for details.

        The basis and penalty are constructed natively in JAX. The mgcv
        documentation describes the corresponding mathematical smooth family.


        References
        ----------

        - Wood, S.N. (2017) Generalized Additive Models: An Introduction with R (2nd
          edition). Chapman and Hall/CRC.
        - R package mgcv https://cran.r-project.org/web/packages/mgcv/index.html

        Examples
        ---------
        >>> import liesel_gam as gam
        >>> df = gam.demo_data(n=100)
        >>> registry = gam.PandasRegistry(df)
        >>> bb = gam.BasisBuilder(registry)
        >>> bb.cr("x_nonlin", k=20)
        Basis(name="B(x_nonlin)")
        """
        _validate_penalty_order(penalty_order)
        x_var, x_array = self._get_var_and_value(x)
        smooth = smoothcon.cubic_regression(x_array, k=k, knots=knots)
        return self._native_basis(
            x_var,
            xname=x_var.name,
            smooth=smooth,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name=basis_name,
            approximation=approximation,
        )

    def cs(
        self,
        x: str | lsl.Var,
        *,
        k: int,
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
        approximation: bool | ApproximationSpec | None = None,
    ) -> Basis:
        """
        Cubic regression spline basis and penalty matrix with null space penalty.

        Parameters
        ----------
        x
            Name of input variable.
        k
            Number of (unconstrained) bases.
        penalty_order
            Order of the penalty.
        knots
            Knots used to set up the basis. If ``None`` (default), a set of equidistant
            knots will be set up automatically, with the domain boundaries inferred from
            the minimum and maximum of the observed values.
        absorb_cons
            Whether the default identification constraint should be applied by
            reparameterization and absorbing the reparameterization matrix into the
            basis and penalty matrices for computational efficiency. If ``False``, the
            basis is unconstrained, if ``True`` it receives a sum to zero constrained.
            Also see :meth:`.Basis.constrain`.
        diagonal_penalty
            Whether the penalty matrix associated with this term should be
            reparameterized into a diagonal matrix. In this case, the basis matrix is
            reparameterized accordingly. This can be beneficial for posterior geometry,
            which is why it is the default. Also see :meth:`.Basis.diagonalize_penalty`.
        scale_penalty
            Whether to use design-aware penalty scaling. Also see
            :meth:`.Basis.scale_penalty`.
        basis_name
            Function-name for the basis matrix. If ``"B"``, and the basis is a function
            of the variable ``"x"``, the full name of the :class:`.Basis` object will be
            ``"B(x)"``. Names are made unique by appending a counter if necessary.

        Notes
        -----

        This native JAX basis uses ``use_callback=False`` and ``cache_basis=True``.
        See :class:`.Basis` for details.

        The basis and penalty are constructed natively in JAX. The mgcv
        documentation describes the corresponding mathematical smooth family.

        References
        ----------

        - Wood, S.N. (2017) Generalized Additive Models: An Introduction with R (2nd
          edition). Chapman and Hall/CRC.
        - R package mgcv https://cran.r-project.org/web/packages/mgcv/index.html

        Examples
        ---------
        >>> import liesel_gam as gam
        >>> df = gam.demo_data(n=100)
        >>> registry = gam.PandasRegistry(df)
        >>> bb = gam.BasisBuilder(registry)
        >>> bb.cs("x_nonlin", k=20)
        Basis(name="B(x_nonlin)")
        """
        _validate_penalty_order(penalty_order)
        x_var, x_array = self._get_var_and_value(x)
        smooth = smoothcon.cubic_regression(x_array, k=k, knots=knots, shrinkage=True)
        return self._native_basis(
            x_var,
            xname=x_var.name,
            smooth=smooth,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name=basis_name,
            approximation=approximation,
        )

    def cc(
        self,
        x: str | lsl.Var,
        *,
        k: int,
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
        approximation: bool | ApproximationSpec | None = None,
    ) -> Basis:
        """
        Cyclic cubic regression spline basis and penalty matrix.

        Basis for  a penalized cubic regression spline whose ends match, up to second
        derivative.

        Parameters
        ----------
        x
            Name of input variable.
        k
            Number of (unconstrained) bases.
        penalty_order
            Order of the penalty.
        knots
            Knots used to set up the basis. If ``None`` (default), a set of equidistant
            knots will be set up automatically, with the domain boundaries inferred from
            the minimum and maximum of the observed values.
        absorb_cons
            Whether the default identification constraint should be applied by
            reparameterization and absorbing the reparameterization matrix into the
            basis and penalty matrices for computational efficiency. If ``False``, the
            basis is unconstrained, if ``True`` it receives a sum to zero constrained.
            Also see :meth:`.Basis.constrain`.
        diagonal_penalty
            Whether the penalty matrix associated with this term should be
            reparameterized into a diagonal matrix. In this case, the basis matrix is
            reparameterized accordingly. This can be beneficial for posterior geometry,
            which is why it is the default. Also see :meth:`.Basis.diagonalize_penalty`.
        scale_penalty
            Whether to use design-aware penalty scaling. Also see
            :meth:`.Basis.scale_penalty`.
        basis_name
            Function-name for the basis matrix. If ``"B"``, and the basis is a function
            of the variable ``"x"``, the full name of the :class:`.Basis` object will be
            ``"B(x)"``. Names are made unique by appending a counter if necessary.

        Notes
        -----

        This native JAX basis uses ``use_callback=False`` and ``cache_basis=True``.
        See :class:`.Basis` for details.

        Cyclicity is enforced by matching the function and its derivatives at the domain
        boundaries. The basis and penalty are constructed natively in JAX.

        References
        ----------

        - Wood, S.N. (2017) Generalized Additive Models: An Introduction with R (2nd
          edition). Chapman and Hall/CRC.
        - R package mgcv https://cran.r-project.org/web/packages/mgcv/index.html

        Examples
        ---------
        >>> import liesel_gam as gam
        >>> df = gam.demo_data(n=100)
        >>> registry = gam.PandasRegistry(df)
        >>> bb = gam.BasisBuilder(registry)
        >>> bb.cc("x_nonlin", k=20)
        Basis(name="B(x_nonlin)")
        """
        _validate_penalty_order(penalty_order)
        x_var, x_array = self._get_var_and_value(x)
        smooth = smoothcon.cyclic_cubic(x_array, k=k, knots=knots)
        return self._native_basis(
            x_var,
            xname=x_var.name,
            smooth=smooth,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name=basis_name,
            approximation=approximation,
        )

    def bs(
        self,
        x: str | lsl.Var,
        *,
        k: int,
        basis_degree: int = 3,
        penalty_order: int | Sequence[int] = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
        approximation: bool | ApproximationSpec | None = None,
    ) -> Basis:
        """
        B-spline basis with integrated squared derivative penalties.

        Parameters
        ----------
        x
            Name of input variable.
        k
            Number of (unconstrained) bases.
        basis_degree
            Degree of the polynomials used in the B-spline basis function. Default is 3
            for cubic B-splines.
        penalty_order
            Order of the penalty. If this is a sequence of integers, a
            penalty of the integer's order is added for each entry in the sequence.
        knots
            Knots used to set up the basis. If ``None`` (default), a set of equidistant
            knots will be set up automatically, with the domain boundaries inferred from
            the minimum and maximum of the observed values.
        absorb_cons
            Whether the default identification constraint should be applied by
            reparameterization and absorbing the reparameterization matrix into the
            basis and penalty matrices for computational efficiency. If ``False``, the
            basis is unconstrained, if ``True`` it receives a sum to zero constrained.
            Also see :meth:`.Basis.constrain`.
        diagonal_penalty
            Whether the penalty matrix associated with this term should be
            reparameterized into a diagonal matrix. In this case, the basis matrix is
            reparameterized accordingly. This can be beneficial for posterior geometry,
            which is why it is the default. Also see :meth:`.Basis.diagonalize_penalty`.
        scale_penalty
            Whether to use design-aware penalty scaling. Also see
            :meth:`.Basis.scale_penalty`.
        basis_name
            Function-name for the basis matrix. If ``"B"``, and the basis is a function
            of the variable ``"x"``, the full name of the :class:`.Basis` object will be
            ``"B(x)"``. Names are made unique by appending a counter if necessary.

        Notes
        -----

        This native JAX basis uses ``use_callback=False`` and ``cache_basis=True``.
        See :class:`.Basis` for details.

        The basis and penalty are constructed natively in JAX.

        References
        ----------

        - Wood, S.N. (2017) Generalized Additive Models: An Introduction with R (2nd
          edition). Chapman and Hall/CRC.
        - R package mgcv https://cran.r-project.org/web/packages/mgcv/index.html

        Examples
        ---------
        >>> import liesel_gam as gam
        >>> df = gam.demo_data(n=100)
        >>> registry = gam.PandasRegistry(df)
        >>> bb = gam.BasisBuilder(registry)
        >>> bb.bs("x_nonlin", k=20)
        Basis(name="B(x_nonlin)")
        """
        if not isinstance(penalty_order, int):
            for order in penalty_order:
                _validate_penalty_order(order)
            raise ValueError(
                "Multiple B-spline penalties are not supported by the current "
                "liesel-gam public API."
            )
        _validate_penalty_order(penalty_order)

        x_var, x_array = self._get_var_and_value(x)
        smooth = smoothcon.bspline(
            x_array,
            k=k,
            degree=basis_degree,
            penalty_order=penalty_order,
            knots=knots,
        )
        return self._native_basis(
            x_var,
            xname=x_var.name,
            smooth=smooth,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name=basis_name,
            approximation=approximation,
        )

    def cp(
        self,
        x: str | lsl.Var,
        *,
        k: int,
        basis_degree: int = 3,
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
        approximation: bool | ApproximationSpec | None = None,
    ) -> Basis:
        """
        Cyclic P-spline basis and penalty matrix.

        Parameters
        ----------
        x
            Name of input variable.
        k
            Number of (unconstrained) bases.
        basis_degree
            Degree of the polynomials used in the B-spline basis function. Default is 3
            for cubic B-splines.
        penalty_order
            Order of the penalty.
        knots
            Knots used to set up the basis. If ``None`` (default), a set of equidistant
            knots will be set up automatically, with the domain boundaries inferred from
            the minimum and maximum of the observed values. The number of knots must be
            ``k + basis_degree + 1``, and for the observed data, it must be true that
            ``knots[basis_degree] < min(x)`` and ``max(x) < knots[-basis_degree]``.
        absorb_cons
            Whether the default identification constraint should be applied by
            reparameterization and absorbing the reparameterization matrix into the
            basis and penalty matrices for computational efficiency. If ``False``, the
            basis is unconstrained, if ``True`` it receives a sum to zero constrained.
            Also see :meth:`.Basis.constrain`.
        diagonal_penalty
            Whether the penalty matrix associated with this term should be
            reparameterized into a diagonal matrix. In this case, the basis matrix is
            reparameterized accordingly. This can be beneficial for posterior geometry,
            which is why it is the default. Also see :meth:`.Basis.diagonalize_penalty`.
        scale_penalty
            Whether to use design-aware penalty scaling. Also see
            :meth:`.Basis.scale_penalty`.
        basis_name
            Function-name for the basis matrix. If ``"B"``, and the basis is a function
            of the variable ``"x"``, the full name of the :class:`.Basis` object will be
            ``"B(x)"``. Names are made unique by appending a counter if necessary.

        Notes
        -----

        This native JAX basis uses ``use_callback=False`` and ``cache_basis=True``.
        See :class:`.Basis` for details.

        The basis and penalty are constructed natively in JAX. The mgcv
        documentation describes the corresponding mathematical smooth family.

        References
        ----------
        - Lang, S., & Brezger, A. (2004). Bayesian P-splines. Journal of Computational
          and Graphical Statistics, 13(1), 183–212.
          https://doi.org/10.1198/1061860043010
        - Wood, S.N. (2017) Generalized Additive Models: An Introduction with R (2nd
          edition). Chapman and Hall/CRC.
        - R package mgcv https://cran.r-project.org/web/packages/mgcv/index.html

        Examples
        --------
        >>> import liesel_gam as gam
        >>> df = gam.demo_data(n=100)
        >>> registry = gam.PandasRegistry(df)
        >>> bb = gam.BasisBuilder(registry)
        >>> bb.cp("x_nonlin", k=20)
        Basis(name="B(x_nonlin)")
        """
        _validate_penalty_order(penalty_order)
        x_var, x_array = self._get_var_and_value(x)
        smooth = smoothcon.cyclic_pspline(
            x_array,
            k=k,
            degree=basis_degree,
            penalty_order=penalty_order,
            knots=knots,
        )
        return self._native_basis(
            x_var,
            xname=x_var.name,
            smooth=smooth,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name=basis_name,
            approximation=approximation,
        )

    def _s(
        self,
        *x: str | lsl.Var,
        k: int,
        bs: BasisTypes,
        m: str = "NA",
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
        approximation: bool | ApproximationSpec | None = None,
    ) -> Basis:
        _validate_bs(bs)
        if not x:
            raise ValueError("At least one covariate is required.")
        obs_vars = [self._get_var_and_value(item)[0] for item in x]
        obs_names = [variable.name for variable in obs_vars]
        xname = ",".join(obs_names)
        if len(obs_vars) > 1:
            xvar: lsl.Calc | lsl.TransientCalc | lsl.Var = self._get_matrix(*x)
            values = jnp.column_stack([variable.value for variable in obs_vars])
        else:
            xvar = obs_vars[0]
            values = jnp.asarray(obs_vars[0].value)
        if knots is not None and len(obs_vars) > 1:
            raise ValueError("Multidimensional custom knots are not supported.")

        numbers = [
            float(value) for value in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", str(m))
        ]
        if bs in ("tp", "ts"):
            order = int(numbers[0]) if numbers else 0
            smooth = smoothcon.thin_plate(
                values,
                k=k,
                penalty_order=order,
                knots=knots,
                shrinkage=bs == "ts",
            )
        elif bs == "gp":
            kernel_codes = {
                1: "spherical",
                2: "power_exponential",
                3: "matern1.5",
                4: "matern2.5",
                5: "matern3.5",
            }
            code = int(numbers[0]) if numbers else 3
            smooth = smoothcon.gaussian_process(
                values,
                k=k,
                kernel_name=kernel_codes[abs(code)],
                linear_trend=code >= 0,
                range_=numbers[1] if len(numbers) > 1 and numbers[1] > 0 else None,
                power=numbers[2] if len(numbers) > 2 else 1.0,
                knots=knots,
            )
        elif len(obs_vars) != 1:
            raise ValueError(f"The {bs!r} basis only supports one covariate.")
        elif bs == "cr":
            smooth = smoothcon.cubic_regression(values, k=k, knots=knots)
        elif bs == "cs":
            smooth = smoothcon.cubic_regression(
                values, k=k, knots=knots, shrinkage=True
            )
        elif bs == "cc":
            smooth = smoothcon.cyclic_cubic(values, k=k, knots=knots)
        elif bs in ("ps", "bs", "cp"):
            degree = int(numbers[0]) + 1 if numbers else 3
            penalty_order = int(numbers[1]) if len(numbers) > 1 else 2
            constructor = {
                "ps": smoothcon.pspline,
                "bs": smoothcon.bspline,
                "cp": smoothcon.cyclic_pspline,
            }[bs]
            smooth = constructor(
                values,
                k=k,
                degree=degree,
                penalty_order=penalty_order,
                knots=knots,
            )
        else:
            raise ValueError(f"Unsupported native smooth family {bs!r}.")

        return self._native_basis(
            xvar,
            xname=xname,
            smooth=smooth,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name=basis_name,
            approximation=approximation,
            approximation_eligible=len(obs_vars) == 1,
            input_name=xname if all(isinstance(item, str) for item in x) else xvar.name,
        )

    def tp(
        self,
        *x: str | lsl.Var,
        k: int,
        penalty_order: int | None = None,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
        remove_null_space_completely: bool = False,
        approximation: bool | ApproximationSpec | None = None,
    ) -> Basis:
        """
        Thin plate spline basis and penalty matrix.

        Parameters
        ----------
        *x
            Names of input variables (one or more).
        k
            Number of (unconstrained) bases.
        penalty_order
            Order of the penalty. Quote from mgcv: "The default is to set this to the
            smallest value satisfying ``2*penalty_order > d+1`` where ``d`` is the
            number of covariates of the term."
        knots
            Knots used to set up the basis. If ``None`` (default), a set knots will be
            set up automatically.
        absorb_cons
            Whether the default identification constraint should be applied by
            reparameterization and absorbing the reparameterization matrix into the
            basis and penalty matrices for computational efficiency. If ``False``, the
            basis is unconstrained, if ``True`` it receives a sum to zero constrained.
            Also see :meth:`.Basis.constrain`.
        diagonal_penalty
            Whether the penalty matrix associated with this term should be
            reparameterized into a diagonal matrix. In this case, the basis matrix is
            reparameterized accordingly. This can be beneficial for posterior geometry,
            which is why it is the default. Also see :meth:`.Basis.diagonalize_penalty`.
        scale_penalty
            Whether to use design-aware penalty scaling. Also see
            :meth:`.Basis.scale_penalty`.
        basis_name
            Function-name for the basis matrix. If ``"B"``, and the basis is a function
            of the variable ``"x"``, the full name of the :class:`.Basis` object will be
            ``"B(x)"``. Names are made unique by appending a counter if necessary.
        remove_null_space_completely
            If ``True``, the unpenalized part of the smooth, corresponding to the null
            space of the penalty matrix, is removed completely.

        Notes
        -----

        This native JAX basis uses ``use_callback=False`` and ``cache_basis=True``.
        See :class:`.Basis` for details.

        The basis and penalty are constructed natively in JAX. The mgcv
        documentation describes the corresponding mathematical smooth family.

        References
        ----------
        - Wood, S.N. (2003) Thin-plate regression splines. Journal of the Royal
          Statistical Society (B) 65(1):95-114.
        - Wood, S.N. (2017) Generalized Additive Models: An Introduction with R (2nd
          edition). Chapman and Hall/CRC.
        - R package mgcv https://cran.r-project.org/web/packages/mgcv/index.html

        Examples
        --------
        >>> import liesel_gam as gam
        >>> df = gam.demo_data(n=100)
        >>> registry = gam.PandasRegistry(df)
        >>> bb = gam.BasisBuilder(registry)
        >>> bb.tp("x_nonlin", k=20)
        Basis(name="B(x_nonlin)")
        """
        if penalty_order is not None:
            _validate_penalty_order(penalty_order)
        if not x:
            raise ValueError("At least one covariate is required.")
        obs_vars = [self._get_var_and_value(item)[0] for item in x]
        xname = ",".join(variable.name for variable in obs_vars)
        if len(obs_vars) > 1:
            xvar: lsl.Calc | lsl.TransientCalc | lsl.Var = self._get_matrix(*x)
            values = jnp.column_stack([variable.value for variable in obs_vars])
            if knots is not None:
                raise ValueError("Multidimensional custom knots are not supported.")
        else:
            xvar = obs_vars[0]
            values = jnp.asarray(obs_vars[0].value)
        smooth = smoothcon.thin_plate(
            values,
            k=k,
            penalty_order=penalty_order or 0,
            knots=knots,
            remove_null_space=remove_null_space_completely,
        )
        return self._native_basis(
            xvar,
            xname=xname,
            smooth=smooth,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name=basis_name,
            skip_constraint=remove_null_space_completely,
            approximation=approximation,
            approximation_eligible=len(obs_vars) == 1,
            input_name=xname if all(isinstance(item, str) for item in x) else xvar.name,
        )

    def ts(
        self,
        *x: str | lsl.Var,
        k: int,
        penalty_order: int | None = None,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
        approximation: bool | ApproximationSpec | None = None,
    ) -> Basis:
        """
        Thin plate spline basis and penalty matrix with null space penalty.

        Parameters
        ----------
        *x
            Names of input variables (one or more).
        k
            Number of (unconstrained) bases.
        penalty_order
            Order of the penalty. Quote from mgcv: "The default is to set this to the
            smallest value satisfying ``2*penalty_order > d+1`` where ``d`` is the
            number of covariates of the term."
        knots
            Knots used to set up the basis. If ``None`` (default), a set knots will be
            set up automatically.
        absorb_cons
            Whether the default identification constraint should be applied by
            reparameterization and absorbing the reparameterization matrix into the
            basis and penalty matrices for computational efficiency. If ``False``, the
            basis is unconstrained, if ``True`` it receives a sum to zero constrained.
            Also see :meth:`.Basis.constrain`.
        diagonal_penalty
            Whether the penalty matrix associated with this term should be
            reparameterized into a diagonal matrix. In this case, the basis matrix is
            reparameterized accordingly. This can be beneficial for posterior geometry,
            which is why it is the default. Also see :meth:`.Basis.diagonalize_penalty`.
        scale_penalty
            Whether to use design-aware penalty scaling. Also see
            :meth:`.Basis.scale_penalty`.
        basis_name
            Function-name for the basis matrix. If ``"B"``, and the basis is a function
            of the variable ``"x"``, the full name of the :class:`.Basis` object will be
            ``"B(x)"``. Names are made unique by appending a counter if necessary.

        Notes
        -----

        This native JAX basis uses ``use_callback=False`` and ``cache_basis=True``.
        See :class:`.Basis` for details.

        The basis and penalty are constructed natively in JAX. The mgcv
        documentation describes the corresponding mathematical smooth family.

        References
        ----------
        - Wood, S.N. (2003) Thin-plate regression splines. Journal of the Royal
          Statistical Society (B) 65(1):95-114.
        - Wood, S.N. (2017) Generalized Additive Models: An Introduction with R (2nd
          edition). Chapman and Hall/CRC.
        - R package mgcv https://cran.r-project.org/web/packages/mgcv/index.html

        Examples
        --------
        >>> import liesel_gam as gam
        >>> df = gam.demo_data(n=100)
        >>> registry = gam.PandasRegistry(df)
        >>> bb = gam.BasisBuilder(registry)
        >>> bb.ts("x_nonlin", k=20)
        Basis(name="B(x_nonlin)")
        """
        if penalty_order is None:
            m_str = "NA"
        else:
            _validate_penalty_order(penalty_order)
            m_str = f"c({penalty_order})"

        basis = self._s(
            *x,
            k=k,
            bs="ts",
            m=m_str,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name=basis_name,
            approximation=approximation,
        )
        return basis

    def kriging(
        self,
        *x: str | lsl.Var,
        k: int,
        kernel_name: Literal[
            "spherical",
            "power_exponential",
            "matern1.5",
            "matern2.5",
            "matern3.5",
        ] = "matern1.5",
        linear_trend: bool = True,
        range: float | None = None,
        power_exponential_power: float = 1.0,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
        approximation: bool | ApproximationSpec | None = None,
    ) -> Basis:
        """
        Gaussian process models with a fixed range parameter in a
        basis-penalty-parameterization, often referred to as Kriging.

        Parameters
        ----------
        *x
            Name of input variables (one or more).
        k
            Number of (unconstrained) bases.
        kernel_name
            Selects the kernel / covariance function to use.
        linear_trend
            Whether to include or remove a linear trend.
        range
            Range parameter. If ``None``, estimated as in Kamman & Wand (2003).
        power_exponential_power
            Power for the power exponential kernel.
        absorb_cons
            Whether the default identification constraint should be applied by
            reparameterization and absorbing the reparameterization matrix into the
            basis and penalty matrices for computational efficiency. If ``False``, the
            basis is unconstrained, if ``True`` it receives a sum to zero constrained.
            Also see :meth:`.Basis.constrain`.
        diagonal_penalty
            Whether the penalty matrix associated with this term should be
            reparameterized into a diagonal matrix. In this case, the basis matrix is
            reparameterized accordingly. This can be beneficial for posterior geometry,
            which is why it is the default. Also see :meth:`.Basis.diagonalize_penalty`.
        scale_penalty
            Whether to use design-aware penalty scaling. Also see
            :meth:`.Basis.scale_penalty`.
        basis_name
            Function-name for the basis matrix. If ``"B"``, and the basis is a function
            of the variable ``"x"``, the full name of the :class:`.Basis` object will be
            ``"B(x)"``. Names are made unique by appending a counter if necessary.

        Notes
        -----

        This native JAX basis uses ``use_callback=False`` and ``cache_basis=True``.
        See :class:`.Basis` for details.

        The basis and penalty are constructed natively in JAX. The mgcv
        documentation describes the corresponding mathematical smooth family.

        References
        ----------
        - Kammann, E. E. and M.P. Wand (2003) Geoadditive Models. Applied Statistics
          52(1):1-18.
        - Wood, S.N. (2017) Generalized Additive Models: An Introduction with R (2nd
          edition). Chapman and Hall/CRC.
        - R package mgcv https://cran.r-project.org/web/packages/mgcv/index.html

        Examples
        --------
        >>> import liesel_gam as gam
        >>> df = gam.demo_data(n=100)
        >>> registry = gam.PandasRegistry(df)
        >>> bb = gam.BasisBuilder(registry)
        >>> bb.kriging("x_nonlin", k=20)
        Basis(name="B(x_nonlin)")

        """
        m_kernel_dict = {
            "spherical": 1,
            "power_exponential": 2,
            "matern1.5": 3,
            "matern2.5": 4,
            "matern3.5": 5,
        }
        m_linear = 1.0 if linear_trend else -1.0

        m_args = []
        m_kernel = str(int(m_linear * m_kernel_dict[kernel_name]))
        m_args.append(m_kernel)
        if range:
            m_range = str(range)
            m_args.append(m_range)
        if power_exponential_power:
            if not range:
                m_args.append(str(-1.0))
            if not 0.0 < power_exponential_power <= 2.0:
                raise ValueError(
                    "'power_exponential_power' must be in (0, 2.0], "
                    f"got {power_exponential_power}"
                )
            m_args.append(str(power_exponential_power))

        m_str = "c(" + ", ".join(m_args) + ")"

        basis = self._s(
            *x,
            k=k,
            bs="gp",
            m=m_str,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name=basis_name,
            approximation=approximation,
        )

        return basis

    def lin(
        self,
        formula: str,
        xname: str = "",
        basis_name: str = "X",
        include_intercept: bool = False,
        context: dict[str, Any] | None = None,
    ) -> LinBasis:
        """
        Linear design matrix without penalty.

        Parameters
        ----------
        formula
            Right-hand side of a model formula, as understood by formulaic_. Most of
            formulaic's grammar_ is supported. See notes for details.
        xname
            If provided, the design matrix will be named ``{basis_name}({xname})``, for
            example ``B(x)``, is ``basis_name="B"`` and ``xname="x"``.
        basis_name
            Name of the basis variable.
        include_intercept
            Whether to include an intercept column in the basis.
        context
            Dictionary of additional Python objects that should be made available to
            formulaic when constructing the design matrix. Gets passed to
            ``formulaic.ModelSpec.get_model_matrix()``.

        Notes
        -----

        The following formulaic syntax is supported:

        - ``+`` for adding a term
        - ``a:b`` for simple interactions
        - ``a*b`` for expanding to ``a + b + a:b``
        - ``(a + b)**n`` for n-th order interactions
        - ``a / b`` for nesting
        - ``C(a, ...)`` for categorical effects
        - ``b %in% a`` for inverted nesting
        - ``{a+1}`` for quoted Python code to be executed
        - ```weird name``` backtick-strings for weird names
        - Other transformations like ``center(a)``, ``scale(a)``, or ``lag(a)``, see
          grammar_.
        - Python functions

        Not supported:

        - String literals
        - Numeric literals
        - Wildcard ``"."``
        - ``\\|`` for splitting a formula
        - ``"~"`` in formula, since this method supports only the right-hand side of a
          Wilkinson formula.
        - ``1 +``, ``0 +``, or ``-1`` in formula, since intercept addition is handled
          via the argument ``include_intercept``.

        References
        ----------

        - Python library formulaic: https://matthewwardrop.github.io/formulaic/latest/

        Examples
        --------

        Simple example:

        >>> import liesel_gam as gam
        >>> df = gam.demo_data(n=100)
        >>> registry = gam.PandasRegistry(df)
        >>> bb = gam.BasisBuilder(registry)
        >>> bb.lin("x_lin + x_nonlin + x_cat")
        LinBasis(name="X")

        Customized categorical encoding:

        >>> import liesel_gam as gam
        >>> df = gam.demo_data(n=100)
        >>> registry = gam.PandasRegistry(df)
        >>> bb = gam.BasisBuilder(registry)
        >>> bb.lin("x_lin + x_nonlin + C(x_cat, contr.sum)")
        LinBasis(name="X")

        Interaction:

        >>> import liesel_gam as gam
        >>> df = gam.demo_data(n=100)
        >>> registry = gam.PandasRegistry(df)
        >>> bb = gam.BasisBuilder(registry)
        >>> bb.lin("x_lin * x_cat")
        LinBasis(name="X")


        .. _formulaic: https://matthewwardrop.github.io/formulaic/latest/
        .. _grammar: https://matthewwardrop.github.io/formulaic/latest/guides/grammar/
        """
        _validate_formula(formula)
        parsed_formula = fo.Formula(formula)
        if not isinstance(parsed_formula, fo.SimpleFormula):
            raise ValueError("Structured formulas are not supported.")

        spec = fo.ModelSpec(parsed_formula, output="numpy")

        # evaluate model matrix once to get a spec with structure information
        # also necessary to populate spec with the correct information for
        # transformations like center, scale, standardize
        try:
            evaluated_spec = spec.get_model_matrix(
                self.data, context=context
            ).model_spec
            if evaluated_spec is None:
                raise RuntimeError("Formulaic did not return a model specification.")
        except Exception as e:
            raise RuntimeError(
                "Could not build model matrix. This could be caused by "
                "unsupported data dtypes like dates. Please check your input data. "
                "Also check the original error message, included above."
            ) from e
        spec = evaluated_spec

        # get column names. There may be a more efficient way to do it
        # that does not require building the model matrix a second time, but this
        # works robustly for now: we take the names that formulaic creates
        column_names = list(
            fo.ModelSpec(parsed_formula, output="pandas")
            .get_model_matrix(self.data, context=context)
            .columns
        )
        if not include_intercept:
            column_names = column_names[1:]

        required_set = {str(var) for var in spec.required_variables}
        # Formulaic 1.2 does not expose the data variable wrapped by Patsy's Q()
        # through ``required_variables``. It is nevertheless a genuine input node.
        quoted = re.findall(r"\bQ\(\s*(['\"])(.*?)\1\s*\)", formula)
        required_set.update(name for _, name in quoted)
        required = sorted(required_set)
        df_subset = self.data.loc[:, required]
        df_colnames = df_subset.columns

        variables = {}

        mappings = {}
        for col in df_colnames:
            result = self.registry.get_obs_and_mapping(col)
            variables[col] = result.var

            if result.mapping is not None:
                self.mappings[col] = result.mapping
                mappings[col] = result.mapping

        xvar = lsl.TransientCalc(  # for memory-efficiency
            lambda *args: jnp.vstack(args).T,
            *list(variables.values()),
            _name=self.names.create(f"[{xname}]") if xname else xname,
        )

        def basis_fn(x):
            df = pd.DataFrame(x, columns=df_colnames)

            # for categorical variables: convert integer representation back to
            # labels
            for col in df_colnames:
                if col in self.mappings:
                    integers = df[col].to_numpy()
                    df[col] = self.mappings[col].integers_to_labels(integers)

            basis = np.asarray(spec.get_model_matrix(df, context=context))
            if not include_intercept:
                basis = basis[:, 1:]
            return jnp.asarray(basis, dtype=float)

        if xname:
            bname = self.names.create(basis_name + "(" + xname + ")")
        else:
            bname = self.names.create(basis_name)

        basis = LinBasis(
            xvar,
            basis_fn=basis_fn,
            use_callback=True,
            cache_basis=True,
            name=bname,
            penalty=None,
        )
        basis._input_name = xname or xvar.name

        basis.model_spec = spec
        basis.mappings = mappings
        basis.column_names = column_names

        return basis

    def ri(
        self,
        cluster: str,
        basis_name: str = "B",
        penalty: ArrayLike | None = None,
    ) -> Basis:
        """
        Random intercept basis.

        Parameters
        ----------
        cluster
            Name of the cluster variable.
        basis_name
            Name of the basis variable.
        penalty
            Custom penalty matrix to use. Default is an iid penalty.

        Notes
        ------
        If the penalty is iid, then each column of the basis consists only of binary
        (0/1) entries, and each row has only one non-zero entry. In this case it is not
        necessary to store the full matrix in memory and evaluate the term as a dot
        product ``basis @ coef``.

        Instead, we can simply store a 1d array of indices, identifying the nonzero
        column for each row of the basis matrix, and use this index to access the
        corresponding coefficient. This scenario is common for independent random
        intercepts.

        This method returns such a sparse representation of the random intercept
        basis if ``penalty=None``.

        Examples
        --------
        >>> import liesel_gam as gam
        >>> df = gam.demo_data(n=100)
        >>> registry = gam.PandasRegistry(df)
        >>> bb = gam.BasisBuilder(registry)
        >>> bb.ri("x_cat")
        Basis(name="B(x_cat)")

        """
        if penalty is not None:
            penalty = jnp.asarray(penalty)
        result = self.registry.get_obs_and_mapping(cluster)

        if not result.is_categorical:
            raise TypeError(f"{cluster=} must be categorical.")

        if result.mapping is not None:
            self.mappings[cluster] = result.mapping

        basis = Basis(
            value=result.var,
            basis_fn=lambda x: x,
            name=self.names.create(basis_name + "(" + cluster + ")"),
            use_callback=False,
            cache_basis=False,
            penalty=jnp.asarray(penalty) if penalty is not None else penalty,
        )

        return basis

    def mrf(
        self,
        x: str,
        k: int = -1,
        polys: dict[str, ArrayLike] | None = None,
        nb: Mapping[str, ArrayLike | list[str] | list[int]] | None = None,
        penalty: ArrayLike | None = None,
        penalty_labels: Sequence[str] | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
    ) -> MRFBasis:
        """
        Gaussian Markov random field basis and penalty.

        The preferred way to initialize these is by supplying ``polys``, because this
        enables plotting via :func:`.plot_regions`.

        Parameters
        ----------
        x
            Name of the region variable.
        k
            If ``-1``, this is a "full-rank" (up to identifiability constraint) Markov
            random field. If ``k`` is an integer smaller than the number of unique
            regions, a low-rank field will be returned, see Wood (2017), Sections 5.8.1
            and 5.4.2.
        polys
            Dictionary of arrays. The keys of the dict are the region labels. The
            corresponding values define the region by defining polygons. The
            neighborhood structure can be inferred from this polygon information.
        nb
            Dictionary of array. The keys of the dict are the region labels. The
            corresponding values indicate the neighbors of the region. If the values are
            lists or arrays of strings, the values are the labels of the neighbors. If
            they are lists or arrays of integers, the values are the indices of the
            neighbors. Indices correspond to regions based on an alphabetical ordering
            of regions.
        penalty
            If a penalty is supplied explicitly, it takes precedence over a potential
            penalty derived from both nb and polys.
        penalty_labels
            If a penalty is supplied explicitly, labels must also be specified. The
            labels create the association between penalty columns and region labels. The
            values of this sequence should be the string labels of unique regions in
            ``x``.
        absorb_cons
            Whether the default identification constraint should be applied by
            reparameterization and absorbing the reparameterization matrix into the
            basis and penalty matrices for computational efficiency. If ``False``, the
            basis is unconstrained, if ``True`` it receives a sum to zero constrained.
            Also see :meth:`.Basis.constrain`.
        diagonal_penalty
            Whether the penalty matrix associated with this term should be
            reparameterized into a diagonal matrix. In this case, the basis matrix is
            reparameterized accordingly. This can be beneficial for posterior geometry,
            which is why it is the default. Also see :meth:`.Basis.diagonalize_penalty`.
        scale_penalty
            Whether to use design-aware penalty scaling. Also see
            :meth:`.Basis.scale_penalty`.
        basis_name
            Function-name for the basis matrix. If ``"B"``, and the basis is a function
            of the variable ``"x"``, the full name of the :class:`.Basis` object will be
            ``"B(x)"``. Names are made unique by appending a counter if necessary.

        See Also
        --------
        .plot_regions : Plots MCMC results on a map of the regions.
        .plot_polys : Plots a map based on polygons.
        .plot_forest : Plots regions with uncertainty in a forest plot.

        Notes
        -----

        This native JAX basis uses ``use_callback=False`` and ``cache_basis=True``.
        See :class:`.Basis` for details.

        The basis and penalty are constructed natively in JAX. The mgcv
        documentation describes the corresponding mathematical smooth family.

        Returns
        -------

            Comments on the :class:`.MRFSpec` attached to the returned
            :class:`.MRFBasis` variable:

            - If either polys or nb are supplied, the returned MRFSpec will contain
              nb.
            - If only a penalty matrix is supplied, the returned MRFSpec will *not*
              contain nb.
            - Returning the label order only makes sense if the basis is *not*
              reparameterized, because only then we have a clear correspondence of
              parameters to labels. If the basis is reparameterized, with
              ``absorb_cons=True`` or of low rank with ``k ≠ -1``, there is no such
              correspondence in a clear way, so the label order is None.


        Examples
        --------
        >>> import liesel_gam as gam
        >>> df = gam.demo_data(n=100)
        >>> print(df.x_cat.unique().tolist())
        ['a', 'b', 'c']
        >>> registry = gam.PandasRegistry(df)
        >>> bb = gam.BasisBuilder(registry)
        >>> nb = {"a": ["b", "c"], "b": ["a"], "c": ["a"]}
        >>> bb.mrf("x_cat", nb=nb)
        MRFBasis(name="B(x_cat)")

        To inspect the penalty and the dummy-coded basis matrix:

        >>> basis = bb.mrf(
        ...     "x_cat",
        ...     nb=nb,
        ...     absorb_cons=False,
        ...     diagonal_penalty=False,
        ...     scale_penalty=False,
        ... )

        >>> basis.penalty.value
        Array([[ 2., -1., -1.],
               [-1.,  1.,  0.],
               [-1.,  0.,  1.]], dtype=float32)

        >>> basis.value[:5, ...]
        Array([[1., 0., 0.],
               [0., 1., 0.],
               [1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]], dtype=float32)

        >>> basis.mrf_spec.ordered_labels
        ['a', 'b', 'c']


        References
        ----------
        - Wood, S.N. (2017) Generalized Additive Models: An Introduction with R (2nd
          edition). Chapman and Hall/CRC.
        - R package mgcv https://cran.r-project.org/web/packages/mgcv/index.html

        """

        if not isinstance(k, int):
            raise TypeError(f"'k' must be int, got {type(k)}.")
        if k < -1:
            raise ValueError(f"'k' cannot be smaller than -1, got {k=}.")

        if polys is None and nb is None and penalty is None:
            raise ValueError("At least one of polys, nb, or penalty must be provided.")

        var, mapping = self.registry.get_categorical_obs(x)
        self.mappings[x] = mapping

        # mgcv orders factor levels lexicographically. Keep that established
        # public behaviour even when pandas carries an explicit category order.
        ordered_labels = sorted(mapping.labels_to_integers_map)
        labels = set(ordered_labels)
        mapping_labels = [
            mapping.integers_to_labels_map[i]
            for i in range(len(mapping.integers_to_labels_map))
        ]
        code_to_region = np.asarray(
            [ordered_labels.index(label) for label in mapping_labels], dtype=np.int32
        )

        if penalty is not None:
            if penalty_labels is None:
                raise ValueError(
                    "If 'penalty' is supplied, 'penalty_labels' must also be supplied."
                )
            if len(penalty_labels) != len(labels):
                raise ValueError(
                    f"Variable {x} has {len(labels)} unique entries, but "
                    f"'penalty_labels' has {len(penalty_labels)}. Both must match."
                )

        if polys is not None and labels != set(polys):
            raise ValueError("Names in 'polys' must correspond to the levels of 'x'.")
        if nb is not None and labels != set(nb):
            raise ValueError("Names in 'nb' must correspond to the levels of 'x'.")

        nb_out = (
            # Numeric neighbor entries index the insertion order of the mapping,
            # exactly like positions in mgcv's named R list.
            smoothcon.normalize_neighbors(nb, ordered_labels, list(nb))
            if nb is not None
            else None
        )
        if nb_out is None and polys is not None and penalty is None:
            nb_out = smoothcon.infer_neighbors_from_polygons(polys)

        if penalty is not None:
            penalty = np.asarray(penalty)
            if penalty.ndim != 2 or penalty.shape[0] != penalty.shape[1]:
                raise ValueError(f"Penalty must be square, got {np.shape(penalty)=}")
            if penalty.shape[1] != len(labels):
                raise ValueError(
                    "Dimensions of 'penalty' must correspond to the levels of 'x'."
                )
            assert penalty_labels is not None
            if set(penalty_labels) != labels:
                raise ValueError("'penalty_labels' must match the levels of 'x'.")
            pen_rank = np.linalg.matrix_rank(penalty)
            pen_dim = penalty.shape[-1]
            if (pen_dim - pen_rank) != 1:
                logger.warning(
                    f"Supplied penalty has dimension {penalty.shape} and rank "
                    f"{pen_rank}. The expected rank deficiency is 1. "
                    "This may indicate a problem. There might be disconnected sets "
                    "of regions in the data represented by this penalty. "
                    "In this case, you probably need more elaborate constraints "
                    "than the ones provided here. You might consider splitting the "
                    "disconnected regions into several mrf terms. "
                    "Otherwise, please only continue if you are certain that you "
                    "know what is happening."
                )
            indices = [list(penalty_labels).index(label) for label in ordered_labels]
            penalty_array = penalty[np.ix_(indices, indices)]
        else:
            assert nb_out is not None
            penalty_array = smoothcon.build_mrf_penalty(nb_out, ordered_labels)

        if nb is not None and penalty is not None:
            logger.warning(
                "Both 'nb' and 'penalty' were supplied. 'penalty' will be used to "
                "setup this basis."
            )
        if polys is not None and penalty is not None:
            logger.warning(
                "Both 'polys' and 'penalty' were supplied. 'penalty' will be used "
                "to setup this basis."
            )

        region_codes = code_to_region[np.asarray(var.value, dtype=np.int32)]
        smooth = smoothcon.mrf(region_codes, penalty=penalty_array, k=k)
        native_basis_fn = smooth.basis
        code_to_region_jax = jnp.asarray(code_to_region)

        def basis_fn(values: Array) -> Array:
            codes = jnp.asarray(values, dtype=jnp.int32)
            return native_basis_fn(code_to_region_jax[codes])

        basis = MRFBasis(
            value=var,
            basis_fn=basis_fn,
            name=self.names.create(basis_name + "(" + x + ")"),
            cache_basis=True,
            use_callback=False,
            penalty=smooth.penalty,
        )
        basis._penalty_rank = smooth.rank
        if scale_penalty:
            basis.scale_penalty()
        if absorb_cons:
            basis.constrain("sumzero_term")
        if diagonal_penalty:
            basis.diagonalize_penalty()

        label_order: list[str] | None = ordered_labels
        if absorb_cons or diagonal_penalty or (k != -1 and k < len(labels)):
            label_order = None
        basis.mrf_spec = MRFSpec(mapping, nb_out, label_order, polys)

        return basis
