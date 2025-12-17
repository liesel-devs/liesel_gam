from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from math import ceil
from typing import Any, Literal, get_args

import formulaic as fo
import jax
import jax.numpy as jnp
import liesel.model as lsl
import numpy as np
import pandas as pd

try:
    # readthedocs safeguard: R is not installed in the readthedocs build environment
    import smoothcon as scon
    from ryp import r, to_py, to_r
except RuntimeError as e:
    import os

    on_rtd = os.environ.get("READTHEDOCS", "False") == "True"
    if on_rtd:
        scon = None
        r = None
        to_py = None
        to_r = None
        pass
    else:
        raise e

from .basis import Basis, LinBasis, MRFBasis, MRFSpec
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

    See Also
    --------

    .TermBuilder : Initializes structured additive terms.
    .Basis : Basic basis class.
    .LinBasis : Specialized basis for linear effects.
    .MRFBasis : Specialized basis for Gaussian Markov random fields.

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
        self, registry: PandasRegistry, names: NameManager | None = None
    ) -> None:
        self.registry = registry
        self.mappings: dict[str, CategoryMapping] = {}
        self.names = NameManager() if names is None else names

    def __repr__(self) -> str:
        return f"{type(self).__name__}(data_shape={self.registry.data.shape})"

    @property
    def data(self) -> pd.DataFrame:
        """The dataframe wrapped by this builder's registry."""
        return self.registry.data

    def basis(
        self,
        *x: str,
        basis_fn: Callable[[Array], Array],
        use_callback: bool = True,
        cache_basis: bool = True,
        penalty: ArrayLike | lsl.Value | None = None,
        basis_name: str = "B",
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
        for x_name in x:
            x_var = self.registry.get_numeric_obs(x_name)
            x_vars.append(x_var)

        Xname = self.registry.prefix + ",".join(x)

        Xvar = lsl.TransientCalc(
            lambda *x: jnp.column_stack(x),
            *x_vars,
            _name=Xname,
        )

        basis = Basis(
            value=Xvar,
            basis_fn=basis_fn,
            name=self.names.create(basis_name + "(" + Xname + ")"),
            use_callback=use_callback,
            cache_basis=cache_basis,
            penalty=penalty,
        )

        return basis

    def ps(
        self,
        x: str,
        *,
        k: int,
        basis_degree: int = 3,
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
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
            Whether the penalty matrix should be scaled such that its infinity norm is
            one. This can improve numerical stability, which is why it is done by
            default. Also see :meth:`.Basis.scale_penalty`.
        basis_name
            Function-name for the basis matrix. If ``"B"``, and the basis is a function
            of the variable ``"x"``, the full name of the :class:`.Basis` object will be
            ``"B(x)"``. Names are made unique by appending a counter if necessary.

        Notes
        -----

        This basis is initialized with ``use_callback=True`` and ``cache_basis=True``.
        See :class:`.Basis` for details.

        This method internally calls the R package mgcv to set up the basis and penalty.

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
        if knots is not None:
            knots = np.asarray(knots)

        spec = f"s({x}, bs='ps', k={k}, m=c({basis_degree - 1}, {penalty_order}))"
        x_array = jnp.asarray(self.registry.data[x].to_numpy())
        smooth = scon.SmoothCon(
            spec,
            data={x: x_array},
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
        )

        x_var = self.registry.get_numeric_obs(x)
        basis = Basis(
            x_var,
            name=self.names.create(basis_name + "(" + x_var.name + ")"),
            basis_fn=lambda x_: jnp.asarray(smooth.predict({x: x_})),
            penalty=smooth.penalty,
            use_callback=True,
            cache_basis=True,
        )

        if absorb_cons:
            basis._constraint = "absorbed_via_mgcv"
        return basis

    def cr(
        self,
        x: str,
        *,
        k: int,
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
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
            Whether the penalty matrix should be scaled such that its infinity norm is
            one. This can improve numerical stability, which is why it is done by
            default. Also see :meth:`.Basis.scale_penalty`.
        basis_name
            Function-name for the basis matrix. If ``"B"``, and the basis is a function
            of the variable ``"x"``, the full name of the :class:`.Basis` object will be
            ``"B(x)"``. Names are made unique by appending a counter if necessary.

        See Also
        --------

        .cs : Cubic regression splines with additinal shrinkage on the null space.

        Notes
        -----

        This basis is initialized with ``use_callback=True`` and ``cache_basis=True``.
        See :class:`.Basis` for details.

        This method internally calls the R package mgcv to set up the basis and penalty.
        The mgcv documentation provides further details.


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
        if knots is not None:
            knots = np.asarray(knots)
        spec = f"s({x}, bs='cr', k={k}, m=c({penalty_order}))"
        x_array = jnp.asarray(self.registry.data[x].to_numpy())
        smooth = scon.SmoothCon(
            spec,
            data={x: x_array},
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
        )

        x_var = self.registry.get_numeric_obs(x)
        basis = Basis(
            x_var,
            name=self.names.create(basis_name + "(" + x_var.name + ")"),
            basis_fn=lambda x_: jnp.asarray(smooth.predict({x: x_})),
            penalty=smooth.penalty,
            use_callback=True,
            cache_basis=True,
        )

        if absorb_cons:
            basis._constraint = "absorbed_via_mgcv"
        return basis

    def cs(
        self,
        x: str,
        *,
        k: int,
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
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
            Whether the penalty matrix should be scaled such that its infinity norm is
            one. This can improve numerical stability, which is why it is done by
            default. Also see :meth:`.Basis.scale_penalty`.
        basis_name
            Function-name for the basis matrix. If ``"B"``, and the basis is a function
            of the variable ``"x"``, the full name of the :class:`.Basis` object will be
            ``"B(x)"``. Names are made unique by appending a counter if necessary.

        Notes
        -----

        This basis is initialized with ``use_callback=True`` and ``cache_basis=True``.
        See :class:`.Basis` for details.

        This method internally calls the R package mgcv to set up the basis and penalty.
        The mgcv documentation provides further details.

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
        if knots is not None:
            knots = np.asarray(knots)
        spec = f"s({x}, bs='cs', k={k}, m=c({penalty_order}))"
        x_array = jnp.asarray(self.registry.data[x].to_numpy())
        smooth = scon.SmoothCon(
            spec,
            data={x: x_array},
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
        )

        x_var = self.registry.get_numeric_obs(x)
        basis = Basis(
            x_var,
            name=self.names.create(basis_name + "(" + x_var.name + ")"),
            basis_fn=lambda x_: jnp.asarray(smooth.predict({x: x_})),
            penalty=smooth.penalty,
            use_callback=True,
            cache_basis=True,
        )

        if absorb_cons:
            basis._constraint = "absorbed_via_mgcv"
        return basis

    def cc(
        self,
        x: str,
        *,
        k: int,
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
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
            Whether the penalty matrix should be scaled such that its infinity norm is
            one. This can improve numerical stability, which is why it is done by
            default. Also see :meth:`.Basis.scale_penalty`.
        basis_name
            Function-name for the basis matrix. If ``"B"``, and the basis is a function
            of the variable ``"x"``, the full name of the :class:`.Basis` object will be
            ``"B(x)"``. Names are made unique by appending a counter if necessary.

        Notes
        -----

        This basis is initialized with ``use_callback=True`` and ``cache_basis=True``.
        See :class:`.Basis` for details.

        Cyclicity is enforced by matching the function and its derivatives at the domain
        boundaries. This method internally calls the R package mgcv to set up the basis
        and penalty. The mgcv documentation provides further details.

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
        if knots is not None:
            knots = np.asarray(knots)
        spec = f"s({x}, bs='cc', k={k}, m=c({penalty_order}))"
        x_array = jnp.asarray(self.registry.data[x].to_numpy())
        smooth = scon.SmoothCon(
            spec,
            data={x: x_array},
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
        )

        x_var = self.registry.get_numeric_obs(x)
        basis = Basis(
            x_var,
            name=self.names.create(basis_name + "(" + x_var.name + ")"),
            basis_fn=lambda x_: jnp.asarray(smooth.predict({x: x_})),
            penalty=smooth.penalty,
            use_callback=True,
            cache_basis=True,
        )

        if absorb_cons:
            basis._constraint = "absorbed_via_mgcv"
        return basis

    def bs(
        self,
        x: str,
        *,
        k: int,
        basis_degree: int = 3,
        penalty_order: int | Sequence[int] = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
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
            Whether the penalty matrix should be scaled such that its infinity norm is
            one. This can improve numerical stability, which is why it is done by
            default. Also see :meth:`.Basis.scale_penalty`.
        basis_name
            Function-name for the basis matrix. If ``"B"``, and the basis is a function
            of the variable ``"x"``, the full name of the :class:`.Basis` object will be
            ``"B(x)"``. Names are made unique by appending a counter if necessary.

        Notes
        -----

        This basis is initialized with ``use_callback=True`` and ``cache_basis=True``.
        See :class:`.Basis` for details.

        This method internally calls the R package mgcv to set up the basis
        and penalty. The mgcv documentation provides further details.

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
        if knots is not None:
            knots = np.asarray(knots)
        if isinstance(penalty_order, int):
            _validate_penalty_order(penalty_order)
            penalty_order_seq: Sequence[str] = [str(penalty_order)]
        else:
            [_validate_penalty_order(p) for p in penalty_order]
            penalty_order_seq = [str(p) for p in penalty_order]

        spec = (
            f"s({x}, bs='bs', k={k}, "
            f"m=c({basis_degree}, {', '.join(penalty_order_seq)}))"
        )
        x_array = jnp.asarray(self.registry.data[x].to_numpy())
        smooth = scon.SmoothCon(
            spec,
            data={x: x_array},
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
        )

        x_var = self.registry.get_numeric_obs(x)
        basis = Basis(
            x_var,
            name=self.names.create(basis_name + "(" + x_var.name + ")"),
            basis_fn=lambda x_: jnp.asarray(smooth.predict({x: x_})),
            penalty=smooth.penalty,
            use_callback=True,
            cache_basis=True,
        )

        if absorb_cons:
            basis._constraint = "absorbed_via_mgcv"
        return basis

    def cp(
        self,
        x: str,
        *,
        k: int,
        basis_degree: int = 3,
        penalty_order: int = 2,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
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
            Whether the penalty matrix should be scaled such that its infinity norm is
            one. This can improve numerical stability, which is why it is done by
            default. Also see :meth:`.Basis.scale_penalty`.
        basis_name
            Function-name for the basis matrix. If ``"B"``, and the basis is a function
            of the variable ``"x"``, the full name of the :class:`.Basis` object will be
            ``"B(x)"``. Names are made unique by appending a counter if necessary.

        Notes
        -----

        This basis is initialized with ``use_callback=True`` and ``cache_basis=True``.
        See :class:`.Basis` for details.

        This method internally calls the R package mgcv to set up the basis and penalty.
        The mgcv documentation provides further details.

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
        if knots is not None:
            knots = np.asarray(knots)
        spec = f"s({x}, bs='cp', k={k}, m=c({basis_degree - 1}, {penalty_order}))"
        x_array = jnp.asarray(self.registry.data[x].to_numpy())
        smooth = scon.SmoothCon(
            spec,
            data={x: x_array},
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
        )

        x_var = self.registry.get_numeric_obs(x)
        basis = Basis(
            x_var,
            name=self.names.create(basis_name + "(" + x_var.name + ")"),
            basis_fn=lambda x_: jnp.asarray(smooth.predict({x: x_})),
            penalty=smooth.penalty,
            use_callback=True,
            cache_basis=True,
        )

        if absorb_cons:
            basis._constraint = "absorbed_via_mgcv"
        return basis

    def _s(
        self,
        *x: str,
        k: int,
        bs: BasisTypes,
        m: str = "NA",
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
    ) -> Basis:
        if knots is not None:
            knots = np.asarray(knots)
        _validate_bs(bs)
        bs_arg = f"'{bs}'"
        spec = f"s({','.join(x)}, bs={bs_arg}, k={k}, m={m})"

        obs_vars = {}
        for xname in x:
            obs_vars[xname] = self.registry.get_numeric_obs(xname)
        obs_values = {k: np.asarray(v.value) for k, v in obs_vars.items()}

        smooth = scon.SmoothCon(
            spec,
            data=pd.DataFrame.from_dict(obs_values),
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
        )

        xname = ",".join([v.name for v in obs_vars.values()])

        if len(obs_vars) > 1:
            xvar: lsl.Var | lsl.TransientCalc = (
                lsl.TransientCalc(  # for memory-efficiency
                    lambda *args: jnp.vstack(args).T,
                    *list(obs_vars.values()),
                    _name=self.names.create(xname),
                )
            )
        else:
            xvar = obs_vars[xname]

        def basis_fn(x):
            df = pd.DataFrame(x, columns=list(obs_vars))
            return jnp.asarray(smooth.predict(df))

        basis = Basis(
            xvar,
            name=self.names.create(basis_name + "(" + xname + ")"),
            basis_fn=basis_fn,
            penalty=smooth.penalty,
            use_callback=True,
            cache_basis=True,
        )
        if absorb_cons:
            basis._constraint = "absorbed_via_mgcv"
        return basis

    def tp(
        self,
        *x: str,
        k: int,
        penalty_order: int | None = None,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
        remove_null_space_completely: bool = False,
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
            Whether the penalty matrix should be scaled such that its infinity norm is
            one. This can improve numerical stability, which is why it is done by
            default. Also see :meth:`.Basis.scale_penalty`.
        basis_name
            Function-name for the basis matrix. If ``"B"``, and the basis is a function
            of the variable ``"x"``, the full name of the :class:`.Basis` object will be
            ``"B(x)"``. Names are made unique by appending a counter if necessary.
        remove_null_space_completely
            If ``True``, the unpenalized part of the smooth, corresponding to the null
            space of the penalty matrix, is removed completely.

        Notes
        -----

        This basis is initialized with ``use_callback=True`` and ``cache_basis=True``.
        See :class:`.Basis` for details.

        This method internally calls the R package mgcv to set up the basis and penalty.
        The mgcv documentation provides further details.

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
        d = len(x)
        m_args = []
        if penalty_order is None:
            penalty_order_default = ceil((d + 1) / 2)
            i = 0
            while not 2 * penalty_order_default > (d + 1) and i < 20:
                penalty_order_default += 1
                i += 1

            m_args.append(str(penalty_order_default))
        else:
            _validate_penalty_order(penalty_order)
            m_args.append(str(penalty_order))

        if remove_null_space_completely:
            m_args.append("0")
        m_str = "c(" + ", ".join(m_args) + ")"

        basis = self._s(
            *x,
            k=k,
            bs="tp",
            m=m_str,
            knots=knots,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
            basis_name=basis_name,
        )
        return basis

    def ts(
        self,
        *x: str,
        k: int,
        penalty_order: int | None = None,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        basis_name: str = "B",
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
            Whether the penalty matrix should be scaled such that its infinity norm is
            one. This can improve numerical stability, which is why it is done by
            default. Also see :meth:`.Basis.scale_penalty`.
        basis_name
            Function-name for the basis matrix. If ``"B"``, and the basis is a function
            of the variable ``"x"``, the full name of the :class:`.Basis` object will be
            ``"B(x)"``. Names are made unique by appending a counter if necessary.

        Notes
        -----

        This basis is initialized with ``use_callback=True`` and ``cache_basis=True``.
        See :class:`.Basis` for details.

        This method internally calls the R package mgcv to set up the basis and penalty.
        The mgcv documentation provides further details.

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
        d = len(x)
        m_args = []
        if not penalty_order:
            m_args.append(str(ceil((d + 1) / 2)))
        else:
            _validate_penalty_order(penalty_order)
            m_args.append(str(penalty_order))

        m_str = "c(" + ", ".join(m_args) + ")"

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
        )
        return basis

    def kriging(
        self,
        *x: str,
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
            Whether the penalty matrix should be scaled such that its infinity norm is
            one. This can improve numerical stability, which is why it is done by
            default. Also see :meth:`.Basis.scale_penalty`.
        basis_name
            Function-name for the basis matrix. If ``"B"``, and the basis is a function
            of the variable ``"x"``, the full name of the :class:`.Basis` object will be
            ``"B(x)"``. Names are made unique by appending a counter if necessary.

        Notes
        -----

        This basis is initialized with ``use_callback=True`` and ``cache_basis=True``.
        See :class:`.Basis` for details.

        This method internally calls the R package mgcv to set up the basis and penalty.
        The mgcv documentation provides further details.

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
        >>> bb.gp("x_nonlin", k=20)
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
        r"""
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
        - ``\|`` for splitting a formula
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
        spec = fo.ModelSpec(formula, output="numpy")

        # evaluate model matrix once to get a spec with structure information
        # also necessary to populate spec with the correct information for
        # transformations like center, scale, standardize
        try:
            spec = spec.get_model_matrix(self.data, context=context).model_spec
        except Exception as e:
            raise RuntimeError(
                "Could not build model matrix. This could be caused by "
                "unsupported data dtypes like dates. Please check your input data. "
                "Also check the original error message, included above."
            ) from e

        # get column names. There may be a more efficient way to do it
        # that does not require building the model matrix a second time, but this
        # works robustly for now: we take the names that formulaic creates
        column_names = list(
            fo.ModelSpec(formula, output="pandas")
            .get_model_matrix(self.data, context=context)
            .columns
        )[1:]

        required = sorted(str(var) for var in spec.required_variables)
        df_subset = self.data.loc[:, required]
        df_colnames = df_subset.columns

        variables = dict()

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
            _name=self.names.create(xname) if xname else xname,
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
            bname = self.names.create(basis_name + "(" + xvar.name + ")")
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
        absorb_cons: bool = False,
        diagonal_penalty: bool = False,
        scale_penalty: bool = False,
        basis_name: str = "B",
    ) -> MRFBasis:
        """
        Polys: Dictionary of arrays. The keys of the dict are the region labels.
            The corresponding values define the region by defining polygons.
        nb: Dictionary of array. The keys of the dict are the region labels.
            The corresponding values indicate the neighbors of the region.
            If it is a list or array of strings, the values are the labels of the
            neighbors.
            If it is a list or array of integers, the values are the indices of the
            neighbors. Indices correspond to regions based on an alphabetical ordering
            of regions.
        penalty: If penalty is supplied, it takes precedence over both nb and polys,
            and the arguments '


        mgcv does not concern itself with your category ordering. It *will* order
        categories alphabetically. Penalty columns have to take this into account.

        Comments on return value:

        - If either polys or nb are supplied, the returned container will contain nb.
        - If only a penalty matrix is supplied, the returned container will *not*
          contain nb.
        - Returning the label order only makes sense if the basis is *not*
          reparameterized, because only then we have a clear correspondence of
          parameters to labels.
          If the basis is reparameterized, there's no such correspondence in a clear
          way, so the returned label order is None.

        """

        if not isinstance(k, int):
            raise TypeError(f"'k' must be int, got {type(k)}.")
        if k < -1:
            raise ValueError(f"'k' cannot be smaller than -1, got {k=}.")

        if polys is None and nb is None and penalty is None:
            raise ValueError("At least one of polys, nb, or penalty must be provided.")

        var, mapping = self.registry.get_categorical_obs(x)
        self.mappings[x] = mapping

        labels = set(list(mapping.labels_to_integers_map))

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

        xt_args = []
        pass_to_r: dict[str, np.typing.NDArray | dict[str, np.typing.NDArray]] = {}
        if polys is not None:
            xt_args.append("polys=polys")
            if not labels == set(list(polys)):
                raise ValueError(
                    "Names in 'polys' must correspond to the levels of 'x'."
                )
            pass_to_r["polys"] = {key: np.asarray(val) for key, val in polys.items()}

        if nb is not None:
            xt_args.append("nb=nb")
            if not labels == set(list(nb)):
                raise ValueError("Names in 'nb' must correspond to the levels of 'x'.")

            nb_processed = {}
            for key, val in nb.items():
                val_arr = np.asarray(val)
                if val_arr.ndim != 1:
                    raise ValueError(
                        f"Expected 1d arrays in 'nb', got {val_arr.ndim=} for {key}."
                    )
                if np.isdtype(val_arr.dtype, np.dtype("int")):
                    # add one to convert to 1-based indexing for R
                    # and cast to float for R
                    val_arr = np.astype(val_arr + 1, float)
                    # val_arr = np.astype(val_arr, float)
                elif np.isdtype(val_arr.dtype, np.dtype("float")):
                    # add one to convert to 1-based indexing for R
                    val_arr = np.astype(np.astype(val_arr, int) + 1, float)
                elif val_arr.dtype.kind == "U":  # must be unicode strings then
                    pass
                else:
                    raise TypeError(f"Unsupported dtype: {val_arr.dtype!r}")

                nb_processed[key] = val_arr

            pass_to_r["nb"] = nb_processed

        if penalty is not None:
            penalty = np.asarray(penalty)
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

            xt_args.append("penalty=penalty")
            if not np.shape(penalty)[0] == np.shape(penalty)[1]:
                raise ValueError(f"Penalty must be square, got {np.shape(penalty)=}")

            if not np.shape(penalty)[1] == len(labels):
                raise ValueError(
                    "Dimensions of 'penalty' must correspond to the levels of 'x'."
                )
            pass_to_r["penalty"] = penalty

        if "nb" in pass_to_r and "penalty" in pass_to_r:
            logger.warning(
                "Both 'nb' and 'penalty' were supplied. 'penalty' will be used to "
                "setup this basis."
            )

        if "polys" in pass_to_r and "penalty" in pass_to_r:
            logger.warning(
                "Both 'polys' and 'penalty' were supplied. 'penalty' will be used "
                "to setup this basis."
            )

        xt = "list("
        xt += ",".join(xt_args)
        xt += ")"

        if penalty is not None:
            # removing penalty from the pass_to_r dict, because we are giving it
            # special treatment here.
            # specifically, we have to equip it with row and column names to make
            # sure that penalty entries get correctly matched to clusters by mgcv
            penalty_prelim_arr = np.asarray(pass_to_r.pop("penalty"))
            to_r(penalty_prelim_arr, "penalty")
            to_r(np.array(penalty_labels), "penalty_labels")
            r("colnames(penalty) <- penalty_labels")
            r("rownames(penalty) <- penalty_labels")

        spec = f"s({x}, k={k}, bs='mrf', xt={xt})"

        observed = mapping.integers_to_labels(var.value)
        regions = list(mapping.labels_to_integers_map)
        df = pd.DataFrame({x: pd.Categorical(observed, categories=regions)})

        smooth = scon.SmoothCon(
            spec,
            data=df,
            diagonal_penalty=diagonal_penalty,
            absorb_cons=absorb_cons,
            scale_penalty=scale_penalty,
            pass_to_r=pass_to_r,
        )

        x_name = x

        def basis_fun(x):
            """
            The array outputted by this smooth contains column names.
            Here, we remove these column names and convert to jax.
            """
            # disabling warnings about "mrf should be a factor"
            r("old_warn <- getOption('warn')")
            r("options(warn = -1)")
            labels = mapping.integers_to_labels(x)
            df = pd.DataFrame({x_name: pd.Categorical(labels, categories=regions)})
            basis = jnp.asarray(np.astype(smooth.predict(df)[:, 1:], "float"))
            r("options(warn = old_warn)")
            return basis

        smooth_penalty = smooth.penalty
        if np.shape(smooth_penalty)[1] > len(labels):
            smooth_penalty = smooth_penalty[:, 1:]
        elif np.shape(smooth_penalty)[0] < np.shape(smooth_penalty)[1]:
            smooth_penalty = smooth_penalty[:, 1:]

        try:
            penalty_arr = jnp.asarray(np.astype(smooth_penalty, "float"))
        except ValueError:
            penalty_arr = jnp.asarray(np.astype(smooth_penalty[:, 1:], "float"))

        basis = MRFBasis(
            value=var,
            basis_fn=basis_fun,
            name=self.names.create(basis_name + "(" + x + ")"),
            cache_basis=True,
            use_callback=True,
            penalty=penalty_arr,
        )
        if absorb_cons:
            basis._constraint = "absorbed_via_mgcv"

        try:
            nb_out = to_py(f"{smooth._smooth_r_name}[[1]]$xt$nb", format="numpy")
        except TypeError:
            nb_out = None
        # nb_out = {key: np.astype(val, "int") for key, val in nb_out.items()}

        if absorb_cons:
            label_order = None
        else:
            label_order = list(
                to_py(f"{smooth._smooth_r_name}[[1]]$X", format="pandas").columns
            )
            label_order = [lab[1:] for lab in label_order]  # removes leading x from R

        if nb_out is not None:

            def to_label(code):
                try:
                    label_array = mapping.integers_to_labels(code - 1)
                except TypeError:
                    label_array = code
                return np.atleast_1d(label_array).tolist()

            nb_out = {k: to_label(v) for k, v in nb_out.items()}

        basis.mrf_spec = MRFSpec(mapping, nb_out, label_order, polys)

        return basis
