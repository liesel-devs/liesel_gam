from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial
from liesel.contrib.splines import basis_matrix

Array = jax.Array
ArrayLike = jax.typing.ArrayLike


def _bspline_basis(x: ArrayLike, knots: ArrayLike, order: int) -> Array:
    """Return B-spline basis, allowing values outside knots."""
    x = jnp.asarray(x)
    knots = jnp.asarray(knots)
    return basis_matrix(x, knots, order, outer_ok=True)


@partial(jax.jit, static_argnums=2)
@partial(jnp.vectorize, excluded=(1, 2), signature="(n)->(n,p)")
def bspline_basis(x: ArrayLike, knots: ArrayLike, order: int) -> Array:
    """
    Vectorized B-spline basis function evaluation.

    Values outside the support are set to zero, where::

        min = knots[order]
        max = knots[-(order + 1)]

    Parameters
    ----------
    x
        Input array.
    knots
        Array of knots.
    order
        Order of the spline (``order=3`` for a cubic spline).

    Returns
    -------
    B-spline basis matrix.
    """
    x = jnp.asarray(x)
    knots = jnp.asarray(knots)

    min_knot = knots[order]
    max_knot = knots[-(order + 1)]
    basis = _bspline_basis(x, knots, order)
    mask = jnp.logical_or(x < min_knot, x > max_knot)
    mask = jnp.expand_dims(mask, -1)
    return jnp.where(mask, 0.0, basis)


@partial(jax.jit, static_argnums=2)
@partial(jnp.vectorize, excluded=(1, 2), signature="(n)->(n,p)")
def bspline_basis_deriv(x: ArrayLike, knots: ArrayLike, order: int) -> Array:
    """
    Evaluate matrix of first derivatives of B-spline bases.

    Values outside the support are set to zero.
    """
    x = jnp.asarray(x)
    knots = jnp.asarray(knots)

    min_knot = knots[order]
    max_knot = knots[-(order + 1)]

    basis = _bspline_basis(x, knots[1:-1], order - 1)
    dknots = jnp.diff(knots).mean()
    D = jnp.diff(jnp.identity(jnp.shape(knots)[-1] - order - 1)).T
    basis_grad = basis @ (D / dknots)

    mask = jnp.logical_or(x < min_knot, x > max_knot)
    mask = jnp.expand_dims(mask, -1)
    return jnp.where(mask, 0.0, basis_grad)


@partial(jax.jit, static_argnums=2)
@partial(jnp.vectorize, excluded=(1, 2), signature="(n)->(n,p)")
def bspline_basis_deriv2(x, knots, order):
    """Evaluate matrix of second derivatives of B-spline bases."""
    x = jnp.asarray(x)
    knots = jnp.asarray(knots)

    min_knot = knots[order]
    max_knot = knots[-(order + 1)]
    basis = _bspline_basis(x, knots[2:-2], order - 2)

    dknots = jnp.diff(knots).mean()
    D = jnp.diff(jnp.identity(jnp.shape(knots)[-1] - order - 1)).T
    basis_grad = basis @ D[1::, 1:] @ (D / (dknots**2))

    mask = jnp.logical_or(x < min_knot, x > max_knot)
    mask = jnp.expand_dims(mask, -1)
    return jnp.where(mask, 0.0, basis_grad)


class BSplineApprox:
    """
    Approximate B-spline evaluations on a fixed grid.

    Parameters
    ----------
    knots
        Knot positions.
    degree
        Degree of the spline, with ``degree=3`` indicating a cubic spline.
    ngrid
        Number of grid points used to precompute the basis (default 1000).
    Z
        Optional matrix to post-multiply the basis. In ``B(x) @ Z``, ``B(x)`` is the
        basis matrix and ``Z`` is the postmultiplication matrix. Can be used to apply
        linear constraints via reparameterization matrices such as those returned by
        :class:`.LinearConstraintEVD`.

    See Also
    --------
    .LinearConstraintEVD : Compute reparameterization matrices for linear constraints.

    Examples
    --------

    Basic example.

    >>> import jax
    >>> from liesel.contrib.splines import equidistant_knots
    >>> import liesel_gam as gam

    >>> nbases = 20
    >>> x = jax.random.uniform(jax.random.key(1234), shape=(40,))
    >>> coef = jax.random.normal(jax.random.key(4321), shape=(nbases,))
    >>> knots = equidistant_knots(x, n_param=nbases)

    >>> bspline = gam.experimental.BSplineApprox(knots)
    >>> bspline.dot(x, coef).shape
    (40,)

    Applying a linear constraint matrix.

    >>> import jax
    >>> from liesel.contrib.splines import basis_matrix, equidistant_knots
    >>> import liesel_gam as gam

    >>> nbases = 20
    >>> x = jax.random.uniform(jax.random.key(1234), shape=(40,))
    >>> coef = jax.random.normal(jax.random.key(4321), shape=((nbases - 1),))
    >>> knots = equidistant_knots(x, n_param=nbases)

    >>> basis = basis_matrix(x, knots)
    >>> Z = gam.LinearConstraintEVD.sumzero_term(basis)

    >>> bspline = gam.experimental.BSplineApprox(knots, Z=Z)
    >>> bspline.dot(x, coef).shape
    (40,)

    """

    def __init__(
        self,
        knots: Array,
        degree: int = 3,
        ngrid: int = 1000,
        Z: Array | None = None,
        subscripts: str = "...ij,...j->...i",
    ) -> None:
        self.knots = jnp.asarray(knots)
        self.dknots = jnp.mean(jnp.diff(knots))
        self.degree = degree
        self.nparam = jnp.shape(knots)[0] - degree - 1
        self.subscripts = subscripts

        self.min_knot = self.knots[degree]
        self.max_knot = self.knots[-(degree + 1)]

        grid = jnp.linspace(self.min_knot, self.max_knot, ngrid)
        self.step = (self.max_knot - self.min_knot) / ngrid
        prepend = jnp.array([self.min_knot - self.step])
        append = jnp.array([self.max_knot + self.step])
        self.ngrid = ngrid
        self.grid = jnp.concatenate((prepend, grid, append))

        self.Z = Z

        self.basis_grid = self.compute_basis(grid)
        self.basis_deriv_grid = self.compute_deriv(grid)
        self.basis_deriv2_grid = self.compute_deriv2(grid)

        self._dot_fn = self._get_dot_fn()
        self._dot_and_deriv_fn = self._get_dot_and_deriv_fn()

    def compute_basis(self, x: Array) -> Array:
        """Computes the basis matrix."""
        basis = bspline_basis(x, self.knots, self.degree)
        if self.Z is not None:
            basis = basis @ self.Z
        return basis

    def compute_deriv(self, x: Array) -> Array:
        """Computes the matrix of basis function derivatives with respect to x."""
        deriv = bspline_basis_deriv(x, self.knots, self.degree)
        if self.Z is not None:
            deriv = deriv @ self.Z
        return deriv

    def compute_deriv2(self, x: Array) -> Array:
        """
        Computes the matrix of basis function second derivatives with respect to x.
        """
        deriv2 = bspline_basis_deriv2(x, self.knots, self.degree)
        if self.Z is not None:
            deriv2 = deriv2 @ self.Z
        return deriv2

    @partial(jax.jit, static_argnums=0)
    def _approx_basis(self, x: Array) -> Array:
        i = jnp.searchsorted(self.grid, x, side="right") - 1
        lo = self.grid[i]
        k = jnp.expand_dims((x - lo) / self.step, -1)

        basis = (1.0 - k) * self.basis_grid[i, :] + (k * self.basis_grid[i + 1, :])
        return jnp.atleast_2d(basis)

    @partial(jax.jit, static_argnums=0)
    def _approx_basis_and_deriv(self, x: Array) -> tuple[Array, Array]:
        """
        Returns the basis matrix approximation and its gradient with
        respect to the data.
        """

        i = jnp.searchsorted(self.grid, x, side="right") - 1
        lo = self.grid[i]
        k = jnp.expand_dims((x - lo) / self.step, -1)

        basis = (1.0 - k) * self.basis_grid[i, :] + (k * self.basis_grid[i + 1, :])
        basis_deriv = (1.0 - k) * self.basis_deriv_grid[i, :] + (
            k * self.basis_deriv_grid[i + 1, :]
        )
        basis = jnp.atleast_2d(basis)
        basis_deriv = jnp.atleast_2d(basis_deriv)
        return basis, basis_deriv

    @partial(jax.jit, static_argnums=0)
    def _approx_basis_deriv_and_deriv2(self, x: Array) -> tuple[Array, Array, Array]:
        """
        Returns the basis matrix approximation and its first and second
        derivative with respect to the data.
        """
        i = jnp.searchsorted(self.grid, x, side="right") - 1
        lo = self.grid[i]
        k = jnp.expand_dims((x - lo) / self.step, -1)

        basis = (1.0 - k) * self.basis_grid[i, :] + (k * self.basis_grid[i + 1, :])
        basis_deriv = (1.0 - k) * self.basis_deriv_grid[i, :] + (
            k * self.basis_deriv_grid[i + 1, :]
        )
        basis_deriv2 = (1.0 - k) * self.basis_deriv2_grid[i, :] + (
            k * self.basis_deriv2_grid[i + 1, :]
        )
        basis = jnp.atleast_2d(basis)
        basis_deriv = jnp.atleast_2d(basis_deriv)
        basis_deriv2 = jnp.atleast_2d(basis_deriv2)
        return basis, basis_deriv, basis_deriv2

    def approx_basis(self, x: Array) -> Array:
        """
        Approximates B-spline basis for input data.

        Parameters
        ----------
        x
            Input data.

        Returns
        -------
        Basis matrix.
        """
        return self._approx_basis(x)

    def approx_basis_and_deriv(self, x: Array) -> tuple[Array, Array]:
        """
        Approximates basis and matrix of first derivatives.

        Parameters
        ----------
        x
            Input data.

        Returns
        -------
        basis
            Basis matrix.
        deriv
            First derivative.
        """
        return self._approx_basis_and_deriv(x)

    def approx_basis_and_deriv2(self, x: Array) -> tuple[Array, Array, Array]:
        """
        Approximates basis matrix and matrices of first and second derivatives.

        Parameters
        ----------
        x
            Input data.

        Returns
        -------
        basis : Array
            Basis matrix.
        deriv : Array
            First derivative.
        deriv2 : Array
            Second derivative.
        """
        return self._approx_basis_deriv_and_deriv2(x)

    def _compute_dot(self, x: Array, coef: Array) -> Array:
        return jnp.einsum(self.subscripts, x, coef)

    def _get_dot_fn(self) -> Callable[[Array, Array], Array]:
        @jax.custom_jvp
        def _dot(
            x: Array,
            coef: Array,
        ) -> Array:
            basis = self.approx_basis(x)
            smooth = self._compute_dot(basis, coef)
            return smooth

        @_dot.defjvp
        def _dot_jvp(primals, tangents):
            # x and x_dot: (n,)
            # coef and coef_dot: (k,)
            x, coef = primals
            x_dot, coef_dot = tangents

            # both shape (n, k)
            basis, basis_deriv = self.approx_basis_and_deriv(x)

            # shape (n,)
            smooth = self._compute_dot(basis, coef)

            tangent_x = self._compute_dot(basis_deriv, coef) * x_dot
            tangent_coef = self._compute_dot(basis, coef_dot)

            tangent = tangent_x + tangent_coef

            return smooth, tangent

        return jax.jit(_dot)

    def _get_dot_and_deriv_fn(
        self,
    ) -> Callable[[Array, Array], tuple[Array, Array]]:
        @jax.custom_jvp
        def _dot_and_deriv(
            x: Array,
            coef: Array,
        ) -> tuple[Array, Array]:
            """
            Assumes x is (,)
            And coef is (p,)
            """
            basis, basis_deriv = self.approx_basis_and_deriv(x)  # (p,) and (p,) shapes
            smooth = self._compute_dot(basis, coef)
            smooth_deriv = self._compute_dot(basis_deriv, coef)
            return smooth, smooth_deriv  # (,) and (,) shapes

        @_dot_and_deriv.defjvp
        def _dot_and_deriv_jvp(primals, tangents):
            x, coef = primals
            x_dot, coef_dot = tangents

            basis, basis_deriv, basis_deriv2 = self.approx_basis_and_deriv2(x)
            smooth = self._compute_dot(basis, coef)
            smooth_deriv = self._compute_dot(basis_deriv, coef)
            smooth_deriv2 = self._compute_dot(basis_deriv2, coef)

            primal_out = (smooth, smooth_deriv)

            tangent_bdot_x = smooth_deriv * x_dot
            tangent_bdot_coef = self._compute_dot(basis, coef_dot)
            tangent_bdot = tangent_bdot_x + tangent_bdot_coef

            tangent_deriv_x = smooth_deriv2 * x_dot
            tangent_deriv_coef = self._compute_dot(basis_deriv, coef_dot)
            tangent_deriv = tangent_deriv_x + tangent_deriv_coef

            tangent_out = (tangent_bdot, tangent_deriv)

            return primal_out, tangent_out

        return jax.jit(_dot_and_deriv)

    def dot(self, x: Array, coef: Array) -> Array:
        """
        Evaluate spline at given points.

        Parameters
        ----------
        x
            Input data, an array of shape (n,).
        coef
            Spline coefficients.

        """
        return self._dot_fn(x, coef)

    def dot_and_deriv(self, x: Array, coef: Array) -> tuple[Array, Array]:
        """
        Evaluate spline and its derivative.

        Parameters
        ----------
        x
            Input data, an array of shape (n,).
        coef
            Spline coefficients.

        Returns
        -------
        value : Array
            Spline values.
        deriv : Array
            Spline derivatives.
        """
        return self._dot_and_deriv_fn(x, coef)
