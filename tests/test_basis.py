import jax
import jax.numpy as jnp
import liesel.model as lsl
import pytest
import scipy
from jax import Array
from jax.random import key, uniform
from liesel.contrib.splines import basis_matrix, equidistant_knots
from ryp import r, to_py

import liesel_gam as gam


@pytest.fixture(scope="module")
def columb():
    r("library(mgcv)")
    r("data(columb)")
    columb = to_py("columb", format="pandas")
    return columb


def pspline_penalty(nparam: int, random_walk_order: int = 2) -> Array:
    """
    Builds an (nparam x nparam) P-spline penalty matrix.
    """
    D = jnp.diff(jnp.identity(nparam), random_walk_order, axis=0)
    return D.T @ D


class TestBasis:
    def test_identity(self) -> None:
        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x")
        basis = gam.Basis(x, basis_fn=lambda x: x)

        assert jnp.allclose(x.value, basis.value)

    @pytest.mark.parametrize("use_callback", (True, False))
    def test_static_kwargs_in_basis_fn(self, use_callback) -> None:
        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x")
        basis = gam.Basis(x, lambda x, y: x + y, y=2.0, use_callback=use_callback)

        assert jnp.allclose(x.value, basis.value - 2.0)

    @pytest.mark.parametrize("use_callback", (True, False))
    def test_dynamic_kwargs_in_basis_fn(self, use_callback) -> None:
        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x")
        y = lsl.Var.new_obs(2.0, name="y")
        basis = gam.Basis(x, lambda x, y: x + y, y=y, use_callback=use_callback)

        assert jnp.allclose(basis.value, x.value + y.value)

        y.value = 3.0
        basis.update()
        assert jnp.allclose(basis.value, x.value + y.value)

    def test_square(self) -> None:
        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x")
        basis = gam.Basis(x, basis_fn=lambda x: x**2)

        assert jnp.allclose(x.value**2, basis.value)

    def test_cube_does_not_work(self) -> None:
        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x")
        with pytest.raises(RuntimeError):
            gam.Basis(x, basis_fn=lambda x: jnp.expand_dims(x, axis=(1, 2)))

    def test_model_can_be_initialized(self) -> None:
        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x")
        basis = gam.Basis(x, basis_fn=lambda x: x**2)
        lsl.Model([basis])

    def test_scalar_works(self) -> None:
        x = lsl.Var.new_obs(2.0, name="x")
        basis = gam.Basis(x, basis_fn=lambda x: x**2)

        assert jnp.allclose(x.value**2, basis.value)

    def test_matrix(self) -> None:
        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x")
        basis = gam.Basis(x, basis_fn=lambda x: jnp.c_[x, x])

        assert jnp.allclose(x.value, basis.value[:, 0])
        assert jnp.allclose(x.value, basis.value[:, 1])
        assert basis.value.shape == (x.value.shape[0], 2)

    def test_unnamed_value(self) -> None:
        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10))
        basis = gam.Basis(x, basis_fn=lambda x: x)
        assert basis.name == ""

    def test_array_without_name(self) -> None:
        basis = gam.Basis(jnp.linspace(0, 1, 10), basis_fn=lambda x: x)  # type: ignore
        assert basis.name == ""
        assert basis.x.name == ""

    def test_array(self) -> None:
        x = jnp.linspace(0, 1, 10)
        basis = gam.Basis(x, basis_fn=lambda x: x, xname="x")
        assert basis.name == "B(x)"

    def test_custom_name(self) -> None:
        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x")
        basis = gam.Basis(x, basis_fn=lambda x: x, name="custom_basis")

        assert basis.name == "custom_basis"

    def test_jittable_basis_fn_works(self) -> None:
        x = jnp.linspace(0, 1, 10)

        # baseline: everything works with jittable function
        basis = gam.Basis(
            x,
            basis_fn=lambda x: jax.scipy.special.logsumexp(x),
            xname="x",
            use_callback=False,
        )

        model = lsl.Model([basis])

        def basis_update(pos, state):
            state = model.update_state(pos, state)
            return model.state["B(x)_var_value"].value

        pos = model.extract_position(["x"])
        jax.jit(basis_update)(pos, model.state)

    def test_nonjittable_basis_fn_errors(self) -> None:
        # error: code breaks with non-jittable function
        x = jnp.linspace(0, 1, 10)
        basis = gam.Basis(
            x,
            basis_fn=lambda x: scipy.special.logsumexp(x),
            xname="x",
            use_callback=False,
        )

        model = lsl.Model([basis])

        def basis_update(pos, state):
            state = model.update_state(pos, state)
            return model.state["B(x)_var_value"].value

        pos = model.extract_position(["x"])
        with pytest.raises(RuntimeError):
            jax.jit(basis_update)(pos, model.state)

    def test_nonjittable_basis_fn_works_with_callback(self) -> None:
        # solution: code works with non-jittable function
        # when using callback
        x = jnp.linspace(0, 1, 10)
        basis = gam.Basis(
            x,
            basis_fn=lambda x: scipy.special.logsumexp(x),
            xname="x",
            use_callback=True,
        )

        model = lsl.Model([basis])

        def basis_update(pos, state):
            state = model.update_state(pos, state)
            return model.state["B(x)_var_value"].value

        pos = model.extract_position(["x"])
        jax.jit(basis_update)(pos, model.state)

    def test_nonjittable_basis_fn_works_by_default(self) -> None:
        # solution: code works with non-jittable function
        # when using callback
        x = jnp.linspace(0, 1, 10)
        basis = gam.Basis(
            x,
            basis_fn=lambda x: scipy.special.logsumexp(x),
            xname="x",
        )

        model = lsl.Model([basis])

        def basis_update(pos, state):
            state = model.update_state(pos, state)
            return model.state["B(x)_var_value"].value

        pos = model.extract_position(["x"])
        jax.jit(basis_update)(pos, model.state)

    def test_cache_basis(self) -> None:
        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x")
        basis = gam.Basis(x, basis_fn=lambda x: jnp.c_[x, x], cache_basis=True)
        assert isinstance(basis.value_node, lsl.Calc)

        basis = gam.Basis(x, basis_fn=lambda x: jnp.c_[x, x], cache_basis=False)
        assert isinstance(basis.value_node, lsl.TransientCalc)

    def test_linear(self) -> None:
        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x")
        basis = gam.Basis.new_linear(x)
        assert basis.name == "B(x)"
        assert basis.value.shape == (x.value.shape[0], 1)

        basis = gam.Basis.new_linear(x, add_intercept=True)
        assert basis.name == "B(x)"
        assert basis.value.shape == (x.value.shape[0], 2)
        assert jnp.allclose(basis.value[:, 0], 1.0)
        assert jnp.allclose(basis.value[:, 1], x.value)

        basis = gam.Basis.new_linear(x, name="custom_name")
        assert basis.name == "custom_name"

        basis = gam.Basis.new_linear(
            jnp.linspace(0, 1, 10), name="custom_name", xname="y"
        )
        assert basis.name == "custom_name"
        assert basis.x.name == "y"

    def test_liesel_var_constructors(self) -> None:
        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x")

        with pytest.raises(NotImplementedError):
            gam.Basis.new_param(x)

        with pytest.raises(NotImplementedError):
            gam.Basis.new_obs(x)

        with pytest.raises(NotImplementedError):
            gam.Basis.new_value(x)

        with pytest.raises(NotImplementedError):
            gam.Basis.new_calc(x)


@pytest.fixture
def basis() -> gam.Basis:
    x = uniform(key(1), (15,))
    knots = equidistant_knots(x, n_param=7, order=3)

    def bfn(x):
        basis = basis_matrix(x, knots, 3)
        return basis

    nparam = bfn(x).shape[-1]
    K = pspline_penalty(nparam)

    return gam.Basis(x, basis_fn=bfn, penalty=K, xname="x")


def is_diagonal(M, atol=1e-6):
    # mask for off-diagonal elements
    off_diag_mask = ~jnp.eye(M.shape[-1], dtype=bool)
    off_diag_values = M[off_diag_mask]
    return jnp.all(jnp.abs(off_diag_values) < atol)


class TestBasisReparameterization:
    def test_diagonalize_penalty(self, basis: gam.Basis):
        assert basis.penalty is not None
        assert not is_diagonal(basis.penalty.value, 1e-5)
        b1 = basis.value
        basis.diagonalize_penalty()
        b2 = basis.value
        assert is_diagonal(basis.penalty.value, 1e-5)
        assert not jnp.allclose(b1, b2, atol=1e-3)

    def test_diagonalize_penalty_twice(self, basis: gam.Basis):
        assert basis.penalty is not None
        basis.diagonalize_penalty()
        b1 = basis.value
        pen1 = basis.penalty.value

        basis.diagonalize_penalty(1e-5)
        assert is_diagonal(basis.penalty.value, 1e-5)

        b2 = basis.value
        pen2 = basis.penalty.value
        assert jnp.allclose(pen1, pen2, atol=1e-5)
        assert jnp.allclose(b1, b2, atol=1e-5)

    def test_scale_penalty(self, basis: gam.Basis):
        assert basis.penalty is not None
        b1 = basis.value
        pen1 = basis.penalty.value

        basis.scale_penalty()

        b2 = basis.value
        pen2 = basis.penalty.value

        assert jnp.linalg.norm(pen2, ord=jnp.inf) == pytest.approx(1.0)
        assert not jnp.allclose(pen1, pen2, atol=1e-5)
        assert jnp.allclose(b1, b2, atol=1e-5)

    def test_scale_penalty_twice(self, basis: gam.Basis):
        assert basis.penalty is not None
        basis.scale_penalty()
        b1 = basis.value
        pen1 = basis.penalty.value

        basis.scale_penalty()

        b2 = basis.value
        pen2 = basis.penalty.value
        assert jnp.allclose(pen1, pen2, atol=1e-6)
        assert jnp.allclose(b1, b2, atol=1e-6)

    def test_constrain_sumzero_coef(self, basis: gam.Basis):
        assert basis.penalty is not None
        basis.constrain("sumzero_coef")
        term = gam.StrctTerm.f(basis)
        coef = jax.random.normal(key(42), term.coef.value.shape)
        constrained_coef = basis.reparam_matrix @ coef
        assert constrained_coef.sum() == pytest.approx(0.0, abs=1e-5)
        assert basis.constraint == "sumzero_coef"

    def test_constrain_sumzero_term(self, basis: gam.Basis):
        assert basis.penalty is not None
        basis.constrain("sumzero_term")
        term = gam.StrctTerm.f(basis)
        term.coef.value = jax.random.normal(key(42), term.coef.value.shape)
        term.update()
        assert term.value.sum() == pytest.approx(0.0, abs=1e-5)
        assert basis.constraint == "sumzero_term"

    def test_constrain_constant_and_linear(self, basis: gam.Basis):
        assert basis.penalty is not None
        basis.constrain("constant_and_linear")
        term = gam.StrctTerm.f(basis)
        term.coef.value = jax.random.normal(key(42), term.coef.value.shape)
        fx = term.update().value

        # sum to zero
        assert fx.sum() == pytest.approx(0.0, abs=1e-4)

        # no linear trend
        nobs = jnp.shape(basis.value)[0]
        j = jnp.ones(shape=nobs)
        X = jnp.c_[j, basis.x.value]
        A = jnp.linalg.inv(X.T @ X) @ X.T

        g = A @ fx
        assert g.shape == (2,)
        assert jnp.allclose(g, 0.0, atol=1e-4)
        assert basis.constraint == "constant_and_linear"

    def test_constrain_custom(self, basis: gam.Basis):
        assert basis.penalty is not None
        A = jnp.mean(basis.value, axis=0, keepdims=True)
        basis.constrain(A)
        term = gam.StrctTerm.f(basis)
        term.coef.value = jax.random.normal(key(42), term.coef.value.shape)
        term.update()
        assert term.value.sum() == pytest.approx(0.0, abs=1e-5)
        assert basis.constraint == "custom"

    def test_constrain_twice(self, basis: gam.Basis):
        assert basis.penalty is not None
        basis.constrain("sumzero_term")
        with pytest.raises(ValueError):
            basis.constrain("sumzero_term")
