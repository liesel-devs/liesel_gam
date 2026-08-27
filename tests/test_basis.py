import jax
import jax.numpy as jnp
import liesel.model as lsl
import pytest
import scipy
from jax import Array
from jax.random import key, uniform
from liesel.contrib.splines import basis_matrix, equidistant_knots

import liesel_gam as gam

from .mgcv_data import load_columb


@pytest.fixture(scope="module")
def columb():
    return load_columb()


def pspline_penalty(nparam: int, random_walk_order: int = 2) -> Array:
    """
    Builds an (nparam x nparam) P-spline penalty matrix.
    """
    D = jnp.diff(jnp.identity(nparam), random_walk_order, axis=0)
    return D.T @ D


class TestBasis:
    def test_approximate_tracks_changed_input(self) -> None:
        x = lsl.Var.new_obs(jnp.linspace(0.0, 1.0, 20), name="x")

        def basis_fn(value):
            return jnp.column_stack((jnp.sin(value), value**2))

        basis = gam.Basis(x, basis_fn=basis_fn, use_callback=False)
        spec = gam.ApproximationSpec(rtol=1e-3, atol=1e-5)

        assert basis.approximate(spec) is basis

        x.value = jnp.linspace(0.01, 0.99, 37)
        basis.update()
        assert spec.rtol is not None
        assert spec.atol is not None

        assert jnp.allclose(
            basis.value,
            basis_fn(x.value),
            rtol=spec.rtol,
            atol=spec.atol,
        )

    def test_approximation_reports_fitted_spec(self) -> None:
        x = lsl.Var.new_obs(jnp.linspace(2.0, 4.0, 20), name="x")
        basis = gam.Basis(x, basis_fn=lambda value: value[:, None])
        spec = gam.ApproximationSpec()

        assert basis.approximation is None
        assert basis.approximation_grid_size is None

        basis.approximate(spec)

        assert basis.approximation == spec._replace(bounds=(2.0, 4.0))
        assert basis.approximation_grid_size == 1_000

    def test_default_approximation_evaluates_one_fixed_grid(self) -> None:
        evaluated_sizes = []

        def basis_fn(value):
            evaluated_sizes.append(value.shape[0])
            return jnp.column_stack((value, value**2))

        basis = gam.Basis(
            jnp.linspace(0.0, 1.0, 20),
            xname="x",
            basis_fn=basis_fn,
            use_callback=False,
        )
        evaluated_sizes.clear()

        basis.approximate()

        assert evaluated_sizes.count(1_000) == 1
        assert 999 not in evaluated_sizes
        assert basis.approximation_grid_size == 1_000

    @pytest.mark.parametrize("tolerance", ({"atol": 1e-3}, {"rtol": 1e-3}))
    def test_approximation_refinement_is_opt_in(self, tolerance) -> None:
        evaluated_sizes = []

        def basis_fn(value):
            evaluated_sizes.append(value.shape[0])
            return (1.0 + value**2)[:, None]

        basis = gam.Basis(
            jnp.linspace(0.0, 1.0, 20),
            xname="x",
            basis_fn=basis_fn,
            use_callback=False,
        )
        evaluated_sizes.clear()

        basis.approximate(
            gam.ApproximationSpec(grid_size=5, max_grid_size=33, **tolerance)
        )

        assert evaluated_sizes[:7] == [20, 4, 4, 4, 4, 4, 5]
        assert {4, 8, 9, 16, 17}.issubset(evaluated_sizes)
        assert basis.approximation_grid_size == 17

    def test_approximation_uses_exact_whole_batch_fallback(self) -> None:
        x = lsl.Var.new_obs(jnp.linspace(0.0, 1.0, 20), name="x")

        def basis_fn(value):
            return jnp.column_stack((jnp.exp(value), jnp.sin(value)))

        basis = gam.Basis(x, basis_fn=basis_fn, use_callback=False)
        basis.approximate(gam.ApproximationSpec(bounds=(0.0, 1.0)))

        x.value = jnp.array([0.123, 0.456, 1.1])
        basis.update()

        assert jnp.array_equal(basis.value, basis_fn(x.value))

    def test_callback_basis_uses_jitted_approximation(self) -> None:
        x = lsl.Var.new_obs(jnp.linspace(0.0, 1.0, 20), name="x")
        basis = gam.Basis(x, basis_fn=scipy.special.expit, use_callback=True)
        spec = gam.ApproximationSpec(rtol=1e-3, atol=1e-5)
        basis.approximate(spec)

        new_values = jnp.linspace(0.01, 0.99, 37)
        assert spec.rtol is not None
        assert spec.atol is not None
        assert isinstance(basis.value_node, lsl.Calc | lsl.TransientCalc)
        result = jax.jit(basis.value_node.function)(new_values)

        assert jnp.allclose(
            result,
            scipy.special.expit(new_values),
            rtol=spec.rtol,
            atol=spec.atol,
        )

    def test_approximate_rejects_dynamic_basis_kwargs(self) -> None:
        x = lsl.Var.new_obs(jnp.linspace(0.0, 1.0, 20), name="x")
        shift = lsl.Var.new_obs(1.0, name="shift")
        basis = gam.Basis(
            x,
            basis_fn=lambda value, shift: value + shift,
            shift=shift,
        )

        with pytest.raises(ValueError, match="dynamic basis_kwargs"):
            basis.approximate()

    def test_approximate_rejects_batch_dependent_basis(self) -> None:
        x = lsl.Var.new_obs(jnp.linspace(0.0, 1.0, 20), name="x")
        basis = gam.Basis(
            x,
            basis_fn=lambda value: (value - jnp.mean(value))[:, None],
            use_callback=False,
        )

        with pytest.raises(ValueError, match="row-wise"):
            basis.approximate()

    def test_approximate_validates_unknown_basis_without_changing_shape(self) -> None:
        evaluated_sizes = []
        transform = jnp.arange(12.0).reshape(4, 3)

        def basis_fn(value):
            evaluated_sizes.append(value.shape[0])
            raw = jnp.column_stack((value, value**2, value**3, value**4))
            return raw @ transform

        basis = gam.Basis(
            jnp.linspace(0.0, 1.0, 20),
            xname="x",
            basis_fn=basis_fn,
            use_callback=False,
        )
        evaluated_sizes.clear()

        basis.approximate()

        assert evaluated_sizes.count(4) == 5
        assert 1 not in evaluated_sizes
        assert basis.approximation is not None

    def test_declared_row_wise_basis_skips_validation(self) -> None:
        evaluated_sizes = []

        def basis_fn(value):
            evaluated_sizes.append(value.shape[0])
            return jnp.column_stack((value, value**2))

        basis = gam.Basis(
            jnp.linspace(0.0, 1.0, 20),
            xname="x",
            basis_fn=basis_fn,
            row_wise=True,
            use_callback=False,
        )
        evaluated_sizes.clear()

        basis.approximate()

        assert basis.row_wise is True
        assert 4 not in evaluated_sizes

    def test_declared_non_row_wise_basis_rejects_approximation(self) -> None:
        evaluated_sizes = []

        def basis_fn(value):
            evaluated_sizes.append(value.shape[0])
            return value[:, None]

        basis = gam.Basis(
            jnp.linspace(0.0, 1.0, 20),
            xname="x",
            basis_fn=basis_fn,
            row_wise=False,
            use_callback=False,
        )
        evaluated_sizes.clear()

        with pytest.raises(ValueError, match="row-wise"):
            basis.approximate()

        assert basis.row_wise is False
        assert evaluated_sizes == []

    def test_approximate_rejects_nonfinite_basis(self) -> None:
        x = lsl.Var.new_obs(jnp.linspace(0.0, 1.0, 20), name="x")
        basis = gam.Basis(
            x,
            basis_fn=lambda value: jnp.where(value < 1.0, value, jnp.nan)[:, None],
            use_callback=False,
        )

        with pytest.raises(ValueError, match="finite basis values"):
            basis.approximate()

    def test_approximate_rejects_constant_basis(self) -> None:
        x = lsl.Var.new_obs(jnp.linspace(0.0, 1.0, 20), name="x")
        basis = gam.Basis(
            x,
            basis_fn=lambda value: jnp.ones((value.shape[0], 2)),
            use_callback=False,
        )

        with pytest.raises(ValueError, match="nonconstant"):
            basis.approximate()

    def test_approximate_accepts_periodic_basis(self) -> None:
        x = lsl.Var.new_obs(jnp.linspace(0.0, 1.0, 20), name="x")
        basis = gam.Basis(
            x,
            basis_fn=lambda value: jnp.sin(2.0 * jnp.pi * value)[:, None],
            use_callback=False,
        )

        basis.approximate()

        assert basis.approximation is not None

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
        basis = gam.Basis(jnp.linspace(0, 1, 10), basis_fn=lambda x: x)
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

    def test_x_node_and_xname_supplied(self):
        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x")
        with pytest.raises(ValueError, match="`xname` must not be used"):
            gam.Basis(x, basis_fn=lambda x: x**2, xname="test")

    def test_update_penalty(self):
        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x")
        basis = gam.Basis(x, basis_fn=lambda x: x**2, penalty=None)
        assert basis.penalty is None

        basis.update_penalty(jnp.eye(1))

        basis.update_penalty(jnp.eye(10))
        assert basis.penalty is not None
        assert jnp.allclose(basis.penalty.value, jnp.eye(10))

        basis = gam.Basis(x, basis_fn=lambda x: jnp.expand_dims(x**2, 0), penalty=None)
        assert basis.penalty is None

        with pytest.raises(ValueError, match="columns, replacement"):
            basis.update_penalty(jnp.eye(1))

        basis.update_penalty(jnp.eye(10))
        assert basis.penalty is not None
        assert jnp.allclose(basis.penalty.value, jnp.eye(10))

    def test_penalty_node(self):
        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x")
        pen = lsl.Value(jnp.eye(10))
        basis = gam.Basis(x, basis_fn=lambda x: x**2, penalty=pen)
        assert basis.penalty is not None
        assert jnp.allclose(basis.penalty.value, jnp.eye(10))
        assert basis.penalty is pen

        basis = gam.Basis(x, basis_fn=lambda x: x**2, penalty=jnp.eye(10))
        assert basis.penalty is not None
        assert jnp.allclose(basis.penalty.value, jnp.eye(10))


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
    def test_approximation_is_transparent_to_constraint_order(self):
        values = jnp.linspace(0.0, 1.0, 20)
        x1 = lsl.Var.new_obs(values, name="x1")
        x2 = lsl.Var.new_obs(values, name="x2")

        def basis_fn(value):
            return jnp.column_stack((jnp.ones_like(value), value, value**2))

        spec = gam.ApproximationSpec(rtol=1e-3, atol=1e-5)
        first = gam.Basis(
            x1,
            basis_fn=basis_fn,
            penalty=jnp.eye(3),
            use_callback=False,
        )
        second = gam.Basis(
            x2,
            basis_fn=basis_fn,
            penalty=jnp.eye(3),
            use_callback=False,
        )

        first.approximate(spec).constrain("sumzero_term")
        second.constrain("sumzero_term").approximate(spec)

        new_values = jnp.linspace(0.01, 0.99, 37)
        x1.value = new_values
        x2.value = new_values
        first.update()
        second.update()

        assert spec.rtol is not None
        assert spec.atol is not None
        assert jnp.allclose(
            first.value,
            second.value,
            rtol=spec.rtol,
            atol=spec.atol,
        )

    def test_transient_approximation_can_be_constrained(self):
        values = jnp.linspace(0.0, 1.0, 20)
        basis = gam.Basis(
            values,
            xname="x",
            basis_fn=lambda value: jnp.column_stack((value, value**2)),
            penalty=jnp.eye(2),
            use_callback=False,
            cache_basis=False,
        )

        basis.approximate().constrain("sumzero_term")

        assert basis.constraint == "sumzero_term"

    def test_approximation_is_transparent_to_all_transformations(self):
        values = jnp.linspace(0.0, 1.0, 20)
        knots = equidistant_knots(values, n_param=7, order=3)

        def basis_fn(value):
            return basis_matrix(value, knots, 3)

        penalty = pspline_penalty(7)
        spec = gam.ApproximationSpec(rtol=1e-3, atol=1e-5)
        first = gam.Basis(
            values,
            xname="x1",
            basis_fn=basis_fn,
            penalty=penalty,
            use_callback=True,
        )
        second = gam.Basis(
            values,
            xname="x2",
            basis_fn=basis_fn,
            penalty=penalty,
            use_callback=True,
        )

        first.approximate(spec)
        first.scale_penalty().constrain("sumzero_term").diagonalize_penalty()
        second.scale_penalty().constrain("sumzero_term").diagonalize_penalty()
        second.approximate(spec)

        new_values = jnp.linspace(0.01, 0.99, 37)
        first_x = first.x
        second_x = second.x
        assert isinstance(first_x, lsl.Var)
        assert isinstance(second_x, lsl.Var)
        first_x.value = new_values
        second_x.value = new_values
        first.update()
        second.update()

        assert spec.rtol is not None
        assert spec.atol is not None
        assert jnp.allclose(
            first.value,
            second.value,
            rtol=spec.rtol,
            atol=spec.atol,
        )
        assert first.penalty is not None
        assert second.penalty is not None
        assert jnp.allclose(first.penalty.value, second.penalty.value)

    def test_penalty_is_none(self):
        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x")
        basis = gam.Basis(x, basis_fn=lambda x: jnp.expand_dims(x**2, -1), penalty=None)

        with pytest.raises(TypeError, match="penalty is None"):
            basis.scale_penalty()

        with pytest.raises(TypeError, match="penalty is None"):
            basis.diagonalize_penalty()

        with pytest.raises(TypeError, match="penalty is None"):
            basis.constrain("sumzero_coef")

    def test_1d_basis(self):
        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x")
        basis = gam.Basis(x, basis_fn=lambda x: x**2, penalty=None)
        with pytest.raises(ValueError, match="matrix-valued bases"):
            basis.constrain("sumzero_coef")

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

        assert jnp.linalg.norm(pen2, ord=1) == pytest.approx(
            float(jnp.linalg.norm(b2, ord=jnp.inf) ** 2)
        )
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
        assert basis.reparam_matrix is not None
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
