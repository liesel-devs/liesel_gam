import jax.numpy as jnp
import liesel.model as lsl
import pytest

import liesel_gam as gam


class TestBasis:
    def test_identity(self) -> None:
        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x")
        basis = gam.Basis(x, basis_fn=lambda x: x)

        assert jnp.allclose(x.value, basis.value)

    def test_args_in_basis_fn(self) -> None:
        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x")
        basis = gam.Basis(x, lambda x, y: x + y, 2.0)

        assert jnp.allclose(x.value, basis.value - 2.0)

    def test_kwargs_in_basis_fn(self) -> None:
        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x")
        basis = gam.Basis(x, lambda x, y: x + y, y=2.0)

        assert jnp.allclose(x.value, basis.value - 2.0)

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

    def test_unnamed_value_causes_error(self) -> None:
        with pytest.raises(ValueError):
            x = lsl.Var.new_obs(jnp.linspace(0, 1, 10))
            gam.Basis(x, basis_fn=lambda x: x)

    def test_array_causes_error(self) -> None:
        with pytest.raises(TypeError):
            gam.Basis(jnp.linspace(0, 1, 10), basis_fn=lambda x: x)  # type: ignore

    def test_custom_name(self) -> None:
        x = lsl.Var.new_obs(jnp.linspace(0, 1, 10), name="x")
        basis = gam.Basis(x, basis_fn=lambda x: x, name="custom_basis")

        assert basis.name == "custom_basis"


class TestIntercept:
    def test_init(self) -> None:
        gam.Intercept("test")


class TestLinearTerm:
    def test_univariate_works(self) -> None:
        x = jnp.linspace(0, 1, 5)
        term = gam.LinearTerm(x, name="b0")
        assert jnp.allclose(jnp.zeros_like(x), term.value)
        assert jnp.allclose(x, term.basis.value[:, 0])

    def test_add_intercept(self) -> None:
        x = jnp.linspace(0, 1, 5)
        term = gam.LinearTerm(x, name="b0", add_intercept=True)
        assert jnp.allclose(x, term.basis.value[:, 1])
        assert jnp.allclose(jnp.ones_like(x), term.basis.value[:, 0])

    def test_bivariate_works(self) -> None:
        x = jnp.linspace(0, 1, 5)
        term = gam.LinearTerm(jnp.c_[x, x], name="b0")
        assert jnp.allclose(x, term.basis.value[:, 0])
        assert jnp.allclose(x, term.basis.value[:, 1])
        assert jnp.allclose(jnp.zeros_like(x), term.value)


class TestSmoothTerm:
    def test_init(self) -> None:
        x = jnp.linspace(0, 1, 10)
        term = gam.SmoothTerm(
            basis=lsl.Var(jnp.c_[x, x]),
            penalty=jnp.eye(2),
            scale=lsl.Var(1.0),
            name="t",
        )

        assert term.basis.value.shape == (10, 2)
        assert term.nbases == 2
        assert jnp.allclose(jnp.zeros(2), term.coef.value)
        assert jnp.allclose(jnp.zeros(10), term.value)
        assert not jnp.isnan(term.coef.log_prob)
        assert term.coef.log_prob is not None

    def test_init_ig(self) -> None:
        x = jnp.linspace(0, 1, 10)
        term = gam.SmoothTerm.new_ig(
            basis=lsl.Var(jnp.c_[x, x]),
            penalty=jnp.eye(2),
            name="t",
        )

        assert jnp.allclose(term.scale.value, 1.0)

        assert term.basis.value.shape == (10, 2)
        assert term.nbases == 2
        assert jnp.allclose(jnp.zeros(2), term.coef.value)
        assert jnp.allclose(jnp.zeros(10), term.value)
        assert not jnp.isnan(term.coef.log_prob)
        assert term.coef.log_prob is not None
