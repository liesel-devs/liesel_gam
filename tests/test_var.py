import jax
import jax.numpy as jnp
import liesel.model as lsl
import pytest
import scipy
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_gam as gam


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

    def test_dist_works(self) -> None:
        x = jnp.linspace(0, 1, 5)
        dist = lsl.Dist(tfd.Normal, loc=0.0, scale=2.0)
        term = gam.LinearTerm(
            jnp.c_[x, x],
            name="b0",
            distribution=dist,
        )
        assert term.coef.dist_node is dist

    def test_default_dist_is_none(self) -> None:
        x = jnp.linspace(0, 1, 5)
        term = gam.LinearTerm(
            jnp.c_[x, x],
            name="b0",
        )
        assert term.coef.dist_node is None


class TestLinearTerm2:
    def test_with_intercept(self) -> None:
        x = jnp.linspace(0, 1, 5)
        term = gam.var.LinearTerm2(x, name="b0", add_intercept=True)
        assert jnp.allclose(jnp.zeros_like(x), term.value)

        # intercept column
        assert jnp.allclose(1.0, term.basis.value[:, 0])

        # x column
        assert jnp.allclose(x, term.basis.value[:, 1])

    def test_no_intercept(self) -> None:
        x = jnp.linspace(0, 1, 5)
        term = gam.var.LinearTerm2(x, name="b0")
        assert jnp.allclose(x, term.basis.value[:, 0])

    def test_bivariate_works(self) -> None:
        x = jnp.linspace(0, 1, 5)
        term = gam.var.LinearTerm2(jnp.c_[x, x], name="b0")
        assert jnp.allclose(x, term.basis.value[:, 0])
        assert jnp.allclose(x, term.basis.value[:, 1])
        assert jnp.allclose(jnp.zeros_like(x), term.value)

    def test_default_dist(self) -> None:
        x = jnp.linspace(0, 1, 5)
        term = gam.var.LinearTerm2(x, name="b0")
        dist = term.coef.dist_node
        assert dist is None

    def test_dist_works(self) -> None:
        x = jnp.linspace(0, 1, 5)
        scale = lsl.Var.new_param(1.0, name="scale")
        prior = gam.var.mvn_diag_prior(scale=scale)
        term = gam.var.LinearTerm2(x, name="b0", prior=prior)
        assert term.coef.dist_node is not None
        assert term.coef.dist_node["scale"] is scale


class TestSmoothTerm:
    def test_init(self) -> None:
        x = jnp.linspace(0, 1, 10)
        term = gam.SmoothTerm(
            basis=gam.Basis(jnp.c_[x, x], xname="x"),
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

    def test_scale_none(self) -> None:
        x = jnp.linspace(0, 1, 10)

        with pytest.raises(ValueError):
            gam.SmoothTerm(
                basis=gam.Basis(jnp.c_[x, x], xname="x"),
                penalty=jnp.eye(2),
                scale=None,
                name="t",
            )

        term = gam.SmoothTerm(
            basis=gam.Basis(jnp.c_[x, x], xname="x"),
            penalty=None,
            scale=None,
            name="t",
        )

        assert term.scale is None
        assert term.coef.dist_node is None
        assert term.basis.value.shape == (10, 2)
        assert term.nbases == 2
        assert jnp.allclose(jnp.zeros(2), term.coef.value)
        assert jnp.allclose(jnp.zeros(10), term.value)
        assert not jnp.isnan(term.coef.log_prob)
        assert term.coef.log_prob is not None

    def test_init_ig(self) -> None:
        x = jnp.linspace(0, 1, 10)
        term = gam.SmoothTerm.new_ig(
            basis=gam.Basis(jnp.c_[x, x], xname="x"),
            penalty=jnp.eye(2),
            name="t",
        )

        assert term.scale is not None
        assert jnp.allclose(term.scale.value, 10.0)

        assert term.basis.value.shape == (10, 2)
        assert term.nbases == 2
        assert jnp.allclose(jnp.zeros(2), term.coef.value)
        assert jnp.allclose(jnp.zeros(10), term.value)
        assert not jnp.isnan(term.coef.log_prob)
        assert term.coef.log_prob is not None

    def test_init_ig_1d(self) -> None:
        x = jnp.linspace(0, 1, 10)
        term = gam.SmoothTerm.new_ig(
            basis=gam.Basis(jnp.expand_dims(x, 1), xname="x"),
            penalty=jnp.eye(1),
            name="t",
        )
        model = lsl.Model([term])
        assert isinstance(term.scale, lsl.Var)
        tau2 = term.scale.value_node[0]
        kernel = tau2.inference.kernel([tau2.name], term.coef, term.scale)  # type: ignore
        proposal = kernel._transition_fn(jax.random.key(1), model.state)  # type: ignore
        assert not jnp.isinf(proposal[tau2.name])
        assert not jnp.isnan(proposal[tau2.name])
        assert proposal[tau2.name] > 0.0
        assert proposal[tau2.name].size == 1

    def test_init_ig_2d(self) -> None:
        x = jnp.linspace(0, 1, 10)
        term = gam.SmoothTerm.new_ig(
            basis=gam.Basis(jnp.c_[x, x], xname="x"),
            penalty=jnp.eye(2),
            name="t",
        )
        model = lsl.Model([term])
        assert isinstance(term.scale, lsl.Var)
        tau2 = term.scale.value_node[0]
        kernel = tau2.inference.kernel([tau2.name], term.coef, term.scale)  # type: ignore
        proposal = kernel._transition_fn(jax.random.key(1), model.state)  # type: ignore
        assert not jnp.isinf(proposal[tau2.name])
        assert not jnp.isnan(proposal[tau2.name])
        assert proposal[tau2.name] > 0.0
        assert proposal[tau2.name].size == 1
