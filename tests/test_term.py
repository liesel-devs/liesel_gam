import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import pytest
from jax import Array
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


class TestIntercept:
    def test_init(self) -> None:
        gam.Intercept("test")


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

    def test_init_2d_scale(self) -> None:
        x = jnp.linspace(0, 1, 10)
        with pytest.raises(ValueError):
            gam.SmoothTerm(
                basis=gam.Basis(jnp.c_[x, x], xname="x"),
                penalty=jnp.eye(2),
                scale=lsl.Var(jnp.ones(2)),
                name="t",
            )

        with pytest.raises(ValueError):
            gam.SmoothTerm(
                basis=gam.Basis(jnp.c_[x, x], xname="x"),
                penalty=jnp.eye(2),
                scale=jnp.ones(2),
                name="t",
            )

        gam.SmoothTerm(
            basis=gam.Basis(jnp.c_[x, x], xname="x"),
            penalty=jnp.eye(2),
            scale=lsl.Var(jnp.ones(2)),
            name="t",
            validate_scalar_scale=False,
        )

        gam.SmoothTerm(
            basis=gam.Basis(jnp.c_[x, x], xname="x"),
            penalty=jnp.eye(2),
            scale=jnp.ones(2),
            name="t",
            validate_scalar_scale=False,
        )

        with pytest.raises(ValueError):
            gam.SmoothTerm(
                basis=gam.Basis(jnp.c_[x, x, x], xname="x"),
                penalty=jnp.eye(2),
                scale=lsl.Var(jnp.ones(2)),
                name="t",
            )

        with pytest.raises(ValueError):
            gam.SmoothTerm(
                basis=gam.Basis(jnp.c_[x, x, x], xname="x"),
                penalty=jnp.eye(2),
                scale=jnp.ones(2),
                name="t",
            )

    def test_no_name(self) -> None:
        x = jnp.linspace(0, 1, 10)
        term = gam.SmoothTerm(
            basis=gam.Basis(jnp.c_[x, x]),
            penalty=jnp.eye(2),
            scale=lsl.Var(jnp.array(1.0)),
        )

        assert term.name == ""
        assert term.basis.name == ""
        assert term.coef.name == ""

    def test_scale_ig(self) -> None:
        x = jnp.linspace(0, 1, 10)
        term = gam.SmoothTerm(
            basis=gam.Basis(jnp.c_[x, x]),
            penalty=jnp.eye(2),
            scale=gam.ScaleIG(10.0, 2.0, 0.005),
        )

        assert term.scale is not None
        var = term.scale.value_node[0]  # type: ignore
        assert isinstance(var.inference, gs.MCMCSpec)  # type: ignore

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


class TestTPTerm:
    def test_init(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        s1 = tb.ps("x", k=10)
        s2 = tb.ps("y", k=10)

        ta = gam.TPTerm.f(s1, s2)

        assert ta.coef.value.shape == (9 * 9,)
        assert "x" in ta.input_obs
        assert "y" in ta.input_obs

    def test_tp(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        s1 = tb.tp("x", "area", k=10)
        s2 = tb.ps("y", k=10)

        ta = gam.TPTerm.f(s1, s2)

        assert ta.coef.value.shape == (9 * 9,)
        assert "x" in ta.input_obs
        assert "y" in ta.input_obs
        assert "area" in ta.input_obs

    def test_basis(self, columb):
        x = lsl.Var.new_obs(jnp.expand_dims(columb["x"].to_numpy(), -1), name="x")
        exp_x = lsl.Var.new_calc(jnp.exp, x, name="exp(x)")
        Bx = gam.Basis(exp_x, penalty=jnp.eye(1))
        By = gam.Basis(
            jnp.expand_dims(columb["y"].to_numpy(), -1), xname="y", penalty=jnp.eye(1)
        )

        t1 = gam.Term.f(Bx, scale=1.0)
        t2 = gam.Term.f(By, scale=1.0)

        ta = gam.TPTerm(t1, t2)
        assert "x" in ta.input_obs
        assert "y" in ta.input_obs

    def test_invalid_scale(self, columb):
        x = lsl.Var.new_obs(jnp.expand_dims(columb["x"].to_numpy(), -1), name="x")
        exp_x = lsl.Var.new_calc(jnp.exp, x, name="exp(x)")
        Bx = gam.Basis(exp_x, penalty=jnp.eye(1))
        By = gam.Basis(
            jnp.expand_dims(columb["y"].to_numpy(), -1), xname="y", penalty=jnp.eye(1)
        )

        t1 = gam.Term.f(Bx, scale=1.0)
        t2 = gam.Term.f(By)

        with pytest.raises(ValueError):
            gam.TPTerm(t1, t2)
