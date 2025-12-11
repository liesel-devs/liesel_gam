import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd
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
        assert jnp.allclose(term.scale.value, 100.0)

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

    def test_init_with_weak_penalty(self) -> None:
        a = lsl.Var.new_param(1.0)
        pen = lsl.Var.new_calc(lambda a: a * jnp.eye(5), a)

        x = jax.random.uniform(jax.random.key(1), (10, 5))
        basis = gam.Basis(x)
        with pytest.raises(NotImplementedError):
            gam.StrctTerm(basis, penalty=pen, scale=1.0)

    def test_init_varigprior(sel):
        x = jax.random.uniform(jax.random.key(1), (10, 5))
        basis = gam.Basis(x)
        term = gam.StrctTerm(basis, penalty=None, scale=gam.VarIGPrior(1.0, 0.005, 2.0))
        assert term.scale.value == pytest.approx(jnp.sqrt(2.0))

        assert term.scale.value_node[0].dist_node[
            "concentration"
        ].value == pytest.approx(1.0)
        assert term.scale.value_node[0].dist_node["scale"].value == pytest.approx(0.005)

        with pytest.raises(ValueError):
            gam.StrctTerm(
                basis, penalty=None, scale=gam.VarIGPrior(jnp.ones(2), 0.005, 2.0)
            )

        with pytest.raises(ValueError):
            gam.StrctTerm(
                basis, penalty=None, scale=gam.VarIGPrior(1.0, jnp.ones(2), 2.0)
            )

        with pytest.raises(ValueError):
            gam.StrctTerm(
                basis, penalty=None, scale=gam.VarIGPrior(1.0, 1.0, jnp.ones(2))
            )

        with pytest.raises(ValueError, match="1 or 5, got size 2"):
            gam.StrctTerm(
                basis,
                penalty=None,
                scale=gam.VarIGPrior(1.0, 1.0, jnp.ones(2)),
                validate_scalar_scale=False,
            )

        with pytest.raises(RuntimeError, match="Failed to setup Gibbs kernel"):
            gam.StrctTerm(
                basis,
                penalty=None,
                scale=gam.VarIGPrior(1.0, 1.0, jnp.ones(5)),
                validate_scalar_scale=False,
            )

        gam.StrctTerm(
            basis,
            penalty=None,
            scale=lsl.Var.new_param(jnp.ones(5)),
            validate_scalar_scale=False,
        )

    def test_scale_types(self):
        x = jax.random.uniform(jax.random.key(1), (10, 5))
        basis = gam.Basis(x)
        with pytest.raises(TypeError, match="Unexpected type for scale"):
            gam.StrctTerm(basis, penalty=None, scale="test")

        with pytest.raises(TypeError, match="Unexpected type for scale"):
            gam.StrctTerm(basis, penalty=None, scale=lsl.Var.new_param("test"))


class TestNonCentering:
    def test_scale_is_none(self):
        x = jax.random.uniform(jax.random.key(1), (10, 5))
        basis = gam.Basis(x)
        term = gam.StrctTerm(basis, penalty=None, scale=None)

        with pytest.raises(ValueError, match="Noncentering"):
            term.reparam_noncentered()

    def test_reparam_twice(self):
        x = jax.random.uniform(jax.random.key(1), (10, 5))
        basis = gam.Basis(x)
        scale = lsl.Var(2.0, name="a")
        term = gam.StrctTerm(basis, penalty=None, scale=scale)
        term.reparam_noncentered()
        assert term.scale is scale
        assert term.coef.dist_node["scale"].value == pytest.approx(1.0)

        # does nothing
        term.reparam_noncentered()
        assert term.scale is scale
        assert term.coef.dist_node["scale"].value == pytest.approx(1.0)

    def test_reparam_with_scale_ig(self):
        x = jax.random.uniform(jax.random.key(1), (10, 5))
        basis = gam.Basis(x)
        scale = gam.ScaleIG(1.0, 1.0, 0.005, name="a")
        term = gam.StrctTerm(basis, penalty=None, scale=scale)
        term.reparam_noncentered()
        assert term.scale is scale
        assert term.coef.dist_node["scale"].value == pytest.approx(1.0)

    def test_reparam_with_scale_ig_multivariate(self):
        x = jax.random.uniform(jax.random.key(1), (10, 5))
        basis = gam.Basis(x)
        scale = gam.ScaleIG(1.0, 1.0, 0.005, name="a")
        term = gam.StrctTerm(
            basis, penalty=None, scale=scale, validate_scalar_scale=False
        )
        scale.value_node[0].value = jnp.ones(5)
        scale.update()
        with pytest.raises(RuntimeError):
            term.reparam_noncentered()


class TestStrctTermFConstructor:
    def test_reparam_with_scale_ig(self):
        x = jax.random.uniform(jax.random.key(1), (10, 5))
        basis = gam.Basis(x)
        with pytest.raises(ValueError, match="must be named"):
            gam.StrctTerm.f(basis, scale=1.0)

        basis = gam.Basis(x, xname="x")
        basis.name = ""
        with pytest.raises(ValueError, match="must be named"):
            gam.StrctTerm.f(basis, scale=1.0)

        basis = gam.Basis(x, xname="x")
        gam.StrctTerm.f(basis, scale=1.0)

    def test_name_type(self):
        x = jax.random.uniform(jax.random.key(1), (10, 5))
        basis = gam.Basis(x, xname="x")
        scale = lsl.Var(2.0, name="a")
        with pytest.raises(TypeError):
            gam.StrctTerm.f(basis, scale=scale, noncentered=True, fname=basis)

    def test_init_noncentered(self):
        x = jax.random.uniform(jax.random.key(1), (10, 5))
        basis = gam.Basis(x, xname="x")
        scale = lsl.Var(2.0, name="a")
        term = gam.StrctTerm.f(basis, scale=scale, noncentered=True)

        assert term.scale is scale
        assert term.coef.dist_node["scale"].value == pytest.approx(1.0)

    def test_init_new_ig_noncentered(self):
        x = jax.random.uniform(jax.random.key(1), (10, 5))
        basis = gam.Basis(x, xname="x")
        term = gam.StrctTerm.new_ig(
            basis, penalty=basis.penalty, noncentered=True, name="test"
        )

        assert term.scale.value == pytest.approx(100.0)
        assert term.coef.dist_node["scale"].value == pytest.approx(1.0)


class TestTermWithCustomPenalty:
    def test_init_diag_prior(self):
        x = jax.random.uniform(jax.random.key(1), (10, 5))
        basis = gam.Basis(x)
        term = gam.StrctTerm(basis, penalty=None, scale=1.0)
        assert isinstance(term.coef.dist_node.init_dist(), tfd.Normal)

    def test_penalty_none(self) -> None:
        x = jax.random.uniform(jax.random.key(1), (10, 5))
        basis = gam.Basis(x)
        term = gam.StrctTerm(basis, penalty=None, scale=1.0)
        assert term._penalty is None

        with pytest.raises(ValueError, match="is None"):
            term.scale_penalty()

        with pytest.raises(ValueError, match="is None"):
            term.constrain("sumzero_term")

        with pytest.raises(ValueError, match="is None"):
            term.diagonalize_penalty()

    def test_penalty_diag_different_object(self) -> None:
        x = jax.random.uniform(jax.random.key(1), (10, 5))
        basis = gam.Basis(x)
        term = gam.StrctTerm(basis, penalty=jnp.eye(5), scale=1.0)

        with pytest.raises(ValueError, match="Different penalty"):
            term.scale_penalty()

        with pytest.raises(ValueError, match="Different penalty"):
            term.constrain("sumzero_term")

        with pytest.raises(ValueError, match="Different penalty"):
            term.diagonalize_penalty()

    def test_penalty_diag_same_object(self) -> None:
        x = jax.random.uniform(jax.random.key(1), (10, 5))
        basis = gam.Basis(x)
        term = gam.StrctTerm(basis, penalty=basis.penalty, scale=1.0)

        term.scale_penalty()
        term.constrain("sumzero_term")
        term.diagonalize_penalty()


class TestTPTerm:
    def test_init(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        s1 = tb.ps("x", k=10)
        s2 = tb.ps("y", k=10)

        ta = gam.StrctTensorProdTerm.f(s1, s2)

        assert ta.coef.value.shape == (9 * 9,)
        assert "x" in ta.input_obs
        assert "y" in ta.input_obs

    def test_tp(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        s1 = tb.tp("x", "area", k=10)
        s2 = tb.ps("y", k=10)

        ta = gam.StrctTensorProdTerm.f(s1, s2)

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

        t1 = gam.StrctTerm.f(Bx, scale=1.0)
        t2 = gam.StrctTerm.f(By, scale=1.0)

        ta = gam.StrctTensorProdTerm(t1, t2)
        assert "x" in ta.input_obs
        assert "y" in ta.input_obs

    def test_invalid_scale(self, columb):
        x = lsl.Var.new_obs(jnp.expand_dims(columb["x"].to_numpy(), -1), name="x")
        exp_x = lsl.Var.new_calc(jnp.exp, x, name="exp(x)")
        Bx = gam.Basis(exp_x, penalty=jnp.eye(1))
        By = gam.Basis(
            jnp.expand_dims(columb["y"].to_numpy(), -1), xname="y", penalty=jnp.eye(1)
        )

        t1 = gam.StrctTerm.f(Bx, scale=1.0)
        t2 = gam.StrctTerm.f(By)

        with pytest.raises(ValueError):
            gam.StrctTensorProdTerm(t1, t2)

    def test_non_penalty(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        px = tb.ps("x", k=20)
        py = tb.ps("y", k=20)

        px.basis._penalty = None

        with pytest.raises(TypeError):
            gam.StrctTensorProdTerm(px, py)

    def test_include_main_effects(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        px = tb.ps("x", k=20)
        py = tb.ps("y", k=20)

        tp = gam.StrctTensorProdTerm(px, py, include_main_effects=True)
        assert tp.value_node[0] is px
        assert tp.value_node[1] is py
        assert isinstance(tp.value_node["basis"], lsl.Var)
        assert isinstance(tp.value_node["coef"], lsl.Var)

        tp = gam.StrctTensorProdTerm(px, py)
        with pytest.raises(IndexError):
            tp.value_node[0]
        assert isinstance(tp.value_node["basis"], lsl.Var)
        assert isinstance(tp.value_node["coef"], lsl.Var)

    def test_strong_input_obs(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        px = tb.ps("x", k=20)
        py = tb.ps("y", k=20)

        assert px.basis.x.strong
        assert py.basis.x.strong

        tp = gam.StrctTensorProdTerm(px, py)
        assert tp.input_obs["x"] is px.basis.x
        assert jnp.allclose(tp.input_obs["x"].value, columb["x"].to_numpy())
        assert tp.input_obs["y"] is py.basis.x
        assert jnp.allclose(tp.input_obs["y"].value, columb["y"].to_numpy())

        px.basis.x.name = ""
        with pytest.raises(ValueError):
            tp.input_obs

        x = px.basis.value_node[0]
        x_weak = lsl.Var.new_calc(jnp.square, x, name="x**2")

        px.basis.value_node[0] = x_weak
        with pytest.raises(ValueError):
            tp.input_obs

        x.name = "x"
        "x" in tp.input_obs

    def test_weak_input_obs(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        px = tb.slin("x + area")
        py = tb.ps("y", k=20)

        assert isinstance(px.basis.x, lsl.TransientCalc)
        assert py.basis.x.strong

        tp = gam.StrctTensorProdTerm(px, py)
        assert jnp.allclose(tp.input_obs["x"].value, columb["x"].to_numpy())
        assert jnp.allclose(tp.input_obs["area"].value, columb["area"].to_numpy())
        assert tp.input_obs["y"] is py.basis.x
        assert jnp.allclose(tp.input_obs["y"].value, columb["y"].to_numpy())

        tp.input_obs["x"].name = ""
        with pytest.raises(ValueError):
            tp.input_obs


class TestIndexingTerm:
    def test_init(self):
        x = jnp.arange(10, dtype=jnp.int32)
        basis = gam.Basis(x, xname="x", penalty=None)
        scale = lsl.Var(2.0, name="a")
        term = gam.IndexingTerm.f(basis, scale=scale)
        assert term._penalty is None

    def test_constraints(self):
        x = jnp.arange(10, dtype=jnp.int32)
        basis = gam.Basis(x, xname="x", penalty=None)
        scale = lsl.Var(2.0, name="a")
        term = gam.IndexingTerm.f(basis, scale=scale)

        with pytest.raises(ValueError):
            term.scale_penalty()

        with pytest.raises(ValueError):
            term.diagonalize_penalty()

        with pytest.raises(ValueError):
            term.constrain("sumzero_coef")

    def test_init_validation(self):
        x = jnp.arange(10, dtype=jnp.float32)
        basis = gam.Basis(x, xname="x", penalty=None)
        scale = lsl.Var(2.0, name="a")
        with pytest.raises(TypeError):
            gam.IndexingTerm.f(basis, scale=scale)

        basis = gam.Basis(jnp.c_[x, x], xname="x", penalty=None)
        scale = lsl.Var(2.0, name="a")
        with pytest.raises(ValueError):
            gam.IndexingTerm.f(basis, scale=scale)

    def test_full_basis(self):
        x = jnp.arange(10, dtype=jnp.int32)
        basis = gam.Basis(x, xname="x", penalty=None)
        scale = lsl.Var(2.0, name="a")
        term = gam.IndexingTerm.f(basis, scale=scale)

        b = term.init_full_basis()
        assert b.value.shape == (10, 10)
        assert b.penalty is not None
        assert b.penalty.value.shape == (10, 10)


class TestRITerm:
    def test_full_basis(self):
        x = jnp.arange(10, dtype=jnp.int32)
        basis = gam.Basis(x, xname="x", penalty=None)
        scale = lsl.Var(2.0, name="a")
        term = gam.RITerm.f(basis, scale=scale)

        b = term.init_full_basis()
        assert b.value.shape == (10, 10)
        assert b.penalty is not None
        assert b.penalty.value.shape == (10, 10)

    def test_special_attributes(self):
        x = jnp.arange(10, dtype=jnp.int32)
        basis = gam.Basis(x, xname="x", penalty=None)
        scale = lsl.Var(2.0, name="a")
        term = gam.RITerm.f(basis, scale=scale)

        with pytest.raises(ValueError):
            term.labels = ["a", "b"]

        term.labels = ["a" + str(i) for i in range(10)]
        assert len(term.labels) == 10
