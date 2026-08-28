import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import Array

import liesel_gam as gam
from liesel_gam.term import _factorized_tensor_dot

from .mgcv_data import load_columb


def _as_interaction(
    term: gam.StrctTerm | gam.StrctInteractionTerm,
) -> gam.StrctInteractionTerm:
    assert isinstance(term, gam.StrctInteractionTerm)
    return term


@pytest.fixture(scope="module")
def columb():
    return load_columb()


def pspline_penalty(nparam: int, random_walk_order: int = 2) -> Array:
    """
    Builds an (nparam x nparam) P-spline penalty matrix.
    """
    D = jnp.diff(jnp.identity(nparam), random_walk_order, axis=0)
    return D.T @ D


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
        term = gam.SmoothTerm.f(
            basis=gam.Basis(jnp.c_[x, x], xname="x"),
            scale=gam.VarIGPrior(1.0, 0.005, 100.0**2),
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
        term = gam.SmoothTerm.f(
            basis=gam.Basis(jnp.expand_dims(x, 1), xname="x"),
            scale=gam.VarIGPrior(1.0, 0.005, 100.0**2),
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
        term = gam.SmoothTerm.f(
            basis=gam.Basis(jnp.c_[x, x], xname="x"),
            scale=gam.VarIGPrior(1.0, 0.005, 100.0**2),
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
        assert isinstance(term.scale, lsl.Var)
        scale_var = term.scale.value_node[0]
        assert isinstance(scale_var, lsl.Var)
        assert scale_var.dist_node is not None
        assert term.scale.value == pytest.approx(jnp.sqrt(2.0))

        assert scale_var.dist_node["concentration"].value == pytest.approx(1.0)
        assert scale_var.dist_node["scale"].value == pytest.approx(0.005)

        with pytest.raises(ValueError):
            gam.StrctTerm(
                basis,
                penalty=None,
                scale=gam.VarIGPrior(jnp.ones(2), 0.005, 2.0),  # type: ignore
            )

        with pytest.raises(ValueError):
            gam.StrctTerm(
                basis,
                penalty=None,
                scale=gam.VarIGPrior(1.0, jnp.ones(2), 2.0),  # type: ignore
            )

        with pytest.raises(ValueError):
            gam.StrctTerm(
                basis,
                penalty=None,
                scale=gam.VarIGPrior(1.0, 1.0, jnp.ones(2)),  # type: ignore
            )

        with pytest.raises(ValueError, match="1 or 5, got size 2"):
            gam.StrctTerm(
                basis,
                penalty=None,
                scale=gam.VarIGPrior(1.0, 1.0, jnp.ones(2)),  # type: ignore
                validate_scalar_scale=False,
            )

        with pytest.raises(RuntimeError, match="Failed to setup Gibbs kernel"):
            gam.StrctTerm(
                basis,
                penalty=None,
                scale=gam.VarIGPrior(1.0, 1.0, jnp.ones(5)),  # type: ignore
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
            gam.StrctTerm(basis, penalty=None, scale="test")  # type: ignore

        with pytest.raises(TypeError, match="Unexpected type for scale"):
            scale = lsl.Var.new_param("test", convert=lambda x: x)
            gam.StrctTerm(basis, penalty=None, scale=scale)


class TestNonCentering:
    def test_scale_is_none(self):
        x = jax.random.uniform(jax.random.key(1), (10, 5))
        basis = gam.Basis(x)
        term = gam.StrctTerm(basis, penalty=None, scale=None)

        with pytest.raises(ValueError, match="Scale factorization"):
            term.factor_scale()

    def test_reparam_twice(self):
        x = jax.random.uniform(jax.random.key(1), (10, 5))
        basis = gam.Basis(x)
        scale = lsl.Var(2.0, name="a")
        term = gam.StrctTerm(basis, penalty=None, scale=scale)
        term.factor_scale()
        assert term.scale is scale
        assert term.coef.dist_node is not None
        assert term.coef.dist_node["scale"].value == pytest.approx(1.0)

        # does nothing
        term.factor_scale()
        assert term.scale is scale
        assert term.coef.dist_node is not None
        assert term.coef.dist_node["scale"].value == pytest.approx(1.0)

    def test_reparam_with_scale_ig(self):
        x = jax.random.uniform(jax.random.key(1), (10, 5))
        basis = gam.Basis(x)
        scale = gam.ScaleIG(1.0, 1.0, 0.005, name="a")
        term = gam.StrctTerm(basis, penalty=None, scale=scale)
        term.factor_scale()
        assert term.scale is scale
        assert term.coef.dist_node is not None
        assert term.coef.dist_node["scale"].value == pytest.approx(1.0)

    def test_reparam_with_scale_ig_multivariate(self):
        x = jax.random.uniform(jax.random.key(1), (10, 5))
        basis = gam.Basis(x)
        scale = gam.ScaleIG(1.0, 1.0, 0.005, name="a")
        term = gam.StrctTerm(
            basis, penalty=None, scale=scale, validate_scalar_scale=False
        )
        assert isinstance(scale.value_node[0], lsl.Var)
        scale.value_node[0].value = jnp.ones(5)
        scale.update()
        with pytest.raises(ValueError):
            term.factor_scale()


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
            gam.StrctTerm.f(basis, scale=scale, factor_scale=True, fname=basis)  # type: ignore

    def test_init_factor_scale(self):
        x = jax.random.uniform(jax.random.key(1), (10, 5))
        basis = gam.Basis(x, xname="x")
        scale = lsl.Var(2.0, name="a")
        term = gam.StrctTerm.f(basis, scale=scale, factor_scale=True)

        assert term.scale is scale
        assert term.coef.dist_node is not None
        assert term.coef.dist_node["scale"].value == pytest.approx(1.0)

    def test_init_new_ig_factor_scale(self):
        x = jax.random.uniform(jax.random.key(1), (10, 5))
        basis = gam.Basis(x, xname="x")
        term = gam.StrctTerm.f(
            basis, scale=gam.VarIGPrior(1.0, 0.005, 10.0**2)
        ).factor_scale()

        assert isinstance(term.scale, lsl.Var)
        assert term.coef.dist_node is not None
        assert term.scale.value == pytest.approx(10.0)
        assert term.coef.dist_node["scale"].value == pytest.approx(1.0)


class TestTermWithCustomPenalty:
    def test_init_diag_prior(self):
        x = jax.random.uniform(jax.random.key(1), (10, 5))
        basis = gam.Basis(x)
        term = gam.StrctTerm(basis, penalty=None, scale=1.0)
        assert term.coef.dist_node is not None
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


class TestInteractionTerm:
    def test_requires_at_least_two_marginals(self, columb):
        marginal = gam.TermBuilder.from_df(columb).ps("x", k=10)

        with pytest.raises(ValueError, match="at least two"):
            gam.StrctInteractionTerm(marginal)

    def test_init(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        s1 = tb.ps("x", k=10)
        s2 = tb.ps("y", k=10)

        ta = gam.StrctInteractionTerm.f(s1, s2)

        assert ta.coef.value.shape == (9 * 9,)
        assert "x" in ta.input_obs
        assert "y" in ta.input_obs

    def test_tp(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        s1 = tb.tp("x", "area", k=10)
        s2 = tb.ps("y", k=10)

        ta = gam.StrctInteractionTerm.f(s1, s2)

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

        ta = gam.StrctInteractionTerm(t1, t2)
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
            gam.StrctInteractionTerm(t1, t2)

    def test_non_penalty(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        px = tb.ps("x", k=20)
        py = tb.ps("y", k=20)

        px.basis._penalty = None

        with pytest.raises(TypeError):
            gam.StrctInteractionTerm(px, py)

    def test_include_main_effects(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        px = tb.ps("x", k=20)
        py = tb.ps("y", k=20)

        tp = gam.StrctInteractionTerm(px, py, include_main_effects=True)
        assert tp.value_node[0] is px
        assert tp.value_node[1] is py
        assert tp.value_node[2] is px.basis
        assert tp.value_node[3] is py.basis
        assert isinstance(tp.value_node["coef"], lsl.Var)

        tp = gam.StrctInteractionTerm(px, py)
        assert tp.value_node[0] is px.basis
        assert tp.value_node[1] is py.basis
        assert isinstance(tp.value_node["coef"], lsl.Var)

    def test_factorized_value_and_gradient_match_explicit_basis(self) -> None:
        n = 7
        b1 = jax.random.normal(jax.random.key(1), (n, 2))
        b2 = jax.random.normal(jax.random.key(2), (n, 3))
        b3 = jax.random.normal(jax.random.key(3), (n, 4))
        terms = [
            gam.StrctTerm.f(
                gam.Basis(basis, xname=f"x{i}", penalty=jnp.eye(basis.shape[1])),
                scale=1.0,
            )
            for i, basis in enumerate((b1, b2, b3))
        ]
        term = gam.StrctInteractionTerm(*terms)
        coef = jnp.linspace(-1.0, 1.0, term.nbases)
        explicit_basis = jax.vmap(lambda x, y, z: jnp.kron(jnp.kron(x, y), z))(
            b1, b2, b3
        )

        term.coef.value = coef
        term.update()
        assert not hasattr(term, "basis")
        assert jnp.allclose(term.value, explicit_basis @ coef, atol=1e-5)

        expected_grad = jax.grad(lambda beta: jnp.sum(explicit_basis @ beta))(coef)
        actual_grad = jax.grad(
            lambda beta: jnp.sum(
                _factorized_tensor_dot(
                    beta,
                    (b1, b2, b3),
                    marginal_sizes=(2, 3, 4),
                    indexed=(False, False, False),
                )
            )
        )(coef)
        assert jnp.allclose(actual_grad, expected_grad, atol=1e-5)

    def test_strong_input_obs(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        px = tb.ps("x", k=20)
        py = tb.ps("y", k=20)

        assert isinstance(px.basis.x, lsl.Var)
        assert isinstance(py.basis.x, lsl.Var)
        assert px.basis.x.strong
        assert py.basis.x.strong

        tp = gam.StrctInteractionTerm(px, py)
        assert tp.input_obs["x"] is px.basis.x
        assert jnp.allclose(tp.input_obs["x"].value, columb["x"].to_numpy())
        assert tp.input_obs["y"] is py.basis.x
        assert jnp.allclose(tp.input_obs["y"].value, columb["y"].to_numpy())

        px.basis.x.name = ""
        with pytest.raises(ValueError):
            _ = tp.input_obs

        x = px.basis.value_node[0]
        x_weak = lsl.Var.new_calc(jnp.square, x, name="x**2")

        px.basis.value_node[0] = x_weak
        with pytest.raises(ValueError):
            _ = tp.input_obs

        x.name = "x"
        assert "x" in tp.input_obs

    def test_weak_input_obs(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        px = tb.slin("x + area")
        py = tb.ps("y", k=20)

        assert isinstance(px.basis.x, lsl.TransientCalc)
        assert isinstance(py.basis.x, lsl.Var)
        assert py.basis.x.strong

        tp = gam.StrctInteractionTerm(px, py)
        assert jnp.allclose(tp.input_obs["x"].value, columb["x"].to_numpy())
        assert jnp.allclose(tp.input_obs["area"].value, columb["area"].to_numpy())
        assert tp.input_obs["y"] is py.basis.x
        assert jnp.allclose(tp.input_obs["y"].value, columb["y"].to_numpy())

        tp.input_obs["x"].name = ""
        with pytest.raises(ValueError):
            _ = tp.input_obs


class TestTensorProdTerm:
    def test_init_2d(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        s1 = tb.ps("x", k=10)
        s2 = tb.ps("y", k=10)

        ta = gam.StrctTensorProdTerm(s1, s2)
        assert "x" in ta.input_obs
        assert "y" in ta.input_obs

        assert 1 in ta.terms_by_order
        assert 2 in ta.terms_by_order
        assert len(ta.terms_by_order[1]) == 2
        assert len(ta.terms_by_order[2]) == 1

        ti = _as_interaction(ta.terms_by_order[2][0])

        assert ti.coef.value.shape == (9 * 9,)
        assert "x" in ti.input_obs
        assert "y" in ti.input_obs

    def test_init_3d(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        s1 = tb.ps("x", k=10)
        s2 = tb.ps("y", k=10)
        s3 = tb.ps("area", k=10)

        ta = gam.StrctTensorProdTerm(s1, s2, s3)
        assert "x" in ta.input_obs
        assert "y" in ta.input_obs
        assert "area" in ta.input_obs

        assert 1 in ta.terms_by_order
        assert 2 in ta.terms_by_order
        assert 3 in ta.terms_by_order
        assert len(ta.terms_by_order[1]) == 3
        assert len(ta.terms_by_order[2]) == 3
        assert len(ta.terms_by_order[3]) == 1

        ti = _as_interaction(ta.terms_by_order[2][0])

        assert ti.coef.value.shape == (9 * 9,)
        assert "x" in ti.input_obs
        assert "y" in ti.input_obs

        ti3 = _as_interaction(ta.terms_by_order[3][0])
        assert ti3.coef.value.shape == (9 * 9 * 9,)
        assert "x" in ti3.input_obs
        assert "y" in ti3.input_obs
        assert "area" in ti3.input_obs

    def test_order(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        s1 = tb.ps("x", k=10)
        s2 = tb.ps("y", k=10)
        s3 = tb.ps("area", k=10)

        ta = gam.StrctTensorProdTerm(s1, s2, s3, order=(1,))
        assert "x" in ta.input_obs
        assert "y" in ta.input_obs
        assert "area" in ta.input_obs

        assert 1 in ta.terms_by_order
        assert 2 not in ta.terms_by_order
        assert 3 not in ta.terms_by_order
        assert len(ta.terms_by_order[1]) == 3

        ta = gam.StrctTensorProdTerm(s1, s2, s3, order=(2,))
        assert "x" in ta.input_obs
        assert "y" in ta.input_obs
        assert "area" in ta.input_obs

        assert 1 not in ta.terms_by_order
        assert 2 in ta.terms_by_order
        assert 3 not in ta.terms_by_order
        assert len(ta.terms_by_order[2]) == 3

        ti = _as_interaction(ta.terms_by_order[2][0])

        assert ti.coef.value.shape == (9 * 9,)
        assert "x" in ti.input_obs
        assert "y" in ti.input_obs

        ta = gam.StrctTensorProdTerm(s1, s2, s3, order=(3,))
        assert "x" in ta.input_obs
        assert "y" in ta.input_obs
        assert "area" in ta.input_obs

        assert 1 not in ta.terms_by_order
        assert 2 not in ta.terms_by_order
        assert 3 in ta.terms_by_order
        assert len(ta.terms_by_order[3]) == 1

        ti3 = _as_interaction(ta.terms_by_order[3][0])
        assert ti3.coef.value.shape == (9 * 9 * 9,)
        assert "x" in ti3.input_obs
        assert "y" in ti3.input_obs
        assert "area" in ti3.input_obs

        ta = gam.StrctTensorProdTerm(s1, s2, s3, order=(2, 3))
        assert "x" in ta.input_obs
        assert "y" in ta.input_obs
        assert "area" in ta.input_obs

        assert 1 not in ta.terms_by_order
        assert 2 in ta.terms_by_order
        assert 3 in ta.terms_by_order
        assert len(ta.terms_by_order[2]) == 3
        assert len(ta.terms_by_order[3]) == 1

        ti = _as_interaction(ta.terms_by_order[2][0])

        assert ti.coef.value.shape == (9 * 9,)
        assert "x" in ti.input_obs
        assert "y" in ti.input_obs

        ti3 = _as_interaction(ta.terms_by_order[3][0])
        assert ti3.coef.value.shape == (9 * 9 * 9,)
        assert "x" in ti3.input_obs
        assert "y" in ti3.input_obs
        assert "area" in ti3.input_obs

        ta = gam.StrctTensorProdTerm(s1, s2, s3, order=(1, 3))
        assert "x" in ta.input_obs
        assert "y" in ta.input_obs
        assert "area" in ta.input_obs

        assert 1 in ta.terms_by_order
        assert 2 not in ta.terms_by_order
        assert 3 in ta.terms_by_order
        assert len(ta.terms_by_order[1]) == 3
        assert len(ta.terms_by_order[3]) == 1

        ti3 = _as_interaction(ta.terms_by_order[3][0])
        assert ti3.coef.value.shape == (9 * 9 * 9,)
        assert "x" in ti3.input_obs
        assert "y" in ti3.input_obs
        assert "area" in ti3.input_obs

    def test_names_prefix(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        s1 = tb.ps("x", k=10)
        s2 = tb.ps("y", k=10)
        s3 = tb.ps("area", k=10)

        ta = gam.StrctTensorProdTerm(s1, s2, s3, names_prefix="m.")
        assert ta.name.startswith("m.")

        for term in ta.terms_by_order[2]:
            assert term.name.startswith("m.")
            assert term.coef.name.startswith("m.")

        for term in ta.terms_by_order[3]:
            assert term.name.startswith("m.")
            assert term.coef.name.startswith("m.")

        for term in ta.terms_by_order[1]:
            assert not term.name.startswith("m.")
            assert not term.coef.name.startswith("m.")

    def test_common_scale(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        s1 = tb.ps("x", k=10)
        s2 = tb.ps("y", k=10)
        s3 = tb.ps("area", k=10)

        scale = lsl.Var.new_param(1.0)
        ta = gam.StrctTensorProdTerm(s1, s2, s3, common_scale=scale)

        for term in ta.terms_by_order[1]:
            assert term.scale is scale
            assert term.coef.dist_node is not None
            assert term.coef.dist_node["scale"] is scale

        for i in [2, 3]:
            for term in ta.terms_by_order[i]:
                assert isinstance(term, gam.StrctInteractionTerm)
                for term_scale in term.scales:
                    assert term_scale is scale

        assert s1.scale is scale
        assert s2.scale is scale
        assert s3.scale is scale

        assert s1.coef.dist_node is not None
        assert s2.coef.dist_node is not None
        assert s3.coef.dist_node is not None
        assert s1.coef.dist_node["scale"] is scale
        assert s2.coef.dist_node["scale"] is scale
        assert s3.coef.dist_node["scale"] is scale

    def test_tx_name(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        s1 = tb.ps("x", k=10)
        s2 = tb.ps("y", k=10)
        s3 = tb.ps("area", k=10)

        ta = gam.StrctTensorProdTerm(s1, s2, s3, tx_name="ti")

        for i in [2, 3]:
            for term in ta.terms_by_order[i]:
                assert term.name.startswith("ti(")

    def test_tf_name(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        s1 = tb.ps("x", k=10)
        s2 = tb.ps("y", k=10)
        s3 = tb.ps("area", k=10)

        ta = gam.StrctTensorProdTerm(s1, s2, s3, tf_name="ti")
        assert ta.name.startswith("ti(")

    def test_coef_name(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        s1 = tb.ps("x", k=10)
        s2 = tb.ps("y", k=10)
        s3 = tb.ps("area", k=10)

        ta = gam.StrctTensorProdTerm(s1, s2, s3, coef_name=r"\gamma")

        for term in ta.terms_by_order[1]:
            assert "beta" in term.coef.name

        for i in [2, 3]:
            for term in ta.terms_by_order[i]:
                assert "gamma" in term.coef.name

    def test_interactions_have_only_marginal_bases(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        s1 = tb.ps("x", k=10)
        s2 = tb.ps("y", k=10)
        s3 = tb.ps("area", k=10)

        ta = gam.StrctTensorProdTerm(s1, s2, s3)

        for term in ta.terms_by_order[1]:
            assert isinstance(term, gam.StrctTerm)
            assert isinstance(term.basis, gam.Basis)
            assert term.basis.name == f"B({term.basis.x.name})"

        for i in [2, 3]:
            for term in ta.terms_by_order[i]:
                assert isinstance(term, gam.StrctInteractionTerm)
                assert not hasattr(term, "basis")
                assert term.marginal_bases == list(term.bases)

    def test_group_terms_by_order(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        s1 = tb.ps("x", k=10)
        s2 = tb.ps("y", k=10)
        s3 = tb.ps("area", k=10)

        ta = gam.StrctTensorProdTerm(s1, s2, s3, group_terms_by_order=True)

        assert len(ta.all_input_vars()) == 3

        ta = gam.StrctTensorProdTerm(s1, s2, s3, group_terms_by_order=False)

        assert len(ta.all_input_vars()) == 7


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


class TestMultivariateTPTerm:
    def test_2d_spline(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        s = tb.ps(
            "x",
            k=10,
            absorb_cons=False,
            diagonal_penalty=False,
            scale_penalty=False,
        )

        s2d = gam.MultivariateStrctTerm.f(
            s,
            dimension_penalties=[jnp.eye(2)],
            dimension_scales=[lsl.Var(1.0)],
        )

        assert s2d.value.shape == (49, 2)

        dist = s2d.coef.dist_node.init_dist()  # type: ignore
        K = dist._op.materialize_penalty()  # type: ignore

        Kmarginal = jnp.kron(s.basis.penalty.value, jnp.eye(2)) + jnp.kron(  # type: ignore
            jnp.eye(2), jnp.eye(s.nbases)
        )

        assert jnp.allclose(K, Kmarginal)
