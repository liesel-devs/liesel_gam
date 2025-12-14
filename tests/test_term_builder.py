import logging
from functools import partial

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import pandas as pd
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
from liesel.contrib import splines as spl
from ryp import r, to_py

import liesel_gam as gam
import liesel_gam.term_builder as gb
from liesel_gam.term_builder import _find_parameter, _format_name, _has_star_gibbs

from .make_df import make_test_df


@pytest.fixture(scope="module")
def data():
    return make_test_df()


@pytest.fixture(scope="class")
def bases(data) -> gb.BasisBuilder:
    registry = gb.PandasRegistry(data, na_action="drop")
    bases = gb.BasisBuilder(registry)
    return bases


@pytest.fixture(scope="module")
def columb():
    r("library(mgcv)")
    r("data(columb)")
    columb = to_py("columb", format="pandas")
    return columb


@pytest.fixture(scope="module")
def columb_polys():
    r("library(mgcv)")
    r("data(columb.polys)")
    polys = to_py("columb.polys", format="numpy")
    # turn to zero-based indecing
    polys = {k: v - 1 for k, v in polys.items()}
    return polys


class TestTermBuilder:
    def test_init(self, data) -> None:
        gb.TermBuilder.from_df(data)

    def test_from_dict(self, data):
        gb.TermBuilder.from_dict(data.to_dict())

    def test_default_scale(self, columb):
        tb = gb.TermBuilder.from_df(
            columb, default_scale_fn=lambda: lsl.Var.new_param(3.0)
        )

        term = tb.ps("x", k=20)
        assert term.scale.value == pytest.approx(3.0)

    def test_ri_basis_with_unobserved_cluster(self) -> None:
        data = pd.DataFrame(
            {"x": pd.Categorical(["a", "b"], categories=["a", "b", "c"])}
        )
        tb = gb.TermBuilder.from_df(data)
        ri = tb.ri("x")
        assert ri.coef.value.size == 3

    def test_ri_basis_with_unobserved_cluster_logging(self, caplog) -> None:
        data = pd.DataFrame(
            {"x": pd.Categorical(["a", "b"], categories=["a", "b", "c"])}
        )
        tb = gb.TermBuilder.from_df(data)
        with caplog.at_level(logging.INFO, logger="liesel_gam"):
            ri = tb.ri("x")

        assert ri.coef.value.size == 3
        assert "categories without observations" in caplog.text
        assert caplog.records[0].levelno == logging.INFO


class TestLinTerm:
    def test_slin(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        term = tb.slin("x + y", factor_scale=True, scale=3.0)
        assert term.scale.value == pytest.approx(3.0)
        assert term.coef.dist_node["scale"].value == pytest.approx(1.0)

        term = tb.slin("x + y", scale=3.0)
        assert term.scale.value == pytest.approx(3.0)
        assert term.coef.dist_node["scale"].value == pytest.approx(3.0)

    def test_name(self, columb):
        tb = gam.TermBuilder.from_df(columb)

        term = tb.lin("x + y")
        assert term.name == "lin(X)"

    def test_labels_to_integers(self, data):
        tb = gam.TermBuilder.from_df(data)

        term = tb.lin("cat_unordered")
        labs = list(term.mappings["cat_unordered"].labels_to_integers_map)

        newdata = tb.labels_to_integers(newdata={"cat_unordered": labs})
        assert jnp.allclose(newdata["cat_unordered"], jnp.arange(3))

    def test_lin_term_attributes(self, columb):
        tb = gam.TermBuilder.from_df(columb)
        term = tb.lin("x + y + district")

        assert term.model_spec is not None
        assert term.mappings is not None

        with pytest.raises(TypeError):
            term.model_spec = "test"

        term._model_spec = None
        with pytest.raises(ValueError):
            term.model_spec

        with pytest.raises(TypeError):
            term.mappings = "test"

        with pytest.raises(TypeError):
            term.mappings = {"test": "value"}

        term._column_names = None
        with pytest.raises(ValueError):
            term.column_names

        with pytest.raises(TypeError):
            term.column_names = 2

        with pytest.raises(TypeError):
            term.column_names = "test"

        term = tb.lin("x + y")
        term._mappings = None
        with pytest.raises(ValueError):
            term.mappings

    def test_lin_term_column_names(self, columb, data):
        tb = gam.TermBuilder.from_df(columb)
        term = tb.lin("x + y")
        assert term.column_names == ["x", "y"]

        with pytest.raises(TypeError):
            term.column_names = [1, 2]

        tb = gam.TermBuilder.from_df(data)
        term = tb.lin("y + cat_ordered")
        assert term.column_names == ["y", "cat_ordered[T.med]", "cat_ordered[T.high]"]

        term = tb.lin("`with space`")
        term.column_names == ["with space"]

        term = tb.lin("`weird:col*name`")
        term.column_names == ["weird:col*name"]


class TestMRFTerm:
    def test_unobserved_regions(self, columb, columb_polys) -> None:
        i = np.arange(columb.shape[0])
        i10 = i[:10]
        i20 = i[11:20]
        i30 = i[21:30]
        irest = i[31:]
        i = np.concatenate((i10, i20, i30, irest))
        df = columb.iloc[i, :].reset_index()
        tb = gb.TermBuilder.from_df(df)
        mrf = tb.mrf("district", polys=columb_polys)
        assert mrf.basis.value.shape[-1] == 48
        assert mrf.coef.value.shape[-1] == 48

    def test_labels_unordered(self) -> None:
        nb = {"a": ["b", "c"], "b": ["a"], "c": ["a"]}
        df = pd.DataFrame({"district": ["c", "a", "b", "a"]})
        tb = gb.TermBuilder.from_df(df)
        mrf = tb.mrf(
            "district",
            nb=nb,
            absorb_cons=False,
            diagonal_penalty=False,
            scale_penalty=False,
        )
        assert mrf.ordered_labels == ["a", "b", "c"]

        assert mrf.labels == ["a", "b", "c"]
        assert mrf.neighbors == nb

        mrf._mapping = None
        with pytest.raises(ValueError):
            mrf.mapping

    def test_labels_categorical(self) -> None:
        nb = {"a": ["b", "c"], "b": ["a"], "c": ["a"]}
        df = pd.DataFrame(
            {
                "district": pd.Categorical(
                    ["c", "a", "b", "a"], categories=["c", "a", "b"]
                )
            }
        )
        tb = gb.TermBuilder.from_df(df)
        mrf = tb.mrf(
            "district",
            nb=nb,
            absorb_cons=False,
            diagonal_penalty=False,
            scale_penalty=False,
        )
        assert mrf.ordered_labels == ["a", "b", "c"]

    def test_labels_ordered(self) -> None:
        nb = {"a": ["b", "c"], "b": ["a"], "c": ["a"]}
        df = pd.DataFrame(
            {
                "district": pd.Categorical(
                    ["c", "a", "b", "a"], categories=["c", "a", "b"], ordered=True
                )
            }
        )
        tb = gb.TermBuilder.from_df(df)
        mrf = tb.mrf(
            "district",
            nb=nb,
            absorb_cons=False,
            diagonal_penalty=False,
            scale_penalty=False,
        )
        assert mrf.ordered_labels == ["a", "b", "c"]

    def test_factor_scale(self, columb, columb_polys):
        tb = gb.TermBuilder.from_df(columb)
        term = tb.mrf("district", factor_scale=True, scale=3.0, polys=columb_polys)
        assert term.scale.value == pytest.approx(3.0)
        assert term.coef.dist_node["scale"].value == pytest.approx(1.0)

        term = tb.mrf("district", scale=3.0, polys=columb_polys)
        assert term.scale.value == pytest.approx(3.0)
        assert term.coef.dist_node["scale"].value == pytest.approx(3.0)


def is_diagonal(M, atol=1e-5):
    # mask for off-diagonal elements
    off_diag_mask = ~jnp.eye(M.shape[-1], dtype=bool)
    off_diag_values = M[off_diag_mask]
    return jnp.all(jnp.abs(off_diag_values) < atol)


class TestBasisReparameterization:
    def test_diagonalize_penalty(self, columb: pd.DataFrame):
        tb = gb.TermBuilder.from_df(columb)
        term = tb.ps(
            "x", k=20, absorb_cons=False, diagonal_penalty=False, scale_penalty=False
        )
        assert term.basis.penalty is not None
        p1 = term.basis.penalty.value
        assert not is_diagonal(p1, 1e-3)
        p = term.coef.dist_node["penalty"].value  # type: ignore
        assert not is_diagonal(p, 1e-3)

        term.basis.diagonalize_penalty()
        p2 = term.basis.penalty.value
        assert is_diagonal(p2, 1e-3)

        p = term.coef.dist_node["penalty"].value  # type: ignore
        assert is_diagonal(p, 1e-3)

    def test_scale_penalty(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        term = tb.ps(
            "x", k=20, absorb_cons=False, diagonal_penalty=False, scale_penalty=False
        )
        p1 = term.basis.penalty.value
        p1b = term.coef.dist_node["penalty"].value
        term.basis.scale_penalty()
        p2 = term.basis.penalty.value
        p2b = term.coef.dist_node["penalty"].value

        assert jnp.linalg.norm(p2, ord=jnp.inf) == pytest.approx(1.0)
        assert jnp.linalg.norm(p2b, ord=jnp.inf) == pytest.approx(1.0)
        assert not jnp.allclose(p1, p2, atol=1e-6)
        assert not jnp.allclose(p1b, p2b, atol=1e-6)

    def test_constrain_sumzero_coef(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        term = tb.ps(
            "x", k=20, absorb_cons=False, diagonal_penalty=False, scale_penalty=False
        )
        basis = term.basis
        term.constrain("sumzero_coef")
        term.update()

        coef = jax.random.normal(jax.random.key(42), term.coef.value.shape)

        constrained_coef = basis.reparam_matrix @ coef
        assert constrained_coef.sum() == pytest.approx(0.0, abs=1e-6)
        assert basis.constraint == "sumzero_coef"

    def test_constrain_sumzero_term(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        term = tb.ps(
            "x", k=20, absorb_cons=False, diagonal_penalty=False, scale_penalty=False
        )
        basis = term.basis
        term.constrain("sumzero_term")

        term.coef.value = jax.random.normal(jax.random.key(42), term.coef.value.shape)
        term.update()

        assert term.value.sum() == pytest.approx(0.0, abs=1e-5)
        assert basis.constraint == "sumzero_term"

    def test_constrain_constant_and_linear(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        term = tb.ps(
            "x", k=20, absorb_cons=False, diagonal_penalty=False, scale_penalty=False
        )
        basis = term.basis
        term.constrain("constant_and_linear")
        term.coef.value = jax.random.normal(jax.random.key(42), term.coef.value.shape)
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

    def test_constrain_custom(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        term = tb.ps(
            "x", k=20, absorb_cons=False, diagonal_penalty=False, scale_penalty=False
        )
        basis = term.basis

        A = jnp.mean(basis.value, axis=0, keepdims=True)
        term.constrain(A)
        term.coef.value = jax.random.normal(jax.random.key(42), term.coef.value.shape)
        term.update()
        assert term.value.sum() == pytest.approx(0.0, abs=1e-5)
        assert basis.constraint == "custom"

    def test_constrain_after_absorption(self, columb, columb_polys):
        tb = gb.TermBuilder.from_df(columb)

        term = tb.ps(
            "x", k=20, absorb_cons=True, diagonal_penalty=False, scale_penalty=False
        )
        with pytest.raises(ValueError):
            term.constrain("sumzero_term")

        term = tb.s(
            "x",
            k=10,
            bs="tp",
            absorb_cons=True,
            diagonal_penalty=False,
            scale_penalty=False,
        )
        with pytest.raises(ValueError):
            term.constrain("sumzero_term")

        term = tb.mrf("district", polys=columb_polys)
        with pytest.raises(ValueError):
            term.constrain("sumzero_term")

    def test_constrain_linear(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        term = tb.lin("x + area")

        with pytest.raises(AttributeError):
            term.constrain("sumzero_term")

    def test_constrain_ri(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        term = tb.ri("district")

        with pytest.raises(ValueError):
            term.constrain("sumzero_term")


def _test_term(
    fn, k, constraints, fewer_bases_by, columb, test_absorb_cons: bool = True
):
    constraints = constraints + fewer_bases_by

    smooth = fn("x", k=k)
    model = lsl.Model([smooth])
    fname = fn.__name__
    assert f"{fname}(x)" in model.vars
    assert "B(x)" in model.vars
    assert "$\\tau_{" + fname + "(x)}^2$" in model.vars
    assert "$\\tau_{" + fname + "(x)}$" in model.vars
    assert "$\\beta_{" + fname + "(x)}$" in model.vars

    assert not any(jnp.isnan(smooth.value))
    assert smooth.value.shape == columb.shape[0:1]
    assert smooth.basis.value.shape == (columb.shape[0], k - constraints)
    assert is_diagonal(smooth.basis.penalty.value)

    smooth = fn("x", k=k, diagonal_penalty=False)
    assert not any(jnp.isnan(smooth.value))
    assert smooth.value.shape == columb.shape[0:1]
    assert smooth.basis.value.shape == (columb.shape[0], k - constraints)
    assert not is_diagonal(smooth.basis.penalty.value)

    if test_absorb_cons:
        smooth = fn("x", k=k, absorb_cons=False)
        assert not any(jnp.isnan(smooth.value))
        assert smooth.value.shape == columb.shape[0:1]
        assert smooth.basis.value.shape == (columb.shape[0], k - fewer_bases_by)
        assert is_diagonal(smooth.basis.penalty.value)

    smooth_scaled = fn("x", k=k, scale_penalty=True, diagonal_penalty=False)
    smooth_unscaled = fn("x", k=k, scale_penalty=False, diagonal_penalty=False)
    smooth_unscaled.update()
    assert not any(jnp.isnan(smooth_unscaled.value))
    assert smooth_unscaled.value.shape == columb.shape[0:1]
    assert smooth_unscaled.basis.value.shape == (columb.shape[0], k - constraints)
    assert not is_diagonal(smooth_unscaled.basis.penalty.value)
    assert not jnp.allclose(
        smooth_unscaled.basis.penalty.value, smooth_scaled.basis.penalty.value
    )

    smooth_factor_scale = fn("x", k=k, factor_scale=True, scale=2.0)
    smooth_factor_scale.scale.value == pytest.approx(2.0)
    smooth_factor_scale.coef.dist_node["scale"].value == pytest.approx(1.0)


class TestTerms:
    def test_ps(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        _test_term(tb.ps, k=20, constraints=1, fewer_bases_by=0, columb=columb)

    def test_s(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        s = partial(tb.s, bs="ps")
        s.__name__ = "ps"  # monkey-patch
        _test_term(s, k=20, constraints=1, fewer_bases_by=0, columb=columb)

    def test_np(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        _test_term(
            tb.np,
            k=20,
            constraints=2,
            fewer_bases_by=0,
            columb=columb,
            test_absorb_cons=False,
        )

    def test_bs(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        _test_term(tb.bs, k=20, constraints=1, fewer_bases_by=0, columb=columb)

    def test_f(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        nparam = 10
        knots = spl.equidistant_knots(columb["x"].to_numpy(), n_param=nparam)
        pen = spl.pspline_penalty(d=nparam, diff=2)

        def bfun(x):
            return spl.basis_matrix(x.squeeze(), knots=knots)

        smooth = tb.f("x", basis_fn=bfun, penalty=pen)

        model = lsl.Model([smooth])
        fname = "f"
        assert f"{fname}(x)" in model.vars
        assert "B(x)" in model.vars
        assert "$\\tau_{" + fname + "(x)}^2$" in model.vars
        assert "$\\tau_{" + fname + "(x)}$" in model.vars
        assert "$\\beta_{" + fname + "(x)}$" in model.vars

        assert not any(jnp.isnan(smooth.value))
        assert smooth.value.shape == columb.shape[0:1]
        assert smooth.basis.value.shape == (columb.shape[0], nparam)
        assert not is_diagonal(smooth.basis.penalty.value)

        smooth = tb.f("x", basis_fn=bfun, penalty=pen)
        smooth.diagonalize_penalty()
        assert is_diagonal(smooth.basis.penalty.value)

        smooth_scaled = tb.f("x", basis_fn=bfun, penalty=pen).scale_penalty()
        smooth_unscaled = tb.f("x", basis_fn=bfun, penalty=pen)
        smooth_unscaled.update()
        assert not any(jnp.isnan(smooth_unscaled.value))
        assert smooth_unscaled.value.shape == columb.shape[0:1]
        assert smooth_unscaled.basis.value.shape == (columb.shape[0], nparam)
        assert not is_diagonal(smooth_unscaled.basis.penalty.value)
        assert not jnp.allclose(
            smooth_unscaled.basis.penalty.value, smooth_scaled.basis.penalty.value
        )

        with pytest.raises(ValueError):
            smooth_factor_scale = tb.f(
                "x", basis_fn=bfun, penalty=pen, factor_scale=True
            )
        smooth_factor_scale = tb.f(
            "x", basis_fn=bfun, penalty=jnp.eye(pen.shape[-1]), factor_scale=True
        )
        smooth_factor_scale.scale.value == pytest.approx(2.0)
        smooth_factor_scale.coef.dist_node["scale"].value == pytest.approx(1.0)

    def test_cp(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        _test_term(tb.cp, k=20, constraints=1, fewer_bases_by=0, columb=columb)

    def test_tp(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        _test_term(tb.tp, k=20, constraints=1, fewer_bases_by=0, columb=columb)

    def test_ts(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        _test_term(tb.ts, k=20, constraints=1, fewer_bases_by=0, columb=columb)

    def test_cr(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        _test_term(tb.cr, k=20, constraints=1, fewer_bases_by=0, columb=columb)

    def test_cs(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        _test_term(tb.cs, k=20, constraints=1, fewer_bases_by=0, columb=columb)

    def test_cc(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        _test_term(tb.cc, k=20, constraints=1, fewer_bases_by=1, columb=columb)

    def test_kriging(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        _test_term(tb.kriging, k=20, constraints=1, fewer_bases_by=0, columb=columb)


class TestGetParameter:
    def test_scale_ig(self):
        scale = gam.ScaleIG(1.0, 0.01, 0.01)
        var = _find_parameter(scale)
        assert var is scale._variance_param
        assert not var.model

    def test_strong(self):
        a = lsl.Var.new_param(1.0, name="a")
        b = _find_parameter(a)
        assert a is b

        a = lsl.Var.new_param(1.0)
        b = _find_parameter(a)
        assert a is b
        assert not b.name

    def test_no_param(self):
        a = lsl.Var.new_value(1.0, name="a")
        with pytest.raises(ValueError):
            _find_parameter(a)

    def test_multiple_params(self):
        a = lsl.Var.new_param(1.0, name="a")
        b = lsl.Var.new_param(1.0, name="b")
        c = lsl.Var.new_calc(lambda a, b: a + b, a, b)
        with pytest.raises(ValueError):
            _find_parameter(c)

    def test_weak(self):
        a = lsl.Var.new_param(1.0, name="a")
        b = lsl.Var.new_calc(jnp.exp, a)
        c = _find_parameter(b)
        assert c is a


class TestRITerm:
    def test_full_basis(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        ri = tb.ri("district")
        assert ri.init_full_basis().value.shape == (49, 49)

        ri._mapping = None  # should also work without mapping
        assert ri.init_full_basis().value.shape == (49, 49)

    def test_labels(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        ri = tb.ri("district")

        assert len(ri.labels) == 49

        ri._labels = None
        with pytest.raises(ValueError):
            ri.labels

    def test_factor_scale(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        term = tb.ri("district", factor_scale=True, scale=3.0)
        assert term.scale.value == pytest.approx(3.0)
        assert term.coef.dist_node["scale"].value == pytest.approx(1.0)

        term = tb.ri("district", scale=3.0)
        assert term.scale.value == pytest.approx(3.0)
        assert term.coef.dist_node["scale"].value == pytest.approx(3.0)


class TestVaryincCoefficient:
    def test_vc(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        psx = tb.ps("x", k=20)
        tb.vc("y", by=psx)


class TestTPTerm:
    def test_ps_ri(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        ri = tb.ri("district")
        ps = tb.ps("x", k=10)
        ta = tb.tx(ri, ps)
        assert ta.basis.value.shape == (49, 9 * 49)

    def test_common_scale(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        psy = tb.ps("y", k=10)
        psx = tb.ps("x", k=10)
        ta = tb.tx(psy, psx, common_scale=gam.VarIGPrior(1.0, 0.005))
        assert ta.basis.value.shape == (49, 9 * 9)
        for i in range(len(ta.scales)):
            assert ta.scales[i].value_node[0].value_node[0].inference is not None
            if i > 0:
                assert ta.scales[i] is ta.scales[i - 1]

        ta = tb.tf(psy, psx, common_scale=gam.VarIGPrior(1.0, 0.005))
        assert ta.basis.value.shape == (49, 9 * 9)
        for i in range(len(ta.scales)):
            assert ta.scales[i].value_node[0].value_node[0].inference is not None
            if i > 0:
                assert ta.scales[i] is ta.scales[i - 1]

        ta = tb.tf(psy, psx, common_scale=1.0)
        assert ta.basis.value.shape == (49, 9 * 9)
        for i in range(len(ta.scales)):
            assert ta.scales[i].strong
            assert ta.scales[i].inference is None
            if i > 0:
                assert ta.scales[i] is ta.scales[i - 1]

    def test_intentional_inference(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        psy = tb.ps("y", k=10)
        psx = tb.ps("x", k=10)
        psy.scale.value_node[0].inference = gs.MCMCSpec(gs.NUTSKernel)
        tb.tx(psy, psx)

        assert psy.scale.value_node[0].inference.kernel is gs.NUTSKernel

    def test_no_inference(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        psy = tb.ps("y", k=10)
        psx = tb.ps("x", k=10)
        psy.scale.value_node[0].inference = None
        tb.tx(psy, psx)

        assert psy.scale.value_node[0].inference is None

        psy = tb.ps("y", k=10)
        psx = tb.ps("x", k=10)
        psy.scale.value_node[0].inference = None
        psx.scale.value_node[0].inference = None
        tb.tx(psy, psx)

        assert psy.scale.value_node[0].inference is None

    def test_no_parameter(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        psy = tb.ps("y", k=10, scale=lsl.Var(1.0, name="noparam"))
        psx = tb.ps("x", k=10)
        with pytest.raises(ValueError):
            tb.tx(psy, psx)

    def test_unnamed_scale(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        psy = tb.ps("y", k=10)
        psx = tb.ps("x", k=10)
        psy.scale.value_node[0].name = ""
        tb.tx(psy, psx)

        assert psy.scale.value_node[0].value_node[0].name == ""

    def test_ps_mrf(self, columb, columb_polys):
        tb = gb.TermBuilder.from_df(columb)
        mrf = tb.mrf("district", polys=columb_polys)
        ps = tb.ps("x", k=10)
        ta = tb.tx(mrf, ps)
        assert ta.basis.value.shape == (49, 9 * 48)


class TestHasStarGibbs:
    def test_term(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        ri = tb.ri("district")
        assert _has_star_gibbs(ri.scale)

        ri2 = tb.ri("district", scale=lsl.Var.new_param(1.0, name="test"))
        assert not _has_star_gibbs(ri2.scale)

        ri3 = tb.ri("district", scale=1.0)
        assert not _has_star_gibbs(ri3.scale)

    def test_other_inference(self):
        v = lsl.Var.new_param(1.0, inference=gs.MCMCSpec(gs.NUTSKernel))
        assert not _has_star_gibbs(v)

    def test_inference_dict(self):
        v = lsl.Var.new_param(1.0, inference={"a": gs.MCMCSpec(gs.NUTSKernel)})
        assert not _has_star_gibbs(v)

    def test_inference_dict_one_of_them_star_ig(self, columb):
        tb = gb.TermBuilder.from_df(columb)
        ri = tb.ri("district")
        v = _find_parameter(ri.scale)
        inference_ig = v.inference
        v.inference = {"a": gs.MCMCSpec(gs.NUTSKernel), "b": inference_ig}
        assert _has_star_gibbs(ri.scale)

    def test_inference_unexpected_type(self):
        v = lsl.Var.new_param(1.0, inference="test")
        with pytest.raises(TypeError, match="Could not handle type"):
            _has_star_gibbs(v)


class TestFillName:
    def test_strong_var_unnamed(self):
        a = lsl.Var(1.0)
        b = _format_name(a, fill="test")

        assert a is b
        assert a.name == "test"

    def test_strong_var_named_without_placeholder(self):
        a = lsl.Var(1.0, name="myname")
        _format_name(a, fill="test")

        assert a.name == "myname"

    def test_strong_var_named_with_placeholder(self):
        a = lsl.Var(1.0, name="{x}")
        _format_name(a, fill="test")

        assert a.name == "test"

    def test_transformed_var(self):
        tau2 = lsl.Var.new_param(1.0, name="{x}2")
        tau = lsl.Var.new_calc(jnp.sqrt, tau2, name="{x}")
        tau2_transformed = tau2.transform(bijector=tfb.Exp())

        _format_name(tau, fill="tau")

        assert tau.name == "tau"
        assert tau2.name == "tau2"
        assert tau2_transformed.name == "tau2_transformed"
        assert tau.value_node.name == "tau_calc"
        assert tau2.value_node.name == "tau2_value"
        assert tau2_transformed.value_node.name == "tau2_transformed_value"
