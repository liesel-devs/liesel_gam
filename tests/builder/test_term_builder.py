import logging

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from ryp import r, to_py

import liesel_gam.builder as gb

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


def is_diagonal(M, atol=1e-6):
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

        assert term.value.sum() == pytest.approx(0.0, abs=1e-6)
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

        term = tb.te("x", "y")
        with pytest.raises(ValueError):
            term.constrain("sumzero_term")

        term = tb.ti("x", "y")
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
