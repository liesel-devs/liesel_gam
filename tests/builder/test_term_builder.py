import logging

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
