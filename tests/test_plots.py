"""
These tests are just smoke tests, ensuring that the plotting functions
run without error.
"""

import liesel.model as lsl
import numpy as np
import pandas as pd
import pytest
from jax.random import key as jkey
from ryp import r, to_py

import liesel_gam as gam


@pytest.fixture(scope="module")
def columb() -> pd.DataFrame:
    """
    'area', 'home.value', 'income', 'crime', 'open.space', 'district',
    'x', 'y', 'home_value'
    """
    r("library(mgcv)")
    r("data(columb)")
    return to_py("columb", format="pandas").reset_index()


@pytest.fixture(scope="module")
def polys() -> np.typing.NDArray:
    r("library(mgcv)")
    r("data(columb.polys)")
    return to_py("columb.polys", format="numpy")


@pytest.fixture
def tb(columb) -> gam.TermBuilder:
    df = columb
    return gam.TermBuilder.from_df(df)


class TestPlots:
    def test_plot_1d_smooth(self, columb):
        tb = gam.TermBuilder.from_df(columb)
        term = tb.ps("x", k=10)
        model = lsl.Model([term])

        samples = model.sample((4, 20), jkey(0))

        gam.plot_1d_smooth(term, samples)
        gam.plot_1d_smooth(term, samples, hdi_prob=0.7)

    def test_plot_2d_smooth(self, columb):
        tb = gam.TermBuilder.from_df(columb)
        term = tb.tp("x", "y", k=10)
        px, py = tb.ps("x", k=10), tb.ps("y", k=10)
        ta = tb.tx(px, py)
        model = lsl.Model([term, ta])

        samples = model.sample((4, 20), jkey(0))

        gam.plot_2d_smooth(term, samples)
        gam.plot_2d_smooth(ta, samples)

    def test_plot_polys(self, columb, polys):
        gam.plot_polys("district", "crime", columb, polys)
        gam.plot_polys("district", "crime", columb, polys, show_unobserved=True)

    def test_plot_forest(self, columb):
        tb = gam.TermBuilder.from_df(columb)
        term = tb.ri("district")
        term2 = tb.slin("x + y")
        model = lsl.Model([term, term2])

        samples = model.sample((4, 20), jkey(0))

        gam.plot_forest(term, samples)
        gam.plot_forest(term2, samples)
        gam.plot_forest(term, samples, indices=range(10))
        gam.plot_forest(term, samples, indices=range(10), show_unobserved=False)

        term._mapping = None
        gam.plot_forest(term, samples, indices=range(10), show_unobserved=False)

    def test_plot_regions(self, columb, polys):
        tb = gam.TermBuilder.from_df(columb)
        term = tb.ri("district")
        term2 = tb.mrf("district", polys=polys)
        model = lsl.Model([term, term2])

        samples = model.sample((4, 20), jkey(0))

        gam.plot_regions(term, samples, polys=polys)
        gam.plot_regions(term2, samples)
        gam.plot_regions(term2, samples, show_unobserved=False)

    def test_plot_1d_smooth_clustered(self, columb):
        tb = gam.TermBuilder.from_df(columb)
        term = tb.ri("district")
        term2 = tb.ps("x", k=20)
        term3 = tb.rs("x", "district")
        term4 = tb.rs(term2, "district")
        model = lsl.Model([term, term2, term3, term4])

        samples = model.sample((4, 20), jkey(0))

        gam.plot_1d_smooth_clustered(term3, samples)
        gam.plot_1d_smooth_clustered(term4, samples)

        term._mapping = None
        gam.plot_1d_smooth_clustered(term3, samples)
