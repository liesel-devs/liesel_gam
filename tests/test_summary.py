from collections.abc import Mapping, Sequence

import jax.numpy as jnp
import liesel.model as lsl
import numpy as np
import pandas as pd
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.random import key as jkey
from jax.random import normal
from jax.typing import ArrayLike
from ryp import r, to_py

import liesel_gam as gam
from liesel_gam.summary import grid_nd, input_grid_nd_smooth, summarise_by_samples


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


class TestSummariseBySamples:
    def test_runs(self, tb: gam.TermBuilder) -> None:
        term = tb._s("x", k=10, bs="ps")
        _ = lsl.Model([term])

        samples = term.coef.sample((4, 20), jkey(0))
        su = summarise_by_samples(
            key=jkey(1), a=samples[term.coef.name], name=term.name, n=5
        )

        coef_dim = term.coef.value.shape[-1]
        nsamples = 5
        assert su.shape[0] == (coef_dim * nsamples)


class Test1dSmoothSummary:
    def test_runs(self, tb: gam.TermBuilder) -> None:
        term = tb._s("x", k=10, bs="ps")
        _ = lsl.Model([term])

        samples = term.coef.sample((4, 20), jkey(0))
        su = gam.summarise_1d_smooth(term=term, samples=samples)
        assert su.shape[0] == 150

    def test_newdata(self, tb: gam.TermBuilder) -> None:
        term = tb._s("x", k=10, bs="ps")
        _ = lsl.Model([term])

        newdata = {term.basis.x.name: jnp.linspace(-1, 2, 13)}

        samples = term.coef.sample((4, 20), jkey(0))
        su = gam.summarise_1d_smooth(term=term, samples=samples, newdata=newdata)
        assert su.shape[0] == 13

        su = gam.summarise_1d_smooth(
            term=term, samples=samples, newdata=newdata, ngrid=100
        )
        assert su.shape[0] == 13

    def test_hdi_prob(self, tb: gam.TermBuilder) -> None:
        term = tb._s("x", k=10, bs="ps")
        _ = lsl.Model([term])

        samples = term.coef.sample((4, 20), jkey(0))
        su1 = gam.summarise_1d_smooth(term=term, samples=samples, hdi_prob=0.9)

        su2 = gam.summarise_1d_smooth(term=term, samples=samples, hdi_prob=0.99)

        assert sum(su1["hdi_low"] > su2["hdi_low"]) > 100
        assert sum(su1["hdi_high"] < su2["hdi_high"]) > 100

    def test_quantiles(self, tb: gam.TermBuilder) -> None:
        term = tb._s("x", k=10, bs="ps")
        _ = lsl.Model([term])

        samples = term.coef.sample((4, 20), jkey(0))
        su1 = gam.summarise_1d_smooth(term=term, samples=samples)
        assert "q_0.05" in su1.columns
        assert "q_0.95" in su1.columns
        assert "q_0.5" in su1.columns

        su2 = gam.summarise_1d_smooth(term=term, samples=samples, quantiles=(0.1, 0.9))
        assert "q_0.1" in su2.columns
        assert "q_0.9" in su2.columns

    def test_ngrid(self, tb: gam.TermBuilder) -> None:
        term = tb._s("x", k=10, bs="ps")
        _ = lsl.Model([term])

        samples = term.coef.sample((4, 20), jkey(0))
        su1 = gam.summarise_1d_smooth(term=term, samples=samples, ngrid=40)
        assert su1.shape[0] == 40


class Test1dSmoothClusteredSummary:
    def test_runs(self, columb: pd.DataFrame) -> None:
        df = columb.iloc[:10, :].copy()
        df["district"] = pd.Categorical(df["district"].to_list())

        tb = gam.TermBuilder.from_df(df)
        smooth = tb._s("x", k=10, bs="ps")
        term = tb.rs(smooth, cluster="district")
        model = lsl.Model([term])

        samples = model.sample((4, 20), jkey(0))
        su = gam.summarise_1d_smooth_clustered(term, samples=samples)
        assert su.shape[0] == 20 * 10

    def test_runs_with_tp_multivar(self, columb: pd.DataFrame) -> None:
        df = columb.iloc[:10, :].copy()
        df["district"] = pd.Categorical(df["district"].to_list())

        tb = gam.TermBuilder.from_df(df)
        smooth = tb.tp("x", "y", k=10)
        term = tb.rs(smooth, cluster="district")
        model = lsl.Model([term])

        samples = model.sample((4, 20), jkey(0))
        su = gam.summarise_1d_smooth_clustered(term, samples=samples)
        assert su.shape[0] == 16 * 10

        term.value_node["cluster"]._mapping = None  # type: ignore
        su = gam.summarise_1d_smooth_clustered(term, samples=samples)
        assert su.shape[0] == 16 * 10

    def test_runs_with_tp_univar(self, columb: pd.DataFrame) -> None:
        df = columb.iloc[:10, :].copy()
        df["district"] = pd.Categorical(df["district"].to_list())

        tb = gam.TermBuilder.from_df(df)
        smooth = tb.tp("x", k=10)
        term = tb.rs(smooth, cluster="district")
        model = lsl.Model([term])

        samples = model.sample((4, 20), jkey(0))
        su = gam.summarise_1d_smooth_clustered(term, samples=samples)
        assert su.shape[0] == 20 * 10

    def test_unobserved(self, columb: pd.DataFrame) -> None:
        df = columb.iloc[:10, :].copy()

        tb = gam.TermBuilder.from_df(df)
        smooth = tb._s("x", k=10, bs="ps")
        term = tb.rs(smooth, cluster="district")
        model = lsl.Model([term])

        samples = model.sample((4, 20), jkey(0))
        su = gam.summarise_1d_smooth_clustered(term, samples=samples, ngrid=5)
        assert su["observed"].sum() == 5 * 10  # only observed
        assert su.shape[0] == 5 * 49  # all 49 categories are present

    def test_newdata_missing_key(self, columb: pd.DataFrame) -> None:
        df = columb.iloc[:10, :].copy()

        tb = gam.TermBuilder.from_df(df)
        smooth = tb._s("x", k=10, bs="ps")
        term = tb.rs(smooth, cluster="district")
        model = lsl.Model([term])

        samples = model.sample((4, 20), jkey(0))

        # missing key raises error
        with pytest.raises(KeyError):
            newdata = {smooth.basis.x.name: jnp.linspace(-1, 2, 13)}
            gam.summarise_1d_smooth_clustered(term, samples=samples, newdata=newdata)

        with pytest.raises(KeyError):
            newdata = {"district": columb["district"].to_numpy()[:13]}
            gam.summarise_1d_smooth_clustered(term, samples=samples, newdata=newdata)

    def test_newdata_direct(self, columb: pd.DataFrame) -> None:
        df = columb.iloc[:10, :].copy()

        tb = gam.TermBuilder.from_df(df)
        smooth = tb._s("x", k=10, bs="ps")
        term = tb.rs(smooth, cluster="district")
        model = lsl.Model([term])

        samples = model.sample((4, 20), jkey(0))

        newdata = {
            "x": jnp.linspace(-1, 2, 13),
            "district": columb["district"].to_numpy()[:13],
        }

        su = gam.summarise_1d_smooth_clustered(term, samples=samples, newdata=newdata)
        assert su.shape[0] == 13
        assert su["observed"].sum() == 10

    def test_newdata_codes(self, columb: pd.DataFrame) -> None:
        """If the newdata for the cluster is supplied as codes, that works, too."""
        df = columb.iloc[:10, :].copy()

        tb = gam.TermBuilder.from_df(df)
        smooth = tb._s("x", k=10, bs="ps")
        term = tb.rs(smooth, cluster="district")
        model = lsl.Model([term])

        samples = model.sample((4, 20), jkey(0))

        newdata = {
            "x": jnp.linspace(-1, 2, 13),
            "district": jnp.array(list(range(13))),
        }

        su = gam.summarise_1d_smooth_clustered(term, samples=samples, newdata=newdata)
        assert su.shape[0] == 13
        assert su["observed"].sum() == 10

    def test_newdata_meshgrid(self, columb: pd.DataFrame) -> None:
        df = columb.iloc[:10, :].copy()

        tb = gam.TermBuilder.from_df(df)
        smooth = tb._s("x", k=10, bs="ps")
        term = tb.rs(smooth, cluster="district")
        model = lsl.Model([term])

        samples = model.sample((4, 20), jkey(0))

        newdata = {
            "x": jnp.linspace(-1, 2, 13),
            "district": columb["district"].to_numpy()[:13],
        }

        su = gam.summarise_1d_smooth_clustered(
            term, samples=samples, newdata=newdata, newdata_meshgrid=True
        )
        assert su.shape[0] == 13 * 13
        assert su["observed"].sum() == 10 * 13

        term.value_node["cluster"]._mapping = None  # type: ignore
        newdata = {
            "x": jnp.linspace(-1, 2, 13),
            "district": columb["district"].to_numpy()[:13].astype(int),
        }
        su = gam.summarise_1d_smooth_clustered(
            term, samples=samples, newdata=newdata, newdata_meshgrid=True
        )
        assert su.shape[0] == 13 * 13
        assert su["observed"].sum() == 10 * 13

    def test_hdi_prob(self, columb: pd.DataFrame) -> None:
        df = columb.iloc[:10, :].copy()
        df["district"] = pd.Categorical(df["district"].to_list())

        tb = gam.TermBuilder.from_df(df)
        smooth = tb._s("x", k=10, bs="ps")
        term = tb.rs(smooth, cluster="district")
        model = lsl.Model([term])

        samples = model.sample((4, 20), jkey(0))
        su1 = gam.summarise_1d_smooth_clustered(term, samples=samples, hdi_prob=0.9)
        su2 = gam.summarise_1d_smooth_clustered(term, samples=samples, hdi_prob=0.99)
        assert sum(su1["hdi_low"] > su2["hdi_low"]) > 100
        assert sum(su1["hdi_high"] < su2["hdi_high"]) > 100

    def test_quantiles(self, columb: pd.DataFrame) -> None:
        df = columb.iloc[:10, :].copy()
        df["district"] = pd.Categorical(df["district"].to_list())

        tb = gam.TermBuilder.from_df(df)
        smooth = tb._s("x", k=10, bs="ps")
        term = tb.rs(smooth, cluster="district")
        model = lsl.Model([term])

        samples = model.sample((4, 20), jkey(0))
        su1 = gam.summarise_1d_smooth_clustered(term, samples=samples)
        assert "q_0.05" in su1.columns
        assert "q_0.95" in su1.columns
        assert "q_0.5" in su1.columns
        su2 = gam.summarise_1d_smooth_clustered(
            term, samples=samples, quantiles=(0.1, 0.9)
        )
        assert "q_0.1" in su2.columns
        assert "q_0.9" in su2.columns

    def test_labels_mapping(self, columb: pd.DataFrame) -> None:
        """
        If there is no mapping defined on the cluster,
        the integer codes are taken as they are as the cluster labels.

        If there is no mapping defined on the cluster and a custom mapping is given,
        the custom mapping is taken.
        """
        df = columb.iloc[:10, :].copy()
        df["district"] = pd.Categorical(df["district"].to_list())

        tb = gam.TermBuilder.from_df(df)
        smooth = tb._s("x", k=10, bs="ps")
        term = tb.rs(smooth, cluster="district")
        model = lsl.Model([term])
        samples = model.sample((4, 20), jkey(0))

        labels = term.value_node["cluster"].mapping  # type: ignore
        term.value_node["cluster"]._mapping = None  # type: ignore
        su1 = gam.summarise_1d_smooth_clustered(term, samples=samples, labels=labels)
        assert su1.shape[0] == 200
        assert all(su1["district"].unique() == df["district"].unique())

    def test_labels_list(self, columb: pd.DataFrame) -> None:
        """
        If there is no mapping defined on the cluster,
        the integer codes are taken as they are as the cluster labels.

        If there is no mapping defined on the cluster and a custom mapping is given,
        the custom mapping is taken.
        """
        df = columb.iloc[:10, :].copy()
        df["district"] = pd.Categorical(df["district"].to_list())

        tb = gam.TermBuilder.from_df(df)
        smooth = tb._s("x", k=10, bs="ps")
        term = tb.rs(smooth, cluster="district")
        model = lsl.Model([term])
        samples = model.sample((4, 20), jkey(0))

        newdata = {
            "x": jnp.linspace(-1, 2, 3),
            "district": columb["district"].to_numpy()[:3],
        }

        with pytest.raises(ValueError):
            gam.summarise_1d_smooth_clustered(
                term, samples=samples, labels=["a", "b", "c"]
            )

        su1 = gam.summarise_1d_smooth_clustered(
            term, samples=samples, newdata=newdata, labels=["a", "b", "c"]
        )
        assert su1.shape[0] == 3
        assert su1["district"].to_list() == ["a", "b", "c"]

        su1 = gam.summarise_1d_smooth_clustered(
            term, samples=samples, newdata=newdata, labels=None
        )
        assert su1.shape[0] == 3
        assert su1["district"].to_list() == ["0", "1", "2"]

        su1 = gam.summarise_1d_smooth_clustered(
            term, samples=samples, newdata=newdata, labels=None
        )
        assert su1.shape[0] == 3
        assert su1["district"].to_list() == ["0", "1", "2"]

    def test_random_slope(self, columb: pd.DataFrame) -> None:
        df = columb.iloc[:10, :].copy()
        df["district"] = pd.Categorical(df["district"].to_list())

        tb = gam.TermBuilder.from_df(df)
        term = tb.rs("x", cluster="district")
        model = lsl.Model([term])

        samples = model.sample((4, 20), jkey(0))
        su = gam.summarise_1d_smooth_clustered(term, samples=samples)
        assert su.shape[0] == 20 * 10

    def test_lin_term(self, columb: pd.DataFrame) -> None:
        df = columb.iloc[:10, :].copy()
        df["district"] = pd.Categorical(df["district"].to_list())

        tb = gam.TermBuilder.from_df(df)
        smooth = tb.lin("x + area", prior=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0))
        term = tb.rs(smooth, cluster="district")
        model = lsl.Model([term])

        samples = model.sample((4, 20), jkey(0))
        su = gam.summarise_1d_smooth_clustered(term, samples=samples, ngrid=10)
        assert "area" in list(su.columns)
        assert "x" in list(su.columns)
        assert "district" in list(su.columns)

        assert su.shape[0] == 3 * 3 * 10

    def test_ri(self, columb: pd.DataFrame) -> None:
        df = columb.iloc[:10, :].copy()
        df["district"] = pd.Categorical(df["district"].to_list())

        tb = gam.TermBuilder.from_df(df)
        smooth = tb.ri("district")
        term = tb.rs(smooth, cluster="district")
        model = lsl.Model([term])

        samples = model.sample((4, 20), jkey(0))
        with pytest.raises(TypeError):
            gam.summarise_1d_smooth_clustered(term, samples=samples, ngrid=10)


class TestNDSmoothSummary:
    def test_grid(self):
        in_grid = {
            "x1": np.linspace(0, 1, 4),
            "x2": np.linspace(1, 2, 3),
            "x3": np.linspace(2, 3, 5),
        }
        grid = grid_nd(in_grid, ngrid=3)
        for v in grid.values():
            assert np.asarray(v).size == 27

        df = pd.DataFrame(grid)
        assert df.shape == df.drop_duplicates().shape

    def test_input_grid(self, tb: gam.TermBuilder) -> None:
        lin1 = tb.lin("x + area")
        grid = input_grid_nd_smooth(lin1, ngrid=3)
        for v in grid.values():
            assert np.asarray(v).size == 9

        df = pd.DataFrame(grid)
        assert df.shape == df.drop_duplicates().shape

        lin2 = tb.lin("x + area + income")
        grid = input_grid_nd_smooth(lin2, ngrid=3)
        for v in grid.values():
            assert np.asarray(v).size == 27

        df = pd.DataFrame(grid)
        assert df.shape == df.drop_duplicates().shape

        # the grid function will not complain if you enter a variable for a categorical
        # term, because that term is represented numerically in liesel
        lin3 = tb.lin("x + area + district")
        grid = input_grid_nd_smooth(lin3, ngrid=3)
        for v in grid.values():
            assert np.asarray(v).size == 27

        df = pd.DataFrame(grid)
        assert df.shape == df.drop_duplicates().shape

    def test_runs(self, tb: gam.TermBuilder) -> None:
        term = tb.tx(tb.ps("x", k=20), tb.ps("area", k=20))
        _ = lsl.Model([term])

        samples = {term.coef.name: normal(jkey(1), (4, 20) + term.coef.value.shape)}

        su = gam.summarise_nd_smooth(term=term, samples=samples)
        assert su.shape[0] == 400

    def test_newdata_meshgrid(self, tb: gam.TermBuilder) -> None:
        term = tb.tx(tb.ps("x", k=20), tb.ps("area", k=20))
        _ = lsl.Model([term])

        newdata = {"x": jnp.linspace(-1, 2, 13), "area": jnp.linspace(-1, 2, 13)}

        samples = {term.coef.name: normal(jkey(1), (4, 20) + term.coef.value.shape)}
        su = gam.summarise_nd_smooth(
            term=term, samples=samples, newdata=newdata, newdata_meshgrid=True
        )
        assert su.shape[0] == 13 * 13

        su = gam.summarise_nd_smooth(
            term=term,
            samples=samples,
            newdata=newdata,
            ngrid=100,
            newdata_meshgrid=True,
        )
        assert su.shape[0] == 13 * 13

    def test_newdata(self, tb: gam.TermBuilder) -> None:
        term = tb.tx(tb.ps("x", k=20), tb.ps("area", k=20))
        _ = lsl.Model([term])

        newdata = {"x": jnp.linspace(-1, 2, 13), "area": jnp.linspace(-1, 2, 13)}

        samples = {term.coef.name: normal(jkey(1), (4, 20) + term.coef.value.shape)}
        su = gam.summarise_nd_smooth(term=term, samples=samples, newdata=newdata)
        assert su.shape[0] == 13

        su = gam.summarise_nd_smooth(
            term=term,
            samples=samples,
            newdata=newdata,
            ngrid=100,
        )
        assert su.shape[0] == 13

    def test_hdi_prob(self, tb: gam.TermBuilder) -> None:
        term = tb.tx(tb.ps("x", k=20), tb.ps("area", k=20))
        _ = lsl.Model([term])

        samples = {term.coef.name: normal(jkey(1), (4, 20) + term.coef.value.shape)}
        su1 = gam.summarise_nd_smooth(
            term=term, samples=samples, hdi_prob=0.9, which=["hdi_low", "hdi_high"]
        )

        su2 = gam.summarise_nd_smooth(
            term=term, samples=samples, hdi_prob=0.99, which=["hdi_low", "hdi_high"]
        )

        assert "hdi_low" in su1.variable.to_list()
        assert "hdi_high" in su1.variable.to_list()

        assert (
            sum(
                su1.query("variable == 'hdi_low'")["value"]
                > su2.query("variable == 'hdi_low'")["value"]
            )
            > 100
        )
        assert (
            sum(
                su1.query("variable == 'hdi_high'")["value"]
                < su2.query("variable == 'hdi_high'")["value"]
            )
            > 100
        )

    def test_quantiles(self, tb: gam.TermBuilder) -> None:
        term = tb.tx(tb.ps("x", k=20), tb.ps("area", k=20))
        _ = lsl.Model([term])

        samples = {term.coef.name: normal(jkey(1), (4, 20) + term.coef.value.shape)}
        su1 = gam.summarise_nd_smooth(
            term=term, samples=samples, which=["q_0.05", "q_0.5", "q_0.95"]
        )
        assert "q_0.05" in su1.variable.to_list()
        assert "q_0.95" in su1.variable.to_list()
        assert "q_0.5" in su1.variable.to_list()

        su2 = gam.summarise_nd_smooth(
            term=term,
            samples=samples,
            which=["q_0.1", "q_0.9"],  # type: ignore
            quantiles=(0.1, 0.9),
        )
        assert "q_0.1" in su2.variable.to_list()
        assert "q_0.9" in su2.variable.to_list()

    def test_ngrid(self, tb: gam.TermBuilder) -> None:
        term = tb.tx(tb.ps("x", k=20), tb.ps("area", k=20))
        _ = lsl.Model([term])

        samples = {term.coef.name: normal(jkey(1), (4, 20) + term.coef.value.shape)}
        su1 = gam.summarise_nd_smooth(term=term, samples=samples, ngrid=10)
        assert su1.shape[0] == 100


class TestLinSummary:
    def test_runs(self, tb: gam.TermBuilder) -> None:
        term = tb.lin("x + area", prior=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0))
        _ = lsl.Model([term])

        samples = term.coef.sample((4, 20), jkey(0))
        su = gam.summarise_lin(term=term, samples=samples)
        assert su.shape[0] == 2

    def test_indices(self, tb: gam.TermBuilder) -> None:
        term = tb.lin("x + area + y", prior=lsl.Dist(tfd.Normal, loc=0.0, scale=1.0))
        _ = lsl.Model([term])

        samples = term.coef.sample((4, 20), jkey(0))
        su = gam.summarise_lin(term=term, samples=samples, indices=[0, 1])
        assert su.shape[0] == 2

        su = gam.summarise_lin(term=term, samples=samples, indices=[0, 1, 2])
        assert su.shape[0] == 3

        su = gam.summarise_lin(term=term, samples=samples, indices=[2])
        assert su.shape[0] == 1
        assert su["x"].to_list() == ["y"]

        su = gam.summarise_lin(term=term, samples=samples, indices=[1, 0])
        assert su.shape[0] == 2
        assert su["x"].to_list() == ["area", "x"]


class TestClusterSummary:
    def test_runs(self, tb: gam.TermBuilder) -> None:
        term = tb.ri("district")
        _ = lsl.Model([term])

        samples = term.coef.sample((4, 20), jkey(0))
        su = gam.summarise_cluster(term=term, samples=samples)
        assert su.shape[0] == 49

    def test_newdata_labels(self, tb: gam.TermBuilder) -> None:
        term = tb.ri("district")
        _ = lsl.Model([term])

        newdata = {term.basis.x.name: ["1", "3", "4"]}
        samples = term.coef.sample((4, 20), jkey(0))
        su = gam.summarise_cluster(term=term, samples=samples, newdata=newdata)
        assert su.shape[0] == 3

    def test_newdata_labels_no_mapping(self, tb: gam.TermBuilder) -> None:
        term = tb.ri("district")
        term._mapping = None
        _ = lsl.Model([term])

        newdata = {term.basis.x.name: ["1", "3", "4"]}
        samples = term.coef.sample((4, 20), jkey(0))
        with pytest.raises(TypeError):
            gam.summarise_cluster(term=term, samples=samples, newdata=newdata)

    def test_newdata_codes(self, tb: gam.TermBuilder) -> None:
        term = tb.ri("district")
        _ = lsl.Model([term])

        newdata = {term.basis.x.name: [0, 1, 4]}
        samples = term.coef.sample((4, 20), jkey(0))
        su = gam.summarise_cluster(term=term, samples=samples, newdata=newdata)
        assert su.shape[0] == 3

    def test_hdi_prob(self, tb: gam.TermBuilder) -> None:
        term = tb.ri("district")
        _ = lsl.Model([term])

        samples = term.coef.sample((4, 20), jkey(0))
        su1 = gam.summarise_cluster(term=term, samples=samples, hdi_prob=0.9)

        su2 = gam.summarise_cluster(term=term, samples=samples, hdi_prob=0.99)

        assert sum(su1["hdi_low"] > su2["hdi_low"]) > 30
        assert sum(su1["hdi_high"] < su2["hdi_high"]) > 30

    def test_quantiles(self, tb: gam.TermBuilder) -> None:
        term = tb.ri("district")
        _ = lsl.Model([term])

        samples = term.coef.sample((4, 20), jkey(0))
        su1 = gam.summarise_cluster(term=term, samples=samples)
        assert "q_0.05" in su1.columns
        assert "q_0.95" in su1.columns
        assert "q_0.5" in su1.columns

        su2 = gam.summarise_cluster(term=term, samples=samples, quantiles=(0.1, 0.9))
        assert "q_0.1" in su2.columns
        assert "q_0.9" in su2.columns

    def test_labels_mapping(self, columb: pd.DataFrame) -> None:
        """
        If there is no mapping defined on the cluster,
        the integer codes are taken as they are as the cluster labels.

        If there is no mapping defined on the cluster and a custom mapping is given,
        the custom mapping is taken.
        """
        df = columb.iloc[:10, :].copy()
        df["district"] = pd.Categorical(df["district"].to_list())

        tb = gam.TermBuilder.from_df(df)
        term = tb.ri("district")
        model = lsl.Model([term])
        samples = model.sample((4, 20), jkey(0))

        labels = term.mapping  # type: ignore
        term._mapping = None  # type: ignore
        su1 = gam.summarise_cluster(term, samples=samples, labels=labels)
        assert su1.shape[0] == 10
        assert all(su1["district"].unique() == df["district"].unique())

    def test_labels_list(self, columb: pd.DataFrame) -> None:
        """
        If there is no mapping defined on the cluster,
        the integer codes are taken as they are as the cluster labels.

        If there is no mapping defined on the cluster and a custom mapping is given,
        the custom mapping is taken.
        """
        df = columb.iloc[:10, :].copy()
        df["district"] = pd.Categorical(df["district"].to_list())

        tb = gam.TermBuilder.from_df(df)
        term = tb.ri("district")
        model = lsl.Model([term])
        samples = model.sample((4, 20), jkey(0))

        newdata: Mapping[str, ArrayLike | Sequence[str] | Sequence[int]] = {
            term.basis.x.name: [0, 1, 4]
        }
        su = gam.summarise_cluster(
            term=term, samples=samples, newdata=newdata, labels=["a", "b", "c"]
        )
        assert su.shape[0] == 3
        assert su["district"].to_list() == ["a", "b", "c"]

        newdata = {term.basis.x.name: ["0", "1", "4"]}
        su = gam.summarise_cluster(
            term=term, samples=samples, newdata=newdata, labels=["a", "b", "c"]
        )
        assert su.shape[0] == 3
        assert su["district"].to_list() == ["a", "b", "c"]


class TestRegionSummary:
    def test_runs(self, tb: gam.TermBuilder, polys) -> None:
        term = tb.mrf("district", polys=polys)
        _ = lsl.Model([term])

        samples = term.coef.sample((4, 20), jkey(0))
        su = gam.summarise_regions(term=term, samples=samples)
        assert su.shape[0] == 1202
        assert su.observed.sum() == 1202
        assert not any(su.value.isna())

    def test_which(self, tb: gam.TermBuilder, polys) -> None:
        term = tb.mrf("district", polys=polys)
        _ = lsl.Model([term])

        samples = term.coef.sample((4, 20), jkey(0))
        su = gam.summarise_regions(term=term, samples=samples, which=["mean", "sd"])
        assert su.shape[0] == 1202 * 2
        assert su.observed.sum() == 1202 * 2
        assert not any(su.value.isna())
        assert su.variable.unique().tolist() == ["mean", "sd"]

    def test_polys_as_arg(self, tb: gam.TermBuilder, polys) -> None:
        term = tb.ri("district")
        _ = lsl.Model([term])

        samples = term.coef.sample((4, 20), jkey(0))
        su = gam.summarise_regions(term=term, samples=samples, polys=polys)
        assert su.shape[0] == 1202
        assert su.observed.sum() == 1202
        assert not any(su.value.isna())

        term._mapping = None
        with pytest.raises(ValueError):
            gam.summarise_regions(term=term, samples=samples, polys=polys)

    def test_incomplete_polys_as_arg(self, tb: gam.TermBuilder, polys) -> None:
        term = tb.ri("district")
        _ = lsl.Model([term])

        polys_ = {}
        poly_keys = list(polys)
        poly_vals = list(polys.values())
        for i in range(10):
            polys_[poly_keys[i]] = poly_vals[i]

        samples = term.coef.sample((4, 20), jkey(0))
        su = gam.summarise_regions(term=term, samples=samples, polys=polys_)
        assert su.shape[0] == 340
        assert su.observed.sum() == 340
        assert not any(su.value.isna())

    def test_polys_for_unknown_region(self, tb: gam.TermBuilder, polys) -> None:
        term = tb.ri("district")
        _ = lsl.Model([term])

        polys_ = {}
        poly_keys = list(polys)
        poly_vals = list(polys.values())
        for i in range(10):
            polys_[poly_keys[i]] = poly_vals[i]

        polys_["test"] = poly_vals[i + 1]

        samples = term.coef.sample((4, 20), jkey(0))
        with pytest.raises(ValueError):
            gam.summarise_regions(term=term, samples=samples, polys=polys_)

    def test_no_polys(self, tb: gam.TermBuilder) -> None:
        term = tb.ri("district")
        _ = lsl.Model([term])

        samples = term.coef.sample((4, 20), jkey(0))
        with pytest.raises(ValueError):
            gam.summarise_regions(term=term, samples=samples)
