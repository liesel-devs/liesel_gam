from collections.abc import Mapping, Sequence
from typing import Any, Literal

import jax
import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import pandas as pd
import plotnine as p9
from jax import Array
from jax.typing import ArrayLike

from .builder.registry import CategoryMapping
from .var import MRFTerm, Term

KeyArray = Any


def summarise_by_samples(
    key: KeyArray, a: Any, name: str, n: int = 100
) -> pd.DataFrame:
    """
    - index: index of the flattened array
    - sample: sample number
    - obs: observation number (enumerates response values)
    - chain: chain number
    """

    _, iterations, _ = a.shape

    a = np.concatenate(a, axis=0)
    idx = jax.random.choice(key, a.shape[0], shape=(n,), replace=True)

    a_column = a[idx, :].ravel()
    sample_column = np.repeat(np.arange(n), a.shape[-1])
    index_column = np.repeat(idx, a.shape[-1])
    obs_column = np.tile(np.arange(a.shape[-1]), n)

    data = {name: a_column, "sample": sample_column}
    data["index"] = index_column
    data["obs"] = obs_column
    df = pd.DataFrame(data)

    df["chain"] = df["index"] // iterations

    return df


def plot_1d_smooth(
    term: Term,
    samples: dict[str, Array],
    newdata: gs.Position | None = None,
    ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
    hdi_prob: float | None = None,
    show_n_samples: int | None = 50,
    seed: int | KeyArray = 1,
):
    if newdata is None:
        # TODO: Currently, this branch of the function assumes that term.basis.x is
        # a strong node.
        # That is not necessarily always the case.
        xgrid = jnp.linspace(term.basis.x.value.min(), term.basis.x.value.max(), 150)
        newdata_x = {term.basis.x.name: xgrid}
    else:
        newdata_x = newdata

    term_samples = term.predict(samples, newdata=newdata_x)
    ci_quantiles_ = (0.05, 0.95) if ci_quantiles is None else ci_quantiles
    hdi_prob_ = 0.9 if hdi_prob is None else hdi_prob
    term_summary = (
        gs.SamplesSummary.from_array(
            term_samples, name=term.name, quantiles=ci_quantiles_, hdi_prob=hdi_prob_
        )
        .to_dataframe()
        .reset_index()
    )

    term_summary[term.basis.x.name] = xgrid

    p = p9.ggplot(term_summary) + p9.labs(
        title=f"Posterior summary of {term.name}",
        x=term.basis.x.name,
        y=term.name,
    )

    if ci_quantiles is not None:
        p = p + p9.geom_ribbon(
            p9.aes(
                term.basis.x.name,
                ymin=f"q_{str(ci_quantiles[0])}",
                ymax=f"q_{str(ci_quantiles[1])}",
            ),
            fill="#56B4E9",
            alpha=0.5,
            data=term_summary,
        )

    if hdi_prob is not None:
        p = p + p9.geom_line(
            p9.aes(term.basis.x.name, "hdi_low"),
            linetype="dashed",
            data=term_summary,
        )

        p = p + p9.geom_line(
            p9.aes(term.basis.x.name, "hdi_high"),
            linetype="dashed",
            data=term_summary,
        )

    if show_n_samples is not None and show_n_samples > 0:
        key = jax.random.key(seed) if isinstance(seed, int) else seed

        summary_samples_df = summarise_by_samples(
            key=key, a=term_samples, name=term.name, n=show_n_samples
        )

        summary_samples_df[term.basis.x.name] = jnp.tile(
            jnp.squeeze(xgrid), show_n_samples
        )

        p = p + p9.geom_line(
            p9.aes(term.basis.x.name, term.name, group="sample"),
            color="grey",
            data=summary_samples_df,
            alpha=0.3,
        )

    p = p + p9.geom_line(
        p9.aes(term.basis.x.name, "mean"), data=term_summary, size=1.3, color="blue"
    )

    return p


def grid_2d(inputs: dict[str, jax.typing.ArrayLike], ngrid: int) -> dict[str, Any]:
    mins = {k: jnp.min(v) for k, v in inputs.items()}
    maxs = {k: jnp.max(v) for k, v in inputs.items()}
    grids = {k: np.linspace(mins[k], maxs[k], ngrid) for k in inputs}
    full_grid_arrays = [v.flatten() for v in np.meshgrid(*grids.values())]
    full_grids = dict(zip(inputs.keys(), full_grid_arrays))
    return full_grids


def input_grid_2d_smooth(term: Term, ngrid: int) -> dict[str, jax.typing.ArrayLike]:
    if not isinstance(term.basis.x, lsl.TransientCalc | lsl.Calc):
        raise NotImplementedError(
            "Function not implemented for bases with inputs of "
            f"type {type(term.basis.x)}."
        )
    inputs = {n.var.name: n.var.value for n in term.basis.x.all_input_nodes()}  # type: ignore
    return grid_2d(inputs, ngrid)


# using q_0.05 and q_0.95 explicitly here
# even though users could choose to return other quantiles like 0.1 and 0.9
# then they can supply q_0.1 and q_0.9, etc.
PlotVars = Literal[
    "mean", "sd", "var", "hdi_low", "hdi_high", "q_0.05", "q_0.5", "q_0.95"
]


def plot_2d_smooth(
    term: Term,
    samples: dict[str, Array],
    newdata: gs.Position | None = None,
    ngrid: int = 20,
    plot_vars: PlotVars | Sequence[PlotVars] = "mean",
    ci_quantiles: tuple[float, float, float] | None = (0.05, 0.5, 0.95),
    hdi_prob: float | None = None,
):
    if isinstance(plot_vars, str):
        plot_vars = [plot_vars]

    if newdata is None:
        # TODO: Currently, this branch of the function assumes that Basis.x is a
        # Calc or TransientCalc. That is not necessarily always the case.
        newdata_x = input_grid_2d_smooth(term, ngrid=ngrid)
    else:
        full_grid_arrays = [v.flatten() for v in np.meshgrid(*newdata.values())]
        newdata_x = dict(zip(newdata.keys(), full_grid_arrays))

    term_samples = term.predict(samples, newdata=newdata_x)
    ci_quantiles_ = (0.05, 0.95) if ci_quantiles is None else ci_quantiles
    hdi_prob_ = 0.9 if hdi_prob is None else hdi_prob
    term_summary = (
        gs.SamplesSummary.from_array(
            term_samples, name=term.name, quantiles=ci_quantiles_, hdi_prob=hdi_prob_
        )
        .to_dataframe()
        .reset_index()
    )

    for k, v in newdata_x.items():
        term_summary[k] = np.asarray(v)

    term_summary.reset_index(inplace=True)
    term_summary = term_summary.melt(
        id_vars=["index"] + list(newdata_x.keys()),
        value_vars=plot_vars,
        var_name="variable",
        value_name="value",
    )

    term_summary["variable"] = pd.Categorical(
        term_summary["variable"], categories=plot_vars
    )

    p = (
        p9.ggplot(term_summary)
        + p9.labs(
            title=f"Posterior summary of {term.name}",
            x=term.basis.x.name,
            y=term.name,
        )
        + p9.aes(*list(newdata_x.keys()), fill="value")
        + p9.facet_wrap("~variable", labeller="label_both")
    )

    p = p + p9.geom_tile()

    return p


def polys_to_df(polys: Mapping[str, ArrayLike]):
    poly_labels = list(polys)
    poly_coords = list(polys.values())
    poly_coord_dim = np.shape(poly_coords[0])[-1]
    poly_df = pd.concat(
        [
            pd.DataFrame(
                poly_coords[i], columns=[f"V{i}" for i in range(poly_coord_dim)]
            ).assign(vertex=lambda df: df.index + 1, id=i, label=poly_labels[i])
            for i in range(len(polys))
        ],
        ignore_index=True,
    )
    return poly_df


def plot_polys(
    region: str,
    plot_vars: str | Sequence[str],
    df: pd.DataFrame,
    polys: Mapping[str, ArrayLike],
) -> p9.ggplot:
    if isinstance(plot_vars, str):
        plot_vars = [plot_vars]

    poly_df = polys_to_df(polys)

    df["label"] = df[region].astype(str)
    # plot_df = df.merge(poly_df, on="label")
    plot_df = poly_df.merge(df, on="label")

    plot_df = plot_df.melt(
        id_vars=["label", "V0", "V1"],
        value_vars=plot_vars,
        var_name="variable",
        value_name="value",
    )

    plot_df["variable"] = pd.Categorical(plot_df["variable"], categories=plot_vars)

    p = (
        p9.ggplot(plot_df)
        + p9.aes("V0", "V1", group="label", fill="value")
        + p9.geom_polygon()
        + p9.facet_wrap("~variable", labeller="label_both")
    )
    return p


def ri_summary(
    term: MRFTerm | Term,
    samples,
    newdata: gs.Position | None = None,
    labels: CategoryMapping | Sequence[str] | None = None,
    ci_quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    hdi_prob: float = 0.9,
    show_unobserved: bool = True,
) -> pd.DataFrame:
    if newdata is None and isinstance(labels, CategoryMapping) and show_unobserved:
        newdata_x = {term.basis.x.name: np.asarray(list(labels.integers_to_labels_map))}
    elif newdata is None:
        newdata_x = {term.basis.x.name: np.unique(term.basis.x.value)}
    else:
        newdata_x = newdata

    predictions = term.predict(samples=samples, newdata=newdata_x)
    predictions_summary = (
        gs.SamplesSummary.from_array(
            predictions,
            quantiles=ci_quantiles,
            hdi_prob=0.9 if hdi_prob is None else hdi_prob,
        )
        .to_dataframe()
        .reset_index()
    )

    if isinstance(labels, CategoryMapping):
        codes = newdata_x[term.basis.x.name]
        labels_str = list(labels.integers_to_labels(codes))
        predictions_summary[term.basis.x.name] = labels_str
    elif labels is not None:
        labels_str = list(labels)
        predictions_summary[term.basis.x.name] = labels_str
    else:
        predictions_summary[term.basis.x.name] = term.basis.x.value

    return predictions_summary


def plot_regions(
    term: MRFTerm | Term,
    samples,
    newdata: gs.Position | None = None,
    plot_vars: PlotVars | Sequence[PlotVars] = "mean",
    polys: Mapping[str, ArrayLike] | None = None,
    labels: CategoryMapping | Sequence[str] | None = None,
    ci_quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    hdi_prob: float = 0.9,
    show_unobserved: bool = True,
) -> p9.ggplot:
    polygons = None
    if polys is not None:
        polygons = polys
    else:
        try:
            # using type ignore here, since the case of term not having the attribute
            # polygons is handle by the try except
            polygons = term.polygons  # type: ignore
        except AttributeError:
            pass

    if not polygons:
        raise ValueError(
            "When passing a term with term.polygons=None, polygons must "
            "be supplied manually."
        )

    if labels is None:
        try:
            labels = term.mapping  # type: ignore
        except AttributeError:
            labels = None

    df = ri_summary(
        term=term,
        samples=samples,
        newdata=newdata,
        labels=labels,
        ci_quantiles=ci_quantiles,
        hdi_prob=hdi_prob,
        show_unobserved=show_unobserved,
    )
    region = term.basis.x.name
    return plot_polys(region=region, plot_vars=plot_vars, df=df, polys=polygons)


def plot_forest(
    term: MRFTerm | Term,
    samples,
    newdata: gs.Position | None = None,
    labels: CategoryMapping | Sequence[str] | None = None,
    ymin: str = "hdi_low",
    ymax: str = "hdi_high",
    ci_quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    hdi_prob: float = 0.9,
    show_unobserved: bool = True,
) -> p9.ggplot:
    if labels is None:
        try:
            labels = term.mapping  # type: ignore
        except AttributeError:
            labels = None

    df = ri_summary(
        term=term,
        samples=samples,
        newdata=newdata,
        labels=labels,
        ci_quantiles=ci_quantiles,
        hdi_prob=hdi_prob,
        show_unobserved=show_unobserved,
    )
    cluster = term.basis.x.name

    if labels is None:
        xlab = cluster + " (indices)"
    else:
        xlab = cluster + " (labels)"

    df[ymin] = df[ymin].astype(df["mean"].dtype)
    df[ymax] = df[ymax].astype(df["mean"].dtype)

    p = (
        p9.ggplot(df)
        + p9.aes(cluster, "mean", color="mean")
        + p9.geom_hline(yintercept=0, color="grey")
        + p9.geom_pointrange(p9.aes(ymin=ymin, ymax=ymax))
        + p9.coord_flip()
        + p9.labs(x=xlab)
    )
    return p


def plot_1d_smooth_clustered(
    term: lsl.Var,
    x: lsl.Var,
    cluster: lsl.Var,
    samples: dict[str, Array],
    ngrid: int = 150,
    newdata: gs.Position | None = None,
    plot_vars: PlotVars | Sequence[PlotVars] = "mean",
    ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
    hdi_prob: float | None = None,
    labels: Sequence[str] | None = None,
):
    if isinstance(plot_vars, str):
        plot_vars = [plot_vars]

    if newdata is None:
        xgrid = jnp.linspace(x.value.min(), x.value.max(), ngrid)
        cluster_grid = jnp.unique(cluster.value)
        grid = {x.name: xgrid, cluster.name: cluster_grid}
    else:
        grid = newdata

    full_grid_arrays = [v.flatten() for v in np.meshgrid(*grid.values())]
    newdata_x = dict(zip(grid.keys(), full_grid_arrays))

    term_samples = term.predict(samples, newdata=newdata_x)
    ci_quantiles_ = (0.05, 0.95) if ci_quantiles is None else ci_quantiles
    hdi_prob_ = 0.9 if hdi_prob is None else hdi_prob
    term_summary = (
        gs.SamplesSummary.from_array(
            term_samples, name=term.name, quantiles=ci_quantiles_, hdi_prob=hdi_prob_
        )
        .to_dataframe()
        .reset_index()
    )

    for k, v in newdata_x.items():
        term_summary[k] = np.asarray(v)

    if labels is not None:
        term_summary[cluster.name] = np.repeat(labels, ngrid)

    term_summary.reset_index(inplace=True)

    if labels is None:
        clab = cluster.name + " (indices)"
    else:
        clab = cluster.name + " (labels)"

    p = (
        p9.ggplot(term_summary)
        + p9.aes(x.name, "mean", group=cluster.name)
        + p9.aes(color=cluster.name)
        + p9.labs(title=f"Posterior summary of {term.name}", x=x.name, color=clab)
        + p9.facet_wrap("~variable", labeller="label_both")
        + p9.geom_line()
    )

    return p
