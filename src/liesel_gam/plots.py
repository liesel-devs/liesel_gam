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

from .registry import CategoryMapping
from .summary import (
    polys_to_df,
    summarise_1d_smooth,
    summarise_1d_smooth_clustered,
    summarise_by_samples,
    summarise_cluster,
    summarise_lin,
    summarise_nd_smooth,
    summarise_regions,
)
from .term import LinTerm, MRFTerm, RITerm, StrctLinTerm, StrctTensorProdTerm, StrctTerm

KeyArray = Any


def plot_1d_smooth(
    term: StrctTerm,
    samples: dict[str, Array],
    newdata: gs.Position | None | Mapping[str, ArrayLike] = None,
    ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
    hdi_prob: float | None = None,
    show_n_samples: int | None = 50,
    seed: int | KeyArray = 1,
    ngrid: int = 150,
):
    if newdata is None:
        # TODO: Currently, this branch of the function assumes that term.basis.x is
        # a strong node.
        # That is not necessarily always the case.
        xgrid = np.linspace(term.basis.x.value.min(), term.basis.x.value.max(), 150)
        newdata_x: Mapping[str, ArrayLike] = {term.basis.x.name: xgrid}
    else:
        newdata_x = newdata
        xgrid = np.asarray(newdata[term.basis.x.name])

    newdata_x = {k: jnp.asarray(v) for k, v in newdata_x.items()}

    term_samples = term.predict(samples, newdata=newdata_x)

    term_summary = summarise_1d_smooth(
        term=term,
        samples=samples,
        newdata=newdata,
        quantiles=(0.05, 0.95) if ci_quantiles is None else ci_quantiles,
        hdi_prob=0.9 if hdi_prob is None else hdi_prob,
        ngrid=ngrid,
    )

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

        summary_samples_df[term.basis.x.name] = np.tile(
            np.squeeze(xgrid), show_n_samples
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


# using q_0.05 and q_0.95 explicitly here
# even though users could choose to return other quantiles like 0.1 and 0.9
# then they can supply q_0.1 and q_0.9, etc.
PlotVars = Literal[
    "mean", "sd", "var", "hdi_low", "hdi_high", "q_0.05", "q_0.5", "q_0.95"
]


def plot_2d_smooth(
    term: StrctTensorProdTerm | StrctTerm,
    samples: Mapping[str, jax.Array],
    newdata: gs.Position | None | Mapping[str, ArrayLike] = None,
    ngrid: int = 20,
    which: PlotVars | Sequence[PlotVars] = "mean",
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    hdi_prob: float = 0.9,
    newdata_meshgrid: bool = False,
):
    if isinstance(term, StrctTensorProdTerm):
        names = list(term.input_obs)
        if len(names) != 2:
            raise ValueError(
                f"'plot_2d_smooth' can only handle smooths with two inputs, "
                f"got {len(names)} for smooth {term}: {names}"
            )

        for v in term.input_obs.values():
            if jnp.issubdtype(v.value, jnp.integer):
                raise TypeError(
                    "'plot_2d_smooth' expects continuous marginals, got "
                    f"type {v.value.dtype} for {v}"
                )
    else:
        names = [n.var.name for n in term.basis.x.all_input_nodes()]  # type: ignore

    term_summary = summarise_nd_smooth(
        term=term,
        samples=samples,
        newdata=newdata,
        ngrid=ngrid,
        which=which,
        quantiles=quantiles,
        hdi_prob=hdi_prob,
        newdata_meshgrid=newdata_meshgrid,
    )

    p = (
        p9.ggplot(term_summary)
        + p9.labs(title=f"Posterior summary of {term.name}")
        + p9.aes(*names, fill="value")
        + p9.facet_wrap("~variable", labeller="label_both")
    )

    p = p + p9.geom_tile()

    return p


def plot_polys(
    region: str,
    which: str | Sequence[str],
    df: pd.DataFrame,
    polys: Mapping[str, ArrayLike],
    show_unobserved: bool = True,
    observed_color: str = "none",
    unobserved_color: str = "red",
) -> p9.ggplot:
    if isinstance(which, str):
        which = [which]

    poly_df = polys_to_df(polys)

    df["label"] = df[region].astype(str)
    # plot_df = df.merge(poly_df, on="label")

    if "observed" not in df.columns:
        df["observed"] = True

    if df["observed"].all():
        show_unobserved = False

    plot_df = poly_df.merge(df, on="label")

    plot_df = plot_df.melt(
        id_vars=["label", "V0", "V1", "observed"],
        value_vars=which,
        var_name="variable",
        value_name="value",
    )

    plot_df["variable"] = pd.Categorical(plot_df["variable"], categories=which)

    p = (
        p9.ggplot(plot_df)
        + p9.aes("V0", "V1", group="label", fill="value")
        + p9.aes(color="observed")
        + p9.facet_wrap("~variable", labeller="label_both")
        + p9.scale_color_manual({True: observed_color, False: unobserved_color})
        + p9.guides(color=p9.guide_legend(override_aes={"fill": None}))
    )
    if show_unobserved:
        p = p + p9.geom_polygon()
    else:
        p = p + p9.geom_polygon(data=plot_df.query("observed == True"))
        p = p + p9.geom_polygon(data=plot_df.query("observed == False"), fill="none")

    return p


def plot_regions(
    term: RITerm | MRFTerm | StrctTerm,
    samples: Mapping[str, jax.Array],
    newdata: gs.Position | None | Mapping[str, ArrayLike] = None,
    which: PlotVars | Sequence[PlotVars] = "mean",
    polys: Mapping[str, ArrayLike] | None = None,
    labels: CategoryMapping | None = None,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    hdi_prob: float = 0.9,
    show_unobserved: bool = True,
    observed_color: str = "none",
    unobserved_color: str = "red",
) -> p9.ggplot:
    plot_df = summarise_regions(
        term=term,
        samples=samples,
        newdata=newdata,
        which=which,
        polys=polys,
        labels=labels,
        quantiles=quantiles,
        hdi_prob=hdi_prob,
    )
    p = (
        p9.ggplot(plot_df)
        + p9.aes("V0", "V1", group="label", fill="value")
        + p9.aes(color="observed")
        + p9.facet_wrap("~variable", labeller="label_both")
        + p9.scale_color_manual({True: observed_color, False: unobserved_color})
        + p9.guides(color=p9.guide_legend(override_aes={"fill": None}))
    )
    if show_unobserved:
        p = p + p9.geom_polygon()
    else:
        p = p + p9.geom_polygon(data=plot_df.query("observed == True"))
        p = p + p9.geom_polygon(data=plot_df.query("observed == False"), fill="none")

    p += p9.labs(title=f"Plot of {term.name}")
    return p


def plot_forest(
    term: RITerm | MRFTerm | LinTerm | StrctLinTerm,
    samples: Mapping[str, jax.Array],
    newdata: gs.Position | None | Mapping[str, ArrayLike] = None,
    labels: CategoryMapping | None = None,
    ymin: str = "hdi_low",
    ymax: str = "hdi_high",
    ci_quantiles: tuple[float, float] = (0.05, 0.95),
    hdi_prob: float = 0.9,
    show_unobserved: bool = True,
    highlight_unobserved: bool = True,
    unobserved_color: str = "red",
    indices: Sequence[int] | None = None,
) -> p9.ggplot:
    if isinstance(term, RITerm | MRFTerm):
        return plot_forest_clustered(
            term=term,
            samples=samples,
            newdata=newdata,
            labels=labels,
            ymin=ymin,
            ymax=ymax,
            ci_quantiles=ci_quantiles,
            hdi_prob=hdi_prob,
            show_unobserved=show_unobserved,
            highlight_unobserved=highlight_unobserved,
            unobserved_color=unobserved_color,
            indices=indices,
        )
    elif isinstance(term, LinTerm | StrctLinTerm):
        return plot_forest_lin(
            term=term,
            samples=samples,
            ymin=ymin,
            ymax=ymax,
            ci_quantiles=ci_quantiles,
            hdi_prob=hdi_prob,
            indices=indices,
        )
    else:
        raise TypeError(f"term has unsupported type {type(term)}.")


def plot_forest_lin(
    term: LinTerm | StrctLinTerm,
    samples: Mapping[str, jax.Array],
    ymin: str = "hdi_low",
    ymax: str = "hdi_high",
    ci_quantiles: tuple[float, float] = (0.05, 0.95),
    hdi_prob: float = 0.9,
    indices: Sequence[int] | None = None,
) -> p9.ggplot:
    df = summarise_lin(
        term=term,
        samples=samples,
        quantiles=ci_quantiles,
        hdi_prob=hdi_prob,
        indices=indices,
    )

    df[ymin] = df[ymin].astype(df["mean"].dtype)
    df[ymax] = df[ymax].astype(df["mean"].dtype)

    p = (
        p9.ggplot(df)
        + p9.aes("x", "mean")
        + p9.geom_hline(yintercept=0, color="grey")
        + p9.geom_linerange(p9.aes(ymin=ymin, ymax=ymax), color="grey")
        + p9.geom_point()
        + p9.coord_flip()
        + p9.labs(x="x")
    )

    p += p9.labs(title=f"Posterior summary of {term.name}")

    return p


def plot_forest_clustered(
    term: RITerm | MRFTerm | StrctTerm,
    samples: Mapping[str, jax.Array],
    newdata: gs.Position | None | Mapping[str, ArrayLike] = None,
    labels: CategoryMapping | None = None,
    ymin: str = "hdi_low",
    ymax: str = "hdi_high",
    ci_quantiles: tuple[float, float] = (0.05, 0.95),
    hdi_prob: float = 0.9,
    show_unobserved: bool = True,
    highlight_unobserved: bool = True,
    unobserved_color: str = "red",
    indices: Sequence[int] | None = None,
) -> p9.ggplot:
    if labels is None:
        try:
            labels = term.mapping  # type: ignore
        except (AttributeError, ValueError):
            labels = None

    df = summarise_cluster(
        term=term,
        samples=samples,
        newdata=newdata,
        labels=labels,
        quantiles=ci_quantiles,
        hdi_prob=hdi_prob,
    )
    cluster = term.basis.x.name

    if labels is None:
        xlab = cluster + " (indices)"
    else:
        xlab = cluster + " (labels)"

    df[ymin] = df[ymin].astype(df["mean"].dtype)
    df[ymax] = df[ymax].astype(df["mean"].dtype)

    if indices is not None:
        df = df.iloc[indices, :]

    if not show_unobserved:
        df = df.query("observed == True")

    p = (
        p9.ggplot(df)
        + p9.aes(cluster, "mean")
        + p9.geom_hline(yintercept=0, color="grey")
        + p9.geom_linerange(p9.aes(ymin=ymin, ymax=ymax), color="grey")
        + p9.geom_point()
        + p9.coord_flip()
        + p9.labs(x=xlab)
    )

    if highlight_unobserved:
        df_uo = df.query("observed == False")
        p = p + p9.geom_point(
            p9.aes(cluster, "mean"),
            color=unobserved_color,
            shape="x",
            data=df_uo,
        )

    p += p9.labs(title=f"Posterior summary of {term.name}")

    return p


def plot_1d_smooth_clustered(
    clustered_term: lsl.Var,
    samples: Mapping[str, jax.Array],
    ngrid: int = 20,
    newdata: gs.Position | None | Mapping[str, ArrayLike] = None,
    labels: CategoryMapping | None = None,
    color_scale: str = "viridis",
    newdata_meshgrid: bool = False,
):
    ci_quantiles = (0.05, 0.5, 0.95)
    hdi_prob = 0.9

    term = clustered_term.value_node["x"]
    cluster = clustered_term.value_node["cluster"]

    assert isinstance(term, StrctTerm | lsl.Var)
    assert isinstance(cluster, RITerm | MRFTerm)

    if labels is None:
        try:
            labels = cluster.mapping  # type: ignore
        except (AttributeError, ValueError):
            labels = None

    term_summary = summarise_1d_smooth_clustered(
        clustered_term=clustered_term,
        samples=samples,
        ngrid=ngrid,
        ci_quantiles=ci_quantiles,
        hdi_prob=hdi_prob,
        labels=labels,
        newdata=newdata,
        newdata_meshgrid=newdata_meshgrid,
    )

    if labels is None:
        clab = cluster.basis.x.name + " (indices)"
    else:
        clab = cluster.basis.x.name + " (labels)"

    if isinstance(term, StrctTerm):
        x = term.basis.x
    else:
        x = term

    p = (
        p9.ggplot(term_summary)
        + p9.aes(x.name, "mean", group=cluster.basis.x.name)
        + p9.aes(color=cluster.basis.x.name)
        + p9.labs(
            title=f"Posterior summary of {clustered_term.name}", x=x.name, color=clab
        )
        + p9.facet_wrap("~variable", labeller="label_both")
        + p9.scale_color_cmap_d(color_scale)
        + p9.geom_line()
    )

    return p
