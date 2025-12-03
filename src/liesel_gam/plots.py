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
from .var import LinTerm, MRFTerm, RITerm, Term

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


def summarise_1d_smooth(
    term: Term,
    samples: dict[str, Array],
    newdata: gs.Position | None | Mapping[str, ArrayLike] = None,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    hdi_prob: float = 0.9,
    ngrid: int = 150,
):
    if newdata is None:
        # TODO: Currently, this branch of the function assumes that term.basis.x is
        # a strong node.
        # That is not necessarily always the case.
        xgrid = np.linspace(term.basis.x.value.min(), term.basis.x.value.max(), ngrid)
        newdata_x: Mapping[str, ArrayLike] = {term.basis.x.name: xgrid}
    else:
        newdata_x = newdata
        xgrid = np.asarray(newdata[term.basis.x.name])

    newdata_x = {k: jnp.asarray(v) for k, v in newdata_x.items()}

    term_samples = term.predict(samples, newdata=newdata_x)
    term_summary = (
        gs.SamplesSummary.from_array(
            term_samples, name=term.name, quantiles=quantiles, hdi_prob=hdi_prob
        )
        .to_dataframe()
        .reset_index()
    )

    term_summary[term.basis.x.name] = xgrid
    return term_summary


def plot_1d_smooth(
    term: Term,
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


def grid_nd(inputs: dict[str, jax.typing.ArrayLike], ngrid: int) -> dict[str, Any]:
    mins = {k: jnp.min(v) for k, v in inputs.items()}
    maxs = {k: jnp.max(v) for k, v in inputs.items()}
    grids = {k: np.linspace(mins[k], maxs[k], ngrid) for k in inputs}
    full_grid_arrays = [v.flatten() for v in np.meshgrid(*grids.values())]
    full_grids = dict(zip(inputs.keys(), full_grid_arrays))
    return full_grids


def input_grid_nd_smooth(
    term: Term | LinTerm, ngrid: int
) -> dict[str, jax.typing.ArrayLike]:
    if not isinstance(term.basis.x, lsl.TransientCalc | lsl.Calc):
        raise NotImplementedError(
            "Function not implemented for bases with inputs of "
            f"type {type(term.basis.x)}."
        )
    inputs = {n.var.name: n.var.value for n in term.basis.x.all_input_nodes()}  # type: ignore
    return grid_nd(inputs, ngrid)


# using q_0.05 and q_0.95 explicitly here
# even though users could choose to return other quantiles like 0.1 and 0.9
# then they can supply q_0.1 and q_0.9, etc.
PlotVars = Literal[
    "mean", "sd", "var", "hdi_low", "hdi_high", "q_0.05", "q_0.5", "q_0.95"
]


def summarise_nd_smooth(
    term: Term,
    samples: Mapping[str, jax.Array],
    newdata: gs.Position | None | Mapping[str, ArrayLike] = None,
    ngrid: int = 20,
    which: PlotVars | Sequence[PlotVars] = "mean",
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    hdi_prob: float = 0.9,
    newdata_meshgrid: bool = False,
):
    if isinstance(which, str):
        which = [which]

    if newdata is None:
        # TODO: Currently, this branch of the function assumes that Basis.x is a
        # Calc or TransientCalc. That is not necessarily always the case.
        newdata_x: Mapping[str, ArrayLike] = input_grid_nd_smooth(term, ngrid=ngrid)
    elif newdata_meshgrid:
        full_grid_arrays = [v.flatten() for v in np.meshgrid(*newdata.values())]
        newdata_x = dict(zip(newdata.keys(), full_grid_arrays))
    else:
        newdata_x = newdata

    newdata_x = {k: jnp.asarray(v) for k, v in newdata_x.items()}
    term_samples = term.predict(samples, newdata=newdata_x)
    ci_quantiles_ = (0.05, 0.95) if quantiles is None else quantiles
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
        value_vars=which,
        var_name="variable",
        value_name="value",
    )

    term_summary["variable"] = pd.Categorical(
        term_summary["variable"], categories=which
    )
    return term_summary


def plot_2d_smooth(
    term: Term,
    samples: Mapping[str, jax.Array],
    newdata: gs.Position | None | Mapping[str, ArrayLike] = None,
    ngrid: int = 20,
    which: PlotVars | Sequence[PlotVars] = "mean",
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    hdi_prob: float = 0.9,
    newdata_meshgrid: bool = False,
):
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

    names = [n.var.name for n in term.basis.x.all_input_nodes()]  # type: ignore

    p = (
        p9.ggplot(term_summary)
        + p9.labs(
            title=f"Posterior summary of {term.name}",
            x=term.basis.x.name,
            y=term.name,
        )
        + p9.aes(*names, fill="value")
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


def _convert_to_integers(
    grid: np.typing.NDArray,
    labels: Sequence[str] | CategoryMapping | None,
    term: RITerm | MRFTerm | lsl.Var,
) -> np.typing.NDArray[np.int_]:
    if isinstance(labels, CategoryMapping):
        grid = labels.to_integers(grid)
    else:
        try:
            grid = term.mapping.to_integers(grid)  # type: ignore
        except (ValueError, AttributeError):
            if not np.issubdtype(grid.dtype, np.integer):
                raise TypeError(
                    f"There's no mapping available on the term {term}. "
                    "In this case, its values in 'newdata' must be specified "
                    f"as integer codes. Got data type {grid.dtype}"
                )

    return grid


def summarise_cluster(
    term: RITerm | MRFTerm | Term,
    samples: Mapping[str, jax.Array],
    newdata: gs.Position
    | None
    | Mapping[str, ArrayLike | Sequence[int] | Sequence[str]] = None,
    labels: CategoryMapping | Sequence[str] | None = None,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    hdi_prob: float = 0.9,
) -> pd.DataFrame:
    if labels is None:
        try:
            labels = term.mapping  # type: ignore
        except (AttributeError, ValueError):
            labels = None

    if newdata is None and isinstance(labels, CategoryMapping):
        grid = np.asarray(list(labels.integers_to_labels_map))
        unique_x = np.unique(term.basis.x.value)
        newdata_x: Mapping[str, ArrayLike] = {term.basis.x.name: grid}
        observed = [x in unique_x for x in grid]
    elif newdata is None:
        grid = np.unique(term.basis.x.value)
        newdata_x = {term.basis.x.name: grid}
        observed = [True for _ in grid]
    else:
        unique_x = np.unique(term.basis.x.value)
        grid = np.asarray(newdata[term.basis.x.name])
        grid = _convert_to_integers(grid, labels, term)

        observed = [x in unique_x for x in grid]
        newdata_x = {term.basis.x.name: grid}

    newdata_x = {k: jnp.asarray(v) for k, v in newdata_x.items()}
    predictions = term.predict(samples=samples, newdata=newdata_x)
    predictions_summary = (
        gs.SamplesSummary.from_array(
            predictions,
            quantiles=quantiles,
            hdi_prob=0.9 if hdi_prob is None else hdi_prob,
        )
        .to_dataframe()
        .reset_index()
    )

    if isinstance(labels, CategoryMapping):
        codes = newdata_x[term.basis.x.name]
        labels_str = list(labels.integers_to_labels(codes))
        categories = list(labels.labels_to_integers_map)
        predictions_summary[term.basis.x.name] = pd.Categorical(
            labels_str, categories=categories
        )
    elif labels is not None:
        labels_str = list(labels)
        categories = sorted(set(labels_str))
        predictions_summary[term.basis.x.name] = pd.Categorical(
            labels_str, categories=categories
        )
    else:
        predictions_summary[term.basis.x.name] = pd.Categorical(
            np.asarray(term.basis.x.value)
        )

    predictions_summary["observed"] = observed

    return predictions_summary


def summarise_regions(
    term: RITerm | MRFTerm | Term,
    samples: Mapping[str, jax.Array],
    newdata: gs.Position | None | Mapping[str, ArrayLike] = None,
    which: PlotVars | Sequence[PlotVars] = "mean",
    polys: Mapping[str, ArrayLike] | None = None,
    labels: CategoryMapping | Sequence[str] | None = None,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    hdi_prob: float = 0.9,
) -> pd.DataFrame:
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
            "When passing a term without polygons, polygons must be supplied manually "
            "through the argument 'polys'"
        )

    df = summarise_cluster(
        term=term,
        samples=samples,
        newdata=newdata,
        labels=labels,
        quantiles=quantiles,
        hdi_prob=hdi_prob,
    )
    region = term.basis.x.name
    if isinstance(which, str):
        which = [which]

    unique_labels_in_df = df[term.basis.x.name].unique().tolist()
    assert polygons is not None
    for region_label in polygons:
        if region_label not in unique_labels_in_df:
            raise ValueError(
                f"Label '{region_label}' found in polys, but not in cluster summary. "
                f"Known labels: {unique_labels_in_df}"
            )

    poly_df = polys_to_df(polygons)

    df["label"] = df[region].astype(str)

    if "observed" not in df.columns:
        df["observed"] = True

    plot_df = poly_df.merge(df, on="label")

    plot_df = plot_df.melt(
        id_vars=["label", "V0", "V1", "observed"],
        value_vars=which,
        var_name="variable",
        value_name="value",
    )

    plot_df["variable"] = pd.Categorical(plot_df["variable"], categories=which)

    return plot_df


def plot_regions(
    term: RITerm | MRFTerm | Term,
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
    term: RITerm | MRFTerm | LinTerm,
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
    elif isinstance(term, LinTerm):
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


def summarise_lin(
    term: LinTerm,
    samples: Mapping[str, jax.Array],
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    hdi_prob: float = 0.9,
    indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    if indices is not None:
        coef_samples = samples[term.coef.name][..., indices]
        colnames = [term.column_names[i] for i in indices]
    else:
        coef_samples = samples[term.coef.name]
        colnames = term.column_names

    df = (
        gs.SamplesSummary.from_array(
            coef_samples, quantiles=quantiles, hdi_prob=hdi_prob
        )
        .to_dataframe()
        .reset_index()
    )

    df["x"] = colnames
    df.drop(["variable", "var_fqn", "var_index"], axis=1, inplace=True)
    df.insert(0, "x", df.pop("x"))
    return df


def plot_forest_lin(
    term: LinTerm,
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
    term: RITerm | MRFTerm | Term,
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
        except AttributeError:
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


def summarise_1d_smooth_clustered(
    clustered_term: lsl.Var,
    samples: Mapping[str, jax.Array],
    ngrid: int = 20,
    newdata: gs.Position
    | None
    | Mapping[str, ArrayLike | Sequence[int] | Sequence[str]] = None,
    which: PlotVars | Sequence[PlotVars] = "mean",
    ci_quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    hdi_prob: float = 0.9,
    labels: CategoryMapping | None | Sequence[str] = None,
    newdata_meshgrid: bool = False,
):
    if isinstance(which, str):
        which = [which]

    term = clustered_term.value_node["x"]
    cluster = clustered_term.value_node["cluster"]

    assert isinstance(term, Term | lsl.Var)
    assert isinstance(cluster, RITerm | MRFTerm)

    if labels is None:
        try:
            labels = cluster.mapping  # type: ignore
        except (AttributeError, ValueError):
            labels = None

    if isinstance(term, Term):
        x = term.basis.x
    else:
        x = term

    if newdata is None:
        if not jnp.issubdtype(x.value.dtype, jnp.floating):
            raise TypeError(
                "Automatic grid creation is valid only for continuous x, got "
                f"dtype {jnp.dtype(x.value)} for {x}."
            )

    if newdata is None and isinstance(labels, CategoryMapping):
        cgrid = np.asarray(list(labels.integers_to_labels_map))  # integer codes
        unique_clusters = np.unique(cluster.basis.x.value)  # unique codes

        if isinstance(x, lsl.Node) or x.strong:
            xgrid: Mapping[str, ArrayLike] = {
                x.name: jnp.linspace(x.value.min(), x.value.max(), ngrid)
            }
        else:
            assert isinstance(term, Term | LinTerm), (
                f"Wrong type for term: {type(term)}"
            )
            ncols = jnp.shape(term.basis.value)[-1]
            xgrid = input_grid_nd_smooth(term, ngrid=int(np.pow(ngrid, 1 / ncols)))

        grid: Mapping[str, ArrayLike | Sequence[int] | Sequence[str]] = dict(xgrid) | {
            cluster.basis.x.name: cgrid
        }

        # code : bool
        observed = {x: x in unique_clusters for x in cgrid}
    elif newdata is None:
        cgrid = np.unique(cluster.basis.x.value)
        if isinstance(x, lsl.Node) or x.strong:
            xgrid = {x.name: jnp.linspace(x.value.min(), x.value.max(), ngrid)}
        else:
            assert isinstance(term, Term | LinTerm), (
                f"Wrong type for term: {type(term)}"
            )
            ncols = jnp.shape(term.basis.value)[-1]
            xgrid = input_grid_nd_smooth(term, ngrid=int(np.pow(ngrid, 1 / ncols)))

        grid = xgrid | {cluster.basis.x.name: cgrid}

        # code : bool
        observed = {x: True for x in cgrid}
    else:
        pass

    if newdata is not None and newdata_meshgrid:
        cgrid = np.asarray(newdata[cluster.basis.x.name])
        cgrid = _convert_to_integers(cgrid, labels, cluster)

        grid = {x.name: newdata[x.name], cluster.basis.x.name: cgrid}
        full_grid_arrays = [v.flatten() for v in np.meshgrid(*grid.values())]
        newdata_x: dict[str, ArrayLike | Sequence[int] | Sequence[str]] = dict(
            zip(grid.keys(), full_grid_arrays)
        )

        if isinstance(labels, CategoryMapping):
            observed = {x: x in cluster.basis.x.value for x in cgrid}
        else:
            observed = {x: True for x in cgrid}
    elif newdata is not None:
        cgrid = np.asarray(newdata[cluster.basis.x.name])
        cgrid = _convert_to_integers(cgrid, labels, cluster)
        newdata_x = {x.name: newdata[x.name], cluster.basis.x.name: cgrid}
        # code : bool
        if isinstance(labels, CategoryMapping):
            observed = {x: x in cluster.basis.x.value for x in cgrid}
        else:
            observed = {x: True for x in cgrid}
    else:  # then we use the grid created from observed data
        full_grid_arrays = [v.flatten() for v in np.meshgrid(*grid.values())]
        newdata_x = dict(zip(grid.keys(), full_grid_arrays))

    newdata_x = {k: jnp.asarray(v) for k, v in newdata_x.items()}

    term_samples = clustered_term.predict(samples, newdata=newdata_x)
    term_summary = (
        gs.SamplesSummary.from_array(
            term_samples,
            name=clustered_term.name,
            quantiles=ci_quantiles,
            hdi_prob=hdi_prob,
        )
        .to_dataframe()
        .reset_index()
    )

    for k, v in newdata_x.items():
        term_summary[k] = np.asarray(v)

    if labels is not None:
        if isinstance(labels, CategoryMapping):
            labels_long = labels.to_labels(newdata_x[cluster.basis.x.name])
            categories = list(labels.labels_to_integers_map)
            term_summary[cluster.basis.x.name] = pd.Categorical(
                labels_long, categories=categories
            )
        else:
            term_summary[cluster.basis.x.name] = labels

    term_summary["observed"] = [
        observed[x] for x in np.asarray(newdata_x[cluster.basis.x.name])
    ]

    term_summary.reset_index(inplace=True)

    return term_summary


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

    assert isinstance(term, Term | lsl.Var)
    assert isinstance(cluster, RITerm | MRFTerm)

    if labels is None:
        try:
            labels = cluster.mapping  # type: ignore
        except AttributeError:
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

    if isinstance(term, Term):
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
