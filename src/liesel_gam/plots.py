from collections.abc import Sequence
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

from .var import Term

KeyArray = Any


def summarise_by_samples(
    key: KeyArray, a: Array, name: str, n: int = 100
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
    grid: ArrayLike | None = None,
    ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
    hdi_prob: float | None = None,
    show_n_samples: int | None = 50,
    seed: int | KeyArray = 1,
):
    if grid is None:
        xgrid = jnp.linspace(term.basis.x.value.min(), term.basis.x.value.max(), 150)
    else:
        xgrid = jnp.asarray(grid)

    term_samples = term.predict(samples, newdata={term.basis.x.name: xgrid})
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


def grid_2d(
    inputs: dict[str, jax.typing.ArrayLike], ngrid: int
) -> dict[str, jax.typing.ArrayLike]:
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
    inputs = {n.var.name: n.var.value for n in term.basis.x.all_input_nodes()}
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
    grid: dict[str, ArrayLike] | None = None,
    ngrid: int = 20,
    plot_vars: PlotVars | Sequence[PlotVars] = "mean",
    ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
    hdi_prob: float | None = None,
):
    if isinstance(plot_vars, str):
        plot_vars = [plot_vars]

    if grid is None:
        xgrid = input_grid_2d_smooth(term, ngrid=ngrid)
    else:
        full_grid_arrays = [v.flatten() for v in np.meshgrid(*grid.values())]
        xgrid = dict(zip(grid.keys(), full_grid_arrays))

    term_samples = term.predict(samples, newdata=xgrid)
    ci_quantiles_ = (0.05, 0.95) if ci_quantiles is None else ci_quantiles
    hdi_prob_ = 0.9 if hdi_prob is None else hdi_prob
    term_summary = (
        gs.SamplesSummary.from_array(
            term_samples, name=term.name, quantiles=ci_quantiles_, hdi_prob=hdi_prob_
        )
        .to_dataframe()
        .reset_index()
    )

    for k, v in xgrid.items():
        term_summary[k] = np.asarray(v)

    term_summary.reset_index(inplace=True)
    term_summary = term_summary.melt(
        id_vars=["index"] + list(xgrid.keys()),
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
        + p9.aes(*list(xgrid.keys()), fill="value")
        + p9.facet_wrap("~variable", labeller="label_both")
    )

    p = p + p9.geom_tile()

    return p


def plot_regions(): ...


def plot_forest(): ...
