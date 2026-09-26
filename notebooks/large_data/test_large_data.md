---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Large-data model setup

Prepare a model on a representative basis setup sample, then encode the full data
as a raw observed position. Liesel's optimizer can split and batch that position
without materializing all basis matrices. This notebook uses the `optim-base`
optimizer API assumed throughout these guides.

## Prepare the model and full-data position

```{code-cell} ipython3
import liesel.model as lsl
import liesel.optim as opt
import optax
import pandas as pd
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_gam as gam
```

```{code-cell} ipython3
gam.__version__
```

We use 1,000 observations, one P-spline covariate, and one random-intercept grouping variable. In an application, the full data can be much larger.

```{code-cell} ipython3
df = gam.demo_data(n=1_000, seed=1)
continuous = ["x_nonlin"]
categorical = ["x_cat"]
```

```{code-cell} ipython3
pd.Series(
    {
        "rows": len(df),
        "columns": len(df.columns),
        "group_levels": df["x_cat"].nunique(),
    },
).to_frame("value")
```

`category_coverage_indices()` returns positional rows that a splitter should retain in training. Here all rows are training rows for this first setup example.

```{code-cell} ipython3
required_train_indices = gam.category_coverage_indices(df, columns=categorical)
train_indices = list(range(len(df)))
```

```{code-cell} ipython3
pd.Series(
    {
        "category_coverage_rows": required_train_indices.tolist(),
        "training_rows": len(train_indices),
    }
).to_frame("value")
```

`basis_setup_sample()` keeps the eligible continuous boundaries and categorical metadata, then fills the sample randomly without replacement.

```{code-cell} ipython3
setup_df = gam.basis_setup_sample(
    df,
    indices=train_indices,
    continuous=continuous,
    categorical=categorical,
    n=200,
    seed=2,
)

setup_range = setup_df["x_nonlin"].agg(["min", "max"]).round(3).tolist()
```

```{code-cell} ipython3
pd.Series(
    {
        "setup_rows": len(setup_df),
        "x_range": setup_range,
        "categories": setup_df["x_cat"].cat.categories.tolist(),
    }
).to_frame("value")
```

The representative data fixes the P-spline setup and the random-intercept category mapping.

```{code-cell} ipython3
def make_model(data):
    tb = gam.TermBuilder.from_df(data, approximation=True)
    loc = gam.AdditivePredictor(name="loc")
    smooth = tb.ps("x_nonlin", k=20)
    groups = tb.ri("x_cat")
    loc += smooth, groups
    coef_keys = [loc.intercept.name, smooth.coef.name, groups.coef.name]

    y = lsl.Var.new_obs(
        data["y"].to_numpy(),
        dist=lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
        name="y",
    )
    return tb, lsl.Model(y), coef_keys


tb, model, coef_keys = make_model(setup_df)
```

```{code-cell} ipython3
sorted(model.observed)
```

`PandasRegistry.observed_position()` now encodes exactly the observed model variables for all rows. It uses the category mapping established from `setup_df` and ignores unrelated DataFrame columns.

```{code-cell} ipython3
full_position = tb.registry.observed_position(model, df)
position_shapes = {
    name: tuple(value.shape) for name, value in sorted(full_position.items())
}
```

```{code-cell} ipython3
pd.DataFrame.from_dict(position_shapes, orient="index", columns=["rows"])
```

## Correctness constraints

Pass explicit column lists for a wide DataFrame. With `continuous=None` or `categorical=None`, columns are inferred by dtype; an empty list disables that column kind. Numeric category codes must first be cast to a pandas categorical dtype. Category levels outside the setup mapping are errors rather than silently receiving new codes.

Derived model quantities must be row-wise: one row's value may depend on that row and fixed setup metadata, but not on other rows in its current batch. Precompute lags, ranks, or other context-dependent features before batching. Setup-dependent knots, constraints, and penalty scaling are based on `setup_df`, not all training rows.

Keep raw covariates in the observed position; let the model evaluate its derived bases for each batch.

## Initialize coefficients with minibatches

Liesel handles splitting, validation, loss scaling, and optimization. Use only
training rows to choose the basis setup sample. Here Adam updates the intercept
and coefficient vectors while keeping smoothing variances at their current values;
the response standard deviation remains fixed at 1. This short run illustrates
batching, not convergence or a posterior fit.

`train_full_data` evaluates all training rows after every epoch; the held-out
validation rows are reserved for later evaluation and do not determine stopping
here. For data too large for that full pass, see the EMA monitor in the
[Liesel monitoring guide](https://github.com/liesel-devs/liesel/blob/optim-base/docs/source/optimizer-monitoring.rst).

```{code-cell} ipython3
n_validate = max(1, round(0.1 * len(df)))
splitter = opt.Split(
    axis_size=len(df),
    validate_axis_size=n_validate,
    keep_in_train=required_train_indices.tolist(),
    shuffle=True,
    seed=42,
)

# Choose the basis using training rows only.
optim_setup_df = gam.basis_setup_sample(
    df,
    indices=splitter.indices_train.tolist(),
    continuous=continuous,
    categorical=categorical,
    n=200,
    seed=43,
)
optim_tb, optim_model, optim_coef_keys = make_model(optim_setup_df)
optim_position = optim_tb.registry.observed_position(optim_model, df)
split = splitter.split_position(optim_position)
```

With the training model and split ready, fit the selected coefficients:

```{code-cell} ipython3
result = opt.LieselOptim(
    optim_model,
    optimizers=[opt.Optimizer(optim_coef_keys, optax.adam(0.01))],
    loss_monitor="train_full_data",
    split=split,
    batch_size=128,
    stopper=opt.Stopper(epochs=10, patience=5),
    save_position_history=False,
    seed=44,
    show_progress=False,
).fit()
```

```{code-cell} ipython3
pd.Series(
    {
        "training_rows": split.train_axis_size,
        "validation_rows": split.validate_axis_size,
        "status": result.status,
    }
).to_frame("value")
```

The selected position contains coefficients only. After completing held-out
evaluation and all model choices, you can initialize a final MCMC fit on all
rows. Update both the observations and fitted coefficients, so that final fit
uses all rows rather than the small setup sample. The smoothing parameters retain
their samplers. See {doc}`../../guides/initialization` for initialization choices.

```{code-cell} ipython3
optim_model.state = optim_model.update_state(
    dict(optim_position) | dict(result.position_min_monitor)
)
```
