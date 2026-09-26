---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Choose terms

Choose a term for the effect you want to estimate. All terms are Liesel variables,
so they can contribute to any distribution parameter, not just a mean.

```{list-table}
:header-rows: 1
:widths: 30 25 45

* - Your task
  - Constructor
  - Model choice
* - Linear or categorical effect
  - {meth}`~liesel_gam.TermBuilder.lin`
  - Constant coefficient prior by default; set `prior` to change it.
* - Shrink linear coefficients
  - {meth}`~liesel_gam.TermBuilder.slin`
  - Independent normal coefficient priors with a shared scale.
* - Smooth one covariate
  - {meth}`~liesel_gam.TermBuilder.ps`
  - P-spline; choose basis size `k` and estimate smoothing strength.
* - Separate linear and nonlinear effects
  - {meth}`~liesel_gam.TermBuilder.np`
  - Remove the constant and linear trend from the smooth; add a linear term.
* - Smooth a periodic covariate
  - {meth}`~liesel_gam.TermBuilder.cp`
  - Cyclic P-spline; choose the period boundaries deliberately.
* - Independent group intercepts
  - {meth}`~liesel_gam.TermBuilder.ri`
  - Exchangeable group effects, with a shared variance.
* - Neighbouring regions
  - {meth}`~liesel_gam.TermBuilder.mrf`
  - A Markov random field couples effects through supplied adjacency.
* - Thin-plate smooths
  - {meth}`~liesel_gam.TermBuilder.tp`
  - One or several continuous inputs; a thin-plate roughness penalty.
* - Smooth spatial coordinates
  - {meth}`~liesel_gam.TermBuilder.kriging`
  - Low-rank Gaussian-process smooth; coordinates and their units matter.
* - Group-specific slopes
  - {meth}`~liesel_gam.TermBuilder.rs`
  - Multiply a covariate or smooth by random group effects.
* - Varying coefficients
  - {meth}`~liesel_gam.TermBuilder.vc`
  - Multiply one term by another covariate or term.
* - Interactions
  - {meth}`~liesel_gam.TermBuilder.tx`, {meth}`~liesel_gam.TermBuilder.tf`
  - Combine marginal terms; see {doc}`interactions`.
```

## Use formulas

The linear constructors use Formulaic syntax. This complete setup combines a
linear effect, a categorical contrast, and a nonlinear effect from simulated data:

```{code-cell} ipython3
import liesel_gam as gam

df = gam.demo_data(n=200, seed=1)
tb = gam.TermBuilder.from_df(df)

mu = gam.AdditivePredictor("mu")
mu += tb.lin("x_lin + C(x_cat)")
mu += tb.ps("x_nonlin", k=15)
```

```{code-cell} ipython3
---
mystnb:
  image:
    alt: A predictor combines its intercept, linear and categorical coefficients,
      and a centered nonlinear smooth.
---
mu.plot()
```

`AdditivePredictor` already includes an intercept. `lin` and `slin` omit
their intercept column by default. With treatment coding, category coefficients
are contrasts to the reference category; they are not separate category means.
Set the category order or contrast explicitly when that reference matters.
Formula `a * b` expands to `a + b + a:b`; it is a linear interaction, not a
smooth tensor product.

## Choose the basis size

`k` is the number of unconstrained basis functions for `ps`. The centering
constraint reduces the coefficient dimension. Increasing `k` gives a richer
function space; the smoothing prior controls roughness within that space.
Check that your conclusions do not depend on a basis that is too small.
Changing the basis or penalty scaling can also change the interpretation of a
smoothing prior; see {doc}`priors`.

For a linear trend plus a separate nonlinear deviation, use `lin` with `np`.
A regular `ps` term already contains a linear component, so adding another
unconstrained linear effect can make their decomposition unidentified.

See fitted examples for {doc}`random intercepts <../notebooks/univariate/test_ri>`,
{doc}`regional effects <../notebooks/univariate/test_mrf>`, and
{doc}`varying coefficients <../notebooks/composite/test_vc>`.

For other smooth families (`ts`, `cr`, `cs`, `cc`, `bs`), see
{class}`~liesel_gam.TermBuilder`. The example library also includes
{doc}`a thin-plate surface <../notebooks/multivariate/test_tp_2d>` and
{doc}`kriging <../notebooks/multivariate/test_kriging>`.
