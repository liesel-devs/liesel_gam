---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Extend SD prior calibration

Choose a variance-prior family or define a new effect statistic while reusing
{meth}`SDPrior.calibrate <liesel_gam.SDPrior.calibrate>`. The shared workflow
handles fitted geometry, Gaussian draws, numerical search, validation, and
{class}`CalibrationResult <liesel_gam.CalibrationResult>` construction.
Start with {doc}`priors` for ordinary smooth construction and the meaning of
the stochastic-effect bound.

These examples use a small synthetic input and fixed seeds.

```{code-cell} ipython3
from dataclasses import dataclass

import numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_gam as gam

tb = gam.TermBuilder.from_dict({"x": np.linspace(-1.0, 1.0, 31)})
```

## Choose a variance family

Supply a callable from a positive scalar `s` to a scalar TFD distribution on
$\tau^2$. This example uses an inverse-gamma prior. Its scale argument receives
`s`; `s` is the factory argument being calibrated, not a draw of $\tau^2$.

```{code-cell} ipython3
prior = gam.SDPrior.from_max_abs_effect(
    3.0,
    probability=0.95,
    draws=1024,
    variance_prior=lambda s: tfd.InverseGamma(concentration=3.0, scale=s),
)
smooth = tb.ps("x", k=6, scale=prior)
result = prior.calibrate(smooth.basis)
```

```{code-cell} ipython3
{
    "method": result.diagnostics["method"],
    "validation_probability": round(
        float(result.diagnostics["validation_probability"]), 3
    ),
    "suggested_tau_start": round(result.initial_scales[0], 3),
}
```

The validation probability should be close to the 95% target, with Monte Carlo
uncertainty reported in `validation_mc_se`. The suggested start is the median
of the scale prior. The method uses variance CDFs because both this distribution
and {class}`MaxAbsEffect <liesel_gam.MaxAbsEffect>` support that shortcut.

## Use a sample-only distribution

A prior needs sampling and log-probability methods for model construction;
CDF and quantile methods are optional. TFD's absolute-value transformation of
a normal distribution provides an example: sampling works, while its generic
transformed-distribution CDF and quantile methods are unimplemented.

```{code-cell} ipython3
def folded_normal_variance(s):
    return tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=0.0, scale=s),
        bijector=tfb.AbsoluteValue(),
    )


sample_only = gam.SDPrior.from_max_abs_effect(
    3.0,
    probability=0.95,
    draws=1024,
    variance_prior=folded_normal_variance,
)
sampled_smooth = tb.ps("x", k=6, scale=sample_only)
sampled_result = sample_only.calibrate(sampled_smooth.basis)
```

```{code-cell} ipython3
{
    "method": sampled_result.diagnostics["method"],
    "validation_probability": round(
        float(sampled_result.diagnostics["validation_probability"]), 3
    ),
    "suggested_tau_start": round(sampled_result.initial_scales[0], 3),
}
```

The method is now `sampled_variance_quantile`: the workflow samples variances
and Gaussian effects, and adjusts `s` until the requested statistic quantile
reaches the bound. Fitting uses fixed draws during search; validation uses an
independent stream. See {meth}`SDPrior.calibrate <liesel_gam.SDPrior.calibrate>`
for the deterministic median fallback used for initialization.

## Define an effect statistic

An ordinary object satisfying {class}`CalibrationTarget
<liesel_gam.CalibrationTarget>` supplies settings and a
{meth}`statistic(effects) <liesel_gam.CalibrationTarget.statistic>` method.
Subclassing or registration is unnecessary. Here a weighted root mean
square replaces the maximum absolute effect. The weights are external context
held by the target.

```{code-cell} ipython3
@dataclass(frozen=True)
class WeightedRMS:
    threshold: float
    probability: float
    weights: np.ndarray
    grid_size: int | None = 65
    seed: int = 42
    draws: int = 2048

    def statistic(self, effects: np.ndarray) -> np.ndarray:
        return np.sqrt(np.average(effects**2, axis=-1, weights=self.weights))


weights = np.linspace(1.0, 2.0, 65)
target = WeightedRMS(threshold=2.0, probability=0.9, weights=weights)
basis = tb.bases.ps("x", k=6)
custom_result = gam.SDPrior(calibration=target).calibrate(basis)
```

```{code-cell} ipython3
{
    "method": custom_result.diagnostics["method"],
    "validation_probability": round(
        float(custom_result.diagnostics["validation_probability"]), 3
    ),
}
```

This targets a 90% probability that weighted RMS is at most 2; it does not
bound the maximum absolute effect.

The `effects` array has shape `(batch, evaluation_points)`. Return one finite
real statistic per row. Process each row independently: a batch may be smaller
than {attr}`draws <liesel_gam.CalibrationTarget.draws>`. Use the target's context
to interpret evaluation locations. Here
the 65 weights match the 65-point scalar evaluation grid.

### Add an optional CDF shortcut

For a homogeneous statistic such as RMS, you can supply `variance_limits`.
Hold a unit-scale Gaussian draw fixed and ask which variances would still
satisfy the bound. If its statistic is 2 and the bound is 3, scaling by $\tau$
gives $2\tau \le 3$, or $\tau^2 \le 2.25$. The variance-prior CDF at 2.25 is
the conditional probability of success for that fixed draw.

```{code-cell} ipython3
class FastWeightedRMS(WeightedRMS):
    def variance_limits(self, values: np.ndarray) -> np.ndarray:
        return (
            np.divide(
                self.threshold,
                values,
                out=np.full_like(values, np.inf, dtype=float),
                where=values > 0,
            )
            ** 2
        )


fast_target = FastWeightedRMS(threshold=2.0, probability=0.9, weights=weights)
fast_result = gam.SDPrior(calibration=fast_target).calibrate(basis)
```

```{code-cell} ipython3
fast_result.diagnostics["method"]
```

The shared workflow averages these CDF probabilities and solves for `s`.
Zero statistics permit infinite variance. This capability applies to a single
variance and must correctly describe your target's event. A target without
such an inverse relationship can use the sampling workflow above.

## Calibrate evaluated multiple-penalty geometry

Use {class}`GaussianGeometry <liesel_gam.GaussianGeometry>` when you already
have an evaluated design and coefficient penalties. This small example has
two stochastic coefficients and one unpenalized direction.

```{code-cell} ipython3
geometry = gam.GaussianGeometry(
    design=np.eye(3),
    penalties=(np.diag([1.0, 0.0, 0.0]), np.diag([0.0, 2.0, 0.0])),
)
joint_prior = gam.SDPrior.from_max_abs_effect(
    2.0,
    probability=0.9,
    draws=2048,
    variance_prior=(
        lambda s: tfd.Weibull(concentration=0.5, scale=0.4 * s),
        lambda s: tfd.InverseGamma(concentration=3.0, scale=2.0 * s),
    ),
)
joint_result = joint_prior.calibrate(geometry)
```

```{code-cell} ipython3
{
    "rank": joint_result.diagnostics["rank"],
    "nullity": joint_result.diagnostics["nullity"],
    "suggested_tau_starts": np.round(joint_result.initial_scales, 3).tolist(),
    "validation_probability": round(
        float(joint_result.diagnostics["validation_probability"]), 3
    ),
}
```

Each tuple entry corresponds to one penalty and receives the same `s`. The
factors 0.4 and 2.0 preserve relative family preferences; the random variances
are independent. See {attr}`variance_prior <liesel_gam.SDPrior.variance_prior>`
for callable and tuple handling.

Conditional coefficient precision is $K_1/\tau_1^2 + K_2/\tau_2^2$. The common
null space is excluded from calibration. The design rows fix evaluation locations.

## Check the numerical result

Read the method, validation probability and Monte Carlo error, stochastic rank,
nullity, and actual evaluation dimensions in
{attr}`diagnostics <liesel_gam.CalibrationResult.diagnostics>`.
Malformed statistics and unattainable targets
raise errors. Finite grids and Monte Carlo calculations approximate the target.

The shared search needs a meaningful relationship between factory argument `s`
and the statistic. Arbitrary factories or targets need their own monotonicity
argument; the workflow does not prove one. See
{meth}`SDPrior.calibrate <liesel_gam.SDPrior.calibrate>` for bracket, quantile,
and tolerance details.
