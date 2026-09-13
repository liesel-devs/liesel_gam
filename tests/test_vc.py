"""Public varying-coefficient API, prediction, and visualization regressions."""

import warnings

import jax.numpy as jnp
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import pandas as pd
import pytest
from jax.random import key, normal
from jax.typing import ArrayLike
from matplotlib import pyplot as plt

import liesel_gam as gam


def setup_terms(multivariate=False, shared_input=False):
    data = {"z": jnp.linspace(-1.0, 1.0, 8), "x": jnp.linspace(0.0, 2.0, 8)}
    tb = gam.TermBuilder.from_dict(data)
    if multivariate:
        tb = gam.MVTermBuilder.from_term_builder(tb, jnp.eye(2))
        smooth = tb.ps("z", k=5, scale=1.0, dimension_scale=1.0)
    else:
        smooth = tb.ps("z", k=5, scale=1.0)
    effect = tb.vc(smooth, by="z" if shared_input else "x")
    return tb, smooth, effect


def vc_inputs(effect, multivariate):
    if multivariate:
        return effect.by, effect.latent.value_node["x"]
    return effect.value_node["by"], effect.value_node["x"]


@pytest.mark.parametrize("multivariate", [False, True])
@pytest.mark.parametrize(
    "style",
    ["positional", "keyword", "legacy_positional", "legacy_mixed", "legacy_keyword"],
)
def test_call_conventions(multivariate, style):
    tb, smooth, effect = setup_terms(multivariate)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if style == "positional":
            other = tb.vc(smooth, "x")
        elif style == "keyword":
            other = tb.vc(term=smooth, by="x")
        elif style == "legacy_positional":
            other = tb.vc("x", smooth)
        elif style == "legacy_mixed":
            other = tb.vc("x", by=smooth)
        else:
            other = tb.vc(x="x", by=smooth)
    if style.startswith("legacy"):
        assert len(caught) == 1
        assert caught[0].category is FutureWarning
        assert "use vc(term, by=x)" in str(caught[0].message)
        assert caught[0].filename == __file__
    else:
        assert not caught
    coefficient, multiplier = vc_inputs(other, multivariate)
    assert coefficient is smooth
    assert multiplier is vc_inputs(effect, multivariate)[1]
    assert coefficient.coef.inference is smooth.coef.inference
    assert other.info == {}
    _model = lsl.Model([effect, other])
    samples: dict[str, ArrayLike] = {
        smooth.coef.name: normal(key(1), (2, 12, smooth.coef.value.size))
    }
    newdata = {"z": jnp.linspace(-0.5, 0.5, 5), "x": jnp.ones(5) * 2}
    np.testing.assert_allclose(
        effect.predict(samples, newdata=newdata),
        other.predict(samples, newdata=newdata),
    )


@pytest.mark.parametrize("multivariate", [False, True])
def test_errors_and_names(multivariate):
    tb, smooth, _ = setup_terms(multivariate)
    for call in (
        lambda: tb.vc(),
        lambda: tb.vc(smooth),
        lambda: tb.vc(smooth, by=smooth),
        lambda: tb.vc(term=smooth, by="x", x="x"),
        lambda: tb.vc(x="x", by="z"),
        lambda: tb.vc("x", "z"),
    ):
        with pytest.raises(TypeError):
            call()
    with pytest.raises(TypeError, match="numeric"):
        tb.vc(smooth, by=gam.CatVar(["a"] * 8, name="group"))
    x = lsl.Var.new_obs(jnp.ones(8), name="weight")
    effect = tb.vc(smooth, by=x, prefix="p.", name="weighted")
    assert effect.name == "p.weighted"
    if multivariate:
        assert effect.by is smooth
        assert effect.latent.value_node["x"] is x
    else:
        assert effect.value_node["x"] is x
        assert effect.value_node["by"] is smooth


@pytest.mark.parametrize("multivariate", [False, True])
@pytest.mark.parametrize("sample_ndim", [0, 1, 2])
@pytest.mark.parametrize("multiplier", [None, 0.0, 1.0, 2.0, -1.0])
def test_summaries_match_draws(multivariate, sample_ndim, multiplier):
    _, smooth, effect = setup_terms(multivariate)
    _model = lsl.Model(effect)
    draws = normal(key(2), (2, 12, smooth.coef.value.size))
    values = [draws[0, 0], draws[0], draws][sample_ndim]
    samples: dict[str, ArrayLike] = {smooth.coef.name: values}
    grid = jnp.linspace(-1.0, 1.0, 7)
    newdata = None if multiplier is None else {"z": grid, "x": jnp.full(7, multiplier)}
    summary = gam.summarise_1d_smooth(
        effect, samples, newdata=newdata, ngrid=7, quantiles=(0.1, 0.9)
    )
    if multiplier is None:
        predictions = lsl.Model(smooth, copy=True).predict(
            gs.Position(samples),
            predict=[smooth.name],
            newdata=gs.Position({"z": grid}),
        )[smooth.name]
    else:
        predictions = effect.predict(samples, newdata=newdata)
    predictions = np.asarray(predictions).reshape(-1, 7 * (2 if multivariate else 1))
    np.testing.assert_allclose(summary["mean"], predictions.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(
        summary["q_0.1"], np.quantile(predictions, 0.1, axis=0), atol=1e-6
    )
    np.testing.assert_allclose(
        summary["q_0.9"], np.quantile(predictions, 0.9, axis=0), atol=1e-6
    )
    if multivariate:
        np.testing.assert_array_equal(summary["dimension"], np.tile([0, 1], 7))
        np.testing.assert_allclose(summary["z"], np.repeat(grid, 2), atol=1e-6)
    else:
        assert "dimension" not in summary


@pytest.mark.parametrize("multivariate", [False, True])
def test_shared_covariate_and_missing_inputs(multivariate):
    _, smooth, effect = setup_terms(multivariate, shared_input=True)
    _model = lsl.Model(effect)
    samples: dict[str, ArrayLike] = {
        smooth.coef.name: normal(key(3), (2, 12, smooth.coef.value.size))
    }
    grid = jnp.linspace(-1.0, 1.0, 7)
    default = gam.summarise_1d_smooth(effect, samples, ngrid=7)
    expected = np.asarray(
        lsl.Model(smooth, copy=True).predict(
            gs.Position(samples),
            predict=[smooth.name],
            newdata=gs.Position({"z": grid}),
        )[smooth.name]
    ).mean(axis=(0, 1))
    np.testing.assert_allclose(default["mean"], expected.ravel(), atol=1e-6)
    _, smooth, effect = setup_terms(multivariate)
    _model = lsl.Model(effect)
    samples: dict[str, ArrayLike] = {
        smooth.coef.name: normal(key(3), (2, 12, smooth.coef.value.size))
    }
    with pytest.raises(ValueError, match="missing inputs: x"):
        gam.summarise_1d_smooth(effect, samples, newdata={"z": grid})
    with pytest.raises(ValueError, match="missing inputs: z"):
        gam.plot_1d_smooth(effect, samples, newdata={"x": grid})


@pytest.mark.parametrize("multivariate", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_plot_renders(tmp_path, multivariate, explicit):
    _, smooth, effect = setup_terms(multivariate)
    _model = lsl.Model(effect)
    samples: dict[str, ArrayLike] = {
        smooth.coef.name: normal(key(4), (2, 12, smooth.coef.value.size))
    }
    grid = jnp.linspace(-1.0, 1.0, 7)
    newdata = {"z": grid, "x": -jnp.ones(7)} if explicit else None
    plot = gam.plot_1d_smooth(
        effect,
        samples,
        newdata=newdata,
        ngrid=7,
        ci_quantiles=(0.1, 0.9),
        hdi_prob=0.8,
        show_n_samples=3,
    )
    assert plot.labels.y == ("Contribution" if explicit else "Coefficient curve")
    fig = plot.draw()
    assert len(fig.axes) == (2 if multivariate else 1)
    fig.savefig(tmp_path / "vc.png")
    plt.close(fig)
    summary = gam.summarise_1d_smooth(
        effect, samples, newdata=newdata, ngrid=7, quantiles=(0.1, 0.9), hdi_prob=0.8
    )
    assert isinstance(plot.data, pd.DataFrame)
    np.testing.assert_allclose(plot.data["mean"], summary["mean"])
    np.testing.assert_allclose(plot.data["hdi_low"], summary["hdi_low"])
    # Rendered posterior lines must be actual draws of the displayed quantity.
    curves = plot.layers[-2].data
    assert len(curves) == 3 * 7 * (2 if multivariate else 1)
    if multivariate:
        assert set(curves["PANEL"]) == {1, 2}

    target = effect if explicit else smooth
    prediction_data = newdata if explicit else {"z": grid}
    assert prediction_data is not None
    draws = np.asarray(
        lsl.Model(target, copy=True).predict(
            gs.Position(samples),
            predict=[target.name],
            newdata=gs.Position(prediction_data),
        )[target.name]
    )
    draws = draws.reshape(24, 7, 2 if multivariate else 1)
    for group, curve in curves.groupby(["PANEL", "group"], observed=True):
        assert isinstance(group, tuple)
        panel = group[0]
        actual = curve.sort_values("x")["y"].to_numpy()
        expected = draws[:, :, int(panel) - 1]
        assert np.any(np.all(np.isclose(expected, actual, atol=1e-6), axis=1))


@pytest.mark.parametrize("multivariate", [False, True])
def test_copy_and_reload(tmp_path, multivariate):
    _, smooth, effect = setup_terms(multivariate)
    model = lsl.Model(effect)
    samples: dict[str, ArrayLike] = {
        smooth.coef.name: normal(key(5), (2, 12, smooth.coef.value.size))
    }
    expected = gam.summarise_1d_smooth(effect, samples, ngrid=7)
    copied = model.copy_vars()[effect.name]
    assert vc_inputs(copied, multivariate)[0] is not smooth
    filename = tmp_path / "vc.pkl"
    lsl.save_model(model, str(filename))
    restored = lsl.load_model(str(filename)).vars[effect.name]
    for candidate in (copied, restored):
        summary = gam.summarise_1d_smooth(candidate, samples, ngrid=7)
        np.testing.assert_allclose(summary["mean"], expected["mean"])
        assert vc_inputs(candidate, multivariate)[1].name == "x"
        assert candidate.info == {}


def test_scalar_positional_options_and_legacy_variable():
    tb, smooth, _ = setup_terms()
    weight = lsl.Var.new_obs(jnp.ones(8), name="weight")
    new = tb.vc(smooth, weight, "p.", "effect")
    with pytest.warns(FutureWarning, match="use vc"):
        old = tb.vc(weight, smooth, "p.", "effect")
    assert new.name == old.name == "p.effect"
    assert new.value_node["by"] is old.value_node["by"]
    assert new.value_node["x"] is old.value_node["x"]


def test_calculated_multiplier_inputs():
    tb, smooth, _ = setup_terms()
    source = lsl.Var.new_obs(jnp.linspace(0.0, 1.0, 8), name="source")
    multiplier = lsl.Var.new_calc(jnp.square, source, name="weight")
    effect = tb.vc(smooth, by=multiplier)
    model = lsl.Model(effect)
    samples = {smooth.coef.name: jnp.ones(smooth.coef.value.size)}
    grid = jnp.linspace(-1.0, 1.0, 7)
    with pytest.raises(ValueError, match="missing inputs: source"):
        gam.summarise_1d_smooth(effect, samples, newdata={"z": grid})
    summary = gam.summarise_1d_smooth(
        effect, samples, newdata={"z": grid, "source": jnp.ones(7) * 2}
    )
    expected = (
        model.vars[smooth.name].predict(
            gs.Position(samples), newdata=gs.Position({"z": grid})
        )
        * 4
    )
    np.testing.assert_allclose(summary["mean"], np.asarray(expected).ravel())


def test_wrapped_multivariate_reconstructed_dimensions(tmp_path):
    scalar, smooth, _ = setup_terms()
    predictor = gam.MVAdditivePredictor.from_identity("eta", 3)
    predictor.constrain("sumzero_coef")
    builder = gam.MVTermBuilder.from_predictor(predictor, scalar)
    effect = builder.vc(smooth, by="x", dimension_scale=1.0)
    coefficient = getattr(effect, "by")
    assert coefficient.latent.value.shape[-1] == 2
    assert effect.value.shape[-1] == 3
    model = lsl.Model(effect)
    samples: dict[str, ArrayLike] = {
        coefficient.coef.name: normal(key(6), (2, 12, coefficient.coef.value.size))
    }
    summary = gam.summarise_1d_smooth(model.vars[effect.name], samples, ngrid=7)
    assert set(summary["dimension"]) == {0, 1, 2}
    np.testing.assert_allclose(
        summary["mean"].to_numpy().reshape(7, 3).sum(axis=1), 0, atol=1e-6
    )
    plot = gam.plot_1d_smooth(effect, samples, ngrid=7, show_n_samples=2)
    fig = plot.draw()
    assert len(fig.axes) == 3
    fig.savefig(tmp_path / "reconstructed.png")
    plt.close(fig)
