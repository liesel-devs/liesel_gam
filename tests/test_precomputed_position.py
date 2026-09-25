"""Registry positions can replace callback inputs with cached basis rows."""

import jax
import jax.experimental
import jax.numpy as jnp
import liesel.model as lsl
import numpy as np
import pandas as pd
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_gam as gam


@pytest.mark.parametrize("registry_type", [gam.DictRegistry, gam.PandasRegistry])
def test_precomputed_position_evaluates_callback_rows_and_preserves_model(
    registry_type,
):
    setup = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    setup_data = setup if registry_type is gam.PandasRegistry else setup.to_dict("list")
    registry = registry_type(setup_data, prefix_names_by="data.")
    calls = []

    def polynomial(values):
        calls.append(np.asarray(values).copy())
        return np.column_stack((values, values**2))

    x = registry.get_obs("x")
    y = registry.get_obs("y")
    basis = gam.Basis(x, basis_fn=polynomial, name="basis")
    model = lsl.Model([basis, y])
    data = pd.DataFrame({"x": [3.0, 4.0, 5.0], "y": [6.0, 7.0, 8.0]})
    jax.effects_barrier()
    calls.clear()

    position = registry.precomputed_position(model, data)
    jax.effects_barrier()

    assert set(position) == {"basis", "data.y"}
    np.testing.assert_array_equal(position["basis"], [[3, 9], [4, 16], [5, 25]])
    np.testing.assert_array_equal(position["data.y"], [6, 7, 8])
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], [3, 4, 5])
    np.testing.assert_array_equal(x.value, [1, 2])
    np.testing.assert_array_equal(y.value, [3, 4])
    np.testing.assert_array_equal(basis.value, [[1, 1], [2, 4]])
    assert set(registry.observed_position(model, data)) == {"data.x", "data.y"}


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_precomputed_position_keeps_native_inputs_and_shared_covariates(shared, cached):
    setup = pd.DataFrame({"x1": [1.0, 2.0], "x2": [3.0, 4.0], "y": [5.0, 6.0]})
    registry = gam.PandasRegistry(setup)
    x1 = registry.get_obs("x1")
    x2 = x1 if shared else registry.get_obs("x2")
    callback = gam.Basis(x1, basis_fn=np.square, cache_basis=cached, name="callback")
    native_evaluations = []

    def native_basis(values):
        native_evaluations.append(values.shape)
        return jnp.square(values)

    native = gam.Basis(x2, basis_fn=native_basis, use_callback=False, name="native")
    model = lsl.Model([callback, native, registry.get_obs("y")])
    data = pd.DataFrame(
        {"x1": [2.0, 3.0, 4.0], "x2": [5.0, 6.0, 7.0], "y": [8.0, 9.0, 10.0]}
    )
    native_evaluations.clear()
    position = registry.precomputed_position(model, data)
    jax.effects_barrier()

    keys = {"y", "x1" if shared or not cached else "callback"}
    if not shared:
        keys.add("x2")
    assert set(position) == keys
    assert native_evaluations == []
    if "callback" in position:
        np.testing.assert_array_equal(position["callback"], [4, 9, 16])
    else:
        np.testing.assert_array_equal(position["x1"], [2, 3, 4])


def test_nested_callback_bases_return_only_the_final_matrix():
    registry = gam.DictRegistry({"x": [1.0, 2.0]})
    first = gam.Basis(registry.get_obs("x"), basis_fn=np.square, name="first")
    last = gam.Basis(first, basis_fn=np.square, name="last")
    model = lsl.Model(last)

    position = registry.precomputed_position(model, {"x": [2.0, 3.0, 4.0]})
    assert set(position) == {"last"}
    np.testing.assert_array_equal(position["last"], [16, 81, 256])


def test_shared_multi_input_basis_keeps_other_callback_inputs_raw():
    setup = pd.DataFrame({"x": [1.0, 2.0], "z": [3.0, 4.0]})
    registry = gam.PandasRegistry(setup)
    builder = gam.BasisBuilder(registry)
    only_x = builder.lin("x")
    both = builder.lin("x + z")
    native = gam.Basis(
        registry.get_obs("z"), basis_fn=jnp.square, use_callback=False, name="native"
    )
    model = lsl.Model([only_x, both, native])
    data = pd.DataFrame({"x": [2.0, 3.0, 4.0], "z": [5.0, 6.0, 7.0]})

    # Keeping z raw also keeps B(x, z) dynamic, so B(x) must retain x as well.
    position = registry.precomputed_position(model, data)
    assert set(position) == {"x", "z"}
    np.testing.assert_array_equal(position["x"], [2, 3, 4])
    np.testing.assert_array_equal(position["z"], [5, 6, 7])


@pytest.mark.parametrize("kind", ["polynomial", "mixed", "shared"])
def test_precomputed_positions_match_raw_minibatch_fits(kind, monkeypatch):
    opt = pytest.importorskip(
        "liesel.optim", reason="Requires development Liesel optim."
    )
    import optax

    calls = []
    io_callback = jax.experimental.io_callback

    def counted(function, *args, **kwargs):
        def evaluate(*values, **options):
            calls.append(None)
            return function(*values, **options)

        return io_callback(evaluate, *args, **kwargs)

    monkeypatch.setattr(jax.experimental, "io_callback", counted)
    rng = np.random.default_rng(1)
    x = rng.uniform(-2, 2, 200)
    data = pd.DataFrame(
        {
            "x": x,
            "x2": rng.uniform(-2, 2, 200),
            "y": -x + x**2 + np.exp(-0.2 * x) * rng.normal(size=200),
        }
    )
    tb = gam.TermBuilder.from_df(data)
    loc = gam.AdditivePredictor("loc")
    scale = gam.AdditivePredictor("scale", inv_link=jnp.exp)
    loc += tb.lin("x + {x**2}" if kind == "polynomial" else "x")
    if kind == "polynomial":
        scale += tb.lin("x")
    else:
        loc += tb.ps(
            "x" if kind == "shared" else "x2", k=8, scale=1.0, approximation=True
        )
    y = lsl.Var.new_obs(data.y.to_numpy(), lsl.Dist(tfd.Normal, loc, scale), name="y")
    model = lsl.Model(y)
    raw = tb.registry.observed_position(model, data)
    precomputed = tb.registry.precomputed_position(model, data)
    if kind == "shared":
        assert set(precomputed) == set(raw)
    else:
        assert "x" not in precomputed
        assert ("x2" in precomputed) == (kind == "mixed")

    def fit(position):
        split = opt.Split(
            list(position), axis_size=len(data), validate_axis_size=40, seed=1
        ).split_position(position)
        jax.effects_barrier()
        calls.clear()
        result = opt.LieselOptim(
            model,
            split=split,
            batch_size=16,
            optimizers=optax.adam(0.01),
            loss_monitor="validation",
            stopper=opt.Stopper(epochs=4, patience=4),
            seed=2,
            show_progress=False,
        ).fit()
        jax.effects_barrier()
        return result, len(calls)

    result, precomputed_calls = fit(precomputed)
    reference, raw_calls = fit(raw)
    assert raw_calls > 0
    assert (precomputed_calls > 0) if kind == "shared" else (precomputed_calls == 0)
    assert result.n_epochs == reference.n_epochs == 4
    for actual, expected in zip(
        jax.tree.leaves(result.history), jax.tree.leaves(reference.history), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
