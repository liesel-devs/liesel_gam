import jax
import jax.numpy as jnp
import liesel.model as lsl
import numpy as np
import pandas as pd
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_gam as gam


def test_catvar_encodes_labels_and_is_observed() -> None:
    group = gam.CatVar(["b", "a", "b"], name="group")

    assert isinstance(group, gam.UserVar)
    assert group.observed
    assert group.name == "group"
    assert jnp.array_equal(group.value, jnp.array([1, 0, 1]))
    assert dict(group.mapping.labels_to_codes_map) == {"a": 0, "b": 1}


@pytest.mark.parametrize(
    "labels",
    [pd.Series(["b", "a"]), np.array(["b", "a"])],
)
def test_catvar_accepts_pandas_series_and_numpy_strings(labels) -> None:
    group = gam.CatVar(labels)

    assert jnp.array_equal(group.value, jnp.array([1, 0]))


def test_catvar_preserves_pandas_category_order() -> None:
    labels = pd.Categorical(["low", "high"], categories=["low", "medium", "high"])

    group = gam.CatVar(labels)

    assert dict(group.mapping.labels_to_codes_map) == {
        "low": 0,
        "medium": 1,
        "high": 2,
    }
    assert jnp.array_equal(group.value, jnp.array([0, 2]))


def test_catvar_preserves_explicit_category_order() -> None:
    group = gam.CatVar(["a", "b"], categories=["b", "a"])

    assert jnp.array_equal(group.value, jnp.array([1, 0]))


@pytest.mark.parametrize("labels", [[], np.empty((0, 2)), np.empty((2, 0))])
def test_catvar_requires_nonempty_labels(labels) -> None:
    with pytest.raises(ValueError, match="nonempty"):
        gam.CatVar(labels)


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (np.array("a"), np.array(0)),
        (
            np.array([["b", "a"], ["a", "b"]]),
            np.array([[1, 0], [0, 1]]),
        ),
        (
            np.array([[["b"], ["a"]], [["a"], ["b"]]]),
            np.array([[[1], [0]], [[0], [1]]]),
        ),
    ],
)
def test_catvar_preserves_arbitrary_label_shapes(labels, expected) -> None:
    group = gam.CatVar(labels)

    assert group.value.shape == expected.shape
    assert jnp.array_equal(group.value, expected)


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
def test_catvar_rejects_missing_labels(missing) -> None:
    with pytest.raises(ValueError, match="missing"):
        gam.CatVar(["a", missing])


def test_catvar_from_codes_reuses_mapping_with_unobserved_categories() -> None:
    mapping = gam.CategoryMapping({"a": 0, "b": 1, "unused": 2})

    group = gam.CatVar.from_codes([1, 0, 1], mapping=mapping, name="group")

    assert group.mapping is mapping
    assert jnp.array_equal(group.value, jnp.array([1, 0, 1]))


@pytest.mark.parametrize("codes", [[], np.empty((0, 2)), np.empty((2, 0))])
def test_catvar_from_codes_requires_nonempty_codes(codes) -> None:
    mapping = gam.CategoryMapping({"a": 0, "b": 1})

    with pytest.raises(ValueError, match="nonempty"):
        gam.CatVar.from_codes(codes, mapping=mapping)


@pytest.mark.parametrize(
    "codes", [np.array(0), np.array([[1, 0], [0, 1]]), np.zeros((2, 1, 2), dtype=int)]
)
def test_catvar_from_codes_preserves_arbitrary_shapes(codes) -> None:
    mapping = gam.CategoryMapping({"a": 0, "b": 1})

    group = gam.CatVar.from_codes(codes, mapping=mapping)

    assert group.value.shape == codes.shape
    assert jnp.array_equal(group.value, codes)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"labels": [20, 10]},
        {"labels": ["a"], "categories": ["a", 20]},
        {"labels": ["a"], "unknown_category": 20},
    ],
)
def test_catvar_rejects_integer_labels_and_categories(kwargs) -> None:
    with pytest.raises(TypeError, match="must not be integers"):
        gam.CatVar(**kwargs)


def test_catvar_from_codes_requires_integer_dtype() -> None:
    mapping = gam.CategoryMapping({"a": 0, "b": 1})

    with pytest.raises(TypeError, match="integer dtype"):
        gam.CatVar.from_codes([0.0, 1.0], mapping=mapping)


def test_catvar_value_replacement_accepts_labels() -> None:
    group = gam.CatVar(["a", "b"], name="group")

    group.value = ["b", "a"]
    assert jnp.array_equal(group.value, jnp.array([1, 0]))

    group.value = [["a", "b"], ["b", "a"]]
    assert jnp.array_equal(group.value, jnp.array([[0, 1], [1, 0]]))


@pytest.mark.parametrize("codes", [[0, 1], np.array([0, 1])])
def test_label_mode_catvar_rejects_integer_replacement(codes) -> None:
    group = gam.CatVar(["a", "b"])

    with pytest.raises(ValueError, match="ambiguous"):
        group.value = codes


def test_catvar_from_codes_keeps_host_integers_as_codes() -> None:
    mapping = gam.CategoryMapping({10: 0, 20: 1})
    group = gam.CatVar.from_codes([0, 1], mapping=mapping)

    group.value = [1, 0]

    assert jnp.array_equal(group.value, jnp.array([1, 0]))


def test_catvar_from_codes_accepts_label_replacement() -> None:
    mapping = gam.CategoryMapping({"a": 0, "b": 1})
    group = gam.CatVar.from_codes([0, 1], mapping=mapping)

    group.value = ["b", "a"]

    assert jnp.array_equal(group.value, jnp.array([1, 0]))


def test_label_mode_catvar_accepts_valid_internal_jax_codes() -> None:
    group = gam.CatVar(["a", "b"])

    group.value = jnp.array([1, 0])

    assert jnp.array_equal(group.value, jnp.array([1, 0]))


def test_label_mode_catvar_rejects_invalid_internal_jax_codes() -> None:
    group = gam.CatVar(["a", "b"])

    with pytest.raises(ValueError, match="Unknown integer codes"):
        group.value = jnp.array([0, 2])


def test_catvar_value_replacement_rejects_empty_values() -> None:
    group = gam.CatVar(["a", "b"])

    with pytest.raises(ValueError, match="nonempty"):
        group.value = np.empty((2, 0))


def test_catvar_rejects_unknown_labels_by_default() -> None:
    group = gam.CatVar(["a", "b"])

    with pytest.raises(KeyError, match="unknown"):
        group.value = ["new"]


def test_catvar_maps_unknown_labels_to_configured_category() -> None:
    group = gam.CatVar(["a", "b"], name="group", unknown_category="other")

    group.value = [["new", "a"], ["other", "new"]]

    assert dict(group.mapping.labels_to_codes_map) == {
        "a": 0,
        "b": 1,
        "other": 2,
    }
    assert jnp.array_equal(group.value, jnp.array([[2, 0], [2, 2]]))

    with pytest.raises(ValueError, match="ambiguous"):
        group.value = [[0, 3]]


def test_catvar_maps_training_labels_omitted_from_categories_to_catch_all() -> None:
    group = gam.CatVar(
        ["known", "new"],
        categories=["known"],
        unknown_category="other",
    )

    assert dict(group.mapping.labels_to_codes_map) == {"known": 0, "other": 1}
    assert jnp.array_equal(group.value, jnp.array([0, 1]))


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
def test_catvar_catch_all_does_not_replace_missing_values(missing) -> None:
    group = gam.CatVar(["a"], unknown_category="other")

    with pytest.raises(KeyError, match="unknown"):
        group.value = [["a", missing]]


@pytest.mark.parametrize("missing", [np.nan, pd.NA])
def test_catvar_rejects_missing_catch_all_category(missing) -> None:
    with pytest.raises(ValueError, match="missing"):
        gam.CatVar(["a"], unknown_category=missing)


def test_catvar_rejects_unhashable_catch_all_category() -> None:
    with pytest.raises(TypeError, match="hashable"):
        gam.CatVar(["a"], unknown_category=["other"])


def test_catvar_numeric_conversion_is_jittable() -> None:
    group = gam.CatVar(["a", "b"], name="group")
    model = lsl.Model(group)

    def update(codes):
        state = model.update_state({"group": codes})
        return model.extract_position(["group"], model_state=state)["group"]

    codes = jnp.array([[1, 0], [0, 1]])
    assert jnp.array_equal(jax.jit(update)(codes), codes)

    with pytest.raises(TypeError):
        jax.jit(update)(jnp.array([1.0, 0.0]))


def test_catvar_supports_a_distribution_on_encoded_values() -> None:
    dist = lsl.Dist(tfd.Categorical, logits=jnp.zeros(2))
    group = gam.CatVar([["a", "b"], ["b", "a"]], name="group", dist=dist)

    assert group.has_dist
    assert jnp.isfinite(lsl.Model(group).log_prob)


def test_catvar_accepts_explicit_codes_in_prediction_newdata() -> None:
    mapping = gam.CategoryMapping({"a": 0, "b": 1})
    group = gam.CatVar.from_codes([0, 1], mapping=mapping, name="group")
    coef = lsl.Var.new_param(jnp.zeros(2), name="coef")
    effect = lsl.Var.new_calc(
        lambda group, coef: coef[group], group, coef, name="effect"
    )
    model = lsl.Model(effect)

    prediction = model.predict(
        samples=lsl.Position({"coef": jnp.array([[10.0, 20.0]])}),
        predict=["effect"],
        newdata=lsl.Position(
            {"group": group.mapping.labels_to_codes([["b", "a"], ["a", "b"]])}
        ),
    )

    assert jnp.array_equal(
        prediction["effect"], jnp.array([[[20.0, 10.0], [10.0, 20.0]]])
    )


def test_label_mode_catvar_rejects_integer_prediction_newdata() -> None:
    group = gam.CatVar(["a", "b"], name="group")
    coef = lsl.Var.new_param(jnp.zeros(2), name="coef")
    effect = lsl.Var.new_calc(
        lambda group, coef: coef[group], group, coef, name="effect"
    )
    model = lsl.Model(effect)

    with pytest.raises(ValueError, match="ambiguous"):
        model.predict(
            samples=lsl.Position({"coef": jnp.array([[10.0, 20.0]])}),
            predict=["effect"],
            newdata=lsl.Position({"group": np.array([0, 1])}),
        )


def test_catvar_converts_label_valued_prediction_newdata() -> None:
    group = gam.CatVar(["a", "b"], name="group", unknown_category="other")
    coef = lsl.Var.new_param(jnp.zeros(3), name="coef")
    effect = lsl.Var.new_calc(
        lambda group, coef: coef[group], group, coef, name="effect"
    )
    model = lsl.Model(effect)

    prediction = model.predict(
        samples=lsl.Position({"coef": jnp.array([[10.0, 20.0, 30.0]])}),
        predict=["effect"],
        newdata=lsl.Position({"group": ["new", "a", "other"]}),
    )

    assert jnp.array_equal(prediction["effect"], jnp.array([[30.0, 10.0, 30.0]]))


def test_catvar_converts_label_valued_sampling_newdata() -> None:
    group = gam.CatVar(["a", "b"], name="group")
    loc = lsl.Var.new_calc(lambda group: group.astype(float), group, name="loc")
    y = lsl.Var(jnp.zeros(2), lsl.Dist(tfd.Deterministic, loc=loc), name="y")
    model = lsl.Model(y)

    samples = model.sample(
        shape=(1,),
        seed=jax.random.key(1),
        newdata=lsl.Position({"group": [["b", "a"], ["a", "b"]]}),
    )

    assert jnp.array_equal(samples["y"], jnp.array([[[1.0, 0.0], [0.0, 1.0]]]))
