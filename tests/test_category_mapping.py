import pickle

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import liesel_gam as gam
from liesel_gam.category_mapping import UnknownCodeError, UnknownLabelError

df = pd.DataFrame(
    {
        "test_string": ["a", "b", "c"],
        "test_np_array": np.array([1.0, 2.0, 3.0]),
        "test_np_string_array": np.array(["a", "b", "c"]),
        "test_jnp_array": jnp.array([1.0, 2.0, 3.0]),
        "test_integer": [1, 2, 3],
        "test_float": [1.0, 2.0, 3.0],
        "test_boolean": [True, False, True],
        "test_categorical": pd.Categorical(["a", "b", "c"]),
    }
)


class TestIsCategorical:
    def test_string(self) -> None:
        assert gam.series_is_categorical(df["test_string"])

    def test_np_array(self) -> None:
        assert not gam.series_is_categorical(df["test_np_array"])

    def test_np_string_array(self) -> None:
        assert gam.series_is_categorical(df["test_np_string_array"])

    def test_jnp_array(self) -> None:
        assert not gam.series_is_categorical(df["test_jnp_array"])

    def test_integer(self) -> None:
        assert not gam.series_is_categorical(df["test_integer"])

    def test_float(self) -> None:
        assert not gam.series_is_categorical(df["test_float"])

    def test_boolean(self) -> None:
        assert not gam.series_is_categorical(df["test_boolean"])

    def test_pandas_categorical(self) -> None:
        assert gam.series_is_categorical(df["test_categorical"])


class TestIsCategoricalEdgeCases:
    def test_pandas_categorical_numeric(self) -> None:
        s = pd.Series(pd.Categorical([1, 2, 1]))
        assert gam.series_is_categorical(s)

    def test_pandas_categorical_ordered(self) -> None:
        s = pd.Series(pd.Categorical(["low", "med", "high"], ordered=True))
        assert gam.series_is_categorical(s)

    def test_string_with_missing_values(self) -> None:
        s = pd.Series(["a", "b", None])
        assert gam.series_is_categorical(s)

    def test_categorical_with_missing_values(self) -> None:
        s = pd.Series(pd.Categorical(["a", None, "b"], categories=["a", "b"]))
        assert gam.series_is_categorical(s)

    def test_object_series_mixed_types(self) -> None:
        s = pd.Series(["a", 1, "b"])
        assert gam.series_is_categorical(s)

    def test_datetime_series_is_not_categorical(self) -> None:
        s = pd.Series(pd.date_range("2020-01-01", periods=3))
        assert not gam.series_is_categorical(s)

    def test_timedelta_series_is_not_categorical(self) -> None:
        s = pd.Series(pd.to_timedelta([1, 2, 3], unit="D"))
        assert not gam.series_is_categorical(s)

    def test_nullable_integer_series_is_not_categorical(self) -> None:
        s = pd.Series([1, 2, None], dtype="Int64")
        assert not gam.series_is_categorical(s)

    def test_boolean_nullable_series_is_not_categorical(self) -> None:
        s = pd.Series([True, False, None], dtype="boolean")
        assert not gam.series_is_categorical(s)

    def test_empty_string_series(self) -> None:
        s = pd.Series([], dtype=object)
        assert gam.series_is_categorical(s)

    def test_single_level_categorical(self) -> None:
        s = pd.Series(pd.Categorical(["a", "a", "a"]))
        assert gam.series_is_categorical(s)


class TestCategoryMappingFromSeries:
    def test_from_series_with_strings(self) -> None:
        s = pd.Series(["apple", "banana", "apple"])
        cm = gam.CategoryMapping.from_series(s)
        assert set(cm.labels_to_integers_map) == {"apple", "banana"}

    def test_from_series_with_repeated_values(self) -> None:
        s = pd.Series(["a", "a", "b", "b"])
        cm = gam.CategoryMapping.from_series(s)
        assert len(cm.labels_to_integers_map) == 2

    def test_from_categorical_preserves_declared_categories(self) -> None:
        cat = pd.Series(
            pd.Categorical(["low", "high"], categories=["low", "med", "high"])
        )
        cm = gam.CategoryMapping.from_series(cat)
        assert list(cm.labels_to_integers_map) == ["low", "med", "high"]

    def test_from_series_accepts_catch_all_category(self) -> None:
        cm = gam.CategoryMapping.from_series(
            pd.Series(["a", "b"]), unknown_category="other"
        )

        assert cm.labels_to_integers(["new"]).tolist() == [2]

    def test_from_series_invalid_type(self) -> None:
        with pytest.raises(TypeError):
            gam.CategoryMapping.from_series(np.array(["a", "b"]))  # type: ignore


class TestCategoryMappingFromLabels:
    def test_infers_sorted_categories(self) -> None:
        mapping = gam.CategoryMapping.from_labels(["pear", "apple", "pear"])

        assert dict(mapping.labels_to_integers_map) == {"apple": 0, "pear": 1}

    @pytest.mark.parametrize(
        ("labels", "expected"),
        [
            (np.array("pear"), {"pear": 0}),
            (
                np.array([["pear", "apple"], ["apple", "pear"]]),
                {"apple": 0, "pear": 1},
            ),
        ],
    )
    def test_infers_categories_from_arbitrary_shapes(self, labels, expected) -> None:
        mapping = gam.CategoryMapping.from_labels(labels)

        assert dict(mapping.labels_to_integers_map) == expected

    def test_preserves_explicit_categories_including_unobserved(self) -> None:
        mapping = gam.CategoryMapping.from_labels(
            ["used", "used"], categories=["unused", "used"]
        )

        assert dict(mapping.labels_to_integers_map) == {"unused": 0, "used": 1}

    def test_rejects_categories_that_omit_observed_labels(self) -> None:
        with pytest.raises(ValueError, match="omit observed labels"):
            gam.CategoryMapping.from_labels(["known", "missing"], ["known"])

    def test_rejects_duplicate_categories(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            gam.CategoryMapping.from_labels(["a"], ["a", "a"])

    def test_unorderable_inferred_labels_request_explicit_categories(self) -> None:
        with pytest.raises(TypeError, match="categories"):
            gam.CategoryMapping.from_labels(["a", 1])

    def test_rejects_unhashable_labels(self) -> None:
        with pytest.raises(TypeError, match="hashable"):
            gam.CategoryMapping.from_labels(np.array([{"a": 1}], dtype=object))

    def test_rejects_ragged_labels(self) -> None:
        with pytest.raises(TypeError, match="hashable"):
            gam.CategoryMapping.from_labels([["a"], ["b", "c"]])

    def test_rejects_missing_explicit_categories(self) -> None:
        with pytest.raises(ValueError, match="missing"):
            gam.CategoryMapping.from_labels(["a"], ["a", None])


class TestCategoryMappingInvariants:
    @pytest.mark.parametrize(
        "mapping",
        [{"a": 1}, {"a": 0, "b": 2}, {"a": 0, "b": 0}],
    )
    def test_codes_must_be_contiguous_and_unique(self, mapping) -> None:
        with pytest.raises(ValueError, match="contiguous"):
            gam.CategoryMapping(mapping)

    def test_mappings_are_immutable_and_pickleable(self) -> None:
        source = {"a": 0}
        mapping = gam.CategoryMapping(source)
        source["b"] = 1

        with pytest.raises(TypeError):
            mapping.labels_to_integers_map["b"] = 1  # ty: ignore[invalid-assignment]

        restored = pickle.loads(pickle.dumps(mapping))
        assert dict(restored.labels_to_integers_map) == {"a": 0}


def test_code_named_api_and_integer_named_compatibility_aliases() -> None:
    mapping = gam.CategoryMapping({"a": 0, "b": 1})

    assert dict(mapping.labels_to_codes_map) == {"a": 0, "b": 1}
    assert dict(mapping.codes_to_labels_map) == {0: "a", 1: "b"}
    assert mapping.labels_to_codes(["b", "a"]).tolist() == [1, 0]
    assert mapping.codes_to_labels([1, 0]).tolist() == ["b", "a"]
    assert mapping.to_codes([0, 1]).tolist() == [0, 1]

    assert mapping.labels_to_integers_map == mapping.labels_to_codes_map
    assert mapping.integers_to_labels_map == mapping.codes_to_labels_map
    assert mapping.labels_to_integers(["b", "a"]).tolist() == [1, 0]
    assert mapping.integers_to_labels([1, 0]).tolist() == ["b", "a"]
    assert mapping.to_integers([0, 1]).tolist() == [0, 1]


class TestCategoryMappingEncoding:
    def test_labels_to_integers_known_values(self) -> None:
        cm = gam.CategoryMapping.from_series(pd.Series(["red", "blue"]))
        encoded = cm.labels_to_integers(["red", "blue", "red"])
        assert encoded.dtype == np.int_
        assert encoded.tolist() == [
            cm.labels_to_integers_map["red"],
            cm.labels_to_integers_map["blue"],
            cm.labels_to_integers_map["red"],
        ]

    def test_labels_to_integers_unknown_label(self) -> None:
        cm = gam.CategoryMapping.from_series(pd.Series(["red", "blue"]))
        with pytest.raises(UnknownLabelError):
            cm.labels_to_integers(["red", "green"])

    def test_labels_to_integers_maps_unknown_label_to_catch_all(self) -> None:
        cm = gam.CategoryMapping({"red": 0}, unknown_category="other")

        assert dict(cm.labels_to_integers_map) == {"red": 0, "other": 1}
        assert cm.labels_to_integers(["red", "green", "other"]).tolist() == [
            0,
            1,
            1,
        ]

    def test_labels_to_integers_known_but_unobserved_category(self) -> None:
        cat = pd.Categorical(["red", "blue"], categories=["red", "blue", "green"])
        cm = gam.CategoryMapping.from_series(cat)
        assert cm.labels_to_integers(["red", "green"]).tolist() == [
            cm.labels_to_integers_map["red"],
            cm.labels_to_integers_map["green"],
        ]

    def test_labels_to_integers_preserves_shape(self) -> None:
        cm = gam.CategoryMapping.from_series(pd.Series(["red", "blue", "green"]))
        arr = np.array([["red", "blue"], ["green", "red"]])
        encoded = cm.labels_to_integers(arr)
        assert encoded.shape == (2, 2)
        assert encoded.tolist() == [
            [cm.labels_to_integers_map["red"], cm.labels_to_integers_map["blue"]],
            [cm.labels_to_integers_map["green"], cm.labels_to_integers_map["red"]],
        ]


class TestCategoryMappingDecoding:
    def test_integers_to_labels_known_values(self) -> None:
        cm = gam.CategoryMapping({"sun": 0, "moon": 1})
        decoded = cm.integers_to_labels([0, 1, 0])
        assert decoded.tolist() == ["sun", "moon", "sun"]

    def test_integers_to_labels_unknown_code(self) -> None:
        cm = gam.CategoryMapping({"sun": 0})
        with pytest.raises(UnknownCodeError):
            cm.integers_to_labels([0, 99])

    def test_integers_to_labels_preserves_shape(self) -> None:
        cm = gam.CategoryMapping({"cat": 0, "dog": 1})
        arr = np.array([[0, 1], [1, 0]])
        decoded = cm.integers_to_labels(arr)
        assert decoded.shape == (2, 2)
        assert decoded.tolist() == [["cat", "dog"], ["dog", "cat"]]

    def test_roundtrip_labels_and_integers(self) -> None:
        cm = gam.CategoryMapping.from_series(pd.Series(["x", "y", "z"]))
        labels = np.array(["x", "z", "x"])
        encoded = cm.labels_to_integers(labels)
        decoded = cm.integers_to_labels(encoded)
        assert decoded.tolist() == labels.tolist()

    def test_integers_to_labels_accepts_sequences(self) -> None:
        cm = gam.CategoryMapping({"apple": 0, "banana": 1})
        decoded = cm.integers_to_labels([0, 1])
        assert decoded.tolist() == ["apple", "banana"]


class TestCatchAllEncodings:
    def test_to_integers(self):
        cm = gam.CategoryMapping({"apple": 0, "banana": 1})

        ints = jnp.array([0, 1])
        assert jnp.allclose(ints, cm.to_integers([0, 1]))
        assert jnp.allclose(ints, cm.to_integers(["apple", "banana"]))

        with pytest.raises(ValueError):
            cm.to_integers([0, 1, 2])

    def test_to_labels(self):
        cm = gam.CategoryMapping({"apple": 0, "banana": 1})

        labs = np.array(["apple", "banana"])
        assert np.all(labs == cm.to_labels([0, 1]))
        assert np.all(labs == cm.to_labels(["apple", "banana"]))

        with pytest.raises(ValueError):
            cm.to_labels(["apple", "banana", "orange"])
