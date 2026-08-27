import pandas as pd
import pytest

import liesel_gam as gam


def test_category_coverage_indices_select_one_row_per_observed_level():
    df = pd.DataFrame({"group": ["a", "a", "b", "c"], "x": [0.0, 1.0, 2.0, 3.0]})

    indices = gam.category_coverage_indices(df, columns=["group"])

    assert indices.tolist() == [0, 2, 3]


def test_category_coverage_indices_warns_when_inference_reserves_every_row():
    df = pd.DataFrame({"identifier": ["a", "b", "c"], "x": [0.0, 1.0, 2.0]})

    with pytest.warns(UserWarning, match="reserve every row"):
        indices = gam.category_coverage_indices(df)

    assert indices.tolist() == [0, 1, 2]


def test_category_coverage_indices_rejects_numeric_category_codes():
    df = pd.DataFrame({"group": [1, 2, 1]})

    with pytest.raises(TypeError, match="categorical dtype"):
        gam.category_coverage_indices(df, columns=["group"])


def test_basis_setup_sample_includes_training_boundaries_within_target_size():
    df = pd.DataFrame({"x": [5.0, 0.0, 2.0, 10.0, 3.0]})

    sample = gam.basis_setup_sample(
        df, indices=[0, 1, 2, 3, 4], continuous=["x"], categorical=[], n=3, seed=4
    )

    assert len(sample) == 3
    assert sample["x"].min() == 0.0
    assert sample["x"].max() == 10.0


def test_basis_setup_sample_preserves_all_training_category_metadata():
    df = pd.DataFrame({"group": ["a", "b", "c", "test-only"]})

    sample = gam.basis_setup_sample(
        df, indices=[0, 1, 2], continuous=[], categorical=["group"], n=1, seed=2
    )

    assert isinstance(sample["group"].dtype, pd.CategoricalDtype)
    assert sample["group"].cat.categories.tolist() == ["a", "b", "c"]


def test_basis_setup_sample_rejects_numeric_category_codes():
    df = pd.DataFrame({"group": [1, 2, 1]})

    with pytest.raises(TypeError, match="categorical dtype"):
        gam.basis_setup_sample(
            df, indices=[0, 1, 2], continuous=[], categorical=["group"]
        )


def test_basis_setup_sample_warns_when_boundaries_exceed_target_size():
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 0.0, 3.0, 2.0]})

    with pytest.warns(UserWarning, match="boundary rows"):
        sample = gam.basis_setup_sample(
            df,
            indices=[0, 1, 2, 3],
            continuous=["x", "y"],
            categorical=[],
            n=2,
            seed=1,
        )

    assert len(sample) == 4
