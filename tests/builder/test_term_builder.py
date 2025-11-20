import pandas as pd
import pytest

import liesel_gam.builder as gb

from .make_df import make_test_df


@pytest.fixture(scope="module")
def data():
    return make_test_df()


@pytest.fixture(scope="class")
def bases(data) -> gb.BasisBuilder:
    registry = gb.PandasRegistry(data, na_action="drop")
    bases = gb.BasisBuilder(registry)
    return bases


class TestBasisBuilder:
    def test_init(self, data) -> None:
        gb.TermBuilder.from_df(data)

    def test_ri_basis(self) -> None:
        data = pd.DataFrame(
            {"x": pd.Categorical(["a", "b"], categories=["a", "b", "c"])}
        )
        tb = gb.TermBuilder.from_df(data)
        ri = tb.ri("x")
        assert ri.coef.value.size == 3
