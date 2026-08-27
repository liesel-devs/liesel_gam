import importlib.util

import numpy as np
import pandas as pd
import smoothcon

import liesel_gam as gam


def test_pspline_builder_consumes_external_smoothcon(monkeypatch) -> None:
    calls = 0
    constructor = smoothcon.pspline

    def tracked_constructor(*args, **kwargs):
        nonlocal calls
        calls += 1
        return constructor(*args, **kwargs)

    monkeypatch.setattr(smoothcon, "pspline", tracked_constructor)
    data = pd.DataFrame({"x": np.linspace(-1.0, 2.0, 40)})
    basis = gam.BasisBuilder(gam.PandasRegistry(data)).ps(
        "x",
        k=9,
        absorb_cons=False,
        scale_penalty=False,
        diagonal_penalty=False,
    )

    assert calls == 1
    assert basis.value.shape == (40, 9)
    assert importlib.util.find_spec("liesel_gam._smoothcon") is None
