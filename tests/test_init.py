import warnings
from types import SimpleNamespace

import liesel_gam as gam


def _pandas_options(*, string_storage: str = "auto", infer_string: bool = True):
    return SimpleNamespace(
        options=SimpleNamespace(
            mode=SimpleNamespace(string_storage=string_storage),
            future=SimpleNamespace(infer_string=infer_string),
        )
    )


def test_r_46_warning_skipped_when_python_string_storage_active():
    pd = _pandas_options(string_storage="python", infer_string=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gam._warn_if_r_46(lambda *_, **__: "4.6.0", pd)

    assert not caught


def test_r_46_warning_skipped_when_infer_string_disabled():
    pd = _pandas_options(string_storage="auto", infer_string=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gam._warn_if_r_46(lambda *_, **__: "4.6.0", pd)

    assert not caught


def test_r_46_warning_emitted_without_workaround():
    pd = _pandas_options(string_storage="auto", infer_string=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gam._warn_if_r_46(lambda *_, **__: "4.6.0", pd)

    assert len(caught) == 1
    assert "https://github.com/liesel-devs/liesel_gam/issues/67" in str(
        caught[0].message
    )
