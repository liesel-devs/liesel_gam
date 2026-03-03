import concurrent.futures as cf
import os
import queue
import threading
from collections.abc import Callable

_tasks: queue.Queue[tuple[Callable, tuple, dict, cf.Future]] = queue.Queue()
_started = False
_start_lock = threading.Lock()


def _worker():
    import ryp  # noqa: F401

    while True:
        fn, args, kwargs, fut = _tasks.get()
        if fn is None:  # shutdown sentinel
            fut.set_result(None)
            return
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as e:
            fut.set_exception(e)


def ensure_r_thread():
    global _started
    with _start_lock:
        if not _started:
            t = threading.Thread(target=_worker, name="ryp-thread", daemon=True)
            t.start()
            _started = True


def call_in_r_thread(fn, *args, **kwargs):
    ensure_r_thread()
    fut: cf.Future = cf.Future()
    _tasks.put((fn, args, kwargs, fut))
    return fut.result()


def init_r():
    """
    Used in other modules of this package to ensure R is initialized in the
    dedicated R thread. Should not be exported or be used outside the library.
    """
    global _started
    if _started:
        return None
    on_rtd = os.environ.get("READTHEDOCS", "False") == "True"
    # safeguard because R is not installed in the readthedocs build environment
    if not on_rtd:
        import pandas as pd

        from .rthread import call_in_r_thread

        def _call_ryp():
            from ryp import r, to_r

            try:
                to_r(pd.DataFrame({"a": [1.0, 2.0]}), "___test___")
                r("rm('___test___')")
            except ImportError as e:
                raise ImportError(
                    "Testing communication between R and Python failed. "
                    "Probably you need to install the R package 'arrow' using "
                    "install.packages('arrow')."
                    "Also, please consider the original traceback from ryp above."
                ) from e

        call_in_r_thread(_call_ryp)
