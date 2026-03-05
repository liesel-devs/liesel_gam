import concurrent.futures as cf
import os
import queue
import threading
from collections.abc import Callable
from typing import Any, Literal


class ProtectedValue:
    def __init__(self, initial_value):
        self._value = initial_value
        self._lock = threading.Lock()

    def set(self, new_value):
        with self._lock:
            self._value = new_value

    def get(self):
        with self._lock:
            return self._value

    def update(self, func):
        """Thread-safe read-modify-write."""
        with self._lock:
            self._value = func(self._value)


_tasks: queue.Queue[tuple[Callable, tuple, dict, cf.Future]] = queue.Queue()
_started = ProtectedValue(initial_value=False)
_start_lock = threading.Lock()


def _worker():
    global _started
    try:
        while True:
            fn, args, kwargs, fut = _tasks.get()
            if fn is None:  # shutdown sentinel
                fut.set_result(None)
                return
            try:
                fut.set_result(fn(*args, **kwargs))
            except BaseException as e:
                fut.set_exception(e)
    finally:
        _started.set(False)


def ensure_r_thread():
    global _started
    with _start_lock:
        if not _started.get():
            t = threading.Thread(target=_worker, name="ryp-thread", daemon=True)
            t.start()
            _started.set(True)


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
    if _started.get():
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
