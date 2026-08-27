"""Run with: uv run python benchmarks/benchmark_basis_approximation.py."""

import json
import subprocess
import sys
from statistics import median
from time import perf_counter

import jax
import jax.numpy as jnp
import liesel.model as lsl
import pandas as pd

import liesel_gam as gam

FAMILIES = ("ps", "bs", "cp", "cr", "cs", "cc", "tp", "ts", "kriging")
SPECS = (
    gam.ApproximationSpec(),
    gam.ApproximationSpec(rtol=1e-2, atol=1e-4),
    gam.ApproximationSpec(rtol=1e-3, atol=1e-5),
    gam.ApproximationSpec(rtol=1e-4, atol=1e-6),
)


def runtime_us(function, values, repeats=100):
    compiled = jax.jit(function)
    compiled(values).block_until_ready()
    timings = []
    for _ in range(repeats):
        start = perf_counter()
        compiled(values).block_until_ready()
        timings.append(perf_counter() - start)
    return median(timings) * 1e6


def cold_worker(mode):
    data = pd.DataFrame({"x": jnp.linspace(0.0, 1.0, 256)})
    method = gam.BasisBuilder(gam.PandasRegistry(data)).ps
    approximation = {
        "exact": False,
        "fixed": gam.ApproximationSpec(),
        "refined": gam.ApproximationSpec(
            grid_size=65,
            rtol=1e-2,
            atol=1e-4,
        ),
    }[mode]

    start = perf_counter()
    basis = method("x", k=20, approximation=approximation)
    construction_s = perf_counter() - start
    values = jax.random.uniform(jax.random.key(42), (256,))
    assert isinstance(basis.value_node, lsl.Calc | lsl.TransientCalc)
    result = {
        "mode": mode,
        "construction_s": construction_s,
        "evaluation_us": runtime_us(basis.value_node.function, values),
        "grid_size": basis.approximation_grid_size,
    }
    print(json.dumps(result))


def cold_process_benchmark():
    results = []
    for mode in ("exact", "fixed", "refined"):
        process = subprocess.run(
            [sys.executable, __file__, "--cold-worker", mode],
            check=True,
            capture_output=True,
            text=True,
        )
        results.append(json.loads(process.stdout.splitlines()[-1]))

    print("cold_mode,construction_s,evaluation_us,grid_size")
    for result in results:
        print(
            f"{result['mode']},{result['construction_s']:.3f},"
            f"{result['evaluation_us']:.1f},{result['grid_size']}"
        )


def main():
    cold_process_benchmark()
    print()
    data = pd.DataFrame({"x": jnp.linspace(0.0, 1.0, 256)})
    values = jax.random.uniform(jax.random.key(42), (256,))
    print(
        "family,rtol,atol,grid_size,max_abs_error,grid_kib,exact_us,approx_us,speedup"
    )
    for family in FAMILIES:
        method = getattr(gam.BasisBuilder(gam.PandasRegistry(data)), family)
        exact = method("x", k=20, approximation=False)
        for spec in SPECS:
            method = getattr(gam.BasisBuilder(gam.PandasRegistry(data)), family)
            try:
                approximate = method("x", k=20, approximation=spec)
            except ValueError as error:
                print(f"{family},{spec.rtol},{spec.atol},failed,{error},,,,,")
                continue
            assert isinstance(exact.value_node, lsl.Calc | lsl.TransientCalc)
            assert isinstance(approximate.value_node, lsl.Calc | lsl.TransientCalc)
            exact_values = jax.jit(exact.value_node.function)(values)
            approximate_values = jax.jit(approximate.value_node.function)(values)
            max_error = float(jnp.max(jnp.abs(exact_values - approximate_values)))
            exact_us = runtime_us(exact.value_node.function, values)
            approximate_us = runtime_us(approximate.value_node.function, values)
            grid_size = approximate.approximation_grid_size
            assert grid_size is not None
            grid_kib = (
                grid_size * approximate.nbases * approximate.value.dtype.itemsize / 1024
            )
            print(
                f"{family},{spec.rtol},{spec.atol},{grid_size},{max_error},"
                f"{grid_kib:.1f},{exact_us:.1f},{approximate_us:.1f},"
                f"{exact_us / approximate_us:.2f}"
            )


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--cold-worker":
        cold_worker(sys.argv[2])
    elif sys.argv[1:] == ["--cold-only"]:
        cold_process_benchmark()
    else:
        main()
