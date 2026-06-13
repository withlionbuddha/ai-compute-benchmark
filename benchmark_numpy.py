import os
import time
import json
import platform
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from datetime import datetime
from zoneinfo import ZoneInfo

RESULT_DIR = Path("./results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def benchmark_matmul(size: int, threads: int, repeat: int = 3) -> dict:
    rng = np.random.default_rng(42)

    a = rng.random((size, size), dtype=np.float64)
    b = rng.random((size, size), dtype=np.float64)

    # warm-up
    with threadpool_limits(limits=threads, user_api="blas"):
        _ = a @ b

    times = []

    with threadpool_limits(limits=threads, user_api="blas"):
        for _ in range(repeat):
            start = time.perf_counter()
            _ = a @ b
            end = time.perf_counter()
            times.append(end - start)

    return {
        "operation": "matrix_multiplication",
        "matrix_size": size,
        "threads": threads,
        "repeat": repeat,
        "times_sec": times,
        "avg_sec": sum(times) / len(times),
        "min_sec": min(times),
        "max_sec": max(times),
    }


def main():
    print("========== NumPy backend info ==========")
    np.show_config()

    print("\n========== threadpool_info ==========")
    backend_info = threadpool_info()
    for info in backend_info:
        print(info)

    print("\n========== environment ==========")
    env_info = {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }

    for key, value in env_info.items():
        print(f"{key}={value}")

    print("\n========== benchmark ==========")

    results = {
        "environment": env_info,
        "threadpool_info": backend_info,
        "benchmarks": [],
    }

    for size in [1000, 2000, 3000]:
        for threads in [1, 2, 4, 6]:
            result = benchmark_matmul(size=size, threads=threads, repeat=3)
            results["benchmarks"].append(result)

            print(
                f"size={size}, threads={threads}, "
                f"avg={result['avg_sec']:.4f}s, "
                f"min={result['min_sec']:.4f}s"
            )

    today = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")

    output_path = RESULT_DIR / f"numpy_backend_benchmark_{today}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()