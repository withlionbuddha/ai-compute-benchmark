import argparse
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

try:
    from threadpoolctl import threadpool_info
except ImportError:  # threadpoolctl is optional for this script.
    threadpool_info = None


RESULT_DIR = Path("./results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def get_threadpool_info() -> list:
    """Return native thread-pool information when threadpoolctl is available."""
    if threadpool_info is None:
        return []
    return threadpool_info()


def get_torch_backend_info() -> dict:
    """Collect PyTorch build and parallel backend information."""
    backend_info = {
        "torch_config": torch.__config__.show(),
        "threadpool_info": get_threadpool_info(),
    }

    if hasattr(torch.__config__, "parallel_info"):
        backend_info["parallel_info"] = torch.__config__.parallel_info()

    return backend_info


def get_environment_info() -> dict:
    """Collect reproducibility-related runtime environment information."""
    env_info = {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_device_count": torch.cuda.device_count(),
    }

    if torch.cuda.is_available():
        env_info["cuda_devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": torch.cuda.get_device_capability(index),
            }
            for index in range(torch.cuda.device_count())
        ]
    else:
        env_info["cuda_devices"] = []

    return env_info


def benchmark_matmul_cpu(
    size: int,
    threads: int,
    repeat: int = 3,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """Benchmark CPU matrix multiplication with a fixed PyTorch intra-op thread count."""
    if size <= 0:
        raise ValueError("size must be positive")
    if threads <= 0:
        raise ValueError("threads must be positive")
    if repeat <= 0:
        raise ValueError("repeat must be positive")

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)

    try:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(42)

        a = torch.rand((size, size), dtype=dtype, generator=generator, device="cpu")
        b = torch.rand((size, size), dtype=dtype, generator=generator, device="cpu")

        # warm-up
        _ = torch.matmul(a, b)

        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            _ = torch.matmul(a, b)
            end = time.perf_counter()
            times.append(end - start)
    finally:
        torch.set_num_threads(previous_threads)

    return {
        "operation": "matrix_multiplication",
        "library": "pytorch",
        "device": "cpu",
        "dtype": str(dtype).replace("torch.", ""),
        "matrix_size": size,
        "threads": threads,
        "repeat": repeat,
        "times_sec": times,
        "avg_sec": sum(times) / len(times),
        "min_sec": min(times),
        "max_sec": max(times),
    }


def benchmark_matmul_cuda(
    size: int,
    repeat: int = 3,
    dtype: torch.dtype = torch.float64,
    device_index: int = 0,
) -> dict:
    """Benchmark CUDA matrix multiplication using CUDA event timing."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if size <= 0:
        raise ValueError("size must be positive")
    if repeat <= 0:
        raise ValueError("repeat must be positive")

    device = torch.device(f"cuda:{device_index}")
    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    a = torch.rand((size, size), dtype=dtype, generator=generator, device=device)
    b = torch.rand((size, size), dtype=dtype, generator=generator, device=device)

    # warm-up
    _ = torch.matmul(a, b)
    torch.cuda.synchronize(device)

    times = []
    for _ in range(repeat):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        _ = torch.matmul(a, b)
        end_event.record()

        torch.cuda.synchronize(device)
        times.append(start_event.elapsed_time(end_event) / 1000.0)

    return {
        "operation": "matrix_multiplication",
        "library": "pytorch",
        "device": f"cuda:{device_index}",
        "device_name": torch.cuda.get_device_name(device_index),
        "dtype": str(dtype).replace("torch.", ""),
        "matrix_size": size,
        "threads": None,
        "repeat": repeat,
        "times_sec": times,
        "avg_sec": sum(times) / len(times),
        "min_sec": min(times),
        "max_sec": max(times),
    }


def parse_int_list(value: str) -> list[int]:
    """Parse comma-separated integers, for example: 1000,2000,3000."""
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch matrix multiplication benchmark")
    parser.add_argument(
        "--sizes",
        type=parse_int_list,
        default=[1000, 2000, 3000],
        help="comma-separated matrix sizes. default: 1000,2000,3000",
    )
    parser.add_argument(
        "--threads",
        type=parse_int_list,
        default=[1, 2, 4, 6],
        help="comma-separated CPU intra-op thread counts. default: 1,2,4,6",
    )
    parser.add_argument("--repeat", type=int, default=3, help="repeat count. default: 3")
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPE_MAP.keys()),
        default="float64",
        help="tensor dtype. default: float64",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="benchmark target. auto runs CPU and CUDA when CUDA is available. default: auto",
    )
    parser.add_argument(
        "--cuda-device-index",
        type=int,
        default=0,
        help="CUDA device index used when --device is cuda or auto. default: 0",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]

    print("========== PyTorch backend info ==========")
    backend_info = get_torch_backend_info()
    print(backend_info["torch_config"])
    if "parallel_info" in backend_info:
        print("\n========== PyTorch parallel info ==========")
        print(backend_info["parallel_info"])

    print("\n========== threadpool_info ==========")
    for info in backend_info["threadpool_info"]:
        print(info)

    print("\n========== environment ==========")
    env_info = get_environment_info()
    for key, value in env_info.items():
        print(f"{key}={value}")

    print("\n========== benchmark ==========")

    results = {
        "environment": env_info,
        "backend_info": backend_info,
        "benchmarks": [],
    }

    if args.device in {"cpu", "auto"}:
        for size in args.sizes:
            for threads in args.threads:
                result = benchmark_matmul_cpu(
                    size=size,
                    threads=threads,
                    repeat=args.repeat,
                    dtype=dtype,
                )
                results["benchmarks"].append(result)

                print(
                    f"device=cpu, size={size}, threads={threads}, "
                    f"avg={result['avg_sec']:.4f}s, "
                    f"min={result['min_sec']:.4f}s"
                )

    if args.device in {"cuda", "auto"}:
        if torch.cuda.is_available():
            for size in args.sizes:
                result = benchmark_matmul_cuda(
                    size=size,
                    repeat=args.repeat,
                    dtype=dtype,
                    device_index=args.cuda_device_index,
                )
                results["benchmarks"].append(result)

                print(
                    f"device={result['device']}, size={size}, "
                    f"avg={result['avg_sec']:.4f}s, "
                    f"min={result['min_sec']:.4f}s"
                )
        elif args.device == "cuda":
            raise RuntimeError("--device cuda was selected, but CUDA is not available")
        else:
            print("CUDA is not available. Skipped CUDA benchmark.")

    today = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")
    output_path = RESULT_DIR / f"pytorch_backend_benchmark_{today}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
