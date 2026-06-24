import os
import sys
import json
import shutil
import ctypes
import platform
import subprocess
import warnings
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

RESULT_DIR = Path("./results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def now_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def save_report(prefix: str, report: dict) -> Path:
    timestamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
    output_path = RESULT_DIR / f"{prefix}_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return output_path


def command_result(cmd: list[str], max_lines: int = 160) -> dict:
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        stdout_lines = (result.stdout or "").splitlines()
        stderr_lines = (result.stderr or "").splitlines()
        return {
            "cmd": cmd,
            "cmd_str": " ".join(cmd),
            "found": True,
            "returncode": result.returncode,
            "stdout": {
                "line_count": len(stdout_lines),
                "truncated": len(stdout_lines) > max_lines,
                "head": stdout_lines[:max_lines],
            },
            "stderr": {
                "line_count": len(stderr_lines),
                "truncated": len(stderr_lines) > max_lines,
                "head": stderr_lines[:max_lines],
            },
            "error": None,
        }
    except FileNotFoundError:
        return {
            "cmd": cmd,
            "cmd_str": " ".join(cmd),
            "found": False,
            "returncode": None,
            "stdout": {"line_count": 0, "truncated": False, "head": []},
            "stderr": {"line_count": 0, "truncated": False, "head": []},
            "error": f"command not found: {cmd[0]}",
        }


def package_status(package_name: str) -> dict:
    raw = command_result(
        [
            "bash",
            "-lc",
            f"dpkg-query -W -f='${{Package}}\\t${{Status}}\\t${{Version}}\\n' {package_name} 2>/dev/null || true",
        ],
        max_lines=20,
    )
    installed = False
    version = None
    if raw["stdout"]["head"]:
        parts = raw["stdout"]["head"][0].split("\t")
        if len(parts) >= 3:
            installed = parts[1] == "install ok installed"
            version = parts[2]
    return {"name": package_name, "installed": installed, "version": version, "raw": raw}


def check_environment() -> dict:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        "PATH": os.environ.get("PATH"),
    }


def check_wsl_docker_bridge() -> dict:
    wsl_lib = Path("/usr/lib/wsl/lib")
    required_libs = ["libd3d12.so", "libd3d12core.so", "libdxcore.so"]
    result = {
        "dxg_exists": Path("/dev/dxg").exists(),
        "wsl_lib_exists": wsl_lib.exists(),
        "required_libs": {name: (wsl_lib / name).exists() for name in required_libs},
        "ld_library_path_contains_wsl_lib": "/usr/lib/wsl/lib" in (os.environ.get("LD_LIBRARY_PATH") or ""),
        "commands": {
            "ls_dev_dxg": command_result(["ls", "-l", "/dev/dxg"], max_lines=20),
            "ls_wsl_lib": command_result(
                ["bash", "-lc", "ls -l /usr/lib/wsl/lib | grep -E 'd3d12|dxcore' || true"],
                max_lines=40,
            ),
        },
    }
    result["ok"] = (
        result["dxg_exists"]
        and result["wsl_lib_exists"]
        and all(result["required_libs"].values())
        and result["ld_library_path_contains_wsl_lib"]
    )
    return result


def check_intel_runtime_packages() -> dict:
    required = ["intel-opencl-icd", "intel-level-zero-gpu", "level-zero", "clinfo"]
    recommended = ["level-zero-dev", "libigc1", "intel-igc-cm", "libigdgmm12"]
    optional = ["libigc-dev", "libigdfcl-dev", "libigfxcmrt-dev", "intel-ocloc", "libze1", "libze-dev"]
    req = [package_status(name) for name in required]
    rec = [package_status(name) for name in recommended]
    opt = [package_status(name) for name in optional]
    return {
        "ok": all(item["installed"] for item in req),
        "required": req,
        "recommended": rec,
        "optional": opt,
        "commands": {
            "dpkg_related": command_result(
                [
                    "bash",
                    "-lc",
                    "dpkg -l | grep -Ei 'intel-opencl|intel-level-zero|level-zero|clinfo|igc|igdgmm|ocloc|libze' || true",
                ],
                max_lines=120,
            ),
            "ldconfig_related": command_result(
                ["bash", "-lc", "ldconfig -p | grep -Ei 'libze|OpenCL|igc|igdgmm|level' || true"],
                max_lines=120,
            ),
        },
    }


def check_opencl() -> dict:
    tool_path = shutil.which("clinfo")
    result = {
        "tool_path": tool_path,
        "ok": False,
        "reason": None,
        "command": None,
        "checks": {},
    }
    if tool_path is None:
        result["reason"] = "clinfo command not found"
        return result
    cmd = command_result(["clinfo"], max_lines=180)
    out = "\n".join(cmd["stdout"]["head"])
    checks = {
        "returncode_zero": cmd["returncode"] == 0,
        "platform_intel_opencl_graphics": "Intel(R) OpenCL Graphics" in out,
        "has_gpu_device_type": "Device Type" in out and "GPU" in out,
        "device_available_yes": "Device Available" in out and "Yes" in out,
    }
    result["command"] = cmd
    result["checks"] = checks
    result["ok"] = all(checks.values())
    result["reason"] = "OpenCL sees Intel GPU" if result["ok"] else "OpenCL did not confirm Intel GPU"
    return result


def check_level_zero_direct() -> dict:
    result = {
        "ok": False,
        "method": "ctypes libze_loader.so.1",
        "library_loaded": False,
        "library_path": None,
        "zeInit_result": None,
        "zeDriverGet_result": None,
        "driver_count": 0,
        "drivers": [],
        "total_device_count": 0,
        "reason": None,
        "exception": None,
    }
    candidate_libs = [
        "libze_loader.so.1",
        "/lib/x86_64-linux-gnu/libze_loader.so.1",
        "/usr/lib/x86_64-linux-gnu/libze_loader.so.1",
    ]
    lib = None
    loaded_path = None
    for libname in candidate_libs:
        try:
            lib = ctypes.CDLL(libname)
            loaded_path = libname
            break
        except OSError:
            continue
    if lib is None:
        result["reason"] = "libze_loader.so.1 could not be loaded"
        return result
    result["library_loaded"] = True
    result["library_path"] = loaded_path
    try:
        ze_driver_handle_t = ctypes.c_void_p
        ze_device_handle_t = ctypes.c_void_p
        lib.zeInit.argtypes = [ctypes.c_uint32]
        lib.zeInit.restype = ctypes.c_int
        lib.zeDriverGet.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ze_driver_handle_t)]
        lib.zeDriverGet.restype = ctypes.c_int
        lib.zeDeviceGet.argtypes = [ze_driver_handle_t, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ze_device_handle_t)]
        lib.zeDeviceGet.restype = ctypes.c_int
        # ZE_INIT_FLAG_GPU_ONLY = 1
        init_result = lib.zeInit(ctypes.c_uint32(1))
        result["zeInit_result"] = int(init_result)
        if init_result != 0:
            result["reason"] = f"zeInit failed with result code {init_result}"
            return result
        driver_count = ctypes.c_uint32(0)
        driver_get_result = lib.zeDriverGet(ctypes.byref(driver_count), None)
        result["zeDriverGet_result"] = int(driver_get_result)
        result["driver_count"] = int(driver_count.value)
        if driver_get_result != 0:
            result["reason"] = f"zeDriverGet count failed with result code {driver_get_result}"
            return result
        if driver_count.value == 0:
            result["reason"] = "zeDriverGet returned zero drivers"
            return result
        drivers = (ze_driver_handle_t * driver_count.value)()
        driver_get_result2 = lib.zeDriverGet(ctypes.byref(driver_count), drivers)
        if driver_get_result2 != 0:
            result["reason"] = f"zeDriverGet handles failed with result code {driver_get_result2}"
            return result
        total_device_count = 0
        driver_infos = []
        for idx in range(driver_count.value):
            device_count = ctypes.c_uint32(0)
            device_get_result = lib.zeDeviceGet(drivers[idx], ctypes.byref(device_count), None)
            driver_info = {
                "driver_index": idx,
                "driver_handle": int(drivers[idx]) if drivers[idx] else None,
                "zeDeviceGet_count_result": int(device_get_result),
                "device_count": int(device_count.value),
            }
            if device_get_result == 0 and device_count.value > 0:
                devices = (ze_device_handle_t * device_count.value)()
                device_get_result2 = lib.zeDeviceGet(drivers[idx], ctypes.byref(device_count), devices)
                driver_info["zeDeviceGet_handles_result"] = int(device_get_result2)
                driver_info["device_handles"] = [int(devices[j]) if devices[j] else None for j in range(device_count.value)]
            driver_infos.append(driver_info)
            total_device_count += int(device_count.value)
        result["drivers"] = driver_infos
        result["total_device_count"] = total_device_count
        result["ok"] = total_device_count > 0
        result["reason"] = "Level Zero sees at least one device" if result["ok"] else "Level Zero initialized but returned zero devices"
        return result
    except Exception as exc:
        result["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        result["reason"] = "exception during Level Zero ctypes check"
        return result


def check_level_zero_tool() -> dict:
    tool_path = shutil.which("ze_info")
    result = {"tool_path": tool_path, "ok": False, "reason": None, "command": None}
    if tool_path is None:
        result["reason"] = "ze_info command not found"
        return result
    cmd = command_result(["ze_info"], max_lines=220)
    out = "\n".join(cmd["stdout"]["head"])
    result["command"] = cmd
    result["ok"] = cmd["returncode"] == 0 and ("Device" in out or "GPU" in out or "Intel" in out)
    result["reason"] = "ze_info sees a device" if result["ok"] else "ze_info did not confirm a device"
    return result


def check_level_zero() -> dict:
    direct = check_level_zero_direct()
    tool = check_level_zero_tool()
    ok = direct["ok"] or tool["ok"]
    if ok:
        reason = "Level Zero usable by ctypes or ze_info"
    elif direct["library_loaded"] and direct["zeInit_result"] == 0 and direct["driver_count"] > 0 and direct["total_device_count"] == 0:
        reason = "Level Zero loader works but returns zero devices"
    elif not direct["library_loaded"]:
        reason = "Level Zero loader library not loaded"
    else:
        reason = direct["reason"] or tool["reason"]
    return {
        "ok": ok,
        "direct_ctypes": direct,
        "ze_info_tool": tool,
        "ldconfig_libze": command_result(["bash", "-lc", "ldconfig -p | grep -Ei 'libze|level-zero' || true"], max_lines=80),
        "reason": reason,
    }


def check_tensorflow_xpu() -> dict:
    result = {
        "tensorflow_import_ok": False,
        "tensorflow_version": None,
        "itex_import_ok": False,
        "itex_version": None,
        "physical_devices": {},
        "gpu_devices": [],
        "xpu_like_devices": [],
        "simple_tensor_test_ok": False,
        "exception": None,
        "ok": False,
        "reason": None,
    }
    try:
        import tensorflow as tf
        result["tensorflow_import_ok"] = True
        result["tensorflow_version"] = getattr(tf, "__version__", None)
    except Exception as exc:
        result["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        result["reason"] = "tensorflow import failed"
        return result

    try:
        import intel_extension_for_tensorflow as itex
        result["itex_import_ok"] = True
        result["itex_version"] = getattr(itex, "__version__", None)
    except Exception as exc:
        result["itex_import_ok"] = False
        result["itex_exception"] = {"type": type(exc).__name__, "message": str(exc)}

    try:
        device_types = ["CPU", "GPU", "XPU"]
        for device_type in device_types:
            try:
                devices = tf.config.list_physical_devices(device_type)
                result["physical_devices"][device_type] = [str(device) for device in devices]
            except Exception as exc:
                result["physical_devices"][device_type] = {"error": f"{type(exc).__name__}: {exc}"}
        result["gpu_devices"] = result["physical_devices"].get("GPU", []) if isinstance(result["physical_devices"].get("GPU"), list) else []
        xpu_list = result["physical_devices"].get("XPU", []) if isinstance(result["physical_devices"].get("XPU"), list) else []
        result["xpu_like_devices"] = xpu_list + result["gpu_devices"]
        with tf.device("/CPU:0"):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.matmul(a, a)
        result["simple_tensor_test_ok"] = tuple(b.shape) == (2, 2)
        result["ok"] = result["tensorflow_import_ok"] and result["itex_import_ok"] and len(result["xpu_like_devices"]) > 0
        if result["ok"]:
            result["reason"] = "TensorFlow/ITEX sees GPU or XPU-like device"
        elif not result["itex_import_ok"]:
            result["reason"] = "intel_extension_for_tensorflow import failed or is not installed"
        else:
            result["reason"] = "TensorFlow/ITEX did not list GPU/XPU device"
        return result
    except Exception as exc:
        result["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        result["reason"] = "exception during TensorFlow XPU check"
        return result


def summarize(checks: dict) -> dict:
    base_ok = checks["wsl_docker_bridge"]["ok"] and checks["intel_runtime_packages"]["ok"] and checks["opencl"]["ok"] and checks["level_zero"]["ok"]
    tf_ok = checks["tensorflow_xpu"]["ok"]
    if tf_ok:
        diagnosis = "TensorFlow XPU 사용 가능"
        next_focus = "ready"
        actions = ["TensorFlow 모델에서 GPU/XPU device 사용"]
    elif base_ok:
        diagnosis = "Intel XPU base는 가능하지만 TensorFlow XPU 장치 확인이 실패함"
        next_focus = "tensorflow_xpu"
        actions = ["intel-extension-for-tensorflow 설치/버전 확인", "tf.config.list_physical_devices 결과 확인"]
    else:
        diagnosis = "TensorFlow 이전의 Intel XPU base 단계가 아직 완료되지 않음"
        next_focus = "intel_xpu_base"
        actions = ["check_intel_xpu_base.py 먼저 통과 확인"]
    return {
        "intel_xpu_base_usable": base_ok,
        "tensorflow_xpu_usable": tf_ok,
        "next_focus": next_focus,
        "diagnosis": diagnosis,
        "recommended_actions": actions,
    }


def main() -> None:
    started_at = now_kst()
    checks = {
        "wsl_docker_bridge": check_wsl_docker_bridge(),
        "intel_runtime_packages": check_intel_runtime_packages(),
        "opencl": check_opencl(),
        "level_zero": check_level_zero(),
        "tensorflow_xpu": check_tensorflow_xpu(),
    }
    report = {
        "started_at": started_at,
        "finished_at": now_kst(),
        "timezone": "Asia/Seoul",
        "scope": "Intel XPU base + TensorFlow XPU runtime",
        "environment": check_environment(),
        "checks": checks,
        "summary": summarize(checks),
    }
    output_path = save_report("tensorflow_xpu_check", report)
    print("=" * 90)
    print("TensorFlow XPU Check Summary")
    print("=" * 90)
    print("intel_xpu_base_usable:", report["summary"]["intel_xpu_base_usable"])
    print("tensorflow_xpu_usable:", report["summary"]["tensorflow_xpu_usable"])
    print("next_focus:", report["summary"]["next_focus"])
    print("diagnosis:", report["summary"]["diagnosis"])
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
