import os
import sys
import json
import shutil
import ctypes
import platform
import subprocess
import warnings
import signal
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


def decode_exit_status(returncode: int | None) -> dict:
    """Return structured crash information from a subprocess return code."""
    info = {
        "returncode": returncode,
        "signal_number": None,
        "signal_name": None,
        "crashed": False,
        "crash_reason": None,
    }
    if returncode is None:
        return info

    signal_number = None
    if returncode < 0:
        signal_number = -returncode
    elif returncode >= 128:
        signal_number = returncode - 128

    if signal_number is not None:
        info["signal_number"] = signal_number
        try:
            info["signal_name"] = signal.Signals(signal_number).name
        except ValueError:
            info["signal_name"] = f"UNKNOWN_SIGNAL_{signal_number}"

    fatal_signals = {"SIGSEGV", "SIGABRT", "SIGBUS", "SIGILL", "SIGFPE"}
    if info["signal_name"] in fatal_signals:
        info["crashed"] = True
        info["crash_reason"] = f"process terminated by {info['signal_name']}"
    elif returncode in {132, 134, 135, 136, 139}:
        info["crashed"] = True
        info["crash_reason"] = f"process exited with fatal native error code {returncode}"

    return info


def command_result(cmd: list[str], max_lines: int = 160, timeout_sec: int | None = None) -> dict:
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout_sec,
        )
        stdout_lines = (result.stdout or "").splitlines()
        stderr_lines = (result.stderr or "").splitlines()
        exit_status = decode_exit_status(result.returncode)
        return {
            "cmd": cmd,
            "cmd_str": " ".join(cmd),
            "found": True,
            "timeout": False,
            "returncode": result.returncode,
            "exit_status": exit_status,
            "crashed": exit_status["crashed"],
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
    except subprocess.TimeoutExpired as exc:
        stdout_lines = (exc.stdout or "").splitlines() if isinstance(exc.stdout, str) else []
        stderr_lines = (exc.stderr or "").splitlines() if isinstance(exc.stderr, str) else []
        return {
            "cmd": cmd,
            "cmd_str": " ".join(cmd),
            "found": True,
            "timeout": True,
            "returncode": None,
            "exit_status": decode_exit_status(None),
            "crashed": False,
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
            "error": f"command timed out after {timeout_sec} seconds",
        }
    except FileNotFoundError:
        return {
            "cmd": cmd,
            "cmd_str": " ".join(cmd),
            "found": False,
            "timeout": False,
            "returncode": None,
            "exit_status": decode_exit_status(None),
            "crashed": False,
            "stdout": {"line_count": 0, "truncated": False, "head": []},
            "stderr": {"line_count": 0, "truncated": False, "head": []},
            "error": f"command not found: {cmd[0]}",
        }


PROBE_PREFIX = "__XPU_CHECK_JSON__"


def parse_probe_payload(command: dict) -> dict | None:
    for line in reversed(command.get("stdout", {}).get("head", [])):
        if line.startswith(PROBE_PREFIX):
            raw = line[len(PROBE_PREFIX):]
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"json_parse_error": raw}
    return None


def run_python_probe(name: str, code: str, timeout_sec: int = 45) -> dict:
    cmd = [sys.executable, "-X", "faulthandler", "-c", code]
    command = command_result(cmd, max_lines=220, timeout_sec=timeout_sec)
    payload = parse_probe_payload(command)
    ok = bool(command["returncode"] == 0 and not command["crashed"] and not command["timeout"])
    return {
        "name": name,
        "ok": ok,
        "payload": payload,
        "command": command,
        "crashed": command["crashed"],
        "timeout": command["timeout"],
        "reason": (
            command["exit_status"]["crash_reason"]
            if command["crashed"]
            else command["error"]
            if command["error"]
            else "completed"
            if ok
            else f"non-zero return code: {command['returncode']}"
        ),
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


def check_pytorch_xpu() -> dict:
    """
    Run PyTorch/XPU checks in child Python processes.

    Reason:
    Native crashes such as segmentation fault cannot be caught by Python try/except
    in the same process. Subprocess isolation keeps this report script alive and
    records the exact stage that crashed.
    """
    result = {
        "subprocess_mode": True,
        "import_ok": False,
        "torch_version": None,
        "torch_file": None,
        "has_xpu": False,
        "xpu_available": False,
        "xpu_device_count": 0,
        "devices": [],
        "matmul_test_ok": False,
        "warnings": [],
        "stages": {},
        "first_failed_stage": None,
        "first_crashed_stage": None,
        "exception": None,
        "ok": False,
        "reason": None,
    }

    probes = [
        (
            "import_torch",
            r"""
import json
import faulthandler
faulthandler.enable()
import torch
print("__XPU_CHECK_JSON__" + json.dumps({
    "import_ok": True,
    "torch_version": torch.__version__,
    "torch_file": getattr(torch, "__file__", None),
    "has_xpu": hasattr(torch, "xpu"),
}, ensure_ascii=False))
""",
        ),
        (
            "xpu_available",
            r"""
import json
import warnings
import faulthandler
faulthandler.enable()
import torch
with warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter("always")
    available = torch.xpu.is_available() if hasattr(torch, "xpu") else False
    count = torch.xpu.device_count() if hasattr(torch, "xpu") else 0
print("__XPU_CHECK_JSON__" + json.dumps({
    "xpu_available": bool(available),
    "xpu_device_count": int(count),
    "warnings": [
        {"category": w.category.__name__, "message": str(w.message)}
        for w in captured
    ],
}, ensure_ascii=False))
""",
        ),
        (
            "xpu_device_names",
            r"""
import json
import faulthandler
faulthandler.enable()
import torch
devices = []
count = torch.xpu.device_count() if hasattr(torch, "xpu") else 0
for idx in range(count):
    devices.append(torch.xpu.get_device_name(idx))
print("__XPU_CHECK_JSON__" + json.dumps({
    "xpu_device_count": int(count),
    "devices": devices,
}, ensure_ascii=False))
""",
        ),
        (
            "xpu_matmul",
            r"""
import json
import faulthandler
faulthandler.enable()
import torch
if not hasattr(torch, "xpu"):
    print("__XPU_CHECK_JSON__" + json.dumps({
        "matmul_test_ok": False,
        "reason": "torch.xpu is not present",
    }, ensure_ascii=False))
    raise SystemExit(2)
if not torch.xpu.is_available():
    print("__XPU_CHECK_JSON__" + json.dumps({
        "matmul_test_ok": False,
        "reason": "torch.xpu.is_available() is False",
    }, ensure_ascii=False))
    raise SystemExit(3)

def get_best_device() -> str:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

device = get_best_device()
print("best_device=", device)
x = torch.randn(256, 256, device=device)
y = torch.randn(256, 256, device=device)

z = x @ y
torch.xpu.synchronize()
print("__XPU_CHECK_JSON__" + json.dumps({
    "matmul_test_ok": tuple(z.shape) == (256, 256),
    "device": str(z.device),
    "shape": list(z.shape),
}, ensure_ascii=False))
""",
        ),
    ]

    for stage_name, code in probes:
        stage = run_python_probe(stage_name, code, timeout_sec=20)
        result["stages"][stage_name] = stage

        if stage["payload"]:
            payload = stage["payload"]
            if "import_ok" in payload:
                result["import_ok"] = bool(payload["import_ok"])
            if "torch_version" in payload:
                result["torch_version"] = payload["torch_version"]
            if "torch_file" in payload:
                result["torch_file"] = payload["torch_file"]
            if "has_xpu" in payload:
                result["has_xpu"] = bool(payload["has_xpu"])
            if "xpu_available" in payload:
                result["xpu_available"] = bool(payload["xpu_available"])
            if "xpu_device_count" in payload:
                result["xpu_device_count"] = int(payload["xpu_device_count"])
            if "devices" in payload:
                result["devices"] = payload["devices"]
            if "warnings" in payload:
                result["warnings"] = payload["warnings"]
            if "matmul_test_ok" in payload:
                result["matmul_test_ok"] = bool(payload["matmul_test_ok"])

        if stage["crashed"] and result["first_crashed_stage"] is None:
            result["first_crashed_stage"] = stage_name

        if not stage["ok"] and result["first_failed_stage"] is None:
            result["first_failed_stage"] = stage_name

        # Later stages depend on earlier stages. Stop after first native crash.
        if stage["crashed"]:
            break

        # If torch import succeeds but no torch.xpu exists, later XPU stages add little value.
        if stage_name == "import_torch" and stage["ok"] and stage["payload"] and not stage["payload"].get("has_xpu", False):
            break

    result["ok"] = bool(result["import_ok"] and result["has_xpu"] and result["xpu_available"] and result["matmul_test_ok"])

    if result["ok"]:
        result["reason"] = "PyTorch XPU matmul succeeded"
    elif result["first_crashed_stage"]:
        result["reason"] = f"native crash detected at stage: {result['first_crashed_stage']}"
    elif result["first_failed_stage"]:
        result["reason"] = f"PyTorch XPU check failed at stage: {result['first_failed_stage']}"
    elif not result["has_xpu"]:
        result["reason"] = "torch.xpu is not present"
    elif not result["xpu_available"]:
        result["reason"] = "torch.xpu.is_available() is False"
    else:
        result["reason"] = "PyTorch XPU matmul failed"

    return result


def summarize(checks: dict) -> dict:
    base_ok = checks["wsl_docker_bridge"]["ok"] and checks["intel_runtime_packages"]["ok"] and checks["opencl"]["ok"] and checks["level_zero"]["ok"]
    pytorch_ok = checks["pytorch_xpu"]["ok"]
    if pytorch_ok:
        diagnosis = "PyTorch XPU 사용 가능"
        next_focus = "ready"
        actions = ["PyTorch 학습/benchmark에서 device='xpu' 사용"]
    elif base_ok:
        diagnosis = "Intel XPU base는 가능하지만 PyTorch XPU 초기화가 실패함"
        next_focus = "pytorch_xpu"
        actions = ["torch XPU wheel 버전 확인", "torch.xpu warning 확인", "지원 GPU 모델 확인"]
    else:
        diagnosis = "PyTorch 이전의 Intel XPU base 단계가 아직 완료되지 않음"
        next_focus = "intel_xpu_base"
        actions = ["check_intel_xpu_base.py 먼저 통과 확인"]
    return {
        "intel_xpu_base_usable": base_ok,
        "pytorch_xpu_usable": pytorch_ok,
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
        "pytorch_xpu": check_pytorch_xpu(),
    }
    report = {
        "started_at": started_at,
        "finished_at": now_kst(),
        "timezone": "Asia/Seoul",
        "scope": "Intel XPU base + PyTorch XPU runtime",
        "environment": check_environment(),
        "checks": checks,
        "summary": summarize(checks),
    }
    output_path = save_report("pytorch_xpu_check", report)
    print("=" * 90)
    print("PyTorch XPU Check Summary")
    print("=" * 90)
    print("intel_xpu_base_usable:", report["summary"]["intel_xpu_base_usable"])
    print("pytorch_xpu_usable:", report["summary"]["pytorch_xpu_usable"])
    print("next_focus:", report["summary"]["next_focus"])
    print("diagnosis:", report["summary"]["diagnosis"])
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
