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


def installed_package_names(packages_result: dict) -> list[str]:
    names: list[str] = []

    for group_name in ["required", "recommended", "optional"]:
        for item in packages_result.get(group_name, []):
            if item.get("installed"):
                names.append(item.get("name"))

    return names


def installed_runtime_libraries_from_ldconfig(packages_result: dict) -> list[str]:
    ldconfig_result = packages_result.get("commands", {}).get("ldconfig_related", {})
    lines = ldconfig_result.get("stdout", {}).get("head", [])
    libraries: list[str] = []

    for line in lines:
        stripped = line.strip()
        if "=>" in stripped:
            lib_name = stripped.split()[0]
            libraries.append(lib_name)

    return libraries


def format_items(items: list[str]) -> str:
    return ", ".join(items) if items else "none"


def build_installed_library_architecture(checks: dict) -> dict:
    packages_result = checks.get("intel_runtime_packages", {})
    installed_packages = installed_package_names(packages_result)
    installed_libs = installed_runtime_libraries_from_ldconfig(packages_result)

    opencl_packages = [
        name
        for name in installed_packages
        if name in {
            "intel-opencl-icd",
            "clinfo",
            "libigc1",
            "intel-igc-cm",
            "libigdgmm12",
            "ocl-icd-libopencl1",
            "ocl-icd-opencl-dev",
        }
    ]

    opencl_libraries = [
        name
        for name in installed_libs
        if (
            "OpenCL" in name
            or "opencl" in name
            or "igc" in name
            or "igdgmm" in name
        )
    ]

    level_zero_packages = [
        name
        for name in installed_packages
        if name in {
            "intel-level-zero-gpu",
            "level-zero",
            "level-zero-dev",
            "libze1",
            "libze-dev",
        }
    ]

    level_zero_libraries = [
        name
        for name in installed_libs
        if name.startswith("libze")
    ]

    bridge_libraries = []
    wsl_bridge = checks.get("wsl_docker_bridge", {})
    required_bridge_libs = wsl_bridge.get("required_libs", {})
    for lib_name, exists in required_bridge_libs.items():
        if exists:
            bridge_libraries.append(lib_name)

    layers = [
        {
            "order": 1,
            "name": "Intel OpenCL runtime",
            "description": "Intel GPU를 OpenCL API로 사용할 수 있게 하는 runtime 계층",
            "installed_packages": opencl_packages,
            "runtime_libraries": opencl_libraries,
            "checks": {
                "opencl": checks.get("opencl", {}).get("ok"),
            },
        },
        {
            "order": 2,
            "name": "Intel Level Zero runtime",
            "description": "Intel GPU를 Level Zero API로 사용할 수 있게 하는 runtime 계층",
            "installed_packages": level_zero_packages,
            "runtime_libraries": level_zero_libraries,
            "checks": {
                "level_zero": checks.get("level_zero", {}).get("ok"),
            },
        },
        {
            "order": 3,
            "name": "Windows / WSL2 / Docker bridge",
            "description": "Windows GPU를 WSL2 Docker 컨테이너에 노출하는 연결 계층",
            "installed_packages": [],
            "runtime_libraries": bridge_libraries,
            "items": {
                "device": "/dev/dxg" if wsl_bridge.get("dxg_exists") else None,
                "wsl_lib_path": "/usr/lib/wsl/lib" if wsl_bridge.get("wsl_lib_exists") else None,
                "ld_library_path": checks.get("environment", {}).get("LD_LIBRARY_PATH"),
            },
            "checks": {
                "wsl_docker_bridge": wsl_bridge.get("ok"),
            },
        },
        {
            "order": 4,
            "name": "Intel GPU hardware",
            "description": "물리 Intel GPU 장치. 계층도 기준 맨 아래 계층",
            "installed_packages": [],
            "runtime_libraries": [],
            "checks": {
                "detected_by_opencl": checks.get("opencl", {}).get("ok"),
                "detected_by_level_zero": checks.get("level_zero", {}).get("ok"),
            },
        },
    ]

    diagram_lines = [
        f"Intel OpenCL runtime (available: {checks.get('opencl', {}).get('ok')})",
        f"├─ installed packages: {format_items(opencl_packages)}",
        f"└─ runtime libraries: {format_items(opencl_libraries)}",
        "↓",
        f"Intel Level Zero runtime (available: {checks.get('level_zero', {}).get('ok')})",
        f"├─ installed packages: {format_items(level_zero_packages)}",
        f"└─ runtime libraries: {format_items(level_zero_libraries)}",
        "↓",
        "Windows / WSL2 / Docker bridge",
        "├─ device: /dev/dxg",
        "├─ wsl lib path: /usr/lib/wsl/lib",
        f"└─ bridge libraries: {format_items(bridge_libraries)}",
        "↓",
        f"Intel GPU hardware (available: {checks.get('opencl', {}).get('ok') and checks.get('level_zero', {}).get('ok')})",
        f"CPU hardware (available: {os.cpu_count() is not None and os.cpu_count() > 0}, logical_count: {os.cpu_count()})",
    ]

    return {
        "diagram": "\n".join(diagram_lines),
        "layers": layers,
    }


def print_installed_library_architecture(architecture: dict) -> None:
    print("=" * 90)
    print("Installed Intel GPU Runtime Architecture")
    print("=" * 90)
    print(architecture["diagram"])
    print("=" * 90)


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


def check_cpu_availability() -> dict:
    logical_count = os.cpu_count()

    result = {
        "ok": logical_count is not None and logical_count > 0,
        "available": logical_count is not None and logical_count > 0,
        "logical_cpu_count": logical_count,
        "platform_machine": platform.machine(),
        "processor": platform.processor(),
        "reason": None,
    }

    if result["ok"]:
        result["reason"] = "CPU is available in the container"
    else:
        result["reason"] = "CPU count could not be detected"

    return result


def build_availability(checks: dict) -> dict:
    cpu = check_cpu_availability()

    opencl_ok = checks.get("opencl", {}).get("ok", False)
    level_zero_ok = checks.get("level_zero", {}).get("ok", False)
    bridge_ok = checks.get("wsl_docker_bridge", {}).get("ok", False)
    packages_ok = checks.get("intel_runtime_packages", {}).get("ok", False)

    intel_gpu_base_usable = bridge_ok and packages_ok and opencl_ok and level_zero_ok

    intel_gpu = {
        "available": intel_gpu_base_usable,
        "opencl_available": opencl_ok,
        "level_zero_available": level_zero_ok,
        "wsl_docker_bridge_available": bridge_ok,
        "runtime_packages_available": packages_ok,
        "reason": (
            "Intel GPU is available through OpenCL and Level Zero runtime"
            if intel_gpu_base_usable
            else "Intel GPU base is not fully available; check failed layer"
        ),
    }

    return {
        "cpu": cpu,
        "intel_gpu": intel_gpu,
    }


def summarize(checks: dict) -> dict:
    bridge_ok = checks["wsl_docker_bridge"]["ok"]
    packages_ok = checks["intel_runtime_packages"]["ok"]
    opencl_ok = checks["opencl"]["ok"]
    level_zero_ok = checks["level_zero"]["ok"]
    base_ok = bridge_ok and packages_ok and opencl_ok and level_zero_ok
    if base_ok:
        next_focus = "ready_for_framework"
        diagnosis = "Intel XPU Docker base 사용 가능: OpenCL과 Level Zero runtime까지 확인됨"
        actions = ["필요에 따라 PyTorch XPU, TensorFlow XPU, OpenVINO, SYCL 파생 이미지를 추가"]
    elif bridge_ok and packages_ok and opencl_ok and not level_zero_ok:
        next_focus = "level_zero"
        diagnosis = "OpenCL은 성공했지만 Level Zero 장치 확인이 실패함"
        actions = ["direct_ctypes.zeInit_result 확인", "driver_count / total_device_count 확인", "libze_loader / libze_intel_gpu 확인"]
    elif bridge_ok and packages_ok and not opencl_ok:
        next_focus = "opencl"
        diagnosis = "WSL2/Docker bridge와 패키지는 있으나 OpenCL에서 GPU 확인 실패"
        actions = ["clinfo 결과 확인", "intel-opencl-icd 설치 확인", "Windows Intel GPU driver 확인"]
    else:
        next_focus = "wsl_docker_bridge_or_packages"
        diagnosis = "WSL2/Docker bridge 또는 Intel runtime 패키지 단계에서 실패"
        actions = ["/dev/dxg", "/usr/lib/wsl/lib", "LD_LIBRARY_PATH", "필수 패키지 설치 확인"]
    return {
        "intel_xpu_base_usable": base_ok,
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
    }
    checks["environment"] = check_environment()
    architecture = build_installed_library_architecture(checks)
    availability = build_availability(checks)
    print_installed_library_architecture(architecture)

    report = {
        "started_at": started_at,
        "finished_at": now_kst(),
        "timezone": "Asia/Seoul",
        "scope": "Intel OpenCL runtime -> Intel Level Zero runtime -> Windows/WSL2/Docker bridge -> Intel GPU hardware",
        "architecture": architecture,
        "availability": availability,
        "environment": checks["environment"],
        "checks": checks,
        "summary": summarize(checks),
    }
    output_path = save_report("intel_xpu_base_check", report)
    print("=" * 90)
    print("Intel XPU Check Summary")
    print("=" * 90)
    print("wsl_docker_bridge:", checks["wsl_docker_bridge"]["ok"])
    print("intel_runtime_packages:", checks["intel_runtime_packages"]["ok"])
    print("opencl:", checks["opencl"]["ok"])
    print("level_zero:", checks["level_zero"]["ok"])
    print("cpu_available:", availability["cpu"]["available"])
    print("cpu_logical_count:", availability["cpu"]["logical_cpu_count"])
    print("intel_gpu_available:", availability["intel_gpu"]["available"])
    print("intel_gpu_opencl_available:", availability["intel_gpu"]["opencl_available"])
    print("intel_gpu_level_zero_available:", availability["intel_gpu"]["level_zero_available"])
    print("intel_xpu_base_usable:", report["summary"]["intel_xpu_base_usable"])
    print("next_focus:", report["summary"]["next_focus"])
    print("diagnosis:", report["summary"]["diagnosis"])
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
