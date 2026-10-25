"""Telemetria do hardware de um pod com GPU, para o painel Dev (Engine Observatory).

  python3 scripts/collect_telemetry.py runs/<run> [intervalo_s]

A cada intervalo (padrão 10 s) grava <run>/system_telemetry.json (gravação atômica) com o que é
do container, não da máquina física: CPU e RAM vêm do cgroup (v1 ou v2), porque /proc/stat e
`free` mostram o host inteiro num pod. GPU (uso, VRAM, temperatura, potência) vem do nvidia-smi.
Guarda as últimas 60 amostras para o painel desenhar a tendência. Sai sozinho quando o run tiver
sft-report.json, ou com Ctrl-C.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CG = Path("/sys/fs/cgroup")
HISTORY = 60


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def cpu_usage_ns():
    """CPU consumida pelo container desde o início, em ns."""
    v1 = read(CG / "cpuacct/cpuacct.usage")
    if v1:
        return int(v1)
    stat = read(CG / "cpu.stat")  # v2: usage_usec
    if stat:
        for line in stat.splitlines():
            key, value = line.split()
            if key == "usage_usec":
                return int(value) * 1000
    return None


def cpu_limit():
    """vCPUs alugadas (quota ÷ período do cgroup); sem limite, os núcleos visíveis."""
    quota, period = read(CG / "cpu/cpu.cfs_quota_us"), read(CG / "cpu/cpu.cfs_period_us")
    if quota and period and int(quota) > 0:
        return int(quota) / int(period)
    v2 = read(CG / "cpu.max")
    if v2 and not v2.startswith("max"):
        q, p = v2.split()
        return int(q) / int(p)
    return os.cpu_count()


def memory():
    used = read(CG / "memory/memory.usage_in_bytes") or read(CG / "memory.current")
    limit = read(CG / "memory/memory.limit_in_bytes") or read(CG / "memory.max")
    if used is None:
        return None, None
    total = int(limit) if limit and limit != "max" and int(limit) < 1 << 60 else None
    return int(used), total


def gpus():
    fields = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    result = []
    for line in out.strip().splitlines():
        name, util, used, total, temp, power, limit = [v.strip() for v in line.split(",")]
        number = lambda v: float(v) if v not in ("[N/A]", "N/A", "") else None  # noqa: E731
        result.append({"name": name, "utilization_pct": number(util), "vram_used_mib": number(used),
                       "vram_total_mib": number(total), "temperature_c": number(temp),
                       "power_w": number(power), "power_limit_w": number(limit)})
    return result


def uptime_s():
    try:
        out = subprocess.run(["ps", "-o", "etimes=", "-p", "1"], capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def main(run, interval=10.0):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    target, history = run / "system_telemetry.json", []
    last_cpu, last_t = cpu_usage_ns(), time.monotonic()
    while True:
        time.sleep(interval)
        cpu, now = cpu_usage_ns(), time.monotonic()
        limit = cpu_limit()
        cpu_pct = None
        if cpu is not None and last_cpu is not None and limit:
            cpu_pct = round(100 * (cpu - last_cpu) / ((now - last_t) * 1e9 * limit), 1)
        last_cpu, last_t = cpu, now
        used, total = memory()
        disk = shutil.disk_usage("/")
        sample = {
            "at": datetime.now(timezone.utc).isoformat(), "uptime_s": uptime_s(),
            "cpu": {"usage_pct": cpu_pct, "vcpus": limit},
            "ram": {"used_bytes": used, "total_bytes": total},
            "disk": {"used_bytes": disk.used, "total_bytes": disk.total},
            "gpus": gpus(),
        }
        history = (history + [{"at": sample["at"], "cpu_pct": cpu_pct,
                               "gpu_pct": sample["gpus"][0]["utilization_pct"] if sample["gpus"] else None}])[-HISTORY:]
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({**sample, "history": history, "source": "pod"}))
        os.replace(temporary, target)
        if (run / "sft-report.json").exists():
            return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 10.0))
