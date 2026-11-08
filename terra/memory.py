"""Make sure only one model is ever in memory on this Mac.

Qwen in bf16 takes about 15 GB and Gemma 4 31B at 4 bits about 17.5 GB, out
of 36 GB of unified memory. Two at once would push the machine into swap.

Two guards:
* A lock file (models/.model_in_memory.lock) records which process holds a
  model. Any other process that tries to load one is refused with a clear
  message; a lock left by a process that has died is ignored.
* Before loading, we read the free memory from `vm_stat` and print a warning
  if it looks too low for the model.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import subprocess
import time
from pathlib import Path

from . import registry


class ModelBusyError(RuntimeError):
    pass


def lock_path() -> Path:
    return registry.artefact_root() / "models" / ".model_in_memory.lock"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def holder() -> dict | None:
    """The live process holding a model, if any (other than us)."""
    p = lock_path()
    if not p.exists():
        return None
    try:
        info = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    pid = int(info.get("pid", -1))
    if pid == os.getpid() or not _alive(pid):
        return None
    return info


def claim(key: str) -> None:
    """Record that this process is about to hold `key`; refuse if another does."""
    other = holder()
    if other is not None:
        raise ModelBusyError(
            f"Another process (pid {other['pid']}) already has {other.get('key')} loaded. "
            "Stop it (the server or a script) before loading another model.")
    p = lock_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"pid": os.getpid(), "key": key, "since": time.strftime("%Y-%m-%d %H:%M:%S")}))


def release() -> None:
    p = lock_path()
    try:
        if p.exists() and int(json.loads(p.read_text()).get("pid", -1)) == os.getpid():
            p.unlink()
    except (OSError, ValueError):
        pass


atexit.register(release)


def available_memory_gb() -> float | None:
    """Free plus reclaimable memory from macOS's vm_stat, or None if unknown."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"page size of (\d+) bytes", out)
    if not m:
        return None
    page = int(m.group(1))
    pages = 0
    for label in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        mm = re.search(label + r":\s+(\d+)", out)
        if mm:
            pages += int(mm.group(1))
    return pages * page / 1e9


def check_before_load(key: str) -> None:
    """Refuse if another process holds a model; warn if memory looks short."""
    claim(key)
    if registry.fake_mode():
        return
    need = registry.spec(key).approx_gb
    free = available_memory_gb()
    if free is not None and free < need:
        print(f"[memory] warning: about {free:.1f} GB looks free but {key} needs about {need:.1f} GB. "
              "Close other large programs (Ollama, games, browsers with many tabs) if loading is slow.")
    if registry.spec(key).backend == "mlx":
        limit = metal_working_set_gb()
        if limit is not None and need > limit:
            print(f"[memory] warning: {key} needs about {need:.1f} GB but Metal recommends at most {limit:.1f} GB "
                  "for the GPU on this Mac; expect swapping.")


def metal_working_set_gb() -> float | None:
    """Metal's recommended ceiling for GPU memory on this Mac (None if unknown).

    MLX keeps the weights in GPU-visible memory; above this ceiling macOS
    starts paging and everything slows down sharply.
    """
    try:
        import mlx.core as mx
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        return float(info["max_recommended_working_set_size"]) / 1e9
    except Exception:
        return None
