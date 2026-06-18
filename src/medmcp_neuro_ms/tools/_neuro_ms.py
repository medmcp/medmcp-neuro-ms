"""Shared internal utilities for medmcp-neuro-ms tools."""

import os
import shutil
import subprocess
from pathlib import Path

# LST-AI lives in its own venv (its torch + onnxruntime-gpu are kept apart from the
# lightweight mcp wrapper env); the image installs the `lst` CLI here.
_LST_VENV = Path(os.environ.get("LST_AI_VENV", "/opt/lst-venv"))


def nii_stem(path: Path) -> str:
    """Return the NIfTI stem, stripping a .nii.gz or .nii suffix."""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def find_lst() -> str:
    """Locate the LST-AI ``lst`` CLI.

    Checks $LST_AI_BIN, then the isolated venv (/opt/lst-venv/bin/lst), then PATH.

    Raises:
        RuntimeError: If the lst CLI cannot be found.
    """
    override = os.environ.get("LST_AI_BIN")
    if override:
        return override
    candidate = _LST_VENV / "bin" / "lst"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    on_path = shutil.which("lst")
    if on_path:
        return on_path
    raise RuntimeError(
        "LST-AI 'lst' CLI not found. The medmcp-neuro-ms image installs it at "
        "/opt/lst-venv/bin/lst; set $LST_AI_BIN to override."
    )


def gpu_present() -> bool:
    """Return True if a CUDA GPU is visible, without importing torch.

    The wrapper venv has no torch, so probe the device nodes a CDI GPU container
    exposes (/dev/nvidia0…) and fall back to ``nvidia-smi -L``.
    """
    if any(Path("/dev").glob("nvidia[0-9]*")):
        return True
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and "GPU" in out.stdout


def resolve_device(device: str) -> str:
    """Map a requested device to an LST-AI device token.

    LST-AI takes a GPU id ('0', '1', …) or 'cpu'. 'auto' picks the first GPU when one
    is visible, else 'cpu'.
    """
    if device != "auto":
        return device
    return "0" if gpu_present() else "cpu"


def lst_subprocess_env() -> dict[str, str]:
    """Build the environment for the ``lst`` subprocess.

    LST-AI shells out to ``hd-bet`` (and runs in its own venv), so the venv's bin dir
    must be on PATH. onnxruntime-gpu also needs the CUDA runtime/cuDNN/cuBLAS shared
    libs; in /opt/lst-venv these come from torch's bundled ``nvidia-*-cu12`` packages
    (and torch/lib), so expose them on LD_LIBRARY_PATH for the CUDA execution provider.
    The CA bundle env (set on the image) is inherited for any first-run downloads.
    """
    env = dict(os.environ)
    bin_dir = _LST_VENV / "bin"
    if bin_dir.is_dir():
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}".rstrip(":")
    lib_dirs = [str(p) for p in _LST_VENV.glob("lib/python*/site-packages/nvidia/*/lib")]
    lib_dirs += [str(p) for p in _LST_VENV.glob("lib/python*/site-packages/torch/lib")]
    if lib_dirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))
    return env
