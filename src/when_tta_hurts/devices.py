"""Explicit device selection and environment capture.

Never silently move only part of a model/tensor pipeline to another device:
callers get back a single torch.device and are expected to .to(device) the
whole model and every batch consistently.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass, field

import torch


class DeviceUnavailableError(RuntimeError):
    """Raised when an explicitly requested device is not usable."""


def _mps_available() -> bool:
    return torch.backends.mps.is_built() and torch.backends.mps.is_available()


def select_device(requested: str = "auto") -> torch.device:
    """Resolve a device selection of 'auto', 'mps', or 'cpu'.

    'mps' fails loudly (DeviceUnavailableError) if MPS is not built/available.
    'auto' prefers MPS but falls back to CPU with a printed warning.
    """
    requested = requested.lower()
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "mps":
        if not _mps_available():
            raise DeviceUnavailableError(
                "device='mps' was explicitly requested but MPS is not built "
                "and available in this PyTorch install (torch.backends.mps."
                f"is_built()={torch.backends.mps.is_built()}, "
                f"is_available()={torch.backends.mps.is_available()})."
            )
        return torch.device("mps")

    if requested == "auto":
        if _mps_available():
            return torch.device("mps")
        print(
            "WARNING: device='auto' requested but MPS is not available on this "
            "machine; falling back to CPU. Training/inference will be slower.",
            file=sys.stderr,
        )
        return torch.device("cpu")

    raise ValueError(f"Unknown device selection '{requested}'; expected auto/mps/cpu.")


def _macos_version() -> str:
    try:
        return subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip()
    except Exception:
        return "unknown"


def _chip_brand() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return "unknown"


@dataclass
class EnvironmentManifest:
    """Machine-readable snapshot of the environment a run executed in."""

    device: str
    mps_built: bool
    mps_available: bool
    macos_version: str
    chip: str
    python_version: str
    platform_machine: str
    torch_version: str
    torchvision_version: str
    package_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "mps_built": self.mps_built,
            "mps_available": self.mps_available,
            "macos_version": self.macos_version,
            "chip": self.chip,
            "python_version": self.python_version,
            "platform_machine": self.platform_machine,
            "torch_version": self.torch_version,
            "torchvision_version": self.torchvision_version,
            "package_versions": self.package_versions,
        }


def capture_environment(device: torch.device) -> EnvironmentManifest:
    import torchvision

    package_versions: dict[str, str] = {}
    for mod_name in ("numpy", "pandas", "scipy", "sklearn", "medmnist", "yaml", "kornia"):
        try:
            mod = __import__(mod_name)
            package_versions[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            package_versions[mod_name] = "not installed"

    return EnvironmentManifest(
        device=str(device),
        mps_built=torch.backends.mps.is_built(),
        mps_available=torch.backends.mps.is_available(),
        macos_version=_macos_version(),
        chip=_chip_brand(),
        python_version=sys.version.split()[0],
        platform_machine=platform.machine(),
        torch_version=torch.__version__,
        torchvision_version=torchvision.__version__,
        package_versions=package_versions,
    )
