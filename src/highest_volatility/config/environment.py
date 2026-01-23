"""Environment defaults for Windows-specific paths."""

from __future__ import annotations

import os
from pathlib import Path


def ensure_windows_environment() -> None:
    """Populate missing Windows path environment variables."""

    if os.name != "nt":
        return
    drive = Path.home().drive or Path.cwd().drive
    if not drive:
        return
    os.environ.setdefault("SystemDrive", drive)
    os.environ.setdefault("ProgramData", str(Path(drive) / "ProgramData"))
    os.environ.setdefault("ProgramFiles", str(Path(drive) / "Program Files"))
    os.environ.setdefault("ProgramFiles(x86)", str(Path(drive) / "Program Files (x86)"))
