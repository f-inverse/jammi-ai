"""Run a script's real command line on a host that lacks one Python package.

The arm of a script that handles a missing package is a property of the
script, not of the host testing it: the child process here sees the package
as absent whatever this host has installed.
"""

from __future__ import annotations

import subprocess
import sys

# `sys.modules[name] = None` makes every `import name` raise ImportError.
_CHILD = (
    "import runpy, sys; "
    "sys.modules[sys.argv[1]] = None; "
    "sys.argv = sys.argv[2:]; "
    "runpy.run_path(sys.argv[0], run_name='__main__')"
)


def run_script_without(package: str, script: str, *args: str) -> subprocess.CompletedProcess[str]:
    """`script args...` as `__main__` in a child where `import package` fails."""
    return subprocess.run(
        [sys.executable, "-c", _CHILD, package, script, *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
