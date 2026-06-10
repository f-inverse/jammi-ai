"""Console-script entry point for the CPU `jammi-server` wheel.

The wheel bundles the compiled binary at `jammi_server/bin/jammi-server`. The
`jammi-server` console script resolves here; `main()` hands the process over to
that binary with `os.execv`, replacing the Python interpreter so signals, exit
codes, and stdio pass through unmediated.
"""

import os
import sys


def _binary_path() -> str:
    return os.path.join(os.path.dirname(__file__), "bin", "jammi-server")


def main() -> None:
    binary = _binary_path()
    if not os.path.exists(binary):
        sys.exit(
            f"jammi-server: bundled binary not found at {binary}; "
            "the wheel was built without its compiled binary"
        )
    # Replace this process with the server binary. argv[0] is the binary itself;
    # the caller's arguments follow. os.execv does not return on success.
    os.execv(binary, [binary, *sys.argv[1:]])
