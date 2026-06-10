"""Console-script entry point for the CUDA (cu12) `jammi-server` wheel.

candle-core 0.9.2 pins cudarc with `dynamic-linking`, so the bundled binary has
hard `DT_NEEDED` entries for libcudart / libcublas / libcublasLt / libcurand /
libnvrtc. The dynamic loader resolves those *before* `main` runs — so the binary
will not exec at all unless they are on the loader path (CPU fallback is a
runtime decision made by the binary itself, long after load).

The `nvidia-*-cu12` dependency wheels install those libraries under
`nvidia/<component>/lib/`. `main()` discovers each present component's `lib/`
directory, prepends them to `LD_LIBRARY_PATH`, then hands the process over to the
binary with `os.execve`, replacing the Python interpreter so signals, exit codes,
and stdio pass through unmediated.
"""

import os
import sys

# Component subdirectories of the `nvidia` namespace package that ship the CUDA
# runtime libraries the binary links against (cudarc 0.19.3's
# cuda_runtime / cublas+cublasLt / curand / nvrtc feature set). cublasLt ships
# inside the `cublas` component, so it needs no separate entry.
_CUDA_COMPONENTS = ("cuda_runtime", "cublas", "curand", "cuda_nvrtc")


def _binary_path() -> str:
    return os.path.join(os.path.dirname(__file__), "bin", "jammi-server")


def _cuda_lib_dirs() -> list:
    """The `lib/` directory of each installed `nvidia` CUDA component, in link order."""
    try:
        import nvidia
    except ImportError:
        return []
    # The `nvidia` namespace package may span multiple site-packages locations
    # (one per `nvidia-*-cu12` wheel); walk every path entry it exposes.
    roots = list(getattr(nvidia, "__path__", []))
    dirs = []
    for root in roots:
        for component in _CUDA_COMPONENTS:
            lib = os.path.join(root, component, "lib")
            if os.path.isdir(lib) and lib not in dirs:
                dirs.append(lib)
    return dirs


def main() -> None:
    binary = _binary_path()
    if not os.path.exists(binary):
        sys.exit(
            f"jammi-server: bundled binary not found at {binary}; "
            "the wheel was built without its compiled binary"
        )

    env = dict(os.environ)
    lib_dirs = _cuda_lib_dirs()
    if lib_dirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        prefix = os.pathsep.join(lib_dirs)
        env["LD_LIBRARY_PATH"] = f"{prefix}{os.pathsep}{existing}" if existing else prefix

    # Replace this process with the server binary. argv[0] is the binary itself;
    # the caller's arguments follow. os.execve does not return on success.
    os.execve(binary, [binary, *sys.argv[1:]], env)
