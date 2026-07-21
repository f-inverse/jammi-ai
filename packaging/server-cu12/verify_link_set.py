#!/usr/bin/env python3
"""Fail the cu12 wheel build if the binary needs a CUDA library the wheel does
not deliver.

The `-cu12` wheel ships no CUDA `.so`s itself: it depends on `nvidia-*-cu12`
wheels and the console-script shim (`jammi_server/_entry.py`) prepends each
present component's `lib/` to `LD_LIBRARY_PATH` before `execve`. That contract is
only correct if every `DT_NEEDED` CUDA library the binary carries is provided by
one of those components — a library the shim does not cover would resolve
nowhere and the binary would fail to `exec` on a user's machine, long after CI is
green. cudarc links `dynamic-linking`, so the set is fixed at build time and
knowable here.

This check reads the binary's `DT_NEEDED` entries (`readelf -d`), and for every
CUDA library among them (SONAME beginning `libcu` or `libnv`) requires that it be
either driver-provided (the host NVIDIA driver, never bundled) or delivered by a
declared wheel component. A newly-required library (e.g. a candle bump that pulls
in cuSPARSE) fails the build with the exact `nvidia-*-cu12` dependency and
`_CUDA_COMPONENTS` entry to add — never a runtime surprise.

Usage: verify_link_set.py <path-to-jammi-server-binary>
"""

import re
import subprocess
import sys

# CUDA libraries delivered by each declared wheel component. The keys are the
# `nvidia` namespace subdirectories `_entry.py`'s `_CUDA_COMPONENTS` prepends to
# `LD_LIBRARY_PATH`; the values are the SONAME stems each `nvidia-*-cu12` wheel
# installs under `nvidia/<component>/lib/`. Keep this in lockstep with
# `_CUDA_COMPONENTS` in `jammi_server/_entry.py` and the `nvidia-*-cu12`
# dependency pins in `pyproject.toml`: the three describe one contract.
COVERED = {
    "cuda_runtime": {"libcudart"},              # nvidia-cuda-runtime-cu12
    "cublas": {"libcublas", "libcublasLt"},     # nvidia-cublas-cu12 (cublasLt rides along)
    "curand": {"libcurand"},                    # nvidia-curand-cu12
    "cuda_nvrtc": {"libnvrtc", "libnvrtc-builtins"},  # nvidia-cuda-nvrtc-cu12
}

# CUDA libraries that come from the host NVIDIA driver, never from a wheel. The
# driver API (`libcuda.so.1`) and NVVM/driver-side JIT (`libnvidia-*`) are
# installed with the driver the user already runs; bundling them would be wrong.
DRIVER_PROVIDED = {"libcuda", "libnvidia-ptxjitcompiler", "libnvidia-nvvm"}


def needed_libs(binary: str) -> list:
    """The binary's `DT_NEEDED` SONAMEs, in link order."""
    out = subprocess.run(
        ["readelf", "-d", binary], capture_output=True, text=True, check=True
    ).stdout
    # Lines read: `0x... (NEEDED)  Shared library: [libcudart.so.12]`
    return re.findall(r"\(NEEDED\).*\[([^\]]+)\]", out)


def soname_stem(soname: str) -> str:
    """`libcublasLt.so.12` -> `libcublasLt` (drop `.so` and everything after)."""
    return soname.split(".so", 1)[0]


def check(binary: str) -> int:
    covered_stems = {stem for stems in COVERED.values() for stem in stems}
    uncovered = []
    for soname in needed_libs(binary):
        stem = soname_stem(soname)
        # Only CUDA/NVIDIA libraries are the wheel's responsibility; system libs
        # (libc, libm, libstdc++, libgcc_s, ld-linux, …) are the platform's.
        if not (stem.startswith("libcu") or stem.startswith("libnv")):
            continue
        if stem in DRIVER_PROVIDED or stem in covered_stems:
            continue
        uncovered.append(soname)

    if uncovered:
        print(
            "cu12 link-set check FAILED — the binary needs CUDA libraries the "
            "wheel does not deliver:",
            file=sys.stderr,
        )
        for soname in uncovered:
            print(f"  - {soname}", file=sys.stderr)
        print(
            "\nAdd the owning `nvidia-*-cu12` wheel to `pyproject.toml`, its "
            "component dir to `_CUDA_COMPONENTS` in `jammi_server/_entry.py`, and "
            "its SONAME stem(s) to `COVERED` here. If it is genuinely "
            "driver-provided, add it to `DRIVER_PROVIDED` instead.",
            file=sys.stderr,
        )
        return 1

    print("cu12 link-set check OK — every CUDA DT_NEEDED library is covered.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <path-to-jammi-server-binary>")
    sys.exit(check(sys.argv[1]))
