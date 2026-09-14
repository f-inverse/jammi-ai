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

This check reads the binary's `DT_NEEDED` entries (`readelf -d`) and classifies
EVERY one of them. Extracting NONE of them is itself a failure — an empty list
is not "everything is classified", it is "nothing was read", and the two must
never print the same line. A soname is acceptable only if it is on the `PLATFORM`
allowlist (a base-image library the manylinux platform tag already promises), in
`DRIVER_PROVIDED` (the host NVIDIA driver, never bundled), or in `COVERED` (a
declared `nvidia-*-cu12` wheel component). Anything else FAILS the build, naming
the soname.

The classification is an allowlist rather than a `libcu`/`libnv` prefix test
because the prefix test fails OPEN on exactly the libraries this file exists to
catch. `libnccl.so.2` — the soname a `candle-core/nccl` build adds — begins with
neither prefix, so under the prefix test the wheel would print "link-set check
OK" while shipping a binary that cannot `exec` anywhere NCCL is absent. Every
future library is in the same position: a new dependency is unclassified until a
human classifies it, and unclassified is a FAIL. That FAIL is the whole point,
including when the new library turns out to be a harmless platform one — adding
it to `PLATFORM` is a one-line, reviewed decision, whereas a silent skip is
nobody's decision at all.

`PLATFORM` is not a guess: it is the non-CUDA half of the actual `DT_NEEDED`
list of the binary this check runs on (`target/release/jammi-server`, built by
`_pypi-server.yml`'s `build-binary` job in the manylinux_2_28 CUDA image and
verified at `_pypi-server.yml:144`), read off the `server-cu12-binary` artifact
of run 34717957779. Note `libmvec.so.1`, glibc's vector-math library: it is on
that list, and a hand-written "libc, libm, libstdc++, libgcc_s, …" allowlist
would have reddened the lane on it.

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
    # nvidia-nccl-cu12, which installs exactly one object,
    # `nvidia/nccl/lib/libnccl.so.2` (measured: the central directory of
    # `nvidia_nccl_cu12-2.23.4-py3-none-manylinux2014_x86_64.whl`). The binary
    # links it because `jammi-ai`'s `cuda` feature includes `candle-core/nccl`
    # and cudarc's build script then emits `cargo:rustc-link-lib=dylib=nccl`.
    "nccl": {"libnccl"},
}

# CUDA libraries that come from the host NVIDIA driver, never from a wheel. The
# driver API (`libcuda.so.1`) and NVVM/driver-side JIT (`libnvidia-*`) are
# installed with the driver the user already runs; bundling them would be wrong.
DRIVER_PROVIDED = {"libcuda", "libnvidia-ptxjitcompiler", "libnvidia-nvvm"}

# Base-image libraries the `manylinux_2_28_x86_64` platform tag itself
# promises — the wheel neither ships nor declares them. This is the measured
# non-CUDA `DT_NEEDED` set of the binary this check runs on (see the module
# doc), not a general list of "system libraries": a soname that is not here is
# unclassified, and unclassified is a FAIL that a human resolves by adding it
# here, to `COVERED`, or to `DRIVER_PROVIDED`.
PLATFORM = {
    "libc",
    "libdl",
    "libgcc_s",
    "libm",
    "libmvec",  # glibc's vector math library; linked by the release build
    "libpthread",
    "librt",
    "libstdc++",
}

# The dynamic loader itself. Its soname carries the architecture
# (`ld-linux-x86-64.so.2` on x86-64, `ld-linux-aarch64.so.1` on arm64), so it is
# matched by prefix rather than by exact stem — the one prefix rule here, and it
# admits no library that is not the loader.
PLATFORM_PREFIXES = ("ld-linux",)


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


def is_platform(stem: str) -> bool:
    """Base-image library the platform tag promises — never the wheel's."""
    return stem in PLATFORM or any(stem.startswith(p) for p in PLATFORM_PREFIXES)


def classify(soname: str) -> str:
    """`platform` | `driver` | `covered` | `unclassified` for one DT_NEEDED
    soname. Every entry lands in exactly one bucket, and `unclassified` is the
    fail bucket: nothing is skipped for looking unlike a CUDA library."""
    stem = soname_stem(soname)
    if is_platform(stem):
        return "platform"
    if stem in DRIVER_PROVIDED:
        return "driver"
    if stem in {s for stems in COVERED.values() for s in stems}:
        return "covered"
    return "unclassified"


def check(binary: str) -> int:
    needed = needed_libs(binary)

    # Empty extraction is a FAILURE, not a pass. Classifying zero sonames
    # says nothing about what the wheel must deliver, but the old code
    # printed the same "OK" it prints for a fully classified binary. The cu12
    # `jammi-server` links `libcudart` at minimum (cudarc's `dynamic-linking`),
    # so an empty list means the file is not the dynamically-linked ELF this
    # check expects, or `readelf -d`'s output no longer matches the
    # `(NEEDED) ... [soname]` shape `needed_libs` parses -- and the second
    # case is exactly the one that would hide an `libnccl.so.2` this check
    # exists to catch. The sibling guard `ci/scripts/assert_glibc_floor.sh`
    # fails closed on the same shape, naming both the binary and the tool.
    if not needed:
        print(
            f"cu12 link-set check FAILED -- readelf -d extracted no DT_NEEDED entries from "
            f"{binary}: there is no link set to classify, so this check can assert nothing "
            "about what the wheel must deliver. A cu12 jammi-server binary always needs at "
            "least libcudart, so either that path is not the dynamically-linked ELF this "
            "check expects (wrong artifact, or a build with no dynamic section) or readelf's "
            "output no longer matches the '(NEEDED) Shared library: [soname]' shape "
            "needed_libs parses. Re-run 'readelf -d' on that path by hand and compare.",
            file=sys.stderr,
        )
        return 1

    unclassified = [s for s in needed if classify(s) == "unclassified"]

    if unclassified:
        print(
            "cu12 link-set check FAILED — the binary needs libraries the wheel "
            "neither delivers nor classifies:",
            file=sys.stderr,
        )
        for soname in unclassified:
            print(f"  - {soname}", file=sys.stderr)
        print(
            "\nClassify each one. If a wheel component delivers it: add the "
            "owning `nvidia-*-cu12` wheel to `pyproject.toml`, its component dir "
            "to `_CUDA_COMPONENTS` in `jammi_server/_entry.py`, and its SONAME "
            "stem(s) to `COVERED` here. If the host NVIDIA driver provides it, "
            "add it to `DRIVER_PROVIDED`. If it is a base-image library the "
            "manylinux platform tag already promises, add it to `PLATFORM`. "
            "Leaving it unclassified is the one thing that is not an option: an "
            "unresolvable `DT_NEEDED` entry fails at `execve` on a user's "
            "machine, with no CI signal at all.",
            file=sys.stderr,
        )
        return 1

    print("cu12 link-set check OK — every DT_NEEDED library is classified and covered.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <path-to-jammi-server-binary>")
    sys.exit(check(sys.argv[1]))
