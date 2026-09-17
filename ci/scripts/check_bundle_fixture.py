#!/usr/bin/env python3
"""Refuses a PROVISIONAL cu12 loader-report fixture on the merge path, with
NO environment-variable escape.

`ci/scripts/fixtures/cu12_loader_report_real.txt` must hold the VERBATIM
`ldd` report `release-binaries.yml`'s `server-cu12-build` job captures from
the real cu12 `jammi-server` binary and its real staged `lib/` (the build
container: CUDA toolkit present, NVIDIA driver absent — see that job's
`cu12-loader-report` artifact and this fixture's own header). Until the lead
downloads that artifact and commits it over the placeholder, the fixture
carries a CLEARLY-LABELLED `# captured: pending` header instead — and T4(3)'s
own property ("the loader's self-named line is parsed as a resolved platform
entry, so a CORRECT [real] stage passes") is UNPROVEN, not merely untested,
while that header stands.

`ci/scripts/test_bundle_cuda_libs.sh` also refuses to run its own T4(3)
integration check against a provisional fixture, but ONLY when the caller
sets `BUNDLE_FIXTURE_PROVISIONAL=1` does it skip that refusal and run the
REST of the suite anyway — a knob meant for a developer iterating on the
other 100+ checks in that file locally, before the real report exists. This
gate is that knob's floor: it is wired into `ci.yml`'s Guard matrix with NO
environment variable, no flag, and no allowlist entry that could suppress
it — the merge path stays red on this file until the real report replaces
the placeholder, full stop.

Run: `python3 ci/scripts/check_bundle_fixture.py`
Self-test: `python3 ci/scripts/check_bundle_fixture.py --self-test`
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "ci" / "scripts" / "fixtures" / "cu12_loader_report_real.txt"
MARKER = "# captured: pending"


def check(fixture: Path) -> int:
    if not fixture.exists():
        print(
            f"check_bundle_fixture.py: FAILED -- no fixture at {fixture}. "
            "ci/scripts/test_bundle_cuda_libs.sh's T4(3) check has nothing to read.",
            file=sys.stderr,
        )
        return 1

    text = fixture.read_text()
    if not text.strip():
        print(
            f"check_bundle_fixture.py: FAILED -- {fixture} is empty. "
            "ci/scripts/test_bundle_cuda_libs.sh's T4(3) check has nothing to read.",
            file=sys.stderr,
        )
        return 1

    first_line = text.splitlines()[0].strip()

    # A PREFIX match, not an exact one, and not a substring-anywhere scan
    # either: matches `test_bundle_cuda_libs.sh`'s own `"# captured:
    # pending"*)` case-glob on the same file, and closes the loophole an
    # exact-equality check would leave open (trailing text appended to the
    # same marker line would otherwise read as "already real").
    if first_line.startswith(MARKER):
        print(
            "check_bundle_fixture.py: FAILED -- "
            f"{fixture} is still the provisional '{MARKER}' placeholder. "
            "The loader-verification arm's T4(3) property (a correct REAL stage "
            "passes) is unproven until the real ldd report -- captured by "
            "release-binaries.yml's server-cu12-build job and uploaded as its "
            "'cu12-loader-report' workflow artifact -- replaces this placeholder "
            "as its own commit. No environment variable escapes this check.",
            file=sys.stderr,
        )
        return 1

    print(
        f"check_bundle_fixture.py: OK -- {fixture} carries a real captured "
        "report, not the provisional placeholder."
    )
    return 0


def self_test() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        provisional = tmp / "provisional.txt"
        provisional.write_text(f"{MARKER}\nsome explanatory header text\n")
        assert check(provisional) == 1, "expected FAILED on a provisional fixture"

        real = tmp / "real.txt"
        real.write_text("libcudart.so.12 => /opt/lib/libcudart.so.12 (0x1)\n")
        assert check(real) == 0, "expected OK on a real (non-provisional) fixture"

        real_leading_ws = tmp / "real_leading_ws.txt"
        real_leading_ws.write_text("  libcudart.so.12 => /opt/lib/libcudart.so.12 (0x1)\n")
        assert check(real_leading_ws) == 0, "expected OK on a real fixture with leading whitespace"

        empty = tmp / "empty.txt"
        empty.write_text("")
        assert check(empty) == 1, "expected FAILED on an empty fixture (no real content either)"

        absent = tmp / "does-not-exist.txt"
        assert check(absent) == 1, "expected FAILED when the fixture file does not exist"

        # Trailing text appended to the SAME marker line (someone editing the
        # header in place rather than deleting it) must still FAIL -- the
        # match is a PREFIX of the first line, so this cannot be defeated by
        # appending text after "pending" without actually removing the marker.
        near_marker = tmp / "near_marker.txt"
        near_marker.write_text(f"{MARKER} -- actually already replaced below\nlibcudart.so.12 => /x (0x1)\n")
        assert check(near_marker) == 1, "expected FAILED -- the marker is a prefix of the first line"

    print("check_bundle_fixture.py --self-test: all self-test cases passed.")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(self_test())
    sys.exit(check(FIXTURE))
