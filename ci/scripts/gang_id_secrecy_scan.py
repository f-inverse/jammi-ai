#!/usr/bin/env python3
"""F5/M4 — the NCCL id-secrecy scan for the RunPod CLUSTER leg
(`ci/scripts/runpod_gpu_cluster.sh`).

The 128-byte NCCL id `Nccl::new_id()` mints on rank 0 is a CAPABILITY (the
right to join this one gang), never evidence — it must never reach a
committed artifact, a CI log, or any file a human later reviews. The id
crosses hosts HEX-encoded (`runpod_gpu_gang.sh`'s own PR-B1 contract, applied
here to the cluster driver's out-of-band `scp` ship), so a leak could show up
raw, hex (either case), or base64 — every encoding the ship step could ever
emit.

This scans, over raw BYTES (`open(p, "rb").read()` — a bash variable or a
grep pattern cannot carry a NUL-bearing 128-byte string, and the id is
exactly that), the CARRIER SET the cluster driver's own run produces:

  1. the pulled artifact directory, recursively (every file the two ranks'
     `rank-<r>.json` reports, and anything else, landed in);
  2. the driver's own run log (the ONE `tee`'d stream F6 requires);
  3. the assembled `gang` artifact JSON;
  4. the staging copy's own directory LISTING (never its content a second
     time — its content is the NEEDLE SOURCE, read once, below) — a leak
     spelled into a FILENAME next to it would otherwise go unseen.

Exit lattice (F5): 0 clean; 1 a carrier carries the id in some encoding
(named by carrier and encoding, the bytes themselves are NEVER printed); 2 a
carrier could not be examined at all (missing, unreadable, a dangling
symlink, or an archive member under the pulled artifact directory — this
scan refuses to look inside an archive rather than silently skip it). 2 is
UNEXAMINABLE, never read as clean.

Symlinks are FOLLOWED (a dangling symlink is 2, never silently skipped).
Archive members (`.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`, `.tbz2`, `.tar.xz`,
`.txz`, `.zip`, `.gz` — matched by suffix, not by sniffing magic bytes,
which would itself be a second unaudited parser) anywhere under the pulled
artifact directory are refused outright (2) rather than opened: this driver
never legitimately ships an archive there, so one appearing is itself
suspicious, and this scanner is not an archive-format parser.

The staging copy (the local runner's own copy of the 128-byte id file,
`$RP_WORK/nccl.id` in the driver) is deleted AFTER a clean scan, and ONLY
when `--delete-staging` is passed — never unconditionally, so a caller can
inspect a dirty run's staging file by hand.

Run: `python3 ci/scripts/gang_id_secrecy_scan.py --staging-file <path> \
  --artifact-dir <dir> --log <path> --assembled-artifact <path> \
  [--delete-staging]`
Self-test: `python3 ci/scripts/gang_id_secrecy_scan.py --self-test`
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

ID_BYTES_LEN = 128

# Suffix match only (never a magic-byte sniff, which would be a second,
# unaudited parser) — matched against the LOWERCASED basename so a
# `.TAR.GZ` upload cannot slip past this refusal on case alone.
ARCHIVE_SUFFIXES = (
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
    ".zip",
    ".gz",
)

STATUS_CLEAN = 0
STATUS_HIT = 1
STATUS_UNEXAMINABLE = 2


def _is_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suf) for suf in ARCHIVE_SUFFIXES)


def id_needles(id_bytes: bytes) -> list[tuple[str, bytes]]:
    """Every encoding the ship step could emit, as the RAW BYTES to search
    for — hex is checked in BOTH cases independently (a case-INsensitive
    search over the raw string would also match unrelated mixed-case text
    that merely contains the right hex digits in the wrong case pattern by
    accident far less often than it would hide a genuine same-case leak, so
    this checks lower and upper as two DISTINCT literal needles, never a
    single case-folded one)."""
    hex_lower = id_bytes.hex().encode("ascii")
    hex_upper = id_bytes.hex().upper().encode("ascii")
    b64 = base64.b64encode(id_bytes)
    return [
        ("raw", id_bytes),
        ("hex-lower", hex_lower),
        ("hex-upper", hex_upper),
        ("base64", b64),
    ]


def _scan_bytes(data: bytes, needles: list[tuple[str, bytes]]) -> str | None:
    for name, needle in needles:
        if needle and needle in data:
            return name
    return None


def scan_file(path: Path, needles: list[tuple[str, bytes]]) -> tuple[int, str]:
    """(status, message) for ONE file-shaped carrier. Symlinks are followed;
    a dangling one is UNEXAMINABLE (2), never skipped. An archive member is
    refused outright (2) without being opened."""
    try:
        if path.is_symlink():
            real = path.resolve(strict=True)
        else:
            real = path
    except (OSError, RuntimeError) as exc:
        return STATUS_UNEXAMINABLE, f"{path}: dangling symlink or unresolvable ({exc})"
    if _is_archive(real):
        return STATUS_UNEXAMINABLE, f"{path}: archive carrier refused (never opened)"
    try:
        st = real.stat()
        if stat.S_ISDIR(st.st_mode):
            return STATUS_UNEXAMINABLE, f"{path}: is a directory, not a file"
        data = real.read_bytes()
    except OSError as exc:
        return STATUS_UNEXAMINABLE, f"{path}: unreadable ({exc})"
    hit = _scan_bytes(data, needles)
    if hit is not None:
        return STATUS_HIT, f"{path}: carries the id ({hit} encoding)"
    return STATUS_CLEAN, ""


def scan_dir(root: Path, needles: list[tuple[str, bytes]]) -> list[tuple[int, str]]:
    """Every file under `root`, recursively, symlinks followed. A directory
    that cannot itself be listed is one UNEXAMINABLE finding for the whole
    subtree (fail-closed: an incomplete listing is worse than no listing at
    all — the same doctrine `rp_cluster_sweep`'s enumeration failures use)."""
    if not root.is_dir():
        return [(STATUS_UNEXAMINABLE, f"{root}: not a directory (pulled artifact dir missing?)")]
    findings: list[tuple[int, str]] = []
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        dpath = Path(dirpath)
        for fname in filenames:
            findings.append(scan_file(dpath / fname, needles))
    return [f for f in findings if f[0] != STATUS_CLEAN]


def scan_directory_listing(staging: Path, needles: list[tuple[str, bytes]]) -> tuple[int, str | None]:
    """The staging copy's own containing directory LISTING — every entry
    NAME in it (never re-opening the staging file's content, which is the
    needle SOURCE, read exactly once by the caller). A name carrying the
    id's hex or base64 spelling is a hit; raw bytes are not checked here (a
    POSIX filename cannot carry a NUL byte, so the raw-byte needle can never
    occur in a name)."""
    listing_dir = staging.parent
    try:
        names = [p.name for p in listing_dir.iterdir()]
    except OSError as exc:
        return STATUS_UNEXAMINABLE, f"{listing_dir}: unreadable directory listing ({exc})"
    blob = "\n".join(names).encode("utf-8", errors="surrogateescape")
    for name, needle in needles:
        if name == "raw":
            continue
        if needle and needle in blob:
            return STATUS_HIT, f"{listing_dir}: a directory entry name carries the id ({name} encoding)"
    return STATUS_CLEAN, None


def run_scan(
    staging_file: Path,
    artifact_dir: Path,
    log: Path,
    assembled_artifact: Path,
    delete_staging: bool,
) -> int:
    try:
        if staging_file.is_symlink():
            real_staging = staging_file.resolve(strict=True)
        else:
            real_staging = staging_file
        id_bytes = real_staging.read_bytes()
    except (OSError, RuntimeError) as exc:
        print(
            f"gang-id-secrecy-scan: UNEXAMINABLE: could not read the staging id file {staging_file}: {exc}",
            file=sys.stderr,
        )
        return STATUS_UNEXAMINABLE
    if len(id_bytes) != ID_BYTES_LEN:
        print(
            f"gang-id-secrecy-scan: UNEXAMINABLE: the staging id file {staging_file} is not exactly "
            f"{ID_BYTES_LEN} bytes (got {len(id_bytes)}) -- refusing to scan against a needle that may "
            "not be the real id",
            file=sys.stderr,
        )
        return STATUS_UNEXAMINABLE

    needles = id_needles(id_bytes)
    worst = STATUS_CLEAN
    messages: list[str] = []

    for status, msg in scan_dir(artifact_dir, needles):
        worst = max(worst, status)
        messages.append(msg)

    for single in (log, assembled_artifact):
        if not (single.exists() or single.is_symlink()):
            worst = max(worst, STATUS_UNEXAMINABLE)
            messages.append(f"{single}: missing")
            continue
        status, msg = scan_file(single, needles)
        if status != STATUS_CLEAN:
            worst = max(worst, status)
            messages.append(msg)

    listing_status, listing_msg = scan_directory_listing(staging_file, needles)
    if listing_status != STATUS_CLEAN:
        worst = max(worst, listing_status)
        if listing_msg:
            messages.append(listing_msg)

    if worst == STATUS_CLEAN:
        print(
            "gang-id-secrecy-scan: clean -- no carrier in the pulled artifact dir, the run log, the "
            "assembled artifact, or the staging directory listing carries the id in any encoding"
        )
        if delete_staging:
            try:
                real_staging.unlink()
            except OSError as exc:
                print(
                    f"gang-id-secrecy-scan: WARNING: scan was clean but the staging copy could not be "
                    f"deleted ({exc})",
                    file=sys.stderr,
                )
        return STATUS_CLEAN

    label = "HIT" if worst == STATUS_HIT else "UNEXAMINABLE"
    print(f"gang-id-secrecy-scan: {label}", file=sys.stderr)
    for msg in messages:
        print(f"  - {msg}", file=sys.stderr)
    return worst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--staging-file", type=Path)
    ap.add_argument("--artifact-dir", type=Path)
    ap.add_argument("--log", type=Path)
    ap.add_argument("--assembled-artifact", type=Path)
    ap.add_argument("--delete-staging", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return _run_self_test()

    missing = [
        name
        for name, val in (
            ("--staging-file", args.staging_file),
            ("--artifact-dir", args.artifact_dir),
            ("--log", args.log),
            ("--assembled-artifact", args.assembled_artifact),
        )
        if val is None
    ]
    if missing:
        print(f"gang_id_secrecy_scan.py: missing required argument(s): {missing}", file=sys.stderr)
        return 2

    return run_scan(args.staging_file, args.artifact_dir, args.log, args.assembled_artifact, args.delete_staging)


# --------------------------------------------------------------------------- #
# Self-test: every arm named in the module doc, driven against a real
# tempdir fixture tree -- no mock filesystem, no bash shim.
# --------------------------------------------------------------------------- #
class GangIdSecrecyScanTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        # A real, NUL-bearing 128-byte id -- exactly the shape a bash
        # variable or a grep pattern cannot carry, which is why this scan
        # is python-over-bytes in the first place.
        self.id_bytes = bytes((i * 7 + 1) % 256 for i in range(ID_BYTES_LEN))
        self.id_bytes = self.id_bytes[:64] + b"\x00" * 4 + self.id_bytes[68:]
        self.staging = self.root / "staging" / "nccl.id"
        self.staging.parent.mkdir(parents=True)
        self.staging.write_bytes(self.id_bytes)
        self.artifact_dir = self.root / "pulled"
        self.artifact_dir.mkdir()
        self.log = self.root / "run.log"
        self.log.write_text("clean log, no secrets here\n")
        self.assembled = self.root / "gang.json"
        self.assembled.write_text('{"gang": {"leg": "cluster"}}\n')

    def _run(self, delete_staging: bool = False) -> tuple[int, str]:
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            rc = run_scan(self.staging, self.artifact_dir, self.log, self.assembled, delete_staging)
        return rc, buf_out.getvalue() + buf_err.getvalue()

    def test_clean_scan_passes_and_staging_survives_without_the_flag(self) -> None:
        rc, _out = self._run(delete_staging=False)
        self.assertEqual(rc, STATUS_CLEAN)
        self.assertTrue(self.staging.exists(), "staging copy must survive without --delete-staging")

    def test_clean_scan_deletes_staging_only_with_the_flag(self) -> None:
        rc, _out = self._run(delete_staging=True)
        self.assertEqual(rc, STATUS_CLEAN)
        self.assertFalse(self.staging.exists(), "a clean scan with --delete-staging must remove it")

    def test_raw_id_planted_in_the_artifact_dir_is_a_hit(self) -> None:
        (self.artifact_dir / "rank-0.json").write_bytes(b'{"note": "' + self.id_bytes + b'"}')
        rc, out = self._run()
        self.assertEqual(rc, STATUS_HIT)
        self.assertIn("raw encoding", out)

    def test_hex_lower_id_planted_in_the_run_log_is_a_hit(self) -> None:
        self.log.write_text("debug: id=" + self.id_bytes.hex() + "\n")
        rc, out = self._run()
        self.assertEqual(rc, STATUS_HIT)
        self.assertIn("hex-lower encoding", out)

    def test_hex_upper_id_planted_in_the_assembled_artifact_is_a_hit(self) -> None:
        self.assembled.write_text('{"gang": {"reason": "' + self.id_bytes.hex().upper() + '"}}')
        rc, out = self._run()
        self.assertEqual(rc, STATUS_HIT)
        self.assertIn("hex-upper encoding", out)

    def test_base64_id_planted_in_the_artifact_dir_is_a_hit(self) -> None:
        b64 = base64.b64encode(self.id_bytes).decode("ascii")
        (self.artifact_dir / "notes.txt").write_text("leaked: " + b64)
        rc, out = self._run()
        self.assertEqual(rc, STATUS_HIT)
        self.assertIn("base64 encoding", out)

    def test_id_encoded_into_a_directory_entry_name_is_a_hit(self) -> None:
        # `scan_directory_listing` exercised DIRECTLY (never through
        # `run_scan`'s end-to-end 128-byte staging check): a POSIX filename
        # cannot carry the raw id at all (NUL included), and the FULL
        # 128-byte id's own hex (256 chars) or base64 (172 chars, and often
        # containing `/`, a path separator) spelling exceeds -- or cannot
        # legally form -- a single filename component on a real filesystem.
        # The mechanism under test (a hit anywhere in the joined listing
        # text) does not depend on the id's own length, so a short synthetic
        # id proves the same property without fighting filesystem limits.
        short_id = b"leaked-bytes"
        needles = id_needles(short_id)
        staging_dir = self.staging.parent
        (staging_dir / ("evidence-" + short_id.hex())).write_text("x")
        status, msg = scan_directory_listing(self.staging, needles)
        self.assertEqual(status, STATUS_HIT)
        self.assertIn("directory entry name carries the id", msg or "")

    def test_archive_member_under_the_pulled_dir_is_refused_not_opened(self) -> None:
        (self.artifact_dir / "bundle.tar.gz").write_bytes(self.id_bytes)  # would be a hit if opened
        rc, out = self._run()
        self.assertEqual(rc, STATUS_UNEXAMINABLE)
        self.assertIn("archive carrier refused", out)

    def test_dangling_symlink_in_the_pulled_dir_is_unexaminable(self) -> None:
        (self.artifact_dir / "dangling").symlink_to(self.artifact_dir / "does-not-exist")
        rc, out = self._run()
        self.assertEqual(rc, STATUS_UNEXAMINABLE)
        self.assertIn("dangling symlink", out)

    def test_symlink_to_a_real_clean_file_is_followed_and_stays_clean(self) -> None:
        target = self.root / "real.txt"
        target.write_text("nothing interesting")
        (self.artifact_dir / "link").symlink_to(target)
        rc, _out = self._run()
        self.assertEqual(rc, STATUS_CLEAN)

    def test_symlink_to_a_file_carrying_the_id_is_followed_and_is_a_hit(self) -> None:
        target = self.root / "real_secret.bin"
        target.write_bytes(self.id_bytes)
        (self.artifact_dir / "link").symlink_to(target)
        rc, out = self._run()
        self.assertEqual(rc, STATUS_HIT)
        self.assertIn("raw encoding", out)

    def test_missing_run_log_is_unexaminable_never_clean(self) -> None:
        self.log.unlink()
        rc, out = self._run()
        self.assertEqual(rc, STATUS_UNEXAMINABLE)
        self.assertIn("missing", out)

    def test_missing_assembled_artifact_is_unexaminable_never_clean(self) -> None:
        self.assembled.unlink()
        rc, out = self._run()
        self.assertEqual(rc, STATUS_UNEXAMINABLE)
        self.assertIn("missing", out)

    def test_unreadable_artifact_file_is_unexaminable(self) -> None:
        if os.geteuid() == 0:
            self.skipTest("root bypasses DAC permission bits (chmod_bypassed class)")
        bad = self.artifact_dir / "no-read.json"
        bad.write_text("{}")
        bad.chmod(0)
        try:
            rc, out = self._run()
        finally:
            bad.chmod(0o600)
        self.assertEqual(rc, STATUS_UNEXAMINABLE)
        self.assertIn("unreadable", out)

    def test_missing_pulled_artifact_dir_is_unexaminable(self) -> None:
        import shutil

        shutil.rmtree(self.artifact_dir)
        rc, out = self._run()
        self.assertEqual(rc, STATUS_UNEXAMINABLE)
        self.assertIn("not a directory", out)

    def test_staging_file_not_exactly_128_bytes_is_unexaminable(self) -> None:
        self.staging.write_bytes(b"\x00" * 127)
        rc, out = self._run()
        self.assertEqual(rc, STATUS_UNEXAMINABLE)
        self.assertIn("not exactly 128 bytes", out)

    def test_hex_upper_and_lower_are_distinct_needles(self) -> None:
        # A control proving the two cases are checked independently: an id
        # whose hex happens to contain no letters at all still needs the
        # OTHER cases exercised elsewhere in this suite; here we confirm
        # mixed-case hex of the SAME id is not accidentally matched by the
        # lower-case needle alone when the planted text is upper-case.
        upper = self.id_bytes.hex().upper()
        self.assertNotIn(self.id_bytes.hex(), upper) if any(c.isalpha() for c in upper) else None


def _run_self_test() -> int:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(GangIdSecrecyScanTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
