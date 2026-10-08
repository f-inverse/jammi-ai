#!/usr/bin/env python3
"""Tests for `build_matrix.py`: every architecture maps to one native runner,
each matrix shape carries exactly the legs asked for, and an unknown
platform, target or architecture is refused by name."""

from __future__ import annotations

import io
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_matrix as bm  # noqa: E402


class Runners(unittest.TestCase):
    def test_each_linux_arch_has_one_native_runner_in_every_spelling(self):
        self.assertEqual(bm.runner("amd64"), bm.runner("x86_64"))
        self.assertEqual(bm.runner("arm64"), bm.runner("aarch64"))
        self.assertNotEqual(bm.runner("amd64"), bm.runner("arm64"))

    def test_unknown_arch_is_refused(self):
        with self.assertRaises(ValueError):
            bm.runner("riscv64")


class Matrices(unittest.TestCase):
    def test_images_follow_the_platform_list(self):
        m = bm.images(["linux/amd64", "linux/arm64"])
        self.assertEqual([row["arch"] for row in m["include"]], ["amd64", "arm64"])
        self.assertEqual([row["runner"] for row in m["include"]], [bm.runner("amd64"), bm.runner("arm64")])
        with self.assertRaises(ValueError):
            bm.images(["linux/386"])

    def test_images_cross_a_variant_axis(self):
        m = bm.images(["linux/amd64", "linux/arm64"], {"variant": ["generic", "selfcontained"]})
        self.assertEqual(
            [(row["variant"], row["arch"]) for row in m["include"]],
            [("generic", "amd64"), ("generic", "arm64"), ("selfcontained", "amd64"), ("selfcontained", "arm64")],
        )
        with self.assertRaises(ValueError):
            bm.images(["linux/amd64"], {"variant": []})

    def test_cli_targets_pick_container_or_bare_macos(self):
        m = bm.cli(["x86_64-unknown-linux-gnu", "aarch64-apple-darwin"], "img:tag")
        linux, mac = m["include"]
        self.assertEqual((linux["runner"], linux["container"]), (bm.runner("amd64"), "img:tag"))
        self.assertEqual((mac["runner"], mac["container"]), (bm.MACOS_RUNNER, ""))
        with self.assertRaises(ValueError):
            bm.cli(["x86_64-pc-windows-msvc"], "img")

    def test_wheels_split_linux_from_macos(self):
        m = bm.wheels(["linux-aarch64", "macos-x86_64"])
        self.assertEqual(m["linux"], [{"arch": "aarch64", "runner": bm.runner("arm64")}])
        self.assertEqual(m["macos"][0]["target"], "x86_64-apple-darwin")
        self.assertEqual(m["macos"][0]["assert_arch"], "x86_64")
        self.assertEqual(bm.wheels(["linux-x86_64"])["macos"], [])


class Cli(unittest.TestCase):
    def run_main(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = bm.main(list(argv))
        return rc, out.getvalue(), err.getvalue()

    def test_outputs_are_github_output_lines(self):
        rc, out, _ = self.run_main("server", "--arch", "aarch64")
        self.assertEqual((rc, out), (0, f"runner={bm.runner('arm64')}\n"))
        rc, out, _ = self.run_main("images", "--platforms", "linux/amd64", "--cross", "variant=generic,selfcontained")
        self.assertEqual(rc, 0)
        self.assertEqual(out.count('"variant"'), 2)
        rc, _, err = self.run_main("images", "--platforms", "linux/amd64", "--cross", "variant")
        self.assertEqual(rc, 1)
        self.assertIn("KEY=V1,V2", err)
        rc, out, _ = self.run_main("wheels", "--platforms", "linux-x86_64 macos-arm64")
        self.assertEqual(rc, 0)
        self.assertTrue(out.startswith("linux=[{") and "\nmacos=[{" in out)

    def test_empty_and_unknown_lists_fail_by_name(self):
        rc, _, err = self.run_main("images", "--platforms", " ")
        self.assertEqual(rc, 1)
        self.assertIn("nothing to build", err)
        rc, _, err = self.run_main("cli", "--targets", "sparc-sun-solaris", "--container", "x")
        self.assertEqual(rc, 1)
        self.assertIn("sparc-sun-solaris", err)


if __name__ == "__main__":
    unittest.main()
