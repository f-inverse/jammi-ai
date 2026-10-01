#!/usr/bin/env python3
"""Tests for `ci_image.py`: the content keys, their closure check, and the
wait for `ci.yml`'s build. Pure Python, no network, no registry: the key
tests run over a scratch tree, the wait over injected fakes.

Run directly: `python3 ci/scripts/test_ci_image.py`
"""

from __future__ import annotations

import io
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ci_image  # noqa: E402
import github_api  # noqa: E402

CPU = ci_image.IMAGES["cpu"]
CUDA = ci_image.IMAGES["cuda"]


class ScratchTree:
    """A copy of every key input (and the env files) under a temp root."""

    def __enter__(self) -> Path:
        self._dir = tempfile.TemporaryDirectory()
        root = Path(self._dir.name)
        for rel in {*CPU.inputs, *CUDA.inputs, ci_image.SERVICE_IMAGES, ".github/workflows/ci.yml"}:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ci_image.REPO_ROOT / rel, root / rel)
        return root

    def __exit__(self, *exc):
        self._dir.cleanup()


def append(root: Path, rel: str, text: str) -> None:
    with open(root / rel, "a") as f:
        f.write(text)


class ContentKeyTest(unittest.TestCase):
    def test_the_tag_is_ctx_and_sixteen_hex(self):
        self.assertRegex(ci_image.tag(CPU), r"^ctx-[0-9a-f]{16}$")

    def test_every_cpu_input_moves_both_keys(self):
        for rel in CPU.inputs:
            with self.subTest(rel), ScratchTree() as root:
                cpu, cuda = ci_image.tag(CPU, root), ci_image.tag(CUDA, root)
                append(root, rel, "\n# moved\n")
                self.assertNotEqual(ci_image.tag(CPU, root), cpu)
                self.assertNotEqual(ci_image.tag(CUDA, root), cuda, "the CUDA image builds FROM the CPU image")

    def test_a_cuda_only_change_leaves_the_cpu_image_alone(self):
        with ScratchTree() as root:
            cpu, cuda = ci_image.tag(CPU, root), ci_image.tag(CUDA, root)
            append(root, ".docker/ci-cuda.Dockerfile", "\n# moved\n")
            self.assertEqual(ci_image.tag(CPU, root), cpu)
            self.assertNotEqual(ci_image.tag(CUDA, root), cuda)

    def test_the_recipe_is_an_input_of_every_image(self):
        for rel in ci_image.RECIPE:
            with self.subTest(rel), ScratchTree() as root:
                cpu, cuda = ci_image.tag(CPU, root), ci_image.tag(CUDA, root)
                append(root, rel, "\n# moved\n")
                self.assertNotEqual(ci_image.tag(CPU, root), cpu)
                self.assertNotEqual(ci_image.tag(CUDA, root), cuda)

    def test_a_service_pin_is_not_an_image_input(self):
        with ScratchTree() as root:
            cpu = ci_image.tag(CPU, root)
            append(root, ci_image.SERVICE_IMAGES, "\n# moved\n")
            self.assertEqual(ci_image.tag(CPU, root), cpu)

    def test_refs_name_every_reference_a_lane_uses(self):
        got = ci_image.refs()
        self.assertEqual(set(got), {"cpu", "cuda", "postgres"})
        self.assertTrue(got["cpu"].endswith(":" + ci_image.tag(CPU)))
        self.assertRegex(got["postgres"], r"@sha256:[0-9a-f]{64}$", "the service image is pinned by digest")


class BuildDefinitionTest(unittest.TestCase):
    """How an image is built is defined once: the base it builds FROM on each
    arch, the toolchain pin, its platforms."""

    def test_an_image_is_found_by_its_dockerfile(self):
        self.assertIs(ci_image.image_for(CUDA.dockerfile), CUDA)
        with self.assertRaises(ValueError):
            ci_image.image_for(".docker/other.Dockerfile")

    def test_the_cpu_image_builds_from_the_pinned_base_of_each_arch(self):
        base = ci_image.parse_env(ci_image.REPO_ROOT / ci_image.BASE_IMAGES)
        for arch in ("amd64", "arm64"):
            args = ci_image.build_args(CPU, arch)
            self.assertEqual(args["BASE_IMAGE"], base[f"MANYLINUX_{arch.upper()}"])
            self.assertRegex(args["BASE_IMAGE"], r"@sha256:[0-9a-f]{64}$")

    def test_the_cuda_image_builds_from_the_cpu_image_of_the_same_tree(self):
        self.assertEqual(ci_image.build_args(CUDA, "amd64")["BASE_IMAGE"], ci_image.ref(CPU))

    def test_the_toolchain_is_the_pin(self):
        pin = (ci_image.REPO_ROOT / "rust-toolchain.toml").read_text()
        self.assertIn(f'"{ci_image.build_args(CPU, "amd64")["RUST_VERSION"]}"', pin)

    def test_every_build_argument_is_keyed(self):
        for image in (CPU, CUDA):
            arch = image.platforms[0].split("/")[1]
            self.assertTrue(set(ci_image.build_args(image, arch)) <= ci_image.KEYED_BUILD_ARGS)

    def test_an_arch_the_image_is_not_built_for_is_refused(self):
        with self.assertRaises(ValueError):
            ci_image.build_args(CUDA, "arm64")

    def test_describe_names_the_reference_its_tag_and_platforms(self):
        d = ci_image.describe(CPU)
        self.assertEqual(d["ref"], ci_image.ref(CPU))
        self.assertEqual(d["platforms"], "linux/amd64,linux/arm64")


class CheckTest(unittest.TestCase):
    def test_the_real_dockerfiles_read_only_their_inputs(self):
        self.assertEqual(ci_image.check(), [])

    def test_copying_an_uncovered_file_is_a_finding(self):
        with ScratchTree() as root:
            append(root, CPU.dockerfile, "\nCOPY extra.sh /tmp/extra.sh\n")
            findings = ci_image.check(root)
            self.assertTrue(any("copies extra.sh" in f for f in findings), findings)

    def test_add_is_held_like_copy(self):
        with ScratchTree() as root:
            append(root, CUDA.dockerfile, "\nADD --chmod=755 pinned-tools.sh /tmp/t.sh\n")
            findings = ci_image.check(root)
            self.assertTrue(any("cuda image's key" in f for f in findings), findings)

    def test_a_copy_from_a_stage_is_not_the_context(self):
        with ScratchTree() as root:
            append(root, CPU.dockerfile, "\nCOPY --from=builder /out/bin /usr/local/bin/bin\n")
            self.assertEqual(ci_image.check(root), [])

    def test_a_build_argument_no_input_supplies_is_a_finding(self):
        with ScratchTree() as root:
            append(root, CPU.dockerfile, "\nARG EXTRA_FLAG\n")
            findings = ci_image.check(root)
            self.assertTrue(any("EXTRA_FLAG" in f for f in findings), findings)

    def test_a_build_argument_with_a_default_is_part_of_the_dockerfile(self):
        with ScratchTree() as root:
            append(root, CPU.dockerfile, "\nARG EXTRA_VERSION=1.2.3\n")
            self.assertEqual(ci_image.check(root), [])


class BuilderNameTest(unittest.TestCase):
    CI = """\
name: CI
on:
  pull_request:
jobs:
  ci-image:
    uses: ./.github/workflows/_ci-image.yml
  build-ci-image:
    name: CI image (CPU)
    needs: ci-image
    uses: ./.github/workflows/_ci-base-image.yml
  check:
    name: Format & Lint
    runs-on: ubuntu-latest
"""

    def test_the_real_builders_carry_the_prefix(self):
        names = ci_image.builder_job_names((ci_image.REPO_ROOT / ".github/workflows/ci.yml").read_text())
        self.assertEqual(set(names), {"build-ci-image", "build-ci-image-cuda"})
        self.assertTrue(all(n.startswith(ci_image.BUILDER_JOB_PREFIX) for n in names.values()))

    def test_only_jobs_that_call_the_builder_are_builders(self):
        self.assertEqual(ci_image.builder_job_names(self.CI), {"build-ci-image": "CI image (CPU)"})

    def test_a_builder_named_otherwise_is_a_finding(self):
        with ScratchTree() as root:
            (root / ".github/workflows/ci.yml").write_text(self.CI.replace("CI image (CPU)", "Build image"))
            findings = ci_image.check(root)
            self.assertTrue(any("named 'Build image'" in f for f in findings), findings)


class AwaitTest(unittest.TestCase):
    def _await(self, exists, builder=lambda: None, deadline_s=60.0):
        t = [0.0]
        sleeps: list[float] = []

        def sleep(s):
            sleeps.append(s)
            t[0] += s

        err = io.StringIO()
        rc = ci_image.await_images(
            ["img:a", "img:b"], exists=exists, builder=builder, sleep=sleep, clock=lambda: t[0],
            deadline_s=deadline_s, poll_s=20, err=err,
        )
        return rc, sleeps, err.getvalue()

    def test_present_images_return_at_once(self):
        rc, sleeps, _ = self._await(lambda r: (True, ""))
        self.assertEqual((rc, sleeps), (0, []))

    def test_images_that_appear_later_are_waited_for(self):
        polls = {"img:a": 0, "img:b": 0}

        def exists(r):
            polls[r] += 1
            return (polls[r] > 2, "not found")

        rc, sleeps, _ = self._await(exists)
        self.assertEqual(rc, 0)
        self.assertEqual(len(sleeps), 2)

    def test_a_failed_build_stops_the_wait_naming_it(self):
        rc, sleeps, err = self._await(lambda r: (False, "not found"), builder=lambda: "CI image (CPU) / plan (url)")
        self.assertEqual((rc, sleeps), (1, []))
        self.assertIn("CI image (CPU) / plan", err)

    def test_the_deadline_names_what_is_missing(self):
        rc, _, err = self._await(lambda r: (r == "img:a", "manifest unknown"), deadline_s=60)
        self.assertEqual(rc, 1)
        self.assertIn("img:b", err)
        self.assertNotIn("img:a,", err)
        self.assertIn("manifest unknown", err)

    def test_an_unreadable_build_record_keeps_waiting_on_the_registry(self):
        calls = [0]

        def exists(r):
            calls[0] += 1
            return (calls[0] > 2, "not found")

        def builder():
            raise github_api.ApiError("simulated")

        rc, _, err = self._await(exists, builder=builder)
        self.assertEqual(rc, 0)
        self.assertIn("::warning::", err)


class FailedBuilderTest(unittest.TestCase):
    SHA = "c" * 40

    def _fetch(self, runs, jobs_by_run):
        def fetch(url, token):
            if "/jobs?" in url:
                rid = int(re.search(r"/runs/(\d+)/jobs", url).group(1))
                return {"jobs": jobs_by_run.get(rid, [])}
            return {"workflow_runs": runs}

        return fetch

    def test_a_red_build_job_is_named(self):
        fetch = self._fetch(
            [{"id": 1, "head_sha": self.SHA}],
            {1: [{"name": "Format & Lint", "conclusion": "failure"},
                 {"name": "CI image (CPU) / build-and-push (linux/arm64)", "conclusion": "failure", "html_url": "u"}]},
        )
        self.assertEqual(
            ci_image.failed_builder(fetch, "tok", "o/r", self.SHA), "CI image (CPU) / build-and-push (linux/arm64) (u)"
        )

    def test_a_running_or_skipped_build_is_not_a_failure(self):
        fetch = self._fetch(
            [{"id": 1, "head_sha": self.SHA}],
            {1: [{"name": "CI image (CPU) / plan", "conclusion": None}, {"name": "CI image (CUDA) / plan", "conclusion": "skipped"}]},
        )
        self.assertIsNone(ci_image.failed_builder(fetch, "tok", "o/r", self.SHA))

    def test_another_commits_run_is_ignored(self):
        fetch = self._fetch([{"id": 1, "head_sha": "d" * 40}], {1: [{"name": "CI image (CPU) / plan", "conclusion": "failure"}]})
        self.assertIsNone(ci_image.failed_builder(fetch, "tok", "o/r", self.SHA))


if __name__ == "__main__":
    unittest.main()
