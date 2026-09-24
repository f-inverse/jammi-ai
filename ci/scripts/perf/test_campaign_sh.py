#!/usr/bin/env python3
"""`campaign.sh` against a stand-in `gpu-dev` that reads its stdin as ssh
does: every roster pod is reached with its own knobs, a pod not at a clean pinned commit is
refused before its job launches, the pod-side job provisions what its
producer needs, and pooled ladders merge. Nothing is rented or run.

Run: `python3 ci/scripts/perf/test_campaign_sh.py`
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

PERF_DIR = Path(__file__).resolve().parent
REPO_ROOT = PERF_DIR.parents[2]
SCRIPT = PERF_DIR / "campaign.sh"
SHA = "a" * 40

# The stand-in pod CLI: records each call, READS ITS STDIN as ssh does, and
# answers the dispatch checkout with the HEAD a pod reports (`STUB_HEAD_<session>`,
# else the pinned SHA and a clean tree).
STUB = textwrap.dedent(
    """\
    #!/usr/bin/env bash
    cat > /dev/null
    echo "$*" >> "$STUB_LOG"
    if [ "$1" = exec ]; then
      head_var="STUB_HEAD_${2//-/_}"
      case "$3" in
        *"git checkout"*) echo "${!head_var:-$STUB_SHA 0}" ;;
        *) echo "head=aaaaaaa log-age=5s running | Compiling jammi-bench" ;;
      esac
    fi
    """
)


class Campaign(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.stub = self.tmp / "gpu-dev"
        self.stub.write_text(STUB)
        self.stub.chmod(0o755)
        self.log = self.tmp / "calls.log"
        self.log.touch()

    def roster(self, text: str) -> Path:
        path = self.tmp / "roster"
        path.write_text(textwrap.dedent(text))
        return path

    def run_campaign(self, *args: str, dry: bool = False, **env: str) -> subprocess.CompletedProcess:
        full = {**os.environ, "GPU_DEV": str(self.stub), "STUB_LOG": str(self.log), "STUB_SHA": SHA,
                "CAMPAIGN_LOG_DIR": str(self.tmp / "logs"), **env}
        if dry:
            full["CAMPAIGN_DRY_RUN"] = "1"
        return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=full, timeout=60)

    def calls(self) -> list[str]:
        return self.log.read_text().splitlines()

    def test_every_roster_pod_is_dispatched_with_its_own_knobs(self):
        roster = self.roster(
            """\
            # a comment
            p1 a100 train-run FINETUNE_RUN_AB_SEEDS=1,2

            p2 a100 train-run FINETUNE_RUN_AB_SEEDS=3,4
            p3 l40s cpu-ladder CPU_AB_WORKLOAD=structure
            """
        )
        done = self.run_campaign("dispatch", str(roster), SHA)
        self.assertEqual(done.returncode, 0, done.stderr)
        runs = sorted(c for c in self.calls() if c.startswith("run "))
        self.assertEqual(len(runs), 3, self.calls())
        self.assertIn("run p1 bash ci/scripts/perf/campaign.sh pod-job p1 train-run FINETUNE_RUN_AB_SEEDS=1\\,2", runs)
        self.assertIn("run p3 bash ci/scripts/perf/campaign.sh pod-job p3 cpu-ladder CPU_AB_WORKLOAD=structure", runs)

    def test_a_pod_not_at_a_clean_pinned_commit_is_refused_before_its_job(self):
        roster = self.roster("p1 a100 train-step\np2 a100 encode\n")
        done = self.run_campaign("dispatch", str(roster), SHA, STUB_HEAD_p2=f"{SHA} 3")
        self.assertIn(f"p2: NOT at a clean {SHA}", done.stdout)
        runs = [c for c in self.calls() if c.startswith("run ")]
        self.assertEqual([r.split()[1] for r in runs], ["p1"], "only the clean pod launched")

    def test_the_checkout_precedes_the_launch_on_each_pod(self):
        done = self.run_campaign("dispatch", str(self.roster("p1 a100 train-step\n")), SHA)
        self.assertEqual(done.returncode, 0, done.stderr)
        verbs = [c.split()[0] for c in self.calls()]
        self.assertEqual(verbs[:2], ["wait-seed", "target"])
        self.assertLess(verbs.index("exec"), verbs.index("run"))

    def test_a_short_sha_is_refused(self):
        done = self.run_campaign("dispatch", str(self.roster("p1 a100 train-step\n")), "abc123")
        self.assertEqual(done.returncode, 2)
        self.assertIn("40-hex", done.stderr)

    def test_a_pooled_producer_on_two_gpu_models_is_refused(self):
        roster = self.roster("p1 a100 train-run\np2 l40s train-run\n")
        done = self.run_campaign("rent", str(roster), "main")
        self.assertEqual(done.returncode, 2)
        self.assertIn("one GPU model", done.stderr)
        self.assertEqual(self.calls(), [], "nothing rented")

    def test_status_reports_one_line_per_pod(self):
        roster = self.roster("p1 a100 train-run\np2 a100 encode\n")
        lines = self.run_campaign("status", str(roster)).stdout.splitlines()
        self.assertEqual([line.split()[0] for line in lines], ["p1", "p2"])
        self.assertIn("log-age=5s running", lines[0])

    def test_down_tears_down_exactly_the_roster(self):
        self.run_campaign("down", str(self.roster("p1 a100 train-run\np2 l40s cpu-ladder\n")))
        self.assertEqual(sorted(self.calls()), ["down p1", "down p2"])

    def test_the_pod_job_provisions_what_its_producer_needs(self):
        gpu = self.run_campaign("pod-job", "p1", "train-run", "FINETUNE_RUN_AB_SEEDS=5", dry=True).stdout
        self.assertIn("torch_venv.py --provision", gpu)
        self.assertIn("checkpoint_files.py --fetch answerdotai/ModernBERT-large /root/checkpoints/ModernBERT-large", gpu)
        self.assertIn(".venv-cookbook", gpu)
        self.assertIn("finetune_run_ab.sh", gpu)
        self.assertIn("CAMPAIGN JOB DONE exit=0", gpu)
        cpu = self.run_campaign("pod-job", "p3", "cpu-ladder", "CPU_AB_WORKLOAD=propagate", dry=True).stdout
        self.assertNotIn("checkpoint_files.py", cpu)
        self.assertIn("cpu_ladders_ab.sh", cpu)

    def test_a_knob_that_is_not_key_value_is_refused(self):
        done = self.run_campaign("pod-job", "p1", "train-step", "oops", dry=True)
        self.assertEqual(done.returncode, 2)
        self.assertIn("KEY=VALUE", done.stderr)

    def test_merge_pools_every_session_of_the_producer_and_names_its_ladder(self):
        # The ladder runs in the CI image, which sees only the checkout.
        (REPO_ROOT / ".campaign-out").mkdir(exist_ok=True)
        dest = Path(tempfile.mkdtemp(dir=REPO_ROOT / ".campaign-out", prefix="test-merge-"))
        self.addCleanup(shutil.rmtree, dest)
        for session, seed in (("p1", 1), ("p2", 3)):
            raw = dest / session / "train-run" / "raw"
            raw.mkdir(parents=True)
            (raw / f"resident__seed{seed}__r1.json").write_text("{}")
        (dest / "p3" / "encode" / "legs-plan").mkdir(parents=True)
        roster = self.roster("p1 a100 train-run\np2 a100 train-run\np3 a100 encode\n")
        done = self.run_campaign("merge", str(roster), str(dest), "train-run", "resident", dry=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        merged = sorted(p.name for p in (dest / "train-run" / "legs").iterdir())
        self.assertEqual(merged, ["resident__seed1__r1.json", "resident__seed3__r1.json"])
        self.assertIn("ladder train-run", done.stdout)
        self.assertIn("--from torch --to resident", done.stdout)


if __name__ == "__main__":
    unittest.main()
