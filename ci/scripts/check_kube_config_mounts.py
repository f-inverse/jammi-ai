#!/usr/bin/env python3
"""Every rendered workload mounts a config file the server actually reads.

**Guarded property**: `jammi-server` resolves its configuration from
`JAMMI_CONFIG`, then `./jammi.toml`, then `/etc/jammi/jammi.toml`. A manifest
that mounts a ConfigMap at `/etc/jammi` under any other key renders, validates
and deploys — and the pod then runs on defaults, with its topology role
(scheduler, executor, worker) silently unset. For each kustomization under
`deploy/kubernetes`, this renders the manifests and requires, for every
container that mounts a ConfigMap at `/etc/jammi`, that the ConfigMap carries
a `jammi.toml` key — unless the container names its config file itself through
`JAMMI_CONFIG`, in which case that path must be a key of the mounted ConfigMap.

Decided over the RENDERED documents (`kustomize build`), so generator renames
(`jammi.toml=jammi-compute.toml`), patches and overlays are all seen as the
cluster would see them.

Run: `python3 ci/scripts/check_kube_config_mounts.py [--self-test]`
Needs `kustomize` on PATH.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path, PurePosixPath

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
KUBE_ROOT = REPO_ROOT / "deploy" / "kubernetes"
CONFIG_DIR = PurePosixPath("/etc/jammi")
DEFAULT_FILE = "jammi.toml"


def kustomizations() -> list[Path]:
    return sorted(path.parent for path in KUBE_ROOT.rglob("kustomization.yaml"))


def render(directory: Path) -> list[dict]:
    built = subprocess.run(
        ["kustomize", "build", str(directory)], capture_output=True, text=True, check=False
    )
    if built.returncode != 0:
        raise RuntimeError(f"kustomize build {directory} failed:\n{built.stderr}")
    return [doc for doc in yaml.safe_load_all(built.stdout) if doc]


def pod_spec(doc: dict) -> dict | None:
    spec = doc.get("spec", {})
    if doc.get("kind") == "Pod":
        return spec
    if doc.get("kind") == "CronJob":
        spec = spec.get("jobTemplate", {}).get("spec", {})
    return spec.get("template", {}).get("spec")


def findings_for(documents: list[dict], origin: str) -> list[str]:
    config_maps = {
        doc["metadata"]["name"]: set(doc.get("data", {}))
        for doc in documents
        if doc.get("kind") == "ConfigMap"
    }
    findings = []
    for doc in documents:
        spec = pod_spec(doc)
        if spec is None:
            continue
        volumes = {
            volume["name"]: volume["configMap"]["name"]
            for volume in spec.get("volumes", [])
            if "configMap" in volume
        }
        workload = f"{origin}: {doc['kind']}/{doc['metadata']['name']}"
        for container in spec.get("containers", []):
            wanted = next(
                (
                    env["value"]
                    for env in container.get("env", [])
                    if env.get("name") == "JAMMI_CONFIG" and "value" in env
                ),
                str(CONFIG_DIR / DEFAULT_FILE),
            )
            for mount in container.get("volumeMounts", []):
                if PurePosixPath(mount["mountPath"]) != CONFIG_DIR:
                    continue
                config_map = volumes.get(mount["name"])
                if config_map is None:
                    continue
                key = PurePosixPath(wanted)
                keys = config_maps.get(config_map)
                if keys is None:
                    findings.append(
                        f"{workload}: container `{container['name']}` mounts ConfigMap "
                        f"`{config_map}`, which this kustomization does not render"
                    )
                elif key.parent != CONFIG_DIR or key.name not in keys:
                    findings.append(
                        f"{workload}: container `{container['name']}` reads `{wanted}` but "
                        f"ConfigMap `{config_map}` mounted at {CONFIG_DIR} carries "
                        f"{sorted(keys)} — the server would start on defaults"
                    )
    return findings


def check() -> list[str]:
    directories = kustomizations()
    if not directories:
        return [f"no kustomization.yaml under {KUBE_ROOT.relative_to(REPO_ROOT)}"]
    return [
        finding
        for directory in directories
        for finding in findings_for(render(directory), str(directory.relative_to(REPO_ROOT)))
    ]


_WORKLOAD = """
apiVersion: apps/v1
kind: {kind}
metadata: {{name: compute}}
spec:
  template:
    spec:
      containers:
        - name: server
          env: {env}
          volumeMounts:
            - {{name: config, mountPath: /etc/jammi, readOnly: true}}
      volumes:
        - name: config
          configMap: {{name: compute-config}}
---
apiVersion: v1
kind: ConfigMap
metadata: {{name: compute-config}}
data:
  {key}: "[worker]\\nenabled = true\\n"
"""


def self_test() -> int:
    failures = []

    def expect(label: str, kind: str, key: str, env: str, refused: bool) -> None:
        text = textwrap.dedent(_WORKLOAD).format(kind=kind, key=key, env=env)
        got = findings_for([d for d in yaml.safe_load_all(text) if d], "fixture")
        if bool(got) != refused:
            failures.append(f"{label}: expected refused={refused}, got {got}")

    expect("the default file name passes", "Deployment", "jammi.toml", "[]", False)
    expect("a renamed key is refused", "StatefulSet", "jammi-compute.toml", "[]", True)
    expect(
        "a renamed key the container names through JAMMI_CONFIG passes",
        "StatefulSet",
        "jammi-compute.toml",
        "[{name: JAMMI_CONFIG, value: /etc/jammi/jammi-compute.toml}]",
        False,
    )
    expect(
        "a JAMMI_CONFIG outside the mount is refused",
        "Deployment",
        "jammi.toml",
        "[{name: JAMMI_CONFIG, value: /opt/jammi.toml}]",
        True,
    )
    rendered = [d.relative_to(REPO_ROOT).as_posix() for d in kustomizations()]
    if not any(path.endswith("overlays/shape-d") for path in rendered):
        failures.append(f"the shape-d overlay is not in the checked set: {rendered}")

    if failures:
        print("kube-config-mounts self-test: FAIL", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"kube-config-mounts self-test: OK ({len(rendered)} kustomizations in the checked set)")
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()
    findings = check()
    if findings:
        print("kube-config-mounts: FAIL", file=sys.stderr)
        for finding in findings:
            print(f"  - {finding}", file=sys.stderr)
        return 1
    print("kube-config-mounts: OK — every config mount carries the file its server reads")
    return 0


if __name__ == "__main__":
    sys.exit(main())
