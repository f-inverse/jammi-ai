# Provenance — `profile_421_run2/`

Cut (verbatim — every byte copied unmodified, never hand-edited) from the
REAL pod `p421` run2 pull that produced the committed close-out artifact
`crates/jammi-kernels/artifacts/cuda-runs/2026-09-07-profile-421-towers-c1b0b0ba-a100-sxm4.json`
(`git_sha c1b0b0bad1f79a4ad6c298400e6ea19cc1ca633c`, box `4dc3a4394178`,
`nsys 2025.3.2.474-253236389321v0`, `NVIDIA A100-SXM4-80GB`). Every value
below is re-derived by walking this fixture directory, never hand-typed —
see `gen_provenance_table.py`-equivalent one-liner at the bottom to
regenerate this table.

`profile_421_artifact.py`'s own `producer.input_sha256` names every byte
read directly off `--legs-dir`/`--p2-dir` — every leg's own `manifest.json`
and `census.json`, `census.pre-demangle.json` for every leg in
`KERNEL_IDENTITY_SPLIT_LEGS`, and every witnessed P2 tower's own
`manifest.json` — alongside the three top-level report files
(`--merge-json`/`--attribution-json`/`--identity`); `census.pre-demangle.
json` (the SOLE source of the identity sidecar's own kernel-identity split
count) is committed here for exactly that reason. This directory carries
every byte a producer script reads, real, byte-for-byte, the same way the
`profile_421_clip_text_a1/` (etc.) attribution fixtures carry theirs.

## What is included, and why

Per leg (`legs/<leg_id>/`, 12 legs: `{clip-text,clip-vision,htsat}-
{A1,A2,D1,D2}`):

- `manifest.json` — read by `profile_421_merge.py` (build/checkpoint
  identity) AND by `profile_421_artifact.py` itself (`collect_identity`'s
  `git_sha`/`box` cross-check).
- `census.json` — read by `profile_421_merge.py` (`gpu_kernel_us_per_step`,
  the positive-proof cross-check) AND by `profile_421_artifact.py` itself
  (`build_legs`'s/`compute_findings`'s `launches_per_step`) AND by
  `profile_421_attribute.py` (`by_kernel_and_grid` chain attribution).
- `census.pre-demangle.json` — read by `profile_421_artifact.py` itself
  (`compute_kernel_identity_split_count`, called for `clip-text-A2`, the
  ONE leg in `KERNEL_IDENTITY_SPLIT_LEGS`) — the SOLE source of the
  identity sidecar's own "N DISTINCT ... instantiations" split count.
  Committed for EVERY leg (not just `clip-text-A2`), since it is real,
  byte-for-byte pod output and `kernel_census.py`'s own pre-demangle pass
  produces one per leg regardless of which leg a given run's identity
  sidecar happens to quote a split count from.
- `run_n.json` / `run_m.json` — read by `profile_421_merge.py` (the two
  positive-proof timing samples every `per_step` wall/front/busy/residual
  number is built from). NOT read by `profile_421_artifact.py` itself
  (its own `producer.input_sha256` therefore does not name these — see
  that module's own "Input-completeness" doc section) — committed here
  so the FULL pipeline (`profile_421_merge.py` -> `profile_421_attribute.py`
  -> `profile_421_artifact.py`), not just the close-out producer's own
  narrower read set, is provably reproducible from this one fixture.

Per witnessed P2 (BF16 pre-flight) tower (`p2-bf16/<tower>/`, 3 towers:
`clip-text`, `clip-vision`, `htsat`):

- `manifest.json` — read by `profile_421_merge.py` AND by
  `profile_421_artifact.py` itself (`collect_identity`'s P2 cross-check).
- `run.json` — read by `profile_421_merge.py` only (the P2 pre-flight's
  single untraced run) — like `run_n.json`/`run_m.json` above, committed
  for full-pipeline reproducibility, not named in `profile_421_artifact.
  py`'s own `producer.input_sha256`.

Top level:

- `merge.json` — `profile_421_merge.py`'s own output over the 12 legs +
  3 P2 towers above (`--legs-dir <this>/legs --p2-dir <this>/p2-bf16`).
  Byte-for-byte the SAME file the committed artifact's own
  `producer.input_sha256.merge_json` names (verified: sha256
  `c80cd7eaf480e6e167861b0ac7ae9855d58df0ec84c2799864cdfd6c931db264`,
  matching the committed artifact exactly).
- `attribution.json` — `profile_421_attribute.py`'s own output
  (`--legs-dir <this>/legs --merge-json <this>/merge.json`), REGENERATED
  fresh against this exact fixture (not a copy of some other run's file) —
  byte-for-byte the SAME as the committed artifact's own
  `producer.input_sha256.attribution_json` (verified: sha256
  `3d5f33c280232f1ee2ee4740f3e18d84dffc6532f113b595bb1c9f34b63007a3`,
  matching the committed artifact exactly). `profile_421_attribute.py` is
  deterministic (no timestamp, no wall-clock-dependent field), so
  regenerating it again from this fixture reproduces the identical bytes —
  proven by `test_profile_421_artifact.py`'s `RealFixtureRegenerationTests`.

The non-numeric identity sidecar itself (`ci/scripts/perf/
profile_421_run2_identity.json`) is NOT duplicated into this fixture
directory — it is already a first-class, separately-committed, hand-
authored file (its own module-doc header covers its own provenance), and
`RealFixtureRegenerationTests` reads it straight from its real committed
path, exactly as the original `main` invocation did.

## What is deliberately OMITTED, and why

- `census.stderr` / `census.stdout` / `run_n.stderr` / `run_m.stderr` /
  `run.stderr` (per leg / per P2 tower): captured `nsys`/`jammi-bench`
  process output, never read by ANY producer script (`profile_421_merge.
  py`, `profile_421_attribute.py`, `profile_421_artifact.py`) — per this
  crate's own "commit what is READ, never what merely sat alongside it"
  convention (see the `profile_421_clip_text_a1/` etc. fixtures'
  own `PROVENANCE.md`).
- `p2-merge.json` (a top-level file in the original pod pull): an earlier,
  P2-only merge output from BEFORE the 12-leg run completed, superseded
  once `merge.json` itself carried the SAME P2 rows under its own
  top-level `p2_bf16` key (verified: `merge.json`'s `p2_bf16` has 3 rows,
  `p2_total: 3, p2_pass: 3, p2_fail: 0`) — not read by any producer script
  once superseded.
- `preflight/` (a top-level directory in the original pod pull): the
  `PROFILE_421_LEGS_PREFLIGHT_ONLY=1` dry-run stage's own output (build +
  checkpoint-fetch smoke test, before any of the 12 legs or the P2 towers
  ran) — not read by `profile_421_merge.py`, `profile_421_attribute.py`,
  or `profile_421_artifact.py` at all; its role ends once the pre-flight
  passes and the real run starts.
- Raw `.nsys-rep` exports and the raw `.sqlite` exports `kernel_census.py`
  reads to PRODUCE `census.json`/`census.pre-demangle.json`: per
  `profile_421_run2_identity.json`'s own recorded deviation, these were
  deleted on-pod after each leg's census was produced (disk budget) and
  were never pulled off-pod at all — there is nothing to commit; the
  census JSON files above are the full extent of what survived the run.

## sha256 manifest (every file in this directory except this one)

Regenerate with:

```python
import hashlib
from pathlib import Path
fix = Path("ci/scripts/perf/fixtures/profile_421_run2")
for p in sorted(fix.rglob("*")):
    if p.is_file() and p.name != "PROVENANCE.md":
        print(p.relative_to(fix).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_size)
```

| path | sha256 | bytes |
| --- | --- | --- |
| `attribution.json` | `3d5f33c280232f1ee2ee4740f3e18d84dffc6532f113b595bb1c9f34b63007a3` | 94122 |
| `legs/clip-text-A1/census.json` | `ddb381c27f6eba0e48100aa08f33554890763abf574cdc0e3d6bb169a6dad1e6` | 33902 |
| `legs/clip-text-A1/census.pre-demangle.json` | `d0784dd312f8b8916efc86f96928ef5b1a43c4cd252dd3fed479d3fc91d54ea0` | 31628 |
| `legs/clip-text-A1/manifest.json` | `f7089429823a022363eb7b37ff47e3e1e3feca40a7b59effc8c5ddb4ce01e47c` | 1188 |
| `legs/clip-text-A1/run_m.json` | `3e969b8e8df6c0976f6149da40efd0dc46cd7b91da36cb98c0a2ef8f43c6c6eb` | 4018 |
| `legs/clip-text-A1/run_n.json` | `8c5a1d899ae4c1d701c04ddb2777c3b24878b8c476844d68d4a1b2b25689a4a5` | 4012 |
| `legs/clip-text-A2/census.json` | `b548dd36521fbe8a5155c80edf9fb77b000ae96dbc6b61e4666dcdfba4fb4953` | 41388 |
| `legs/clip-text-A2/census.pre-demangle.json` | `5d4adf1cf47eeff3249b3ae292fd87f6202129458e51453e1acc9fbecee6d12a` | 37977 |
| `legs/clip-text-A2/manifest.json` | `13b976c19c156e7f705481f767209e92de3c075a1f82394d68bbd3ce84733104` | 1189 |
| `legs/clip-text-A2/run_m.json` | `fbbea34d405cb317c168f32bc03d73cbc2bd2a9148bf84d399691d38a40c3d27` | 3988 |
| `legs/clip-text-A2/run_n.json` | `1ed2e6587aecc7bd1dead5724b14c8319c01be99b43edffb7f6989fc864ce252` | 3975 |
| `legs/clip-text-D1/census.json` | `c560b0138481c93037562b7c2719942f3b6b40297e97ef37a854413f57765584` | 35445 |
| `legs/clip-text-D1/census.pre-demangle.json` | `7237716f938e235fef53f9944bb9cbdf3f1af82de3bff6ed85e77357aaa740a8` | 32491 |
| `legs/clip-text-D1/manifest.json` | `2c931c8591f5ef1f931b51351e979e5765e6098700e56296a5d53e4061ec7113` | 1234 |
| `legs/clip-text-D1/run_m.json` | `a06f4c1e4ad1b3e5f6b8bfa015039f24e13bfc982ac74b312c41ba8bb2a30e63` | 4206 |
| `legs/clip-text-D1/run_n.json` | `316e2ab8a7ed31889089f69f8828633463f893c40801c8b4cbaaf922b2c84dee` | 4204 |
| `legs/clip-text-D2/census.json` | `d88b49017f414d1e28aaab764d4818935c7695ed9693580737ad0c64029d4f0b` | 33658 |
| `legs/clip-text-D2/census.pre-demangle.json` | `454fb73216ae5480d0fb02cb0d546e01ff9df9844ca8a0f33e7cfc1e9ddfda59` | 30706 |
| `legs/clip-text-D2/manifest.json` | `d375a00613df48edfbb6b2e2020e9c9fb81d07591dbf2ac42853a38f41b0ac55` | 1212 |
| `legs/clip-text-D2/run_m.json` | `f123c828c968e63615307a794885b174eb24a7a2efd6a56415faa0399a071d18` | 4120 |
| `legs/clip-text-D2/run_n.json` | `30710994a23cf353d86ab3c471d0c35703767ebfa0ccff98ac34f2331eed9958` | 4118 |
| `legs/clip-vision-A1/census.json` | `a32e1fe99a1ab7d34d0cc649b30efd4007888f20566dcafa8762313343835f78` | 35943 |
| `legs/clip-vision-A1/census.pre-demangle.json` | `5c0f1665f7fa108bc46860cbbff204d6547545fa5bd946944e134354666d1d66` | 33091 |
| `legs/clip-vision-A1/manifest.json` | `4b2a6fa28778978c2959b25e46cca14dab3623c780fc8d88a2bb6c17b8eb8447` | 1208 |
| `legs/clip-vision-A1/run_m.json` | `57cfba6e604398de2847cf5ee365571ee1a4ceea0a37f4e39d75bfb94dd19927` | 4107 |
| `legs/clip-vision-A1/run_n.json` | `c6d4e28670ea52df218ce63555f0e3f17ceaee8b8299fc3a3037062b0b28ce01` | 4101 |
| `legs/clip-vision-A2/census.json` | `0899cf0672f19d09c919e87cdc0d599509b1c121a20f7d0fce1bdf53e70fde86` | 41224 |
| `legs/clip-vision-A2/census.pre-demangle.json` | `3cb597654da2b82764090bf6a4344f6def6a0085fa91abbee00f1e2e065f35ab` | 38091 |
| `legs/clip-vision-A2/manifest.json` | `f207112245a3aa457ea4d19ff2975d02dd677239bec6225baf940c9e4aa7b3e5` | 1209 |
| `legs/clip-vision-A2/run_m.json` | `9a66ea4703e91acfd0646fb4789ab335393b36871a407858001bbfea16bcaf5b` | 4103 |
| `legs/clip-vision-A2/run_n.json` | `18e2ec6a38ebc93b19bdf57ad4fbdfb1a89076ff2803f20fe80793a802fc64fd` | 4093 |
| `legs/clip-vision-D1/census.json` | `82243a1215c8d34fefd284e9d8bcede0640839a11eb6bc7483d2812aadecf752` | 38297 |
| `legs/clip-vision-D1/census.pre-demangle.json` | `375212f973b93f3063edb2e613a34c62b9ae842a8e48f69e3fc648aec861eb25` | 34628 |
| `legs/clip-vision-D1/manifest.json` | `e082ff71ae1ecb22137b4aa6a6d0ea7779091c6d22ce526984b6fe635b407b3e` | 1254 |
| `legs/clip-vision-D1/run_m.json` | `898ff83a88049bf7e2f78cbb697bad4791238549dd4838afa6e3174398fab0f0` | 4294 |
| `legs/clip-vision-D1/run_n.json` | `540ad30eeef4ae021c0e6abe15609e83ebada9b67bf4d4869fab3b9f99546d2c` | 4289 |
| `legs/clip-vision-D2/census.json` | `bb7be951a8f8ed83d396200cb3fbcdff2b91493fd213a5544aeb14fe37956596` | 36599 |
| `legs/clip-vision-D2/census.pre-demangle.json` | `8d5487f509681b3782ff3f72416a917db142db8079302937ecb27f72bd122fc3` | 32939 |
| `legs/clip-vision-D2/manifest.json` | `4c5820f355ecf89f0bba4d5134a0e6110abbfa9daabfceb80f7dfe996ecf9b6b` | 1231 |
| `legs/clip-vision-D2/run_m.json` | `2db2ba5877f7d7984f53853fe37694c574c618419822d9bca532a316e449f6ad` | 4211 |
| `legs/clip-vision-D2/run_n.json` | `f8bdef4c489945095ac8ca4b5afc91ff25ac5ed5058338644be72adfe32377b4` | 4208 |
| `legs/htsat-A1/census.json` | `2870be92b7518ed035dc760a489ab4c35a9f12eff6f80fc966ca281263b93f68` | 84328 |
| `legs/htsat-A1/census.pre-demangle.json` | `95c57ec73c4142876279a409a8f859ca197678eaee89946ebca051b2755b8c6d` | 78117 |
| `legs/htsat-A1/manifest.json` | `baa602b397d66d909b80e82711e665b8ce84fc861e86695561d0286e18881710` | 1288 |
| `legs/htsat-A1/run_m.json` | `9a26086aa57c0412afc287a16d19e275a56c4bb4124afe6c1ac3d0972a2d1c04` | 4260 |
| `legs/htsat-A1/run_n.json` | `2b7729bf50f0eb200c5f9b50344e9bbb50cd23a4afdef3744222622a655a0a4a` | 4257 |
| `legs/htsat-A2/census.json` | `a7ceb8c72da1b623045a7c2d7c6335664b0535d0fb2d981ff5cdb8f4779e9ab5` | 96173 |
| `legs/htsat-A2/census.pre-demangle.json` | `cec26c8ec9f2d992e3bbc15c0c44fcec73b01ab2bc9d148e620d82578a46b7ec` | 88331 |
| `legs/htsat-A2/manifest.json` | `2a24f3ff7c5580d58636a8d64f2b3ec230bda3f03f45844166aacc7b0726e4ee` | 1287 |
| `legs/htsat-A2/run_m.json` | `58880dc9ae7e14b62e524f5e4b2b5c57c797091b657cde3d14d5324213742ea9` | 4237 |
| `legs/htsat-A2/run_n.json` | `6e69c3326ad8c9a6c94d0e8d0341d8b6b016a2cbe1f03978dfb68e82f524e76c` | 4232 |
| `legs/htsat-D1/census.json` | `3a8841a9cbe79afddc981c4d17f8bd9bae75f012f6ae349605be27dd46ee9621` | 99581 |
| `legs/htsat-D1/census.pre-demangle.json` | `5f3f9b575a21a7d417803af7431ebb4363a0ae452483a7dc438a26d08bec330e` | 89683 |
| `legs/htsat-D1/manifest.json` | `b04fdd68772c13190950dd3ecab1193c9dfa89a80ee9b550e31739bc8657ebcd` | 1354 |
| `legs/htsat-D1/run_m.json` | `846950b904e311bd8af97a571ad8b3e9047d756af7ea6837c0918a8f826bfc5f` | 4529 |
| `legs/htsat-D1/run_n.json` | `eedfb339bdd24655431ad16218f985330dc8d4cd4d2c562b5bc92ebeea2a042d` | 4526 |
| `legs/htsat-D2/census.json` | `8313ea36b66a6a36f6c48f21874a6ba36b55dcca1852e6688d0305817e4bec4f` | 91455 |
| `legs/htsat-D2/census.pre-demangle.json` | `3877a8aa2feaa6138938c68b79b24c4088480a32eb55c50ee16455aff553afa0` | 81564 |
| `legs/htsat-D2/manifest.json` | `4ed7b7546b926ef313a881785a568ab8bb2c940576a028d280d1836ac5d79c45` | 1312 |
| `legs/htsat-D2/run_m.json` | `6874cbd95802d7f376759f36186b279aa7171c87aec0630890d49c683cf2e9fa` | 4366 |
| `legs/htsat-D2/run_n.json` | `593b6e876848f35b4e255afd04c4ccfce1e31c910b6a536274297bbd6e44b6c8` | 4364 |
| `merge.json` | `c80cd7eaf480e6e167861b0ac7ae9855d58df0ec84c2799864cdfd6c931db264` | 32000 |
| `p2-bf16/clip-text/manifest.json` | `ecc1f5b2b9ce1a33fcd5d145602997d1f1cacc4679937670b0756ac0997e3506` | 486 |
| `p2-bf16/clip-text/run.json` | `25f7da364ec4dfae99e6508d7d6be082a98b9f8e36dce903e8266c6d5f16566a` | 3979 |
| `p2-bf16/clip-vision/manifest.json` | `ff8c0c811555cd0cf28fa2899b65531ac5031b60713510ac3ff5d70caf7f30c5` | 489 |
| `p2-bf16/clip-vision/run.json` | `5c47d8bd51c566d358eaa83ad99bb68c5a752c3a7bc509f924dc0864461f39fb` | 4107 |
| `p2-bf16/htsat/manifest.json` | `6f41b0551b4abeb9692e736f46bf75a4a3c5f1d3c81819802221d6a25eb6e9f8` | 570 |
| `p2-bf16/htsat/run.json` | `58ca1df93e9b1d5e8a35f5d98c49d528049f3c4ba456d2d7453a624251c15428` | 4227 |

Total: 68 files, 1,533,003 bytes.
