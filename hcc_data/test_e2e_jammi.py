"""
End-to-end Jammi test: fine-tune BioClinical-ModernBERT → embed hix_data.csv → ANN search.

Pipeline
--------
1. Generate triplet Parquet from hix_data.csv  (if not already present)
2. Connect to the Jammi artifact store
3. Fine-tune BioClinical-ModernBERT-base on the triplets via LoRA
4. Register hix_data.csv as a Jammi source
5. Generate embeddings for every DX_DSC description using the fine-tuned model
6. Encode the query "asthma" with the same model
7. Run ANN search → print top-5 retrieved diagnosis descriptions

Run from the repo root (after `maturin develop --release`):
    python hcc_data/test_e2e_jammi.py

── Model selection (edit these two variables) ──────────────────────────────────

  USE_MODEL_ID   – paste a model ID string (e.g. "jammi:fine-tuned:<uuid>") to
                   skip fine-tuning and use that adapter directly.
                   Set to None to auto-select the most recent completed job.

  FORCE_FINETUNE – set to True to always train a brand-new adapter, even if a
                   completed job already exists in the catalog.

────────────────────────────────────────────────────────────────────────────────
"""

# ── Model-selection knobs ─────────────────────────────────────────────────────
USE_MODEL_ID:   str | None = None   # e.g. "jammi:fine-tuned:d2e7d045-cf6d-4f9c-887d-2452bc7e23ae"
FORCE_FINETUNE: bool       = False  # True → always train a new adapter
# ─────────────────────────────────────────────────────────────────────────────

import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

# ── Windows: add CUDA DLL directory so _native.pyd loads correctly ─────────────
_cuda_bin = Path(os.environ.get("CUDA_PATH",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4")) / "bin"
if _cuda_bin.exists() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(_cuda_bin))

# Show INFO-level Rust logs (epoch loss, sim, lr).  Override with RUST_LOG=debug.
os.environ.setdefault("RUST_LOG", "jammi_ai=info,jammi_engine=info")

HERE             = Path(__file__).parent
REPO_ROOT        = HERE.parent
TRIPLETS_PARQUET = HERE / "triplets_train.parquet"
HIX_CSV          = HERE / "hix_data.csv"
JAMMI_DB_DIR     = HERE / "jammi_test_db"
CATALOG_DB       = JAMMI_DB_DIR / "catalog.db"

SEPARATOR = "=" * 65


def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


# ══════════════════════════════════════════════════════════════════════
# Step 1 – Generate triplet Parquet
# ══════════════════════════════════════════════════════════════════════
section("Step 1 — Generate triplet Parquet (if missing)")

if TRIPLETS_PARQUET.exists():
    print(f"  Parquet already exists: {TRIPLETS_PARQUET}")
else:
    print("  Parquet not found — running generate_triplets.py …")
    subprocess.check_call([sys.executable, str(HERE / "generate_triplets.py")])
    print("  Done.")

if not HIX_CSV.exists():
    print(f"\nERROR: hix_data.csv not found at {HIX_CSV}")
    sys.exit(1)

print(f"  hix_data.csv found: {HIX_CSV}")


# ══════════════════════════════════════════════════════════════════════
# Step 2 – Import jammi
# ══════════════════════════════════════════════════════════════════════
section("Step 2 — Import jammi")

try:
    import jammi
    print(f"  jammi imported OK (version: {getattr(jammi, '__version__', 'n/a')})")
except ImportError as err:
    err_str = str(err)
    if "DLL load failed" in err_str or "specified module could not be found" in err_str:
        print(f"\n  ERROR: {err}")
        print("  A CUDA DLL could not be found. Ensure CUDA 12.x is installed and set:")
        print(r"    set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4")
        sys.exit(1)
    print("  jammi not installed — building with maturin …")
    r = subprocess.run(
        ["maturin", "develop", "--release", "--features", "cuda"],
        cwd=str(REPO_ROOT),
    )
    if r.returncode != 0:
        print("  Build failed. Install maturin and ensure Rust + CUDA are present.")
        sys.exit(1)
    import jammi


# ══════════════════════════════════════════════════════════════════════
# Step 3 – Connect to the Jammi artifact store
# ══════════════════════════════════════════════════════════════════════
section("Step 3 — Connect to Jammi artifact store")

JAMMI_DB_DIR.mkdir(parents=True, exist_ok=True)
db = jammi.connect(artifact_dir=str(JAMMI_DB_DIR))
print(f"  Connected → {JAMMI_DB_DIR}")


# ══════════════════════════════════════════════════════════════════════
# Step 4 – Fine-tune BioClinical-ModernBERT on triplets
# ══════════════════════════════════════════════════════════════════════
section("Step 4 — Fine-tune BioClinical-ModernBERT-base on HCC triplets")

# ── Register the triplet source ────────────────────────────────────────
TRIPLET_SOURCE = "hcc_triplets"
try:
    db.add_source(TRIPLET_SOURCE, path=str(TRIPLETS_PARQUET), format="parquet")
    print(f"  Registered source '{TRIPLET_SOURCE}'")
except Exception as e:
    print(f"  Source already registered (or note: {e})")

# ── Choose base model ──────────────────────────────────────────────────
LOCAL_MODEL = REPO_ROOT / "models_nexus"
if LOCAL_MODEL.exists():
    BASE_MODEL = str(LOCAL_MODEL)
    print(f"  Using local model: {BASE_MODEL}")
else:
    BASE_MODEL = "hf://thomas-sounack/BioClinical-ModernBERT-base"
    print(f"  Using HuggingFace model: {BASE_MODEL}")
    print("  (set HUGGING_FACE_HUB_TOKEN if you get a 403 / 404)")

# ── List all completed fine-tune jobs from the catalog ────────────────
def list_completed_jobs(catalog_path: Path) -> list[tuple[str, str, str]]:
    """Return [(job_id, output_model_id, created_at), ...] newest-first."""
    if not catalog_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(catalog_path))
        cur  = conn.cursor()
        tables = [r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        if "fine_tune_jobs" not in tables:
            conn.close()
            return []
        rows = cur.execute(
            "SELECT job_id, output_model_id, created_at FROM fine_tune_jobs "
            "WHERE status='completed' AND output_model_id IS NOT NULL "
            "ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        return rows
    except Exception:
        return []

completed_jobs = list_completed_jobs(CATALOG_DB)

print()
if completed_jobs:
    print(f"  Previously completed fine-tune jobs ({len(completed_jobs)} found):")
    print(f"  {'#':<4} {'Created':<26} {'Model ID'}")
    print(f"  {'-'*3} {'-'*25} {'-'*50}")
    for i, (job_id, model_id, created_at) in enumerate(completed_jobs, 1):
        marker = " ◀ most recent" if i == 1 else ""
        print(f"  {i:<4} {str(created_at):<26} {model_id}{marker}")
    print()
    print("  Adapter weights stored under:")
    print(f"    {JAMMI_DB_DIR / 'models'}")
else:
    print("  No completed fine-tune jobs found in catalog yet.")

# ── Resolve which model to use ─────────────────────────────────────────
fine_tuned_model_id: str | None = None

if FORCE_FINETUNE:
    print("  FORCE_FINETUNE=True → will train a fresh adapter regardless.")
elif USE_MODEL_ID:
    fine_tuned_model_id = USE_MODEL_ID
    print(f"  USE_MODEL_ID set → using: {fine_tuned_model_id}")
elif completed_jobs:
    fine_tuned_model_id = completed_jobs[0][1]   # most recent
    print(f"  Auto-selected most recent model: {fine_tuned_model_id}")
    print("  (set USE_MODEL_ID=<id> to pin a specific one, or FORCE_FINETUNE=True to retrain)")

if fine_tuned_model_id:
    print(f"\n  Skipping fine-tuning — using model: {fine_tuned_model_id}")
else:
    print("\n  Starting fine-tuning …")
    print(f"  base_model   : {BASE_MODEL}")
    print(f"  source       : {TRIPLET_SOURCE}")
    print(f"  lora_rank    : 16   lora_alpha : 32   lora_dropout : 0.1")
    print(f"  target_modules : ['Wqkv', 'out_proj', 'Wi', 'Wo']")
    print(f"  epochs       : 10    batch_size : 16   grad_accum : 1")
    print(f"  warmup_steps : 6 (~10% of total)   weight_decay : 0.01   max_grad_norm : 1.0")
    print(f"  triplet_margin : 0.3   backbone_dtype : bf16   max_seq_length : 128")
    print()

    t0 = time.time()
    job = db.fine_tune(
        source=TRIPLET_SOURCE,
        base_model=BASE_MODEL,
        columns=["anchor", "positive", "negative"],
        method="lora",
        task="embedding",

        lora_rank=16,
        lora_alpha=32.0,
        lora_dropout=0.1,
        target_modules=["Wqkv", "out_proj", "Wi", "Wo"],

        epochs=10,
        batch_size=8,
        learning_rate=2e-5,
        warmup_steps=6,
        triplet_margin=0.3,

        validation_fraction=0.1,
        early_stopping_patience=3,
        early_stopping_metric="val_loss",

        gradient_accumulation_steps=1,
        max_seq_length=128,
        backbone_dtype="bf16",
        weight_decay=0.01,
        max_grad_norm=1.0,
    )

    print(f"  Job submitted: {job.job_id}")
    print("  Training logs stream below …")
    print("-" * 65)

    job.wait()

    elapsed = time.time() - t0
    fine_tuned_model_id = job.model_id

    print(f"\n  Fine-tuning COMPLETE in {elapsed:.1f}s")
    print(f"  Model ID : {fine_tuned_model_id}")
    print(f"  Status   : {job.status()}")


# ══════════════════════════════════════════════════════════════════════
# Step 5 – Register hix_data.csv and generate embeddings
# ══════════════════════════════════════════════════════════════════════
section("Step 5 — Register hix_data.csv and generate embeddings")

HIX_SOURCE = "hix_diagnoses"

try:
    db.add_source(HIX_SOURCE, path=str(HIX_CSV), format="csv")
    print(f"  Registered source '{HIX_SOURCE}' ({HIX_CSV.name})")
except Exception as e:
    print(f"  Source already registered (or note: {e})")

# Check whether embeddings for this source+model are already on disk.
# Jammi names each artifact  <source>__text_embedding__<model_id_safe>__<ts>.manifest.json
# We detect any existing manifest rather than unconditionally re-embedding.
JAMMI_DB_SUBDIR  = JAMMI_DB_DIR / "jammi_db"
model_slug       = fine_tuned_model_id.replace(":", "_").replace("/", "_")
existing_manifests = list(JAMMI_DB_SUBDIR.glob(
    f"{HIX_SOURCE}__text_embedding__*{model_slug.split('_')[-1]}*.manifest.json"
)) if JAMMI_DB_SUBDIR.exists() else []

if existing_manifests and not FORCE_FINETUNE:
    manifest = existing_manifests[-1]   # most recent
    print(f"\n  Embeddings already exist — skipping generation.")
    print(f"  Manifest : {manifest.name}")
    import json as _json
    meta = _json.loads(manifest.read_text())
    print(f"  Vectors  : {meta.get('count')} × {meta.get('dimensions')}d  ({meta.get('metric')} distance)")
    print("  (delete hcc_data/jammi_test_db/ and re-run to regenerate)")
else:
    # Embed DX_DSC (diagnosis description); DX_CD is the unique row key.
    print(f"\n  Generating embeddings for DX_DSC using model: {fine_tuned_model_id}")
    print("  (This embeds all ~11 k diagnosis descriptions — may take a few minutes)")

    t0 = time.time()
    db.generate_text_embeddings(
        source=HIX_SOURCE,
        model=fine_tuned_model_id,
        columns=["DX_DSC"],
        key="DX_CD",
    )
    elapsed = time.time() - t0
    print(f"  Embeddings generated in {elapsed:.1f}s")


# ══════════════════════════════════════════════════════════════════════
# Step 6 – Semantic search: "asthma" → top-5 diagnoses
# ══════════════════════════════════════════════════════════════════════
section('Step 6 — ANN search: "asthma" → top-5 diagnoses')

QUERY_TEXT = "asthma"
print(f"  Query : \"{QUERY_TEXT}\"")
print(f"  Model : {fine_tuned_model_id}")

query_vec = db.encode_text_query(fine_tuned_model_id, QUERY_TEXT)
print(f"  Encoded query — dim={len(query_vec)}, first 8: {[round(v,4) for v in query_vec[:8]]}")

result = db.search(HIX_SOURCE, query=query_vec, k=5)
result_tbl = result.run()

print(f"\n  Top-5 results ({result_tbl.num_rows} rows returned):")
print(f"  Columns: {result_tbl.schema.names}")
print()

# Build a code→description lookup from the result table and / or the CSV
result_df = result_tbl.to_pandas()

# The result includes at minimum the key column (DX_CD) and _score.
# If DX_DSC is not in the result, join it from the raw CSV.
if "DX_DSC" not in result_df.columns:
    import pandas as pd
    hix_df = pd.read_csv(HIX_CSV, dtype=str)
    result_df = result_df.merge(hix_df[["DX_CD", "DX_DSC"]], on="DX_CD", how="left")

score_col = next(
    (c for c in result_df.columns if "score" in c.lower() or "similarity" in c.lower()),
    None,
)

print(f"  {'Rank':<5} {'Score':>8}  {'DX_CD':<12} {'DX_DSC'}")
print(f"  {'-'*4} {'-'*8}  {'-'*11} {'-'*40}")
for rank, row in enumerate(result_df.itertuples(), start=1):
    score = getattr(row, score_col, "n/a") if score_col else "n/a"
    score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
    dx_cd  = getattr(row, "DX_CD",  "n/a")
    dx_dsc = getattr(row, "DX_DSC", "n/a")
    print(f"  {rank:<5} {score_str:>8}  {dx_cd:<12} {dx_dsc}")

print(f"\n{SEPARATOR}")
print("  End-to-end test COMPLETE.")
print(SEPARATOR)
