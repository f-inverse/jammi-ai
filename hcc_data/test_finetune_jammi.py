"""
Fine-tune ModernBERT on 100 HCC medical diagnosis triplets using Jammi AI.

Setup (run once from the repo root):
    python -m venv .venv
    .venv\\Scripts\\activate
    pip install maturin pandas pyarrow
    maturin develop --release

Then run:
    python hcc_data/test_finetune_jammi.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# Show INFO-level training logs from the Rust backend (epoch loss, pos/neg sim, lr).
# Override with RUST_LOG=debug for verbose output or RUST_LOG=warn to silence.
os.environ.setdefault("RUST_LOG", "jammi_ai=info,jammi_engine=info")

# Python 3.8+ no longer searches PATH for DLLs when loading extension modules.
# os.add_dll_directory() is the correct API for adding extra search paths on Windows.
# Without this, _native.pyd raises "DLL load failed" because cudart64_124.dll,
# cublas64_12.dll, etc. are not in the trusted DLL search path.
_cuda_bin = Path(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4")) / "bin"
if _cuda_bin.exists() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(_cuda_bin))

HERE      = Path(__file__).parent
REPO_ROOT = HERE.parent
TRIPLETS_PARQUET = HERE / "triplets_train.parquet"
JAMMI_DB_DIR     = HERE / "jammi_test_db"

# ── 1. Generate Parquet data if missing ───────────────────────────────────────
if not TRIPLETS_PARQUET.exists():
    print("Parquet not found — generating ...")
    subprocess.check_call([sys.executable, str(HERE / "generate_triplets.py")])
else:
    print(f"Parquet found: {TRIPLETS_PARQUET}")

# ── 2. Import jammi (build with maturin if needed) ────────────────────────────
try:
    import jammi
    print(f"jammi imported OK")
except ImportError as _import_err:
    _err_str = str(_import_err)
    if "DLL load failed" in _err_str or "specified module could not be found" in _err_str:
        print(f"\nERROR: {_import_err}")
        print("\nA CUDA DLL could not be found. Ensure CUDA 12.x is installed and run:")
        print(f"  set CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4")
        print("  python hcc_data/test_finetune_jammi.py")
        sys.exit(1)
    # Package genuinely not installed — build it.
    print("\njammi not installed — building with maturin (requires Rust toolchain + CUDA) ...")
    r = subprocess.run(
        ["maturin", "develop", "--release", "--features", "cuda"],
        cwd=str(REPO_ROOT),
    )
    if r.returncode != 0:
        print("\nBuild failed. Make sure you have run:")
        print("  pip install maturin")
        print("  (and that Rust + CUDA are installed)")
        sys.exit(1)
    import jammi

# ── 3. Open the Jammi database ────────────────────────────────────────────────
# jammi.connect() creates/opens an InferenceSession backed by a SQLite catalog
# and a file-system artifact store under artifact_dir.
JAMMI_DB_DIR.mkdir(parents=True, exist_ok=True)
print(f"\nConnecting to Jammi database at: {JAMMI_DB_DIR}")
db = jammi.connect(artifact_dir=str(JAMMI_DB_DIR))

# ── 4. Register the Parquet file as a named data source ──────────────────────
# Jammi registers the file with DataFusion so it can be queried with SQL or
# read by the training data loader.
SOURCE = "hcc_triplets"
print(f"Registering source '{SOURCE}' ...")
try:
    db.add_source(SOURCE, path=str(TRIPLETS_PARQUET), format="parquet")
except Exception as e:
    print(f"  (already registered or note: {e})")

# ── 5. Choose the base model ──────────────────────────────────────────────────
# Priority order:
#   1. ./models_nexus   — local directory drop-in
#   2. LOCAL_BIOCLINICAL — explicit local path (set below if you have the model)
#   3. HuggingFace Hub  — requires public access or HUGGING_FACE_HUB_TOKEN set

LOCAL_BIOCLINICAL = None   # e.g. Path(r"C:\models\BioClinical-ModernBERT-large")

LOCAL_MODEL = REPO_ROOT / "models_nexus"
if LOCAL_MODEL.exists():
    BASE_MODEL = str(LOCAL_MODEL)
    print(f"\nUsing local model (models_nexus): {BASE_MODEL}")
elif LOCAL_BIOCLINICAL and Path(LOCAL_BIOCLINICAL).exists():
    BASE_MODEL = f"local:{LOCAL_BIOCLINICAL}"
    print(f"\nUsing local model: {BASE_MODEL}")
else:
    BASE_MODEL = "hf://thomas-sounack/BioClinical-ModernBERT-base"
    print(f"\nDownloading from HuggingFace: {BASE_MODEL}")
    print("  If you get a 404, set HUGGING_FACE_HUB_TOKEN or point LOCAL_BIOCLINICAL to a local copy.")

# ── 6. Launch fine-tuning ─────────────────────────────────────────────────────
# Memory-tuned config for an 8 GB consumer GPU (e.g. RTX 4060 with ~7-8 GB free).
# Notes:
#   - Triplet loss runs THREE forward passes per batch (anchor + positive + negative),
#     so with `batch_size=N` peak activations are equivalent to a 3*N classification
#     batch. We use batch_size=2 here; raise to 4 if VRAM allows.
#   - backbone_dtype="bf16" loads the frozen backbone weights in BFloat16, cutting
#     backbone memory by ~half (e.g. ~450 MB → ~225 MB for ModernBERT-base). LoRA
#     A/B matrices are always kept in FP32 for numerical stability. Use "f32" to
#     disable (safe default on CPUs that don't support BF16).
#   - gradient_accumulation_steps accumulates gradients across micro-batches by
#     calling backward() per step, so activation memory is NOT multiplied. Safe
#     to raise (e.g. to 4) to achieve a larger effective batch without extra VRAM.
#   - max_seq_length=64 is plenty for short medical-diagnosis triplets and roughly
#     halves activation memory across all transformer layers vs. seq=128.
#   - target_modules=["Wqkv"] adapts only attention; the most parameter-efficient
#     LoRA target for embedding fine-tuning. Add "Wo"/"Wi" back later if quality
#     plateaus and VRAM allows.

LORA_RANK            = 16
LORA_ALPHA           = 32.0
LORA_DROPOUT         = 0.1
TARGET_MODULES       = ["Wqkv"]
EPOCHS               = 2
BATCH_SIZE           = 2
GRAD_ACCUM_STEPS     = 4    # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS = 8
LEARNING_RATE        = 2e-5
WARMUP_STEPS         = 50
TRIPLET_MARGIN       = 0.3
MAX_SEQ_LENGTH       = 64
VALIDATION_FRACTION  = 0.1
EARLY_STOP_PATIENCE  = 2
BACKBONE_DTYPE       = "bf16"  # "bf16" saves ~half VRAM; "f32" for CPU / older GPUs

print("\n" + "="*60)
print("Starting Jammi fine-tuning ...")
print(f"  model               : {BASE_MODEL}")
print(f"  source              : {SOURCE}")
print(f"  lora_rank           : {LORA_RANK}")
print(f"  lora_alpha          : {LORA_ALPHA}")
print(f"  target_modules      : {TARGET_MODULES}")
print(f"  triplet_margin      : {TRIPLET_MARGIN}")
print(f"  epochs              : {EPOCHS}  (set to 10 for production)")
print(f"  batch_size          : {BATCH_SIZE}")
print(f"  grad_accum_steps    : {GRAD_ACCUM_STEPS}  (effective batch = {BATCH_SIZE * GRAD_ACCUM_STEPS})")
print(f"  max_seq_length      : {MAX_SEQ_LENGTH}")
print(f"  learning_rate       : {LEARNING_RATE}")
print(f"  backbone_dtype      : {BACKBONE_DTYPE}")
print("="*60)

t0 = time.time()

job = db.fine_tune(
    source=SOURCE,
    base_model=BASE_MODEL,
    columns=["anchor", "positive", "negative"],
    method="lora",
    task="embedding",

    lora_rank=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,

    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    triplet_margin=TRIPLET_MARGIN,

    validation_fraction=VALIDATION_FRACTION,
    early_stopping_patience=EARLY_STOP_PATIENCE,
    early_stopping_metric="val_loss",

    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    max_seq_length=MAX_SEQ_LENGTH,
    backbone_dtype=BACKBONE_DTYPE,
)

# job_id and model_id are properties (not methods)
print(f"\nJob submitted  : {job.job_id}")
print("Training logs (epoch / loss / pos_sim / neg_sim) stream to stderr below:")
print("-" * 60)

job.wait()   # blocks until done or raises on failure; Rust logs stream to stderr

elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"Fine-tuning COMPLETE in {elapsed:.1f}s")
print(f"  Model ID : {job.model_id}")
print(f"  Job ID   : {job.job_id}")
print(f"  Status   : {job.status()}")

# The adapter .safetensors file lands here:
adapter_dir = JAMMI_DB_DIR / "artifacts" / "models" / job.job_id
print(f"\nAdapter output: {adapter_dir}")
if adapter_dir.exists():
    for f in sorted(adapter_dir.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
else:
    print("  (path not yet flushed — check JAMMI_DB_DIR/artifacts/)")

print("="*60)
