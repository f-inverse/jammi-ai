"""
Comprehensive Jammi database test.

Tests (in order):
  1.  Connection          — jammi.connect() opens the DB
  2.  Source registration — add_source() registers the Parquet file
  3.  SQL query           — db.sql() returns correct rows / count
  4.  Schema check        — expected columns present in the data
  5.  Catalog models      — list fine-tuned models from the SQLite catalog
  6.  Encode query        — encode_text_query() returns a non-zero float vector
  7.  Generate embeddings — generate_text_embeddings() runs without error
  8.  Vector search       — search() returns top-k rows with a score column
  9.  Fine-tuned model    — if a fine-tuned adapter exists, repeat steps 6-8

Run from the repo root:
    python hcc_data/test_db_jammi.py
"""

import os
import sqlite3
import sys
import time
from pathlib import Path

# ── CUDA DLL path (Windows) ───────────────────────────────────────────────────
_cuda_bin = Path(os.environ.get("CUDA_PATH",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4")) / "bin"
if _cuda_bin.exists() and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(_cuda_bin))

os.environ.setdefault("RUST_LOG", "jammi_ai=warn,jammi_engine=warn")

HERE          = Path(__file__).parent
REPO_ROOT     = HERE.parent
TRAIN_PARQUET = HERE / "triplets_train.parquet"
JAMMI_DB_DIR  = HERE / "jammi_test_db"
CATALOG_DB    = JAMMI_DB_DIR / "catalog.db"

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

failures: list[str] = []

def ok(msg: str) -> None:
    print(f"  {PASS}  {msg}")

def fail(msg: str, exc: Exception | None = None) -> None:
    detail = f": {exc}" if exc else ""
    print(f"  {FAIL}  {msg}{detail}")
    failures.append(msg)

def skip(msg: str) -> None:
    print(f"  {SKIP}  {msg}")

def section(title: str) -> None:
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")

# ── helper: pretty-print a pyarrow table ─────────────────────────────────────
def show_table(tbl, max_rows: int = 5) -> None:
    try:
        import pyarrow as pa
        df = tbl.to_pandas() if hasattr(tbl, "to_pandas") else tbl
        print(df.head(max_rows).to_string(index=False))
    except Exception:
        print(f"  (table with {tbl.num_rows} rows, {tbl.num_columns} columns)")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Import jammi
# ══════════════════════════════════════════════════════════════════════════════
section("1. Import jammi")
try:
    import jammi
    ok(f"jammi imported (version attribute: {getattr(jammi, '__version__', 'n/a')})")
except ImportError as e:
    fail("import jammi", e)
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Connect to the database
# ══════════════════════════════════════════════════════════════════════════════
section("2. Connect to database")
try:
    JAMMI_DB_DIR.mkdir(parents=True, exist_ok=True)
    db = jammi.connect(artifact_dir=str(JAMMI_DB_DIR))
    ok(f"Connected to {JAMMI_DB_DIR}")
except Exception as e:
    fail("jammi.connect()", e)
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Register data source
# ══════════════════════════════════════════════════════════════════════════════
section("3. Register data source")
if not TRAIN_PARQUET.exists():
    fail(f"Parquet not found: {TRAIN_PARQUET} — run generate_triplets.py first")
    sys.exit(1)

SOURCE = "hcc_triplets"
try:
    db.add_source(SOURCE, path=str(TRAIN_PARQUET), format="parquet")
    ok(f"Source '{SOURCE}' registered")
except Exception as e:
    if "already registered" in str(e).lower() or "Source already registered" in str(e):
        ok(f"Source '{SOURCE}' already registered (skipped re-registration)")
    else:
        fail("add_source()", e)
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 4. SQL queries
# ══════════════════════════════════════════════════════════════════════════════
section("4. SQL queries")

# DataFusion registers sources as:  {source_id}.public."{table_name}"
# where table_name is the parquet filename stem.
TABLE_NAME = TRAIN_PARQUET.stem          # "triplets_train"
SQL_TABLE  = f'{SOURCE}.public."{TABLE_NAME}"'

# 4a. row count
try:
    count_tbl = db.sql(f"SELECT COUNT(*) AS n FROM {SQL_TABLE}")
    n_rows = count_tbl.column("n")[0].as_py()
    if n_rows > 0:
        ok(f"Row count: {n_rows}")
    else:
        fail("Row count returned 0")
except Exception as e:
    fail("sql(COUNT(*))", e)

# 4b. sample rows
try:
    sample = db.sql(
        f'SELECT anchor, positive, negative FROM {SQL_TABLE} LIMIT 3'
    )
    ok(f"Sample query returned {sample.num_rows} rows, {sample.num_columns} columns")
    show_table(sample)
except Exception as e:
    fail("sql(SELECT ... LIMIT 3)", e)

# 4c. schema check
try:
    schema_tbl = db.sql(f"SELECT * FROM {SQL_TABLE} LIMIT 1")
    cols = set(schema_tbl.schema.names)
    required = {"anchor", "positive", "negative"}
    missing = required - cols
    if not missing:
        ok(f"Schema OK — columns: {sorted(cols)}")
    else:
        fail(f"Missing columns: {missing}")
except Exception as e:
    fail("Schema check", e)

# 4d. chapter distribution (if column exists)
try:
    dist = db.sql(
        f'SELECT chapter, COUNT(*) AS n FROM {SQL_TABLE} '
        f"GROUP BY chapter ORDER BY n DESC"
    )
    ok(f"Chapter distribution ({dist.num_rows} chapters):")
    show_table(dist, max_rows=10)
except Exception:
    pass  # column may not exist — not a test failure


# ══════════════════════════════════════════════════════════════════════════════
# 5. Catalog: list fine-tuned models
# ══════════════════════════════════════════════════════════════════════════════
section("5. Catalog: fine-tuned models")
fine_tuned_model_id: str | None = None

if not CATALOG_DB.exists():
    skip(f"catalog.db not found at {CATALOG_DB}")
else:
    try:
        conn = sqlite3.connect(str(CATALOG_DB))
        cur  = conn.cursor()

        # List all tables for diagnostics
        tables = [r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        ok(f"Catalog tables: {tables}")

        # Fine-tune jobs
        if "fine_tune_jobs" in tables:
            jobs = cur.execute(
                "SELECT job_id, base_model_id, output_model_id, status, created_at "
                "FROM fine_tune_jobs ORDER BY created_at DESC LIMIT 10"
            ).fetchall()
            if jobs:
                ok(f"Fine-tune jobs ({len(jobs)}):")
                for job_id, base_model_id, output_model_id, status, created_at in jobs:
                    print(f"    job={job_id[:8]}  model={base_model_id}  status={status}  created={created_at}")
                # Pick the most recent completed job for downstream tests
                completed = [
                    (j, b, o, s, t) for j, b, o, s, t in jobs
                    if s == "completed" and o is not None
                ]
                if completed:
                    fine_tuned_model_id = completed[0][2]  # output_model_id
                    ok(f"Will use fine-tuned model: {fine_tuned_model_id}")
            else:
                skip("No fine-tune jobs recorded in catalog yet")

        # Models table
        if "models" in tables:
            models = cur.execute(
                "SELECT model_id, model_type, created_at FROM models LIMIT 10"
            ).fetchall()
            ok(f"Models in catalog ({len(models)}):")
            for row in models:
                print(f"    {row}")

        conn.close()
    except Exception as e:
        fail("Reading catalog.db", e)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Encode text query (base model)
# ══════════════════════════════════════════════════════════════════════════════
section("6. encode_text_query — base model")

BASE_MODEL = "hf://thomas-sounack/BioClinical-ModernBERT-base"
TEST_QUERY = "Patient with type 2 diabetes and hypertension"

try:
    t0  = time.time()
    vec = db.encode_text_query(BASE_MODEL, TEST_QUERY)
    elapsed = time.time() - t0
    if len(vec) > 0 and any(v != 0.0 for v in vec):
        ok(f"encode_text_query OK — dim={len(vec)}, elapsed={elapsed:.2f}s")
        print(f"    first 8 dims: {[round(v, 4) for v in vec[:8]]}")
    else:
        fail("encode_text_query returned zero/empty vector")
except Exception as e:
    fail("encode_text_query (base model)", e)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Generate embeddings (base model)
# ══════════════════════════════════════════════════════════════════════════════
section("7. generate_text_embeddings — base model")

# key= is a column in the source that uniquely identifies each row (join key).
# It must be DIFFERENT from the columns being embedded to avoid a duplicate
# projection in the SQL query that Jammi builds internally.
# We embed "anchor" text and use "positive" as the row identifier.
EMBED_KEY_COL = "positive"
try:
    t0 = time.time()
    db.generate_text_embeddings(
        source=SOURCE,
        model=BASE_MODEL,
        columns=["anchor"],
        key=EMBED_KEY_COL,
    )
    elapsed = time.time() - t0
    ok(f"generate_text_embeddings OK (key_col='{EMBED_KEY_COL}', elapsed={elapsed:.1f}s)")
except Exception as e:
    fail("generate_text_embeddings (base model)", e)


# ══════════════════════════════════════════════════════════════════════════════
# 8. Vector search (base model)
# ══════════════════════════════════════════════════════════════════════════════
section("8. vector search — base model")
try:
    query_vec = db.encode_text_query(BASE_MODEL, TEST_QUERY)
    # search() takes the source_id (not a key name) and looks up its embedding table
    result = db.search(SOURCE, query=query_vec, k=5)
    result.limit(5)
    result_tbl = result.run()
    ok(f"search() returned {result_tbl.num_rows} rows")
    col_names = result_tbl.schema.names
    ok(f"Result columns: {col_names}")
    if "_score" in col_names or "score" in col_names:
        ok("Score column present")
    show_table(result_tbl)
except Exception as e:
    fail("vector search (base model)", e)


# ══════════════════════════════════════════════════════════════════════════════
# 9. Fine-tuned model tests (if available)
# ══════════════════════════════════════════════════════════════════════════════
if fine_tuned_model_id:
    section(f"9. Fine-tuned model: {fine_tuned_model_id}")

    # 9a. encode_text_query with fine-tuned model
    try:
        t0  = time.time()
        ft_vec = db.encode_text_query(fine_tuned_model_id, TEST_QUERY)
        elapsed = time.time() - t0
        if len(ft_vec) > 0 and any(v != 0.0 for v in ft_vec):
            ok(f"encode_text_query (fine-tuned) OK — dim={len(ft_vec)}, elapsed={elapsed:.2f}s")
            print(f"    first 8 dims: {[round(v, 4) for v in ft_vec[:8]]}")
        else:
            fail("encode_text_query (fine-tuned) returned zero/empty vector")
    except Exception as e:
        fail("encode_text_query (fine-tuned model)", e)

    # 9b. generate_text_embeddings with fine-tuned model
    # We register a second source pointing to the same parquet so the
    # fine-tuned embedding table is stored separately from the base model's.
    SOURCE_FT = "hcc_triplets_ft"
    try:
        db.add_source(SOURCE_FT, path=str(TRAIN_PARQUET), format="parquet")
    except Exception:
        pass  # already registered is fine

    try:
        t0 = time.time()
        db.generate_text_embeddings(
            source=SOURCE_FT,
            model=fine_tuned_model_id,
            columns=["anchor"],
            key=EMBED_KEY_COL,
        )
        elapsed = time.time() - t0
        ok(f"generate_text_embeddings (fine-tuned) OK — elapsed={elapsed:.1f}s")
    except Exception as e:
        fail("generate_text_embeddings (fine-tuned model)", e)

    # 9c. vector search with fine-tuned embeddings
    try:
        ft_query_vec = db.encode_text_query(fine_tuned_model_id, TEST_QUERY)
        result = db.search(SOURCE_FT, query=ft_query_vec, k=5)
        result.limit(5)
        result_tbl = result.run()
        ok(f"search (fine-tuned) returned {result_tbl.num_rows} rows")
        show_table(result_tbl)
    except Exception as e:
        fail("vector search (fine-tuned model)", e)

else:
    section("9. Fine-tuned model tests")
    skip("No completed fine-tune job found in catalog — run test_finetune_jammi.py first")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
if failures:
    print(f"  {len(failures)} FAILURE(S):")
    for f_msg in failures:
        print(f"    FAIL: {f_msg}")
    sys.exit(1)
else:
    print(f"  All tests passed!")
print(f"{'='*60}")
