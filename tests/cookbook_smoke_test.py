"""Cookbook smoke test — verifies every Python recipe actually works.

Each test function creates its own Database to avoid shared-state issues.
"""

import os
import tempfile

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
TINY_BERT = f"local:{os.path.join(FIXTURES, 'tiny_bert')}"


def make_db(tmpdir):
    import jammi
    return jammi.connect(artifact_dir=tmpdir, gpu_device=-1, inference_batch_size=8)


def test_query_data_with_sql():
    """Recipe: Query Your Data with SQL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)

        # Register Parquet source
        db.add_source("patents", path=os.path.join(FIXTURES, "patents.parquet"), format="parquet")

        # Simple query
        table = db.sql("SELECT id, title, year FROM patents.public.patents WHERE year > 2020 LIMIT 5")
        assert table.num_rows > 0
        assert "title" in table.column_names

        # Aggregation
        agg = db.sql("SELECT category, COUNT(*) as count FROM patents.public.patents GROUP BY category ORDER BY count DESC")
        assert agg.num_rows > 0

        # Cross-source join with CSV
        db.add_source("companies", path=os.path.join(FIXTURES, "assignees.csv"), format="csv")
        joined = db.sql("""
            SELECT p.title, c.company_name
            FROM patents.public.patents p
            JOIN companies.public.assignees c ON p.assignee_id = c.id
        """)
        assert joined.num_rows > 0
        assert "company_name" in joined.column_names

    print("  PASS query data with SQL")


def test_generate_embeddings():
    """Recipe: Generate Embeddings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        db.add_source("patents", path=os.path.join(FIXTURES, "patents.parquet"), format="parquet")

        # Basic generate_embeddings
        db.generate_text_embeddings(source="patents", model=TINY_BERT, columns=["abstract"], key="id")

        # Raw inference (no persistence)
        result = db.infer(source="patents", model=TINY_BERT, columns=["abstract"], task="text_embedding", key="id")
        assert result.num_rows > 0
        assert "_status" in result.column_names
        assert "vector" in result.column_names

    print("  PASS generate embeddings")


def test_semantic_search():
    """Recipe: Semantic Search."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        db.add_source("patents", path=os.path.join(FIXTURES, "patents.parquet"), format="parquet")
        db.generate_text_embeddings(source="patents", model=TINY_BERT, columns=["abstract"], key="id")

        # encode_query
        query_vec = db.encode_text_query(TINY_BERT, "quantum computing applications")
        assert len(query_vec) > 0

        # Basic search
        search = db.search("patents", query=query_vec, k=10)
        results = search.run()
        assert results.num_rows > 0
        assert "similarity" in results.column_names
        assert "title" in results.column_names
        assert "retrieved_by" in results.column_names

        # SearchBuilder: filter + sort + limit + select
        search = db.search("patents", query=query_vec, k=20)
        search.filter("year > 2020")
        search.sort("similarity", descending=True)
        search.limit(5)
        search.select(["_row_id", "title", "similarity"])
        filtered = search.run()
        assert filtered.num_rows <= 5
        assert "_row_id" in filtered.column_names
        assert "title" in filtered.column_names
        assert "similarity" in filtered.column_names
        # Evidence columns always appended
        assert "retrieved_by" in filtered.column_names

    print("  PASS semantic search")


def test_enrich_results():
    """Recipe: Enrich Results with Joins and Annotations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        db.add_source("patents", path=os.path.join(FIXTURES, "patents.parquet"), format="parquet")
        db.add_source("companies", path=os.path.join(FIXTURES, "assignees.csv"), format="csv")
        db.generate_text_embeddings(source="patents", model=TINY_BERT, columns=["abstract"], key="id")

        query_vec = [0.5] * 32  # tiny_bert is 32-dim

        # Join
        search = db.search("patents", query=query_vec, k=10)
        search.join("companies", on="assignee_id=id")
        joined = search.run()
        assert joined.num_rows > 0
        assert "company_name" in joined.column_names

        # Annotate
        search = db.search("patents", query=query_vec, k=10)
        search.annotate(model=TINY_BERT, task="text_embedding", columns=["abstract"])
        annotated = search.run()
        assert annotated.num_rows > 0
        assert "annotated_by" in annotated.column_names

        # Compose: join + filter + sort + limit + select
        search = db.search("patents", query=query_vec, k=100)
        search.join("companies", on="assignee_id=id")
        search.filter("country = 'US'")
        search.sort("similarity", descending=True)
        search.limit(10)
        search.select(["title", "company_name", "similarity"])
        composed = search.run()
        assert composed.num_rows <= 10
        assert "company_name" in composed.column_names

    print("  PASS enrich results")


def test_fine_tune():
    """Recipe: Fine-Tune for Your Domain."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        db.add_source("training", path=os.path.join(FIXTURES, "training_pairs.csv"), format="csv")

        # Start fine-tuning job
        job = db.fine_tune(
            source="training",
            base_model=TINY_BERT,
            columns=["text_a", "text_b", "score"],
            method="lora",
            task="text_embedding",
        )
        assert job.job_id

        # Wait for completion
        job.wait()

        # Use the fine-tuned model
        ft_model_id = job.model_id
        assert ft_model_id.startswith("jammi:fine-tuned:")

        # encode_query with fine-tuned model
        query_vec = db.encode_text_query(ft_model_id, "quantum computing")
        assert len(query_vec) > 0

    print("  PASS fine-tune")


def test_evaluation():
    """Recipe: Evaluate and Compare Models."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        db.add_source("patents", path=os.path.join(FIXTURES, "patents.parquet"), format="parquet")
        db.add_source("golden", path=os.path.join(FIXTURES, "golden_relevance.csv"), format="csv")
        db.generate_text_embeddings(source="patents", model=TINY_BERT, columns=["abstract"], key="id")

        # eval_embeddings
        metrics = db.eval_embeddings(
            source="patents",
            golden_source="golden.public.golden_relevance",
            k=10,
        )
        assert 0.0 <= metrics["recall_at_k"] <= 1.0
        assert 0.0 <= metrics["precision_at_k"] <= 1.0
        assert 0.0 <= metrics["mrr"] <= 1.0
        assert 0.0 <= metrics["ndcg"] <= 1.0

    print("  PASS evaluation")


def main():
    print("Python cookbook smoke test")
    print("=" * 40)

    test_query_data_with_sql()
    test_generate_embeddings()
    test_semantic_search()
    test_enrich_results()
    test_fine_tune()
    test_evaluation()

    print("=" * 40)
    print("All Python cookbook tests PASSED")


if __name__ == "__main__":
    main()
