# Compound retrieval and inference in one SQL query

Join sources, filter, and run a model over a relation — in one statement the
engine plans — embedded or against a `jammi-server` over Flight SQL.

**When to use this pattern.** You want results enriched with context from
other sources (who holds a patent, which category it is in) and with model
output computed inside the query, without a per-row round-trip. `search`
returns a bounded top-k; anything caller-shaped rides SQL, where the
`annotate(...)` table function runs a model over a relation.

## What `example.py` does

1. Registers a 20-row patent corpus and a small `assignees` table
2. Enriches by joining: every patent gains its assignee's company and country
3. `annotate('<model>', 'text_embedding', 'corpus.public.tiny_corpus', 'id', 'content')`
   embeds every patent inside the query, joined back to its row and its
   assignee and filtered to US assignees — one statement
4. Runs the identical SQL against a `jammi-server` (`jammi.testing.LiveServer`)
   over the Flight SQL lane and checks the same rows and vectors come back

## Run it

Needs a `jammi-server` on PATH:

```bash
pip install jammi-server
python cookbook/recipes/compound_query/example.py
```

See [Enrich Results with Joins and Annotations](https://f-inverse.github.io/jammi-ai/enrich-results.html)
and [Compound Retrieval and Inference over Flight SQL](https://f-inverse.github.io/jammi-ai/remote-compound-query.html).
