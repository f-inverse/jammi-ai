# 5-minute quickstart

Go from `pip install "jammi-ai[embedded]"` to a vector search over your own
data in five minutes: install, connect, register a source, embed it, search it.
[`quickstart.py`](./quickstart.py) is the program, one step per cell. It is
also the cookbook's [first chapter](https://f-inverse.github.io/jammi-ai/cookbook/chapters/recipes/quickstart.html),
where every step runs in Colab.

## Run it

```bash
pip install "jammi-ai[embedded]" "jammi-cookbook @ git+https://github.com/f-inverse/jammi-ai#subdirectory=cookbook/book"
python cookbook/quickstart/quickstart.py
```

It prints three rows of the corpus, then the three nearest matches to the
query with their cosine similarity, in seconds on a CPU.
