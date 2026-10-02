# Runnable Recipes

Every capability this guide describes also runs as code in
[The Cookbook](https://f-inverse.github.io/jammi-ai/cookbook/), published with
this guide for the same release. It opens with the quickstart and is organized
by what you want to do: search, models and inference, evaluation, fine-tuning,
graphs, data that changes, where data lives, and running Jammi.

Each recipe under
[`cookbook/recipes/`](https://github.com/f-inverse/jammi-ai/tree/main/cookbook/recipes)
is a short program, one step per cell, beside a README. It runs three ways:

- as a chapter of the book, its steps rendered with their output;
- in Colab, from the chapter's Open-in-Colab badge, a step at a time;
- as a script, `python cookbook/recipes/<name>/example.py`.

CI runs every recipe on every change (`tests/cookbook_smoke.py`), and runs
every notebook nightly as a reader does, installed from PyPI. The longer
chapters take one capability deep and measure it against frozen results.
