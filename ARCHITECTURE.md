# Architecture

The structural map below is generated from the **compiler's** view of the workspace by
[`build-graph`](https://github.com/d-tietjen/build-graph) — the crate dependency graph and
source-file membership, extracted from `cargo`'s build artifacts rather than written by hand.

It is refreshed by [`.github/workflows/architecture-graph.yml`](.github/workflows/architecture-graph.yml)
(weekly + on demand), which opens a PR when the facts change. **Edit narrative *outside* the
generated block** — everything between the two markers is overwritten on every refresh.

<!-- build-graph:architecture:start -->
_(populated by the `architecture-graph` workflow on its first run — trigger it manually via
the Actions tab, or run `cargo build-graph build` locally to preview the graph.)_
<!-- build-graph:architecture:end -->
