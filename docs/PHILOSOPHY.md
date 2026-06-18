# Design Philosophy

> **This is a pointer, not a copy.** The canonical, full statement of Jammi's
> design philosophy — the one rule (Jammi names no consumer), the discipline
> test, the boundary table of *what stays in the engine* vs *what lives in the
> consumer's repo*, the `search`-only embedding-consumption stance, and the four
> deployment shapes with the five pluggable backends — lives in the published
> guide:
>
> **→ [`docs/guide/src/philosophy.md`](./guide/src/philosophy.md)** (rendered as the
> *Design Philosophy* page of the [Jammi guide](https://docs.rs/jammi-ai)).

It is kept in one place to avoid drift: a second full copy would inevitably fall
out of sync with the first. Repo-root and plan-relative references to
`docs/PHILOSOPHY.md` (for example from `CLAUDE.md` and the `docs/plans/`
specs, which reach it as `../../PHILOSOPHY.md`) resolve here, and this page
forwards to the canonical source.

The one-line version, for orientation: **Jammi is an engine of generic
primitives, not a substrate-platform.** References point one way only — a
consumer may depend on Jammi; Jammi depends on no consumer. The gate for any
capability entering the engine is the discipline test: *would a user who has
never heard of any particular consumer reach for this on its own?* The guide
page states the rest.
