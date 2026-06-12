# Marathon learnings (v0.25.0 → v0.26.4)

What the cookbook-hardening marathon taught, distilled for two audiences. Evidence-graded:
every lesson carries the real incident that taught it. The bug *record* is the `CHANGELOG`
and git history; these are the transferable *patterns*.

- **[AGENTIC-PLAYBOOK.md](AGENTIC-PLAYBOOK.md)** — for AI agents. How to do rigorous
  agentic engineering here: the rigor chain as a bug-discovery mechanism, why green CI is
  not sufficient, delegation discipline, the operational traps. `CLAUDE.md` links here.
- **[METHODOLOGY.md](METHODOLOGY.md)** — for human readers. The engine↔cookbook loop and
  why it works, as a narrative with named principles.
- **[CASE-STUDIES.md](CASE-STUDIES.md)** — shared. The deepest findings worked end to end
  (the Adam/standardization lesson, domain-validity, the tenant model, the flaky control).

Operational context (build env, gate, release machinery, infra) lives in the hardening
roadmap's §7 and the project memory files.
