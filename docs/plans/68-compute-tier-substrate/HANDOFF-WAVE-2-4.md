# Handoff — resuming at wave 2

Written 2026-09-12 at the close of wave 1. `PROGRAM.md` is still the schedule; this document is what
that schedule does not tell you.

## Where wave 1 left the tree

PR #521 carries wave 1: operability, versioned embeddings, and the distributed data plane, as stacked
commit groups on `feat/482-compute-tier-substrate`. Merged before it: #504 merge-path reliability,
#514 the DataFusion 54 engine line, #518 the first three lead-gate patches. #520 is open and carries a
fourth gate rule.

**The job-dependency unit is NOT in wave 1.** It was removed and deferred to #515, which carries its
round-4 audit as the resume specification. Its migrations 033 and 034 are gone; the ledger ends at
032. Its error-proto tags 37 and 38 are deliberately left vacant so a cherry-pick recovery cannot
collide. Wave 3's schedule lists it as a dependency — read #515 before assuming it exists.

## What to read before you start, in this order

`scratchpad/CONTRACT-RULES.md` from this session, if it survives; otherwise the memory entries it
produced. Nine rules, each traceable to a specific failure that cost a full audit round. The three
that matter most:

A claim that something cannot happen ships with the executed attempt to falsify it, or ships marked
uncovered. Three separate rounds declared something impossible and an auditor refuted each by simply
trying — one in about two seconds, using a pattern already in the test tree.

A property states its own quantifier and a sweep must prove its quantifier matches. A module-scoped
sweep does not discharge a crate-scoped property, and three sites fell in exactly that gap.

A state defined by missing evidence cannot be given a definite consequence. One operator sentence was
wrong three times before that was understood.

## The thing that actually went wrong, so you do not repeat it

Wave 1 took roughly twenty fix rounds. The rounds were not wasted — each found something real — but
the lead answered every audit finding with another round long after the findings had become sentences
and comments. Phase 5, the only gate that can refuse, did not run until the very end. It hard-blocked
immediately, on a fail-closed gate left red.

**Run the oracle early and often, not as a closing ceremony.** Its invariants are orthogonal to what
adversarial audits examine: migrations, tenant isolation per remote call, byte parity, dependency
direction, the frozen surface. Twenty rounds of audits touched none of them.

**And fold or file rather than round.** A finding that is a sentence gets fixed in the next
consolidated pass. A finding that is a defect gets a round.

## Wave-specific knowledge you would otherwise rediscover

**Wave 2 (PR-B).** Independent of every plan-68 unit. Its units edit regions of the worker that
operability did not, but operability landed first, so rebase rather than assume. Spike S1 found the
CUDA continuous-integration image lacks the NCCL development package; U4a carries that as a
precondition and it is not fixed.

**Wave 3 (PR-C).** U5a's job slot wraps the loop operability rewrote. U5b-1 builds the membership
substrate the distributed unit's design sketches but does not implement. U5b-2 uses operability's
lease-release primitive, which changed shape in wave 1: the per-hold pass now returns a value carrying
released, not-required, failed and attempted counts rather than a bare count.

**Wave 4 (PR-D).** Spike S6 refuted three of U8b's premises and the plan documents were corrected:
the framework re-runs consumer work on executor loss regardless of retry settings, jobs do not survive
a scheduler restart (only registrations do), and two schedulers serve only sequential jobs. U8b needs
a bind-time guard and its restart and high-availability oracles rewritten before anyone builds this.
It is an admin merge because it edits a domain card.

## Operational facts worth having

Build directories go OUTSIDE worktrees, under a scratch targets directory — a stamp guard walks
anything in-tree but `target`, six times per substrate run.

A shared worktree with concurrent write agents is hazardous and cost real work in wave 1: one agent
amended another's commit after HEAD moved, another overwrote an unsaved edit that then broke the
branch at head. Commit by explicit pathspec, re-read HEAD in the same command as any amend, and
verify commit contents again at the end of the task, not only when you make it.

The lead gate refuses a verifier dispatch while a block has no fix commit, and it keys on branch
rather than unit, so one unfixed block serializes audits across every unit on a consolidated branch.
Do not open a second slug to route around it.

A relay's probe field is a list of strings, not objects. Objects parse as zero sites and the dispatch
is denied.

Two tests on this tree are load-sensitive and predate this program: one in the cancel suite on a fixed
fifteen-second bound, one on a two-pool-writer lock. Report the NAME of a test that dies, never a
count, or a flake reads as a mutation kill.

## Open issues carrying real specifications

#515 job dependencies, with the full round-4 audit. #507 release-lane feature lists. #516 a
delta-oracle projection gap. #517 a gate heuristic gap. #519 the entry half of a width-attribution
invariant, with the measurement that motivates it.
