//! (g) A `Collective` verb reached from a runtime worker thread is a
//! COMPILE error, never a runtime refusal — proven by `trybuild`, the only
//! oracle that can assert a non-compilation. `tests/ui/` holds the cases
//! that must fail (with their pinned diagnostics) and `tests/ui_pass/` the
//! control: the identical call from a `spawn_blocking` closure compiles.

#[test]
fn a_collective_verb_from_a_runtime_worker_thread_does_not_compile() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/*.rs");
    cases.pass("tests/ui_pass/*.rs");
}
