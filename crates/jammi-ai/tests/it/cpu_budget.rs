//! `[engine] execution_threads` is the one budget every CPU pool takes its size
//! from: DataFusion's partitions, the forwards the CPU device admits at once,
//! and the process-wide rayon pool.
//!
//! The rayon pool is process-global and sized once, so the assertions run in a
//! child process of their own.

use std::num::NonZeroUsize;
use std::time::Duration;

use jammi_ai::concurrency::init_cpu_pool;
use jammi_ai::session::InferenceSession;
use jammi_db::error::JammiError;

const BUDGET: usize = 3;

#[test]
fn every_cpu_pool_takes_the_configured_budget() {
    jammi_test_resources::child_test_stdout(&mut jammi_test_resources::child_test(
        "cpu_budget::child_every_cpu_pool_takes_the_configured_budget",
    ));
}

#[tokio::test]
#[ignore = "child process of every_cpu_pool_takes_the_configured_budget"]
async fn child_every_cpu_pool_takes_the_configured_budget() {
    let budget = NonZeroUsize::new(BUDGET).expect("a positive thread count");
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = jammi_test_utils::test_config(dir.path());
    config.engine.execution_threads = budget;

    init_cpu_pool(budget).expect("the first sizing of this process's pool");
    assert_eq!(rayon::current_num_threads(), BUDGET, "the rayon pool");
    init_cpu_pool(budget).expect("the size the pool already has is accepted");
    let other = NonZeroUsize::new(BUDGET + 2).expect("a positive thread count");
    assert!(
        matches!(init_cpu_pool(other), Err(JammiError::Config(_))),
        "a running pool cannot take another size, and saying so beats keeping the old one quietly"
    );

    let session = InferenceSession::open(config).await.expect("session");
    assert_eq!(
        session.context().state().config().target_partitions(),
        BUDGET,
        "DataFusion's partitions"
    );

    let schedulers = session.model_cache().schedulers();
    let cpu = schedulers
        .get(schedulers.primary())
        .expect("the primary device has a scheduler");
    let mut admitted = Vec::new();
    for _ in 0..BUDGET {
        admitted.push(cpu.admit_forward().await.expect("admitted"));
    }
    assert!(
        tokio::time::timeout(Duration::from_millis(100), cpu.admit_forward())
            .await
            .is_err(),
        "the CPU device admits exactly the budget of forwards at once"
    );
}
