use std::sync::Arc;
use std::time::{Duration, Instant};

use jammi_ai::concurrency::{GpuPermit, GpuPriority, GpuScheduler};
use tokio::time::timeout;

// ─── Permit lifecycle and RAII ───────────────────────────────────────────────
//
// One scheduler, multiple assertions: concurrent acquisition, drop frees
// memory, nested scopes release in order. Setup cost is trivial (no I/O)
// but grouping exercises the full lifecycle in sequence.
// Covers acceptance criteria 1, 3.

#[tokio::test]
async fn permit_lifecycle_and_raii() {
    let scheduler = Arc::new(GpuScheduler::new(1_000_000_000, 0.2));
    // 1 GB total, 20% headroom -> 800 MB usable

    // Two small permits acquired concurrently
    let permit_a = scheduler
        .acquire(200_000_000, GpuPriority::Interactive)
        .await
        .unwrap();
    let permit_b = scheduler
        .acquire(200_000_000, GpuPriority::Interactive)
        .await
        .unwrap();
    assert_eq!(
        scheduler.available(),
        400_000_000,
        "Should have 400 MB remaining after two 200 MB permits"
    );
    drop(permit_a);
    drop(permit_b);
    assert_eq!(
        scheduler.available(),
        800_000_000,
        "All memory restored after both permits dropped"
    );

    // RAII drop in scoped block
    let initial = scheduler.available();
    {
        let _permit = scheduler
            .acquire(300_000_000, GpuPriority::Interactive)
            .await
            .unwrap();
        assert_eq!(scheduler.available(), initial - 300_000_000);
    }
    assert_eq!(scheduler.available(), initial, "Scope drop restores memory");

    // Nested permit scopes release in order
    let sched_no_headroom = Arc::new(GpuScheduler::new(1_000_000_000, 0.0));
    let full = sched_no_headroom.available();
    {
        let _p1 = sched_no_headroom.try_acquire(100_000_000).unwrap();
        {
            let _p2 = sched_no_headroom.try_acquire(200_000_000).unwrap();
            {
                let _p3 = sched_no_headroom.try_acquire(300_000_000).unwrap();
            }
            assert_eq!(sched_no_headroom.available(), full - 300_000_000);
        }
        assert_eq!(sched_no_headroom.available(), full - 100_000_000);
    }
    assert_eq!(sched_no_headroom.available(), full);
}

// ─── Blocking acquire and panic safety ───────────────────────────────────────
//
// Tests that acquire() blocks when memory is insufficient, unblocks when
// freed, and that permits are released even if the holding task panics.
// Covers acceptance criteria 2, 3 (panic path).

#[tokio::test]
async fn blocking_acquire_and_panic_safety() {
    let scheduler = Arc::new(GpuScheduler::new(1_000_000_000, 0.2));

    // Large request blocks until memory freed
    let blocker = scheduler
        .acquire(600_000_000, GpuPriority::Interactive)
        .await
        .unwrap();

    let sched_clone = Arc::clone(&scheduler);
    let large_task = tokio::spawn(async move {
        let start = Instant::now();
        let permit = sched_clone
            .acquire(500_000_000, GpuPriority::Background)
            .await
            .unwrap();
        (permit, start.elapsed())
    });

    tokio::time::sleep(Duration::from_millis(50)).await;
    drop(blocker);

    let (_, waited) = timeout(Duration::from_secs(2), large_task)
        .await
        .expect("large task should complete within timeout")
        .unwrap();
    assert!(
        waited >= Duration::from_millis(30),
        "Large task should have waited for memory to free"
    );

    // Panic safety: permit released even if holder panics
    let before = scheduler.available();
    let sched = Arc::clone(&scheduler);
    let handle = tokio::spawn(async move {
        let _permit = sched
            .acquire(200_000_000, GpuPriority::Interactive)
            .await
            .unwrap();
        panic!("intentional panic");
    });
    assert!(handle.await.is_err(), "Task should have panicked");
    assert_eq!(
        scheduler.available(),
        before,
        "Memory should be fully restored after task panic"
    );
}

// ─── Non-blocking try_acquire semantics ──────────────────────────────────────
//
// try_acquire returns Some when sufficient, None when not, never blocks.
// Covers acceptance criterion 4.

#[tokio::test]
async fn try_acquire_nonblocking_semantics() {
    let scheduler = Arc::new(GpuScheduler::new(1_000_000_000, 0.2));

    // Succeeds when sufficient
    let permit = scheduler.try_acquire(200_000_000);
    assert!(permit.is_some(), "Should succeed with enough memory");
    assert_eq!(scheduler.available(), 600_000_000);
    drop(permit);
    assert_eq!(scheduler.available(), 800_000_000);

    // Returns None when insufficient
    let _blocker = scheduler
        .acquire(700_000_000, GpuPriority::Interactive)
        .await
        .unwrap();
    assert!(
        scheduler.try_acquire(200_000_000).is_none(),
        "Should return None when insufficient memory"
    );
}

// ─── Headroom enforcement ────────────────────────────────────────────────────
//
// Validates headroom limits usable capacity, different fractions work,
// and exact-boundary behavior. Covers acceptance criteria 7, 8.

#[tokio::test]
async fn headroom_fraction_enforced() {
    // 20% headroom: 800 MB usable from 1 GB
    let scheduler = Arc::new(GpuScheduler::new(1_000_000_000, 0.2));
    assert_eq!(scheduler.available(), 800_000_000);

    // Exact limit succeeds
    let permit = scheduler.try_acquire(800_000_000);
    assert!(permit.is_some(), "Exact usable limit must succeed");
    assert_eq!(scheduler.available(), 0);
    drop(permit);

    // Over limit fails even with zero reservations
    assert!(
        scheduler.try_acquire(900_000_000).is_none(),
        "Exceeding usable limit must fail"
    );
    assert_eq!(scheduler.available(), 800_000_000);

    // 50% headroom: 500 MB usable
    let half = Arc::new(GpuScheduler::new(1_000_000_000, 0.5));
    assert_eq!(half.available(), 500_000_000);
    assert!(half.try_acquire(600_000_000).is_none());
    assert!(half.try_acquire(500_000_000).is_some());
}

#[test]
#[should_panic(expected = "headroom_fraction must be between 0.0 and 1.0")]
fn headroom_fraction_rejects_out_of_range() {
    let _ = GpuScheduler::new(1_000_000_000, 1.5);
}

#[test]
#[should_panic(expected = "headroom_fraction must be between 0.0 and 1.0")]
fn headroom_fraction_rejects_negative() {
    let _ = GpuScheduler::new(1_000_000_000, -0.1);
}

// ─── GPU memory detection ────────────────────────────────────────────────────
//
// Covers acceptance criterion 9: Err on CPU-only, no panic.

#[test]
fn detect_gpu_memory_returns_result_not_panic() {
    let result = GpuScheduler::detect_gpu_memory(0);
    match result {
        Ok((free, total)) => {
            assert!(total > 0, "Total GPU memory should be positive");
            assert!(free <= total, "Free memory should not exceed total");
        }
        Err(_) => {
            // CPU fallback is acceptable
        }
    }
}

// ─── Concurrent stress, liveness, and sum conservation ───────────────────────
//
// Consolidated: 20-task stress, 10-waiter liveness, and rapid acquire/release
// sum conservation all exercise the same concurrency machinery. Grouping
// avoids 3 separate tokio runtime setups.
// Covers acceptance criteria 2, 3, 7, 11.

#[tokio::test]
async fn concurrent_acquire_stress_and_liveness() {
    // --- 20 tasks, 100 MB each, 800 MB usable (at most 8 concurrent) ---
    let total = 1_000_000_000_usize;
    let headroom = 0.2;
    let usable = (total as f64 * (1.0 - headroom)) as usize;
    let permit_size = 100_000_000_usize;
    let scheduler = Arc::new(GpuScheduler::new(total, headroom));

    let mut handles = Vec::new();
    for _ in 0..20 {
        let sched = Arc::clone(&scheduler);
        handles.push(tokio::spawn(async move {
            let permit = sched
                .acquire(permit_size, GpuPriority::Interactive)
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(10)).await;
            let avail = sched.available();
            assert!(avail + permit_size <= usable);
            drop(permit);
        }));
    }

    // Sample invariant while tasks run
    for _ in 0..50 {
        assert!(scheduler.available() <= usable);
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    for h in handles {
        h.await.unwrap();
    }
    assert_eq!(
        scheduler.available(),
        usable,
        "All memory freed after stress"
    );

    // --- 10 waiters, 200 MB each, 500 MB pool (only 2 at a time) ---
    let scheduler = Arc::new(GpuScheduler::new(500_000_000, 0.0));
    let mut handles = Vec::new();
    for i in 0..10 {
        let sched = Arc::clone(&scheduler);
        handles.push(tokio::spawn(async move {
            let permit = sched
                .acquire(200_000_000, GpuPriority::Interactive)
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(20)).await;
            drop(permit);
            i
        }));
    }
    let mut completed = Vec::new();
    for h in handles {
        completed.push(
            timeout(Duration::from_secs(10), h)
                .await
                .expect("task should complete within 10s")
                .unwrap(),
        );
    }
    completed.sort();
    assert_eq!(completed, (0..10).collect::<Vec<_>>());

    // --- Rapid acquire/release: sum conservation ---
    let scheduler = Arc::new(GpuScheduler::new(1_000_000_000, 0.2));
    let usable = 800_000_000_usize;
    let mut permits: Vec<GpuPermit> = Vec::new();
    let sizes = [
        50_000_000,
        100_000_000,
        150_000_000,
        200_000_000,
        75_000_000,
    ];
    for &size in sizes.iter().cycle().take(20) {
        if let Some(permit) = scheduler.try_acquire(size) {
            permits.push(permit);
        }
        if permits.len() > 3 {
            permits.remove(0);
        }
        assert!(scheduler.available() <= usable);
    }
    permits.clear();
    assert_eq!(scheduler.available(), usable);
}
