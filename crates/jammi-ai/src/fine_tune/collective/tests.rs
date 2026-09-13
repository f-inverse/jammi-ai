//! Hermetic oracles for the collective arms, on CPU devices.
//!
//! Every gang here runs its ranks on real OS threads through the real
//! rendezvous — the property under test is what a gang of concurrently
//! executing ranks agrees on, and a single-threaded driver would prove
//! nothing about that.

use std::sync::Arc;
use std::time::Duration;

use candle_core::{DType, Device, Tensor};

use super::{Collective, Local, LocalGang, Noop};

/// The exact bits of a tensor's f32 elements, in memory order.
///
/// `to_bits` rather than the `f32`s themselves: the acceptance is
/// bit-for-bit, and `-0.0 == 0.0` and `NaN != NaN` both make a float
/// comparison say the wrong thing about identical or differing buffers.
fn bits(t: &Tensor) -> Vec<u32> {
    t.flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("f32 elements")
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

/// A `rows × cols` f32 tensor whose elements are `base + index`.
fn matrix(rows: usize, cols: usize, base: f32) -> Tensor {
    let data: Vec<f32> = (0..rows * cols).map(|i| base + i as f32).collect();
    Tensor::from_vec(data, (rows, cols), &Device::Cpu).expect("tensor")
}

/// Run one closure per rank of a fresh `world`-rank CPU gang, each on its own
/// thread, and return the results in rank order.
fn run_gang<T, F>(world: usize, body: F) -> Vec<T>
where
    T: Send + 'static,
    F: Fn(Local) -> T + Send + Sync + 'static,
{
    let gang =
        LocalGang::with_timeout(vec![Device::Cpu; world], Duration::from_secs(30)).expect("gang");
    let body = Arc::new(body);
    let handles: Vec<_> = (0..world as u32)
        .map(|rank| {
            let local = gang.rank(rank).expect("rank handle");
            let body = Arc::clone(&body);
            std::thread::spawn(move || body(local))
        })
        .collect();
    handles
        .into_iter()
        .map(|h| h.join().expect("a rank thread panicked"))
        .collect()
}

// ── (a) `Local`: gather layout, reduce order, determinism ───────────────────

/// EQUAL counts: the gather is the rank-ordered concatenation, bit-for-bit
/// equal to the serial reference, and every rank holds the same bytes.
#[test]
fn local_all_gather_at_equal_counts_is_the_rank_ordered_concatenation() {
    let slices = [matrix(2, 3, 0.0), matrix(2, 3, 100.0)];
    let reference = Tensor::cat(&slices, 0).expect("serial concat");

    let per_rank = {
        let slices = slices.clone();
        run_gang(2, move |local| {
            let rank = local.rank() as usize;
            bits(
                &local
                    .all_gather(&slices[rank], &[2, 2])
                    .expect("all_gather"),
            )
        })
    };

    for (rank, got) in per_rank.iter().enumerate() {
        assert_eq!(
            got,
            &bits(&reference),
            "rank {rank}'s gather must equal the serial concatenation bit-for-bit"
        );
    }
}

/// UNEQUAL counts, including a rank that contributes nothing: the layout is
/// still the rank-ordered concatenation of exactly the contributed rows, with
/// no padding visible to the caller. A batch whose row count is not a
/// multiple of `world · batch` produces exactly this shape.
#[test]
fn local_all_gather_at_unequal_counts_including_a_zero_row_rank() {
    let slices = [matrix(3, 2, 0.0), matrix(0, 2, 900.0), matrix(1, 2, 500.0)];
    let reference = Tensor::cat(&[&slices[0], &slices[2]], 0).expect("serial concat");
    let counts = [3usize, 0, 1];

    let per_rank = {
        let slices = slices.clone();
        run_gang(3, move |local| {
            let rank = local.rank() as usize;
            let gathered = local
                .all_gather(&slices[rank], &counts)
                .expect("all_gather");
            (gathered.dims().to_vec(), bits(&gathered))
        })
    };

    for (rank, (dims, got)) in per_rank.iter().enumerate() {
        assert_eq!(
            dims,
            &vec![4, 2],
            "rank {rank}: the gather has exactly the contributed rows, never the padding an \
             unequal-count NCCL gather needs internally"
        );
        assert_eq!(got, &bits(&reference), "rank {rank}");
    }
}

/// A gang whose ranks disagree with the partition rule is refused rather than
/// silently producing a layout the peers do not assume — the two
/// determinants (this rank's own count, a peer's) each raise their own error.
#[test]
fn local_all_gather_refuses_counts_that_contradict_the_ranks() {
    let wrong_length = run_gang(2, move |local| {
        local
            .all_gather(&matrix(1, 2, 0.0), &[1])
            .expect_err("a one-entry count vector cannot describe a two-rank gang")
            .to_string()
    });
    for message in &wrong_length {
        assert!(
            message.contains("counts has 1 entries"),
            "unexpected message: {message}"
        );
    }

    // Determinant 2: this rank's OWN count contradicts the tensor it holds.
    // Checked before the rendezvous, so it needs no peer.
    let own_count = run_gang(1, move |local| {
        local
            .all_gather(&matrix(1, 2, 0.0), &[2])
            .expect_err("a rank that holds one row cannot claim two")
            .to_string()
    });
    assert!(
        own_count[0].contains("rank 0 holds 1 rows but the partition rule says 2"),
        "unexpected message: {}",
        own_count[0]
    );

    // Determinant 3: the ranks derived DIFFERENT count vectors. Each is
    // self-consistent, so only the post-rendezvous check catches it — rank 0
    // is the one whose vector the gathered rows contradict.
    let disagreeing = run_gang(2, move |local| {
        let (rows, counts) = if local.rank() == 0 {
            (1usize, [1usize, 1])
        } else {
            (2, [1, 2])
        };
        local
            .all_gather(&matrix(rows, 2, 0.0), &counts)
            .map(|_| String::new())
            .unwrap_or_else(|e| e.to_string())
    });
    assert!(
        disagreeing[0].contains("rank 1 contributed 2 rows where the partition rule says 1"),
        "unexpected message: {}",
        disagreeing[0]
    );
}

/// `all_reduce_sum` folds in RANK ORDER, and the order is a real determinant:
/// the same three terms folded in reverse produce different bits, so the
/// bit-for-bit assertion below is not one a differently-ordered
/// implementation could also pass.
#[test]
fn local_all_reduce_sum_folds_in_rank_order() {
    // f32 spacing at 2^24 is 2, so (2^24 + 1) + 1 rounds to 2^24 while
    // (1 + 1) + 2^24 is 2^24 + 2. The fold ORDER is the result.
    let terms = [16_777_216.0f32, 1.0, 1.0];
    let serial_rank_order = ((terms[0] + terms[1]) + terms[2]).to_bits();
    let serial_reverse = ((terms[2] + terms[1]) + terms[0]).to_bits();
    assert_ne!(
        serial_rank_order, serial_reverse,
        "the control: these terms must be order-sensitive, or the oracle below is vacuous"
    );

    let per_rank = run_gang(3, move |local| {
        let rank = local.rank() as usize;
        let mut tensors = vec![
            Tensor::from_vec(vec![terms[rank]], 1, &Device::Cpu).expect("tensor"),
            Tensor::from_vec(vec![terms[rank] * 2.0], 1, &Device::Cpu).expect("tensor"),
        ];
        local.all_reduce_sum(&mut tensors).expect("all_reduce_sum");
        tensors.iter().map(bits).collect::<Vec<_>>()
    });

    for (rank, got) in per_rank.iter().enumerate() {
        assert_eq!(
            got[0],
            vec![serial_rank_order],
            "rank {rank}: the sum must equal the rank-ordered serial fold bit-for-bit"
        );
        assert_eq!(
            got[1],
            vec![(((terms[0] * 2.0) + (terms[1] * 2.0)) + (terms[2] * 2.0)).to_bits()],
            "rank {rank}: every index of the canonical order is folded the same way"
        );
    }
}

/// Two runs of the same gang over the same inputs produce byte-identical
/// gathers and sums — the reproducibility the equal-topology oracle rests on.
/// Run as two independent gangs, so the thread interleaving differs between
/// them and only a result that does not depend on arrival order can pass.
#[test]
fn local_collectives_are_deterministic_across_two_runs() {
    let once = || {
        run_gang(3, move |local| {
            let rank = local.rank() as usize;
            let gathered = local
                .all_gather(&matrix(rank + 1, 2, rank as f32 * 10.0), &[1, 2, 3])
                .expect("all_gather");
            let mut tensors = vec![matrix(2, 2, rank as f32 * 0.3)];
            local.all_reduce_sum(&mut tensors).expect("all_reduce_sum");
            (bits(&gathered), bits(&tensors[0]))
        })
    };
    assert_eq!(
        once(),
        once(),
        "two runs of one gang must agree bit-for-bit"
    );
}

/// The lockstep control word and the broadcast: a flag set on ANY rank is
/// seen by all, and every rank leaves the broadcast holding the root's bytes.
#[test]
fn local_flags_and_broadcast_agree_on_every_rank() {
    let flags = run_gang(3, move |local| {
        // Only rank 2 sees the divergence.
        let mine = if local.rank() == 2 { 0b100 } else { 0 };
        local.all_reduce_max_flags(mine).expect("flags")
    });
    assert_eq!(flags, vec![0b100, 0b100, 0b100]);

    // Rank 0 broadcasts its own tensor, which the closure below builds as
    // `matrix(2, 2, 0.0)` — the reference is that tensor, not another one.
    let root_bits = bits(&matrix(2, 2, 0.0));
    let broadcast = run_gang(3, move |local| {
        let mut t = matrix(2, 2, local.rank() as f32 * 1000.0);
        local.broadcast(&mut t, 0).expect("broadcast");
        bits(&t)
    });
    for (rank, got) in broadcast.iter().enumerate() {
        assert_eq!(got, &root_bits, "rank {rank} must hold rank 0's bytes");
    }

    let bad_root = run_gang(2, move |local| {
        let mut t = matrix(1, 1, 0.0);
        local
            .broadcast(&mut t, 5)
            .expect_err("a root outside the gang is refused")
            .to_string()
    });
    for message in &bad_root {
        assert!(
            message.contains("root 5 is not a rank of a gang of 2"),
            "unexpected message: {message}"
        );
    }
}

/// The barrier is a real rendezvous: no rank leaves it before every rank has
/// entered. Each rank records the arrival count it observed on exit; a
/// barrier that let a rank through early would show a count below the world
/// size.
#[test]
fn local_barrier_releases_only_after_every_rank_arrives() {
    let arrived = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let observed = {
        let arrived = Arc::clone(&arrived);
        run_gang(4, move |local| {
            // Stagger the arrivals so a barrier that released early would
            // almost certainly observe a count below four.
            std::thread::sleep(Duration::from_millis(10 * local.rank() as u64));
            arrived.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            local.barrier().expect("barrier");
            arrived.load(std::sync::atomic::Ordering::SeqCst)
        })
    };
    for (rank, count) in observed.iter().enumerate() {
        assert_eq!(
            *count, 4,
            "rank {rank} left the barrier with only {count} of 4 ranks arrived"
        );
    }
}

/// A gang whose ranks are executing DIFFERENT collectives has left lockstep,
/// and is told so rather than pairing a gather with a reduce and returning a
/// meaningless result.
#[test]
fn local_refuses_a_round_whose_ranks_are_at_different_collectives() {
    let messages = run_gang(2, move |local| {
        if local.rank() == 0 {
            local
                .all_reduce_max_flags(1)
                .map(|_| String::new())
                .unwrap_or_else(|e| e.to_string())
        } else {
            local
                .barrier()
                .map(|_| String::new())
                .unwrap_or_else(|e| e.to_string())
        }
    });
    assert!(
        messages.iter().any(|m| m.contains("left lockstep")),
        "neither rank reported the mismatch: {messages:?}"
    );
}

/// A gang whose peer never arrives fails with a typed deadline error rather
/// than parking forever — the property that keeps a crashed rank from
/// wedging its peers with no statement of what they are waiting for.
#[test]
fn local_rendezvous_expires_rather_than_parking_forever() {
    let gang =
        LocalGang::with_timeout(vec![Device::Cpu; 2], Duration::from_millis(50)).expect("gang");
    let rank0 = gang.rank(0).expect("rank 0");
    let error = rank0
        .barrier()
        .expect_err("rank 1 never arrives, so the round must expire");
    assert!(
        error.to_string().contains("timed out"),
        "unexpected message: {error}"
    );
}

/// A gang needs at least one rank and a non-zero deadline: both are domain
/// errors at construction, not conditions a later collective discovers.
#[test]
fn local_gang_refuses_an_empty_device_list_and_a_zero_deadline() {
    LocalGang::new(vec![]).expect_err("a gang of no ranks has no rank 0 to reduce on");
    LocalGang::with_timeout(vec![Device::Cpu], Duration::ZERO)
        .expect_err("a zero deadline expires before any peer can arrive");
    let gang = LocalGang::new(vec![Device::Cpu; 2]).expect("gang");
    gang.rank(2)
        .expect_err("rank 2 is not a rank of a gang of 2");
}

// ── (e) `Noop` at W = 1 is the identity ─────────────────────────────────────

/// Every [`Noop`] operation returns its input unchanged, bit-for-bit, and the
/// gang shape it reports is the single rank.
#[test]
fn noop_is_the_identity_at_a_single_rank() {
    let noop = Noop::new();
    assert_eq!(noop.rank(), 0);
    assert_eq!(noop.world(), 1);

    let local = matrix(3, 4, 1.5);
    let gathered = noop.all_gather(&local, &[3]).expect("all_gather");
    assert_eq!(gathered.dims(), local.dims());
    assert_eq!(bits(&gathered), bits(&local), "the gather is the input");

    let before: Vec<Vec<u32>> = [matrix(2, 2, 3.0), matrix(1, 5, -2.0)]
        .iter()
        .map(bits)
        .collect();
    let mut tensors = vec![matrix(2, 2, 3.0), matrix(1, 5, -2.0)];
    noop.all_reduce_sum(&mut tensors).expect("all_reduce_sum");
    let after: Vec<Vec<u32>> = tensors.iter().map(bits).collect();
    assert_eq!(after, before, "the reduce leaves every tensor untouched");

    for flags in [0u32, 1, 0xFFFF_FFFF] {
        assert_eq!(noop.all_reduce_max_flags(flags).expect("flags"), flags);
    }

    let mut t = matrix(2, 3, 8.25);
    let untouched = bits(&t);
    noop.broadcast(&mut t, 0).expect("broadcast");
    assert_eq!(bits(&t), untouched, "the broadcast leaves the tensor alone");

    noop.barrier().expect("barrier");

    // The single-rank topology does not excuse the domain checks: a partition
    // rule that would be wrong at `world > 1` is wrong here too.
    noop.all_gather(&matrix(3, 4, 0.0), &[3, 3])
        .expect_err("a two-entry count vector cannot describe a gang of one");
    noop.all_gather(&matrix(3, 4, 0.0), &[2])
        .expect_err("a count that contradicts the local row count is refused");
    noop.broadcast(&mut matrix(1, 1, 0.0), 1)
        .expect_err("rank 1 is not a rank of a gang of one");
}

/// A single-rank [`Local`] gang and a [`Noop`] agree bit-for-bit — the
/// property that lets a `world_size = 1` run take either arm and produce the
/// same bytes, which is what makes the `Noop` refactor-parity row a valid
/// baseline for the multi-rank arms.
#[test]
fn a_single_rank_local_gang_matches_noop_bit_for_bit() {
    let input = matrix(3, 2, 4.0);
    let noop = Noop::new();
    let noop_gathered = bits(&noop.all_gather(&input, &[3]).expect("all_gather"));
    let mut noop_tensors = vec![matrix(2, 2, 1.0)];
    noop.all_reduce_sum(&mut noop_tensors).expect("reduce");

    let per_rank = run_gang(1, move |local| {
        let gathered = local
            .all_gather(&matrix(3, 2, 4.0), &[3])
            .expect("all_gather");
        let mut tensors = vec![matrix(2, 2, 1.0)];
        local.all_reduce_sum(&mut tensors).expect("reduce");
        (bits(&gathered), bits(&tensors[0]))
    });

    assert_eq!(per_rank[0].0, noop_gathered);
    assert_eq!(per_rank[0].1, bits(&noop_tensors[0]));
}

/// A gathered remote slot carries no gradient back to the peer that produced
/// it: the rank's OWN slot is the only path backward. Without this the summed
/// gradient of a trainable parameter would be `world` times too large.
#[test]
fn local_all_gather_keeps_only_the_local_slot_attached() {
    let per_rank = run_gang(2, move |local| {
        let rank = local.rank();
        // A trainable var per rank, so a gradient that crossed to the peer
        // would show up as a gradient for a var this rank never touched.
        let var = candle_core::Var::from_tensor(&matrix(1, 2, rank as f32 + 1.0)).expect("var");
        let mine = var.as_tensor().affine(2.0, 0.0).expect("affine");
        let gathered = local.all_gather(&mine, &[1, 1]).expect("all_gather");
        let loss = gathered.sum_all().expect("sum");
        let grads = loss.backward().expect("backward");
        // Exactly one variable has a gradient here: this rank's own.
        (
            grads.get(&var).is_some(),
            grads.get(&var).map(bits).unwrap_or_default(),
        )
    });
    for (rank, (has_grad, grad)) in per_rank.iter().enumerate() {
        assert!(*has_grad, "rank {rank} must retain its own slot's gradient");
        assert_eq!(
            grad,
            &vec![2.0f32.to_bits(), 2.0f32.to_bits()],
            "rank {rank}'s gradient must be its OWN slot's, never scaled by the world size"
        );
    }
}

/// `DType` other than f32 flows through unchanged — the collective is not an
/// f32-only seam, and a bf16 adapter gradient reduces the same way.
#[test]
fn local_reduces_a_non_f32_dtype() {
    let per_rank = run_gang(2, move |local| {
        let rank = local.rank();
        let mut tensors = vec![
            Tensor::from_vec(vec![1u32 + rank, 10 + rank], 2, &Device::Cpu)
                .expect("tensor")
                .to_dtype(DType::U32)
                .expect("dtype"),
        ];
        local.all_reduce_sum(&mut tensors).expect("reduce");
        tensors[0].to_vec1::<u32>().expect("u32 elements")
    });
    for got in &per_rank {
        assert_eq!(got, &vec![3, 21]);
    }
}

// ── The gang's fault state: a failed round is permanent ─────────────────────

/// A round that expired cannot be completed by the peer that caused the
/// expiry.
///
/// The timed-out rank has already been told it exchanged nothing; if the late
/// peer were allowed to deposit into that same round it would fold the
/// timed-out rank's STALE contribution and return `Ok` — two ranks with
/// opposite verdicts about one round, which is the state the lockstep
/// property exists to exclude. The late peer must instead be told the gang
/// has failed, and told which collective it failed in.
#[test]
fn a_late_peer_cannot_complete_a_round_that_already_timed_out() {
    let gang =
        LocalGang::with_timeout(vec![Device::Cpu; 2], Duration::from_millis(50)).expect("gang");
    let rank0 = gang.rank(0).expect("rank 0");
    let rank1 = gang.rank(1).expect("rank 1");

    // Rank 0 sets a control word its peer would see, and expires waiting.
    let expired = rank0
        .all_reduce_max_flags(0b1011)
        .expect_err("rank 1 never arrives inside the deadline");
    assert!(
        expired.to_string().contains("timed out"),
        "unexpected message: {expired}"
    );

    // Rank 1 arrives after the deadline. `Ok(0b1011)` here would be rank 0's
    // stale flags — a decision rank 0 is not making.
    let late = rank1
        .all_reduce_max_flags(0)
        .expect_err("the round rank 1 would complete has already failed");
    let message = late.to_string();
    assert!(
        message.contains("the gang has already failed"),
        "the late peer must be told the gang failed, not handed a result: {message}"
    );
    assert!(
        message.contains("all_reduce_max_flags") && message.contains("timed out"),
        "the fault names the round it failed in: {message}"
    );
}

/// Once a gang has faulted, EVERY collective on EVERY rank refuses — promptly
/// and with a typed error naming the fault, never `Ok` and never a park.
///
/// The elapsed bound is the "never a hang" half of that: the gang's deadline
/// is two seconds, so a single collective that parked instead of refusing
/// would put the sweep over the bound on its own.
#[test]
fn every_collective_after_a_fault_errs_promptly_on_every_rank() {
    let deadline = Duration::from_secs(2);
    let gang = LocalGang::with_timeout(vec![Device::Cpu; 2], deadline).expect("gang");
    let rank0 = gang.rank(0).expect("rank 0");
    let rank1 = gang.rank(1).expect("rank 1");

    // Fault the gang through the lockstep check rather than the deadline, so
    // the deadline below is free to be long enough for a park to be visible.
    let mismatched = std::thread::scope(|scope| {
        let zero = scope.spawn(|| rank0.all_reduce_max_flags(1).map(|_| ()));
        let one = scope.spawn(|| rank1.barrier());
        [
            zero.join().expect("rank 0 thread"),
            one.join().expect("rank 1 thread"),
        ]
    });
    assert!(
        mismatched.iter().any(|r| r
            .as_ref()
            .err()
            .is_some_and(|e| e.to_string().contains("left lockstep"))),
        "the control: the gang must actually be faulted here, or the sweep below is vacuous"
    );

    let started = std::time::Instant::now();
    for (who, rank) in [("rank 0", &rank0), ("rank 1", &rank1)] {
        let mut sum = vec![matrix(1, 1, 0.0)];
        let mut broadcast = matrix(1, 1, 0.0);
        let attempts: [(&str, jammi_db::error::Result<()>); 5] = [
            (
                "all_gather",
                rank.all_gather(&matrix(1, 2, 0.0), &[1, 1]).map(|_| ()),
            ),
            ("all_reduce_sum", rank.all_reduce_sum(&mut sum)),
            (
                "all_reduce_max_flags",
                rank.all_reduce_max_flags(0).map(|_| ()),
            ),
            ("broadcast", rank.broadcast(&mut broadcast, 0)),
            ("barrier", rank.barrier()),
        ];
        for (op, result) in attempts {
            let error = result.expect_err(&format!(
                "{who}'s {op} ran on a gang that has already failed"
            ));
            assert!(
                error.to_string().contains("the gang has already failed"),
                "{who}'s {op}: unexpected message: {error}"
            );
        }
    }
    let elapsed = started.elapsed();
    assert!(
        elapsed < deadline,
        "the ten refusals took {elapsed:?}: at least one parked instead of refusing"
    );
}
