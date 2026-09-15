//! Hermetic oracles for the collective arms, on CPU devices.
//!
//! Every gang here runs its ranks on real OS threads through the real
//! rendezvous — the property under test is what a gang of concurrently
//! executing ranks agrees on, and a single-threaded driver would prove
//! nothing about that.

use std::sync::Arc;
use std::time::{Duration, Instant};

use candle_core::{DType, Device, Tensor};

use super::{BlockingCall, Collective, Local, LocalGang, Noop};

/// Run `f` on a fresh OS thread with a [`BlockingCall`] witness and join it:
/// the test's own thread is a runtime-free OS thread too, but the witness
/// has no constructor outside the three minting sites, so a test body that
/// calls a verb directly routes through one of them.
pub(super) fn witness<T: Send>(f: impl FnOnce(BlockingCall) -> T + Send) -> T {
    std::thread::scope(|scope| {
        BlockingCall::spawn_scoped(scope, f)
            .join()
            .expect("witness thread")
    })
}

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
    F: Fn(Local, BlockingCall) -> T + Send + Sync + 'static,
{
    let gang =
        LocalGang::with_timeout(vec![Device::Cpu; world], Duration::from_secs(30)).expect("gang");
    let body = Arc::new(body);
    let handles: Vec<_> = (0..world as u32)
        .map(|rank| {
            let local = gang.rank(rank).expect("rank handle");
            let body = Arc::clone(&body);
            BlockingCall::spawn_thread(move |call| body(local, call))
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
        run_gang(2, move |local, call| {
            let rank = local.rank() as usize;
            bits(
                &local
                    .all_gather(&call, &slices[rank], &[2, 2])
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
        run_gang(3, move |local, call| {
            let rank = local.rank() as usize;
            let gathered = local
                .all_gather(&call, &slices[rank], &counts)
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
    let wrong_length = run_gang(2, move |local, call| {
        local
            .all_gather(&call, &matrix(1, 2, 0.0), &[1])
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
    let own_count = run_gang(1, move |local, call| {
        local
            .all_gather(&call, &matrix(1, 2, 0.0), &[2])
            .expect_err("a rank that holds one row cannot claim two")
            .to_string()
    });
    assert!(
        own_count[0].contains("rank 0 holds 1 rows but the partition rule says 2"),
        "unexpected message: {}",
        own_count[0]
    );

    // Determinant 3: the ranks derived DIFFERENT count vectors. Each is
    // self-consistent, so only the round descriptor's `counts` field catches
    // it — and it catches it on BOTH ranks, symmetrically, before either is
    // handed a result.
    let disagreeing = run_gang(2, move |local, call| {
        let (rows, counts) = if local.rank() == 0 {
            (1usize, [1usize, 1])
        } else {
            (2, [1, 2])
        };
        local
            .all_gather(&call, &matrix(rows, 2, 0.0), &counts)
            .map(|_| String::new())
            .unwrap_or_else(|e| e.to_string())
    });
    for (rank, message) in disagreeing.iter().enumerate() {
        assert!(
            message.contains("disagree about what this round computes"),
            "rank {rank} must be refused, not silently handed a layout the other rank did not \
             agree to: {message}"
        );
    }
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

    let per_rank = run_gang(3, move |local, call| {
        let rank = local.rank() as usize;
        let mut tensors = vec![
            Tensor::from_vec(vec![terms[rank]], 1, &Device::Cpu).expect("tensor"),
            Tensor::from_vec(vec![terms[rank] * 2.0], 1, &Device::Cpu).expect("tensor"),
        ];
        local
            .all_reduce_sum(&call, &mut tensors)
            .expect("all_reduce_sum");
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
        run_gang(3, move |local, call| {
            let rank = local.rank() as usize;
            let gathered = local
                .all_gather(&call, &matrix(rank + 1, 2, rank as f32 * 10.0), &[1, 2, 3])
                .expect("all_gather");
            let mut tensors = vec![matrix(2, 2, rank as f32 * 0.3)];
            local
                .all_reduce_sum(&call, &mut tensors)
                .expect("all_reduce_sum");
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
    let flags = run_gang(3, move |local, call| {
        // Only rank 2 sees the divergence.
        let mine = if local.rank() == 2 { 0b100 } else { 0 };
        local.all_reduce_max_flags(&call, mine).expect("flags")
    });
    assert_eq!(flags, vec![0b100, 0b100, 0b100]);

    // Rank 0 broadcasts its own tensor, which the closure below builds as
    // `matrix(2, 2, 0.0)` — the reference is that tensor, not another one.
    let root_bits = bits(&matrix(2, 2, 0.0));
    let broadcast = run_gang(3, move |local, call| {
        let mut t = matrix(2, 2, local.rank() as f32 * 1000.0);
        local.broadcast(&call, &mut t, 0).expect("broadcast");
        bits(&t)
    });
    for (rank, got) in broadcast.iter().enumerate() {
        assert_eq!(got, &root_bits, "rank {rank} must hold rank 0's bytes");
    }

    let bad_root = run_gang(2, move |local, call| {
        let mut t = matrix(1, 1, 0.0);
        local
            .broadcast(&call, &mut t, 5)
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
        run_gang(4, move |local, call| {
            // Stagger the arrivals so a barrier that released early would
            // almost certainly observe a count below four.
            std::thread::sleep(Duration::from_millis(10 * local.rank() as u64));
            arrived.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            local.barrier(&call).expect("barrier");
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

/// A gang whose ranks are executing DIFFERENT collectives disagrees at the
/// verb field of the round descriptor, and BOTH ranks are told so — never a
/// gather paired with a reduce and handed back as a meaningless result.
#[test]
fn local_refuses_a_round_whose_ranks_are_at_different_collectives() {
    let messages = run_gang(2, move |local, call| {
        if local.rank() == 0 {
            local
                .all_reduce_max_flags(&call, 1)
                .map(|_| String::new())
                .unwrap_or_else(|e| e.to_string())
        } else {
            local
                .barrier(&call)
                .map(|_| String::new())
                .unwrap_or_else(|e| e.to_string())
        }
    });
    for (rank, message) in messages.iter().enumerate() {
        assert!(
            message.contains("disagree about what this round computes"),
            "rank {rank} must be refused, not handed a meaningless result: {message}"
        );
    }
}

/// A gang whose peer never arrives fails with a typed deadline error rather
/// than parking forever — the property that keeps a crashed rank from
/// wedging its peers with no statement of what they are waiting for.
#[test]
fn local_rendezvous_expires_rather_than_parking_forever() {
    witness(|call| {
        let gang =
            LocalGang::with_timeout(vec![Device::Cpu; 2], Duration::from_millis(50)).expect("gang");
        let rank0 = gang.rank(0).expect("rank 0");
        let error = rank0
            .barrier(&call)
            .expect_err("rank 1 never arrives, so the round must expire");
        assert!(
            error.to_string().contains("timed out"),
            "unexpected message: {error}"
        );
    });
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
    witness(|call| {
        let noop = Noop::new();
        assert_eq!(noop.rank(), 0);
        assert_eq!(noop.world(), 1);

        let local = matrix(3, 4, 1.5);
        let gathered = noop.all_gather(&call, &local, &[3]).expect("all_gather");
        assert_eq!(gathered.dims(), local.dims());
        assert_eq!(bits(&gathered), bits(&local), "the gather is the input");

        let before: Vec<Vec<u32>> = [matrix(2, 2, 3.0), matrix(1, 5, -2.0)]
            .iter()
            .map(bits)
            .collect();
        let mut tensors = vec![matrix(2, 2, 3.0), matrix(1, 5, -2.0)];
        noop.all_reduce_sum(&call, &mut tensors)
            .expect("all_reduce_sum");
        let after: Vec<Vec<u32>> = tensors.iter().map(bits).collect();
        assert_eq!(after, before, "the reduce leaves every tensor untouched");

        for flags in [0u32, 1, 0xFFFF_FFFF] {
            assert_eq!(
                noop.all_reduce_max_flags(&call, flags).expect("flags"),
                flags
            );
        }

        let mut t = matrix(2, 3, 8.25);
        let untouched = bits(&t);
        noop.broadcast(&call, &mut t, 0).expect("broadcast");
        assert_eq!(bits(&t), untouched, "the broadcast leaves the tensor alone");

        noop.barrier(&call).expect("barrier");

        // The single-rank topology does not excuse the domain checks: a partition
        // rule that would be wrong at `world > 1` is wrong here too.
        noop.all_gather(&call, &matrix(3, 4, 0.0), &[3, 3])
            .expect_err("a two-entry count vector cannot describe a gang of one");
        noop.all_gather(&call, &matrix(3, 4, 0.0), &[2])
            .expect_err("a count that contradicts the local row count is refused");
        noop.broadcast(&call, &mut matrix(1, 1, 0.0), 1)
            .expect_err("rank 1 is not a rank of a gang of one");
    });
}

/// A single-rank [`Local`] gang and a [`Noop`] agree bit-for-bit — the
/// property that lets a `world_size = 1` run take either arm and produce the
/// same bytes, which is what makes the `Noop` refactor-parity row a valid
/// baseline for the multi-rank arms.
#[test]
fn a_single_rank_local_gang_matches_noop_bit_for_bit() {
    witness(|call| {
        let input = matrix(3, 2, 4.0);
        let noop = Noop::new();
        let noop_gathered = bits(&noop.all_gather(&call, &input, &[3]).expect("all_gather"));
        let mut noop_tensors = vec![matrix(2, 2, 1.0)];
        noop.all_reduce_sum(&call, &mut noop_tensors)
            .expect("reduce");

        let per_rank = run_gang(1, move |local, call| {
            let gathered = local
                .all_gather(&call, &matrix(3, 2, 4.0), &[3])
                .expect("all_gather");
            let mut tensors = vec![matrix(2, 2, 1.0)];
            local.all_reduce_sum(&call, &mut tensors).expect("reduce");
            (bits(&gathered), bits(&tensors[0]))
        });

        assert_eq!(per_rank[0].0, noop_gathered);
        assert_eq!(per_rank[0].1, bits(&noop_tensors[0]));
    });
}

/// A gathered remote slot carries no gradient back to the peer that produced
/// it: the rank's OWN slot is the only path backward. Without this the summed
/// gradient of a trainable parameter would be `world` times too large.
#[test]
fn local_all_gather_keeps_only_the_local_slot_attached() {
    let per_rank = run_gang(2, move |local, call| {
        let rank = local.rank();
        // A trainable var per rank, so a gradient that crossed to the peer
        // would show up as a gradient for a var this rank never touched.
        let var = candle_core::Var::from_tensor(&matrix(1, 2, rank as f32 + 1.0)).expect("var");
        let mine = var.as_tensor().affine(2.0, 0.0).expect("affine");
        let gathered = local.all_gather(&call, &mine, &[1, 1]).expect("all_gather");
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
    let per_rank = run_gang(2, move |local, call| {
        let rank = local.rank();
        let mut tensors = vec![
            Tensor::from_vec(vec![1u32 + rank, 10 + rank], 2, &Device::Cpu)
                .expect("tensor")
                .to_dtype(DType::U32)
                .expect("dtype"),
        ];
        local.all_reduce_sum(&call, &mut tensors).expect("reduce");
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
    witness(|call| {
        let gang =
            LocalGang::with_timeout(vec![Device::Cpu; 2], Duration::from_millis(50)).expect("gang");
        let rank0 = gang.rank(0).expect("rank 0");
        let rank1 = gang.rank(1).expect("rank 1");

        // Rank 0 sets a control word its peer would see, and expires waiting.
        let expired = rank0
            .all_reduce_max_flags(&call, 0b1011)
            .expect_err("rank 1 never arrives inside the deadline");
        assert!(
            expired.to_string().contains("timed out"),
            "unexpected message: {expired}"
        );

        // Rank 1 arrives after the deadline. `Ok(0b1011)` here would be rank 0's
        // stale flags — a decision rank 0 is not making.
        let late = rank1
            .all_reduce_max_flags(&call, 0)
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
    });
}

/// Once a gang has faulted, EVERY collective on EVERY rank refuses — promptly
/// and with a typed error naming the fault, never `Ok` and never a park.
///
/// The elapsed bound is the "never a hang" half of that: the gang's deadline
/// is two seconds, so a single collective that parked instead of refusing
/// would put the sweep over the bound on its own.
#[test]
fn every_collective_after_a_fault_errs_promptly_on_every_rank() {
    witness(|call| {
        let deadline = Duration::from_secs(2);
        let gang = LocalGang::with_timeout(vec![Device::Cpu; 2], deadline).expect("gang");
        let rank0 = gang.rank(0).expect("rank 0");
        let rank1 = gang.rank(1).expect("rank 1");

        // Fault the gang through the lockstep check rather than the deadline, so
        // the deadline below is free to be long enough for a park to be visible.
        let mismatched = std::thread::scope(|scope| {
            let zero = BlockingCall::spawn_scoped(scope, |call| {
                rank0.all_reduce_max_flags(&call, 1).map(|_| ())
            });
            let one = BlockingCall::spawn_scoped(scope, |call| rank1.barrier(&call));
            [
                zero.join().expect("rank 0 thread"),
                one.join().expect("rank 1 thread"),
            ]
        });
        assert!(
            mismatched.iter().any(|r| r.as_ref().err().is_some_and(|e| e
                .to_string()
                .contains("disagree about what this round computes"))),
            "the control: the gang must actually be faulted here, or the sweep below is vacuous"
        );

        let started = std::time::Instant::now();
        for (who, rank) in [("rank 0", &rank0), ("rank 1", &rank1)] {
            let mut sum = vec![matrix(1, 1, 0.0)];
            let mut broadcast = matrix(1, 1, 0.0);
            let attempts: [(&str, jammi_db::error::Result<()>); 5] = [
                (
                    "all_gather",
                    rank.all_gather(&call, &matrix(1, 2, 0.0), &[1, 1])
                        .map(|_| ()),
                ),
                ("all_reduce_sum", rank.all_reduce_sum(&call, &mut sum)),
                (
                    "all_reduce_max_flags",
                    rank.all_reduce_max_flags(&call, 0).map(|_| ()),
                ),
                ("broadcast", rank.broadcast(&call, &mut broadcast, 0)),
                ("barrier", rank.barrier(&call)),
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
    });
}

// ── The round descriptor: no rank returns `Ok` from a round any rank rejects ─

/// Rank 0 calls `broadcast(root = 0)`, rank 1 calls
/// `broadcast(root = 1)` — each is a valid root of a two-rank gang on its
/// own, so nothing here is a domain error either rank could catch alone. At
/// base (`f74943b5`) both return `Ok` with DIFFERENT bytes: rank 0 broadcasts
/// its own tensor, rank 1 broadcasts its own, and the two never rendezvous
/// about which of them is really the root. With the round descriptor's
/// `root` field agreed before publishing, this is a symmetric typed error on
/// both ranks instead.
#[test]
fn broadcast_with_self_named_roots_faults_both_ranks_symmetrically() {
    let results: Vec<std::result::Result<Vec<u32>, String>> = run_gang(2, move |local, call| {
        let mut t = matrix(1, 1, local.rank() as f32 + 1.0);
        let root = local.rank(); // each rank names ITSELF the root
        local
            .broadcast(&call, &mut t, root)
            .map(|_| bits(&t))
            .map_err(|e| e.to_string())
    });
    for (rank, result) in results.iter().enumerate() {
        let error = match result {
            Err(error) => error,
            Ok(bytes) => panic!(
                "rank {rank} returned Ok({bytes:?}) from a round rank {} also rejected the \
                 premise of — a self-named root must never be handed a peer's answer",
                1 - rank
            ),
        };
        assert!(
            error.contains("disagree about what this round computes"),
            "rank {rank}: unexpected message: {error}"
        );
    }
}

/// Rank 0 calls `all_gather` with `counts = [1, 1]`,
/// rank 1 calls it with `counts = [1, 2]` — each rank's OWN row count agrees
/// with its OWN counts vector, so neither can catch the disagreement from its
/// own inputs alone. At base (`f74943b5`) rank 0 errs (its view of rank 1's
/// row count contradicts rank 0's copy of `counts`) but rank 1 returns `Ok`
/// with a 3-row gather (rank 1's copy of `counts` matches everything rank 1
/// can see). With the round descriptor's `counts` field agreed before
/// publishing, both ranks are refused.
#[test]
fn all_gather_with_disagreeing_counts_faults_both_ranks_symmetrically() {
    let results: Vec<std::result::Result<Vec<usize>, String>> = run_gang(2, move |local, call| {
        let (rows, counts) = if local.rank() == 0 {
            (1usize, [1usize, 1])
        } else {
            (2usize, [1usize, 2])
        };
        local
            .all_gather(&call, &matrix(rows, 2, 0.0), &counts)
            .map(|t| t.dims().to_vec())
            .map_err(|e| e.to_string())
    });
    for (rank, result) in results.iter().enumerate() {
        let error = match result {
            Err(error) => error,
            Ok(dims) => panic!(
                "rank {rank} returned Ok(dims = {dims:?}) from a round rank {} also rejected \
                 the premise of — a disagreeing partition must never be handed a gathered \
                 layout",
                1 - rank
            ),
        };
        assert!(
            error.contains("disagree about what this round computes"),
            "rank {rank}: unexpected message: {error}"
        );
    }
}

/// A PRE-rendezvous domain refusal on one rank (a `counts` vector of the
/// wrong length — `exchange` is never even entered) still faults the WHOLE
/// gang, so the peer's next collective is refused PROMPTLY rather than
/// discovering the same fact only after waiting out its own deadline. This
/// is `Local::guarded`'s `fail_detached` call: delete it and rank 1 below
/// would instead wait the full `deadline` for a rank 0 that never arrives.
#[test]
fn a_pre_rendezvous_domain_refusal_on_one_rank_faults_the_gang_before_its_peer_waits_out_the_deadline(
) {
    witness(|call| {
        let deadline = Duration::from_secs(4);
        let gang = LocalGang::with_timeout(vec![Device::Cpu; 2], deadline).expect("gang");
        let rank0 = gang.rank(0).expect("rank 0");
        let rank1 = gang.rank(1).expect("rank 1");

        // Rank 0's own domain check fails before it ever deposits into a round:
        // a one-entry counts vector cannot describe a two-rank gang.
        rank0
            .all_gather(&call, &matrix(1, 2, 0.0), &[1])
            .expect_err("rank 0's own counts cannot describe this gang");

        let started = Instant::now();
        let peer = rank1
            .barrier(&call)
            .expect_err("the gang is already faulted by rank 0's domain refusal");
        let elapsed = started.elapsed();
        assert!(
            peer.to_string().contains("the gang has already failed"),
            "rank 1 must be told the gang already failed, not handed a fresh timeout: {peer}"
        );
        assert!(
            elapsed < deadline / 4,
            "rank 1 waited {elapsed:?} for a fault that had already happened when it called \
             barrier — a pre-rendezvous domain refusal on one rank must fault the gang promptly, \
             not leave the peer to discover it only after its own deadline"
        );
    });
}

/// `Shared::check_entry` runs BEFORE a collective's own domain checks: on a
/// faulted gang, `all_gather` with counts that are ALSO independently wrong
/// reports the FAULT, never the counts error — the fault is the fact that
/// explains why the round cannot happen at all, and a fresh domain error
/// would mask it. This is what dies if `check_entry` is mutated to return
/// `Ok(())` unconditionally: the counts error would surface instead.
#[test]
fn a_faulted_gang_reports_the_fault_before_a_new_domain_error_on_every_rank() {
    witness(|call| {
        let gang =
            LocalGang::with_timeout(vec![Device::Cpu; 2], Duration::from_millis(50)).expect("gang");
        let rank0 = gang.rank(0).expect("rank 0");
        let rank1 = gang.rank(1).expect("rank 1");

        // Fault the gang via a timeout: rank 1 never arrives.
        rank0
            .barrier(&call)
            .expect_err("rank 1 never arrives inside the deadline");

        // Rank 1 now calls `all_gather` with counts that are ALSO independently
        // wrong (a one-entry vector for a two-rank gang) — `check_entry` must
        // refuse with the fault before that domain check ever runs.
        let error = rank1
            .all_gather(&call, &matrix(1, 2, 0.0), &[1])
            .expect_err("the gang is already faulted");
        let message = error.to_string();
        assert!(
            message.contains("the gang has already failed"),
            "a faulted gang must report the FAULT first, never a fresh domain error that masks \
             it: {message}"
        );
        assert!(
            !message.contains("counts has 1 entries"),
            "the counts error must never surface once the gang is faulted: {message}"
        );
    });
}

// ── The shared seam: a 0-dim tensor is refused before any arm signs it ─────

/// A 0-dim scalar has
/// no row count to check against a partition rule, so `checked_gather_counts`
/// — the ONE seam every arm calls before it does anything else — must refuse
/// it, naming the rank and the shape, before any arm's own gather logic ever
/// runs.
///
/// At `c9d20550` `checked_gather_counts` read a 0-dim tensor's row count as
/// `unwrap_or(0)`, and `Descriptor::of_gather_slice` signed it with
/// `unwrap_or_default()` exactly like a 1-D tensor's trailing shape — so a
/// two-rank gather where rank 0 passes a 0-dim scalar and rank 1 a real 1-D
/// `[0]` tensor (both claiming zero rows) produced two descriptors that
/// happened to agree (`{ dims: [] }` for both) and the round published:
/// `[Ok([]), Ok([0])]`, one rank silently signing a shape the other rank
/// never actually held.
#[test]
fn a_0_dim_tensor_is_refused_before_any_arm_signs_it() {
    witness(|call| {
        // Noop (world = 1): the single-rank topology gets the same domain check.
        let noop = Noop::new();
        let scalar = Tensor::new(1.0f32, &Device::Cpu).expect("0-dim scalar");
        let error = noop
            .all_gather(&call, &scalar, &[0])
            .expect_err("a 0-dim tensor has no row count to gather along");
        assert!(
            error.to_string().contains("0-dim"),
            "unexpected message: {error}"
        );

        // Local (world = 2): the same 0-dim shape — rank 0 a 0-dim
        // scalar, rank 1 a real 1-D `[0]` tensor, both claiming zero rows.
        let results: Vec<std::result::Result<Vec<usize>, String>> =
            run_gang(2, move |local, call| {
                let counts = [0usize, 0];
                let t = if local.rank() == 0 {
                    Tensor::new(1.0f32, &Device::Cpu).expect("0-dim scalar")
                } else {
                    Tensor::from_vec(Vec::<f32>::new(), (0,), &Device::Cpu).expect("1-D [0]")
                };
                local
                    .all_gather(&call, &t, &counts)
                    .map(|g| g.dims().to_vec())
                    .map_err(|e| e.to_string())
            });
        for (rank, result) in results.iter().enumerate() {
            let error = match result {
                Err(error) => error,
                Ok(dims) => panic!(
                    "rank {rank} returned Ok(dims = {dims:?}) — a 0-dim tensor must never be signed \
                     as though it were a 1-D tensor of the same trailing shape (at c9d20550 this was \
                     `[Ok([]), Ok([0])]`)"
                ),
            };
            assert!(
                error.contains("0-dim"),
                "rank {rank}: unexpected message: {error}"
            );
        }
    });
}

/// A second missing end-to-end oracle found while re-aiming the per-field
/// sweep at the descriptor constructors: `all_gather`'s
/// TRAILING shape is a descriptor determinant (`TensorSignature::of_gather_slice`
/// drops only dim 0, which `counts` already governs) but, before this test,
/// no end-to-end case exercised it — a mutation that replaced the
/// constructor's `tensors` field with an empty vector left every other test
/// in this file passing. Two ranks pass the SAME `counts` (so neither rank's
/// own row-count check catches anything) but a DIFFERENT trailing shape —
/// rank 0 a `[1, 2]` tensor, rank 1 a `[1, 3]` tensor — so only the
/// descriptor's `tensors` field can catch the disagreement, symmetrically,
/// before either rank is folded into a concatenation the other never agreed
/// to the shape of.
#[test]
fn a_two_rank_all_gather_with_a_trailing_shape_mismatch_at_equal_counts_faults_both_ranks_symmetrically(
) {
    let results: Vec<std::result::Result<Vec<usize>, String>> = run_gang(2, move |local, call| {
        let cols = if local.rank() == 0 { 2 } else { 3 };
        local
            .all_gather(&call, &matrix(1, cols, 0.0), &[1, 1])
            .map(|t| t.dims().to_vec())
            .map_err(|e| e.to_string())
    });
    for (rank, result) in results.iter().enumerate() {
        let error = match result {
            Err(error) => error,
            Ok(dims) => panic!(
                "rank {rank} returned Ok(dims = {dims:?}) from a gather the peer's trailing \
                 shape disagreed with — equal counts must not excuse a differently shaped \
                 tensor at each rank"
            ),
        };
        assert!(
            error.contains("disagree about what this round computes"),
            "rank {rank}: unexpected message: {error}"
        );
    }
}

/// A reachability probe: three
/// ranks name an ASYMMETRIC set of roots for `broadcast` — two agree with
/// each other, one disagrees — so more than one rank's `Contribution` carries
/// `Some` at once if the round were ever (wrongly) assembled. The property
/// under test is that this is refused before any rank reaches that far, never
/// that a particular unwrap happens not to panic on this input: a thread that
/// panicked would fail `run_gang`'s own `.expect("a rank thread panicked")`.
#[test]
fn reachability_a_three_rank_asymmetric_root_broadcast_never_panics() {
    let roots = [0u32, 1, 0];
    let messages = run_gang(3, move |local, call| {
        let mut t = matrix(1, 1, local.rank() as f32);
        local
            .broadcast(&call, &mut t, roots[local.rank() as usize])
            .map(|_| String::new())
            .unwrap_or_else(|e| e.to_string())
    });
    for (rank, message) in messages.iter().enumerate() {
        assert!(
            !message.is_empty(),
            "rank {rank} returned Ok from an asymmetric-root round — a round naming two \
             different roots must never be handed a result"
        );
        assert!(
            message.contains("disagree about what this round computes"),
            "rank {rank}: unexpected message: {message}"
        );
    }
}

/// A reachability probe: two ranks call `all_reduce_sum` with lists of
/// DIFFERENT lengths. `Local::all_reduce_sum`'s fold loop indexes every
/// peer's contribution at each of the CALLING rank's own tensor indices
/// (`peer_tensors[index]`) — a peer with a shorter list is an out-of-bounds
/// index if such a round were ever assembled, so the length mismatch must be
/// caught before the fold ever runs, not merely happen not to panic for this
/// particular pair of lengths.
#[test]
fn reachability_a_two_rank_uneven_all_reduce_sum_list_never_panics() {
    let messages = run_gang(2, move |local, call| {
        let mut tensors = if local.rank() == 0 {
            vec![matrix(1, 1, 0.0), matrix(1, 1, 1.0)]
        } else {
            vec![matrix(1, 1, 0.0)]
        };
        local
            .all_reduce_sum(&call, &mut tensors)
            .map(|_| String::new())
            .unwrap_or_else(|e| e.to_string())
    });
    for (rank, message) in messages.iter().enumerate() {
        assert!(
            !message.is_empty(),
            "rank {rank} returned Ok from a round whose peer reduced a different number of \
             tensors"
        );
        assert!(
            message.contains("disagree about what this round computes"),
            "rank {rank}: unexpected message: {message}"
        );
    }
}

/// The root passes a `[2, 3]` tensor, the
/// non-root a `[1, 1]` tensor. `broadcast`'s descriptor carries every rank's
/// OWN tensor shape, not only the root's, so a shape mismatch off the root is
/// a symmetric typed error on BOTH ranks, and the gang is left faulted for
/// any later collective.
#[test]
fn broadcast_with_a_shape_mismatch_off_the_root_faults_both_ranks_symmetrically() {
    witness(|call| {
        let gang =
            LocalGang::with_timeout(vec![Device::Cpu; 2], Duration::from_secs(5)).expect("gang");
        let rank0 = gang.rank(0).expect("rank 0");
        let rank1 = gang.rank(1).expect("rank 1");

        let results = std::thread::scope(|scope| {
            let root = BlockingCall::spawn_scoped(scope, move |call| {
                let mut t = matrix(2, 3, 0.0);
                rank0
                    .broadcast(&call, &mut t, 0)
                    .map(|_| String::new())
                    .unwrap_or_else(|e| e.to_string())
            });
            let non_root = BlockingCall::spawn_scoped(scope, move |call| {
                let mut t = matrix(1, 1, 0.0);
                rank1
                    .broadcast(&call, &mut t, 0)
                    .map(|_| String::new())
                    .unwrap_or_else(|e| e.to_string())
            });
            [
                root.join().expect("root thread"),
                non_root.join().expect("non-root thread"),
            ]
        });

        for (who, message) in [("root", &results[0]), ("non-root", &results[1])] {
            assert!(
                !message.is_empty(),
                "{who} returned Ok from a broadcast the peer's shape disagreed with"
            );
            assert!(
                message.contains("disagree about what this round computes"),
                "{who}: unexpected message: {message}"
            );
        }

        // The gang is left faulted: a fresh handle's next collective refuses.
        let after_fault = gang.rank(0).expect("rank 0 handle after the fault");
        let refused = after_fault
            .barrier(&call)
            .expect_err("the gang must stay faulted after the shape mismatch");
        assert!(
            refused.to_string().contains("the gang has already failed"),
            "unexpected message: {refused}"
        );
    });
}

/// A third attempt to break "no rank can return `Ok` from a round any rank
/// rejects": a THREE-rank gang where two ranks agree with EACH OTHER and
/// only the THIRD disagrees, on a field neither the root check nor the
/// counts check exercises — the reduced tensor's dtype. This shape of case is
/// exactly what a wrong implementation (say, one that compared each rank
/// only to its immediate neighbor, or that only checked the field the
/// tests above happened to cover) would get wrong: two agreeing ranks pairing up
/// and only the outlier ever noticing. `agrees_with` compares every rank's
/// descriptor against rank 0's, so this must fault symmetrically too.
#[test]
fn a_third_attempt_two_ranks_agree_a_third_disagrees_on_dtype_still_faults_every_rank() {
    let results: Vec<std::result::Result<(), String>> = run_gang(3, move |local, call| {
        let dtype = if local.rank() == 2 {
            DType::F64
        } else {
            DType::F32
        };
        let mut tensors = vec![Tensor::zeros((2, 2), dtype, &Device::Cpu).expect("tensor")];
        local
            .all_reduce_sum(&call, &mut tensors)
            .map_err(|e| e.to_string())
    });
    for (rank, result) in results.iter().enumerate() {
        let error = match result {
            Err(error) => error,
            Ok(()) => panic!(
                "rank {rank} returned Ok from a round rank 2 also rejected the premise of — \
                 a disagreeing dtype at one rank out of three must never be handed a result \
                 to the other two"
            ),
        };
        assert!(
            error.contains("disagree about what this round computes"),
            "rank {rank}: unexpected message: {error}"
        );
    }
}

/// The missing end-to-end oracle for `all_reduce_sum`'s SHAPE determinant:
/// a two-rank gang where rank 0 reduces a `[2, 2]`
/// tensor and rank 1 a `[2, 3]` tensor at the same trainable-variable index.
/// Neither rank's own tensor is internally inconsistent — each is a valid
/// shape for a reduce on its own — so only the round descriptor's per-tensor
/// `TensorSignature` catches the disagreement, symmetrically, before either
/// rank is folded into the other's sum (which `Tensor::add` would refuse
/// with a backend error anyway, but only for the rank unlucky enough to
/// reach the fold — the descriptor check refuses BOTH, before the fold ever
/// runs, and leaves the gang faulted for every collective after this one).
#[test]
fn a_two_rank_all_reduce_sum_with_a_per_tensor_shape_mismatch_faults_both_ranks_symmetrically() {
    witness(|call| {
        let gang =
            LocalGang::with_timeout(vec![Device::Cpu; 2], Duration::from_secs(5)).expect("gang");
        let rank0 = gang.rank(0).expect("rank 0");
        let rank1 = gang.rank(1).expect("rank 1");

        let results = std::thread::scope(|scope| {
            let a = BlockingCall::spawn_scoped(scope, move |call| {
                let mut tensors = vec![matrix(2, 2, 0.0)];
                rank0
                    .all_reduce_sum(&call, &mut tensors)
                    .map(|_| String::new())
                    .unwrap_or_else(|e| e.to_string())
            });
            let b = BlockingCall::spawn_scoped(scope, move |call| {
                let mut tensors = vec![matrix(2, 3, 0.0)];
                rank1
                    .all_reduce_sum(&call, &mut tensors)
                    .map(|_| String::new())
                    .unwrap_or_else(|e| e.to_string())
            });
            [
                a.join().expect("rank 0 thread"),
                b.join().expect("rank 1 thread"),
            ]
        });

        for (rank, message) in results.iter().enumerate() {
            assert!(
                !message.is_empty(),
                "rank {rank} returned Ok from a reduce the peer's shape disagreed with"
            );
            assert!(
                message.contains("disagree about what this round computes"),
                "rank {rank}: unexpected message: {message}"
            );
        }

        // The gang is left faulted: a fresh handle's next collective refuses.
        let after_fault = gang.rank(0).expect("rank 0 handle after the fault");
        let refused = after_fault
            .barrier(&call)
            .expect_err("the gang must stay faulted after the shape mismatch");
        assert!(
            refused.to_string().contains("the gang has already failed"),
            "unexpected message: {refused}"
        );
    });
}

// ── the verb field, pinned at EACH constructor ──────────
//
// `agrees_with`'s per-field mutation sweep proves the `verb` COMPARISON is
// sound in isolation, but every `Descriptor` in that sweep is hand-built —
// none of it runs through a real verb's constructor. The three tests below
// close that gap: rank 0 calls the verb under test through its real
// `Collective` method (so its descriptor is built by that constructor, not by
// a test helper), rank 1 calls `barrier`, and the property is the same one
// `local_refuses_a_round_whose_ranks_are_at_different_collectives` already
// proves for `all_reduce_max_flags` and `barrier` themselves: a round whose
// ranks are at different verbs is refused on BOTH ranks, symmetrically, and
// the gang is left faulted for the next call on EITHER rank.
//
// The mutation for each is hardcoding `verb: contribution.kind()` to
// `verb: "barrier"` at that ONE constructor. `all_gather`'s descriptor always
// carries `counts: Some(_)` (never `None`) and `broadcast`'s always carries
// `root: Some(_)` (never `None`), so — MEASURED below, not assumed — a
// verb-only hardcode at either of those two constructors leaves a second,
// independent field disagreement against `barrier`'s `{ counts: None, root:
// None }` and the round stays refused for that reason instead: the mutation
// does NOT kill the corresponding test on its own. `all_reduce_sum`'s
// descriptor already carries `root: None` and `counts: None` like
// `barrier`'s, so calling it with an EMPTY tensor slice makes its `tensors`
// field `vec![]` too — the only constructor of the three where hardcoding
// `verb` alone drives every other field into agreement with `barrier`,
// publishing the round; the empty tensor slice then means
// `Local::all_reduce_sum`'s own fold loop (which iterates the CALLING rank's
// tensors, not the round) never runs, so instead of the fold hitting a
// foreign `Contribution::Barrier`, both ranks are silently handed `Ok` — the
// "refused, not published" property fails a different, and arguably worse,
// way: no error, no panic, both ranks agree on nothing.

/// `all_gather`'s constructor (local.rs `Collective::all_gather`): the
/// mutation `verb: "barrier"` is executed and reported below rather than
/// assumed to kill this test.
#[test]
fn a_two_rank_all_gather_and_barrier_verb_mismatch_faults_both_ranks_symmetrically() {
    witness(|call| {
        let gang =
            LocalGang::with_timeout(vec![Device::Cpu; 2], Duration::from_secs(5)).expect("gang");
        let rank0 = gang.rank(0).expect("rank 0");
        let rank1 = gang.rank(1).expect("rank 1");

        let results = std::thread::scope(|scope| {
            let gather = BlockingCall::spawn_scoped(scope, move |call| {
                rank0
                    .all_gather(&call, &matrix(1, 2, 0.0), &[1, 1])
                    .map(|_| String::new())
                    .unwrap_or_else(|e| e.to_string())
            });
            let barrier = BlockingCall::spawn_scoped(scope, move |call| {
                rank1
                    .barrier(&call)
                    .map(|_| String::new())
                    .unwrap_or_else(|e| e.to_string())
            });
            [
                gather.join().expect("rank 0 thread"),
                barrier.join().expect("rank 1 thread"),
            ]
        });

        for (rank, message) in results.iter().enumerate() {
            assert!(
                !message.is_empty(),
                "rank {rank} returned Ok from a round rank {} was running a different collective in",
                1 - rank
            );
            assert!(
                message.contains("disagree about what this round computes"),
                "rank {rank}: unexpected message: {message}"
            );
        }

        let after_fault = gang.rank(0).expect("rank 0 handle after the fault");
        let refused = after_fault
            .barrier(&call)
            .expect_err("the gang must stay faulted after the verb mismatch");
        assert!(
            refused.to_string().contains("the gang has already failed"),
            "unexpected message: {refused}"
        );
    });
}

/// `all_reduce_sum`'s constructor: the tensor slice is EMPTY, which is what
/// makes the `verb: "barrier"` mutation actually publish the round (see the
/// section comment above) rather than being caught by a second, independent
/// field.
#[test]
fn a_two_rank_all_reduce_sum_and_barrier_verb_mismatch_faults_both_ranks_symmetrically() {
    witness(|call| {
        let gang =
            LocalGang::with_timeout(vec![Device::Cpu; 2], Duration::from_secs(5)).expect("gang");
        let rank0 = gang.rank(0).expect("rank 0");
        let rank1 = gang.rank(1).expect("rank 1");

        let results = std::thread::scope(|scope| {
            let reduce = BlockingCall::spawn_scoped(scope, move |call| {
                let mut tensors: Vec<Tensor> = Vec::new();
                rank0
                    .all_reduce_sum(&call, &mut tensors)
                    .map(|_| String::new())
                    .unwrap_or_else(|e| e.to_string())
            });
            let barrier = BlockingCall::spawn_scoped(scope, move |call| {
                rank1
                    .barrier(&call)
                    .map(|_| String::new())
                    .unwrap_or_else(|e| e.to_string())
            });
            [
                reduce.join().expect("rank 0 thread"),
                barrier.join().expect("rank 1 thread"),
            ]
        });

        for (rank, message) in results.iter().enumerate() {
            assert!(
                !message.is_empty(),
                "rank {rank} returned Ok from a round rank {} was running a different collective in",
                1 - rank
            );
            assert!(
                message.contains("disagree about what this round computes"),
                "rank {rank}: unexpected message: {message}"
            );
        }

        let after_fault = gang.rank(0).expect("rank 0 handle after the fault");
        let refused = after_fault
            .barrier(&call)
            .expect_err("the gang must stay faulted after the verb mismatch");
        assert!(
            refused.to_string().contains("the gang has already failed"),
            "unexpected message: {refused}"
        );
    });
}

/// `broadcast`'s constructor: the mutation `verb: "barrier"` is executed and
/// reported below rather than assumed to kill this test.
#[test]
fn a_two_rank_broadcast_and_barrier_verb_mismatch_faults_both_ranks_symmetrically() {
    witness(|call| {
        let gang =
            LocalGang::with_timeout(vec![Device::Cpu; 2], Duration::from_secs(5)).expect("gang");
        let rank0 = gang.rank(0).expect("rank 0");
        let rank1 = gang.rank(1).expect("rank 1");

        let results = std::thread::scope(|scope| {
            let broadcast = BlockingCall::spawn_scoped(scope, move |call| {
                let mut t = matrix(1, 1, 0.0);
                rank0
                    .broadcast(&call, &mut t, 0)
                    .map(|_| String::new())
                    .unwrap_or_else(|e| e.to_string())
            });
            let barrier = BlockingCall::spawn_scoped(scope, move |call| {
                rank1
                    .barrier(&call)
                    .map(|_| String::new())
                    .unwrap_or_else(|e| e.to_string())
            });
            [
                broadcast.join().expect("rank 0 thread"),
                barrier.join().expect("rank 1 thread"),
            ]
        });

        for (rank, message) in results.iter().enumerate() {
            assert!(
                !message.is_empty(),
                "rank {rank} returned Ok from a round rank {} was running a different collective in",
                1 - rank
            );
            assert!(
                message.contains("disagree about what this round computes"),
                "rank {rank}: unexpected message: {message}"
            );
        }

        let after_fault = gang.rank(0).expect("rank 0 handle after the fault");
        let refused = after_fault
            .barrier(&call)
            .expect_err("the gang must stay faulted after the verb mismatch");
        assert!(
            refused.to_string().contains("the gang has already failed"),
            "unexpected message: {refused}"
        );
    });
}
