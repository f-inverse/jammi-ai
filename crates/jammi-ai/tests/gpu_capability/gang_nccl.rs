//! The NCCL transport's device primitive on two real CUDA devices.
//!
//! Not a `ci/scripts/check_gpu_parity_matrix.py` cell: it exercises the
//! collective's device transport across two devices, never an (architecture
//! × verb) CPU↔GPU forward parity pair.
//!
//! Everything ABOUT the transport that can be settled on a host is settled
//! there — the packing, agreement before any byte moves, the deadline, the
//! fault — by the hermetic oracles over an in-memory exchange
//! (`fine_tune::collective::transport_tests`, `peer_tests`). What only real
//! devices can answer is whether NCCL's own primitive agrees with those
//! contracts:
//!
//! - an in-process gang over the NCCL transport ends every verb, at every
//!   dtype, with the INLINE transport's bytes on the same two devices (the
//!   one-fold claim, measured on devices);
//! - `ncclCommAbort` from another thread ends a real NCCL wait against a
//!   rank that never joins the collective;
//! - the multi-process join is bounded: two ranks joining the same id form a
//!   communicator that both release together at the healthy end
//!   (`ncclCommFinalize` then `ncclCommDestroy`), and a join no peer
//!   completes ends at its deadline rather than parking.
//!
//! The training-level proof — a real fine-tune through the worker's
//! topologies, one host and many — is the product path's, not this file's
//! (`jammi-server`'s `gpu::` topology tests and the GPU topology lane).
//!
//! Compiled under `live-gpu-gang-tests`: two CUDA devices on one host, plus
//! NCCL. A host without the second device panics naming it rather than
//! passing vacuously on a one-rank gang.

#[cfg(feature = "cuda")]
use crate::harness;

/// A result as its shape, dtype and raw little-endian element bytes.
#[cfg(feature = "cuda")]
type Bytes = (Vec<usize>, candle_core::DType, Vec<u8>);

/// Every verb at every dtype on one rank of a two-rank gang — an unequal
/// gather with a zero-row rank, a three-dtype sum, a broadcast from rank 1,
/// the control word and a barrier — as the exact bytes each result holds.
#[cfg(feature = "cuda")]
fn every_verb(
    local: jammi_ai::fine_tune::collective::Local,
    call: jammi_ai::fine_tune::collective::BlockingCall,
) -> Vec<Bytes> {
    use candle_core::{DType, Tensor};
    use jammi_ai::fine_tune::collective::Collective;

    let device = local.device().clone();
    let rank = local.rank();
    // A zero-row slice is a view of a one-row tensor, cut LAST: CUDA refuses
    // a zero-length copy and a zero-element kernel launch alike.
    let ramp = |rows: usize, cols: usize, base: f32, dtype: DType| {
        let data: Vec<f32> = (0..rows.max(1) * cols)
            .map(|i| base + i as f32 * 0.37)
            .collect();
        Tensor::from_vec(data, (rows.max(1), cols), &device)
            .and_then(|t| t.to_dtype(dtype))
            .and_then(|t| t.narrow(0, 0, rows))
            .expect("tensor")
    };
    let mut out = Vec::new();

    let counts = [3usize, 0];
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let mine = ramp(counts[rank as usize], 4, rank as f32 * 10.0 + 1.0, dtype);
        let gathered = local.all_gather(&call, &mine, &counts).expect("all_gather");
        out.push(bytes(&gathered));
    }

    let mut tensors = vec![
        ramp(2, 3, rank as f32 + 0.5, DType::F32),
        ramp(4, 1, rank as f32 * 3.0 + 0.1, DType::F16),
        ramp(3, 2, rank as f32 - 1.25, DType::BF16),
    ];
    local
        .all_reduce_sum(&call, &mut tensors)
        .expect("all_reduce_sum");
    out.extend(tensors.iter().map(bytes));

    let mut announced = ramp(2, 2, 100.0 * (rank as f32 + 1.0), DType::F32);
    local
        .broadcast(&call, &mut announced, 1)
        .expect("broadcast");
    out.push(bytes(&announced));

    let flags = local.all_reduce_max_flags(&call, 1 << rank).expect("flags");
    out.push((vec![], DType::U32, flags.to_le_bytes().to_vec()));
    local.barrier(&call).expect("barrier");
    out
}

/// A tensor's shape, dtype and raw little-endian element bytes — equality is
/// bit-for-bit.
#[cfg(feature = "cuda")]
fn bytes(t: &candle_core::Tensor) -> Bytes {
    use candle_core::DType;
    let flat = t
        .flatten_all()
        .and_then(|f| f.to_device(&candle_core::Device::Cpu))
        .expect("flatten to host");
    let raw: Vec<u8> = match t.dtype() {
        DType::F32 => flat
            .to_vec1::<f32>()
            .expect("f32")
            .into_iter()
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect(),
        DType::F16 => flat
            .to_vec1::<half::f16>()
            .expect("f16")
            .into_iter()
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect(),
        DType::BF16 => flat
            .to_vec1::<half::bf16>()
            .expect("bf16")
            .into_iter()
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect(),
        other => panic!("no byte reader for {other:?}"),
    };
    (t.dims().to_vec(), t.dtype(), raw)
}

/// Run `every_verb` on a two-rank in-process gang over `devices` and the
/// transports `kind` builds for them; per-rank results in rank order.
#[cfg(feature = "cuda")]
fn run_every_verb(
    kind: jammi_ai::fine_tune::collective::transport::TransportKind,
    devices: &[candle_core::Device],
) -> Vec<Vec<Bytes>> {
    use jammi_ai::fine_tune::collective::{BlockingCall, LocalGang};
    use std::time::Duration;

    let transports = kind
        .local_transports(devices, Duration::from_secs(60))
        .expect("transports");
    let gang = LocalGang::with_transports(devices.to_vec(), transports, Duration::from_secs(60))
        .expect("gang");
    let handles: Vec<_> = (0..devices.len() as u32)
        .map(|rank| {
            let local = gang.rank(rank).expect("rank");
            BlockingCall::spawn_thread(move |call| every_verb(local, call))
        })
        .collect();
    handles
        .into_iter()
        .map(|h| h.join().expect("a rank thread panicked"))
        .collect()
}

/// The one-fold claim on real devices: the NCCL transport ends every verb, at
/// every dtype, with the inline transport's bytes, on both ranks.
#[cfg(feature = "live-gpu-gang-tests")]
#[test]
fn an_nccl_gang_ends_every_verb_with_the_inline_gangs_bytes_on_two_devices() {
    #[cfg(feature = "cuda")]
    {
        use jammi_ai::fine_tune::collective::transport::TransportKind;

        // The binary's one-at-a-time device slot, held for the whole gang:
        // this test allocates on every visible device.
        let slot = harness::serial_cuda_device();
        let devices = [slot.device().clone(), jammi_test_resources::cuda_device(1)];
        let inline = run_every_verb(TransportKind::Inline, &devices);
        let nccl = run_every_verb(TransportKind::Nccl, &devices);
        assert_eq!(
            inline, nccl,
            "the NCCL transport must end every rank with the inline transport's bytes"
        );
        assert_eq!(nccl[0], nccl[1], "both ranks hold the same result");
        // Not degenerate: the gather is the three contributed rows, not a
        // slice or the padding.
        assert_eq!(nccl[0][0].0, vec![3, 4]);
        drop(slot);
    }
}

/// `ncclCommAbort` from another thread ends a real NCCL wait: rank 0 gathers
/// while rank 1 never joins, and the abort — what the transport's deadline
/// watchdog and a control plane's fault both call — ends rank 0's call with
/// a typed refusal rather than a hang or a garbage buffer.
#[cfg(feature = "live-gpu-gang-tests")]
#[test]
fn an_abort_ends_a_real_nccl_wait_on_a_rank_that_never_joins() {
    #[cfg(feature = "cuda")]
    {
        use std::time::{Duration, Instant};

        use candle_core::Tensor;
        use jammi_ai::fine_tune::collective::transport::{Transport, TransportKind};
        use jammi_ai::fine_tune::collective::BlockingCall;

        let slot = harness::serial_cuda_device();
        let devices = [slot.device().clone(), jammi_test_resources::cuda_device(1)];
        let mut transports = TransportKind::Nccl
            .local_transports(&devices, Duration::from_secs(60))
            .expect("transports")
            .into_iter();
        let Some(Transport::Device(rank0)) = transports.next() else {
            panic!("an NCCL gang's transports are device exchanges");
        };
        // Rank 1's exchange is held, never called.
        let _rank1 = transports.next().expect("rank 1");

        let aborter = {
            let rank0 = std::sync::Arc::clone(&rank0);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_secs(3));
                rank0.abort();
            })
        };
        let started = Instant::now();
        let buf = Tensor::ones(16, candle_core::DType::F32, &devices[0]).expect("buffer");
        let outcome = BlockingCall::spawn_thread(move |call| rank0.all_gather(&call, &buf))
            .join()
            .expect("rank thread");
        aborter.join().expect("aborter");
        let error = outcome.expect_err("a gather no peer joins cannot complete");
        assert!(
            error.to_string().contains("aborted"),
            "the refusal names the abort: {error}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "the abort must end the wait promptly: {:?}",
            started.elapsed()
        );
        drop(slot);
    }
}

/// The multi-process join (`ncclCommInitRankConfig`, non-blocking, polled)
/// forms a communicator when both ranks join the same id — driven here from
/// two threads of one process, which NCCL permits — that both ranks then
/// close together; a join no peer completes ends at its deadline instead of
/// parking.
#[cfg(feature = "live-gpu-gang-tests")]
#[test]
fn a_bounded_join_forms_a_communicator_both_ranks_close_and_a_lone_join_ends_at_its_deadline() {
    #[cfg(feature = "cuda")]
    {
        use std::time::{Duration, Instant};

        use candle_core::{DType, Tensor};
        use jammi_ai::fine_tune::collective::transport::{nccl, Transport};
        use jammi_ai::fine_tune::collective::BlockingCall;

        let slot = harness::serial_cuda_device();
        let devices = [slot.device().clone(), jammi_test_resources::cuda_device(1)];

        // Two ranks, one id: the join completes on both, and the joined
        // communicators gather.
        let id = nccl::mint_id().expect("id");
        let joins: Vec<_> = devices
            .iter()
            .enumerate()
            .map(|(rank, device)| {
                let (device, id) = (device.clone(), id.clone());
                std::thread::spawn(move || {
                    let joined = nccl::join(&device, rank as u32, 2, &id, Duration::from_secs(60))
                        .expect("both ranks join the same id");
                    let Transport::Device(exchange) = joined else {
                        panic!("an NCCL join is a device exchange");
                    };
                    let buf = Tensor::from_vec(vec![rank as f32; 3], 3, &device)
                        .and_then(|t| t.to_dtype(DType::F32))
                        .expect("buffer");
                    let gathered = BlockingCall::spawn_thread({
                        let exchange = std::sync::Arc::clone(&exchange);
                        move |call| exchange.all_gather(&call, &buf)
                    })
                    .join()
                    .expect("rank thread")
                    .expect("gather")
                    .to_device(&candle_core::Device::Cpu)
                    .and_then(|t| t.to_vec1::<f32>())
                    .expect("read back");
                    // The healthy end: both ranks release together.
                    exchange
                        .close(Duration::from_secs(60))
                        .expect("both ranks close the communicator together");
                    gathered
                })
            })
            .collect();
        for join in joins {
            assert_eq!(
                join.join().expect("join thread"),
                vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                "the rank-ordered concatenation of both ranks' buffers"
            );
        }

        // One rank of two, alone: bounded by its deadline.
        let deadline = Duration::from_secs(5);
        let started = Instant::now();
        let lone = nccl::join(&devices[0], 0, 2, &nccl::mint_id().expect("id"), deadline);
        let error = match lone {
            Ok(_) => panic!("a join no peer completes cannot form a communicator"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("did not complete joining"),
            "the refusal names the deadline: {error}"
        );
        assert!(
            started.elapsed() < deadline * 6,
            "the join ends near its deadline, not parked: {:?}",
            started.elapsed()
        );
        drop(slot);
    }
}
