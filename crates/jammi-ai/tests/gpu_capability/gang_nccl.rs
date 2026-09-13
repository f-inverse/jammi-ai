//! The NCCL arm of the collective, over two real CUDA devices.
//!
//! Not a `ci/scripts/check_gpu_parity_matrix.py` cell: it exercises a
//! collective across two devices, never an (architecture × verb) CPU↔GPU
//! forward parity pair.
//!
//! The pod leg for `fine_tune::collective::nccl`. Everything ABOUT the arm
//! that can be settled on a host is settled on a host — the trait contract,
//! the gather layout, the rank-ordered fold, the refusals — by the hermetic
//! `Local` oracles in `crates/jammi-ai/src/fine_tune/collective/tests.rs`.
//! What only a real gang can answer is whether the communicator, the stream
//! discipline and the pad/narrow gather agree with those contracts on device,
//! which is what this asks.
//!
//! Gated exactly like the rest of the suite: the module compiles under
//! `live-gpu-tests` alone (a `cuda`-less build type-checks it, which is what
//! CI's compile-check lane runs), and a meaningful run also needs `cuda`, two
//! visible CUDA devices, and NCCL. Without them it skips loudly, never
//! `#[ignore]`.

#[cfg(feature = "cuda")]
use crate::harness;
use crate::skip_without_gpu;

/// [`harness::serial_cuda_device`], or a hard failure when `JAMMI_REQUIRE_CUDA`
/// is set and no usable CUDA device opens. Same require-gate idiom as
/// `crates/jammi-ai/src/fine_tune/optimizer.rs::cuda_device` and
/// `crates/jammi-ai/tests/gpu_capability/gguf_quantized_gpu.rs::
/// device_memory_used_bytes_or_require`: on a pod leg this test is meant to
/// run on, a missing device is a hard failure, never a silent skip.
#[cfg(feature = "cuda")]
fn serial_cuda_device_or_require(test: &str) -> Option<harness::SerialGpu> {
    match harness::serial_cuda_device() {
        Some(slot) => Some(slot),
        None => {
            if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
                panic!(
                    "{test}: JAMMI_REQUIRE_CUDA is set but no usable CUDA device could be \
                     acquired — a silent skip is not acceptable here"
                );
            }
            None
        }
    }
}

/// A second CUDA device (`candle_core::Device::new_cuda(1)`), or a hard
/// failure when `JAMMI_REQUIRE_CUDA_GANG` is set and only one CUDA device is
/// visible. A two-rank NCCL gang needs two devices to answer anything: it is
/// the gang pod lane's obligation (`ci/scripts/runpod_gpu_gang.sh`'s remote
/// environment, per plan 67) to export this variable, so the pod's own run
/// hard-fails rather than skipping — that export does not exist yet, so a
/// green run of this suite elsewhere is not NCCL coverage on its own. The
/// single-GPU prove lane never sets it, so a one-device host on that lane
/// still skips with the reason rather than failing.
#[cfg(feature = "cuda")]
fn second_cuda_device_or_require(test: &str) -> Option<candle_core::Device> {
    match candle_core::Device::new_cuda(1) {
        Ok(d) => Some(d),
        Err(e) => {
            if std::env::var_os("JAMMI_REQUIRE_CUDA_GANG").is_some() {
                panic!(
                    "{test}: JAMMI_REQUIRE_CUDA_GANG is set but a second CUDA device could not \
                     be acquired — a two-rank NCCL gang needs two visible devices: {e}"
                );
            }
            None
        }
    }
}

/// Two ranks on two devices: the rank-ordered sum of a known vector, an
/// unequal-count gather (including the zero-row rank a remainder batch
/// produces), and the lockstep control word — each compared to the value the
/// host arms are pinned to, bit-for-bit.
///
/// A single-device host cannot answer the question this asks, so it skips
/// with the reason rather than passing vacuously on a one-rank gang.
#[test]
fn gang_nccl_reduces_a_known_vector_over_two_devices() {
    skip_without_gpu!();

    #[cfg(feature = "cuda")]
    {
        use candle_core::Tensor;
        use jammi_ai::fine_tune::collective::nccl::Nccl;
        use jammi_ai::fine_tune::collective::Collective;

        // The binary's one-at-a-time device slot, held for the whole gang:
        // this test allocates on every visible device, so a sibling leg
        // measuring device memory must not run beside it.
        let Some(slot) =
            serial_cuda_device_or_require("gang_nccl_reduces_a_known_vector_over_two_devices")
        else {
            tracing::warn!("SKIP: no usable CUDA device");
            return;
        };
        let first = slot.device().clone();
        let Some(second) =
            second_cuda_device_or_require("gang_nccl_reduces_a_known_vector_over_two_devices")
        else {
            tracing::warn!("SKIP: a two-rank NCCL gang needs two CUDA devices; this host has one");
            return;
        };

        let ranks = match Nccl::single_process(&[first.clone(), second.clone()]) {
            Ok(ranks) => ranks,
            Err(e) => {
                panic!(
                    "two visible CUDA devices must form a two-rank NCCL gang; \
                     ncclCommInitAll failed: {e}"
                );
            }
        };
        assert_eq!(ranks.len(), 2);

        // One thread per communicator: `ncclCommInitAll` comms driven from a
        // single thread would need an explicit group around every collective,
        // and NCCL's own rule is one thread per communicator.
        let devices = [first, second];
        let handles: Vec<_> = ranks
            .into_iter()
            .zip(devices)
            .map(|(rank, device)| {
                std::thread::spawn(move || {
                    let r = rank.rank();
                    assert_eq!(rank.world(), 2);

                    // (1) The rank-ordered sum of a known vector.
                    let scale = (r + 1) as f32;
                    let mut tensors = vec![Tensor::from_vec(
                        vec![1.0 * scale, 2.0 * scale, 3.0 * scale],
                        3,
                        &device,
                    )
                    .expect("tensor")];
                    rank.all_reduce_sum(&mut tensors).expect("all_reduce_sum");
                    let summed = tensors[0].to_vec1::<f32>().expect("read back");
                    assert_eq!(
                        summed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        vec![3.0f32.to_bits(), 6.0f32.to_bits(), 9.0f32.to_bits()],
                        "rank {r}: the reduced vector must equal the serial sum bit-for-bit"
                    );

                    // (2) An unequal-count gather, including the zero-row rank
                    // a remainder batch produces. The result is the
                    // rank-ordered concatenation of exactly the contributed
                    // rows — the pad-to-max NCCL needs internally is never
                    // visible here.
                    let counts = [2usize, 0];
                    let local = if r == 0 {
                        Tensor::from_vec(vec![10.0f32, 11.0, 12.0, 13.0], (2, 2), &device)
                    } else {
                        Tensor::from_vec(Vec::<f32>::new(), (0, 2), &device)
                    }
                    .expect("tensor");
                    let gathered = rank.all_gather(&local, &counts).expect("all_gather");
                    assert_eq!(gathered.dims(), &[2, 2], "rank {r}: gathered shape");
                    assert_eq!(
                        gathered
                            .flatten_all()
                            .expect("flatten")
                            .to_vec1::<f32>()
                            .expect("read back")
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        [10.0f32, 11.0, 12.0, 13.0]
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        "rank {r}: the gather must equal the serial concatenation bit-for-bit"
                    );

                    // (2.5) PROBE-A1 (U4a fix round 3): `checked_gather_counts`
                    // is the ONE seam every arm shares, and it refuses a
                    // 0-dim scalar before this arm's own pad/gather logic
                    // ever runs — the same predicate the hermetic `Noop` and
                    // `Local` oracles pin in
                    // `fine_tune::collective::tests`.
                    let scalar = Tensor::new(1.0f32, &device).expect("0-dim scalar");
                    rank.all_gather(&scalar, &[0, 0])
                        .expect_err("a 0-dim tensor has no row count to gather along");

                    // (3) The lockstep control word: a flag set on ONE rank is
                    // seen by both.
                    let mine = if r == 1 { 0b100 } else { 0 };
                    assert_eq!(
                        rank.all_reduce_max_flags(mine).expect("flags"),
                        0b100,
                        "rank {r}: a flag set on any rank must be seen by all"
                    );

                    rank.barrier().expect("barrier");
                    assert!(!rank.is_aborted(), "rank {r} must not have aborted");
                    rank
                })
            })
            .collect();

        let ranks: Vec<Nccl> = handles
            .into_iter()
            .map(|h| h.join().expect("a rank thread panicked"))
            .collect();

        // The abort is idempotent and leaves no communicator behind to drop:
        // the second call finds the `Option` empty, which is what makes the
        // double `ncclCommAbort` that segfaults unreachable.
        for rank in &ranks {
            rank.abort();
            rank.abort();
            assert!(rank.is_aborted());
            assert!(
                rank.all_reduce_max_flags(0).is_err(),
                "an aborted communicator must refuse rather than read the garbage buffer \
                 an aborted collective leaves behind"
            );
        }
        drop(ranks);
        drop(slot);
    }
}
