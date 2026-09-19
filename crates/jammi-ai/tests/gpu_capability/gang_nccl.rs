//! The NCCL arm of the collective, over two real CUDA devices — two legs.
//!
//! Not a `ci/scripts/check_gpu_parity_matrix.py` cell: it exercises a
//! collective across two devices, never an (architecture × verb) CPU↔GPU
//! forward parity pair.
//!
//! The pod/cluster legs for `fine_tune::collective::nccl`. Everything ABOUT
//! the arm that can be settled on a host is settled on a host — the trait
//! contract, the gather layout, the rank-ordered fold, the refusals — by the
//! hermetic `Local` oracles in `crates/jammi-ai/src/fine_tune/collective/tests.rs`.
//! What only a real gang can answer is whether the communicator, the stream
//! discipline and the pad/narrow gather agree with those contracts on device,
//! which is what this asks.
//!
//! Gated exactly like the rest of the suite: the module compiles under
//! `live-gpu-tests` alone (a `cuda`-less build type-checks it, which is what
//! CI's compile-check lane runs), and a meaningful run also needs `cuda`, two
//! visible CUDA devices, and NCCL. Without them it skips loudly, never
//! `#[ignore]`.
//!
//! # Two legs, one set of assertions
//!
//! [`gang_nccl_reduces_a_known_vector_over_two_devices`] is the SINGLE-PROCESS
//! leg (`ncclCommInitAll` over two devices in one process — the RunPod POD
//! lane, `ci/scripts/runpod_gpu_gang.sh`). [`gang_nccl_two_hosts_reduce_a_known_vector`]
//! is the TWO-HOST leg (`ncclCommInitRank` on each of two separate processes,
//! one per host — the RunPod CLUSTER lane, `ci/scripts/runpod_gpu_cluster.sh`,
//! U7b-A2b). Until a run of that lane has actually happened, or as a
//! fallback if it is ever unavailable, a maintainer may still drive this leg
//! BY HAND with `ci/scripts/runpod_lib.sh`'s own cluster primitives
//! (`rp_cluster_create`/`rp_cluster_get`/`rp_cluster_pods`/`rp_cluster_delete`):
//! rent a 2×1 cluster, `scp` the 128-byte NCCL id between the two members,
//! and export each rank's `JAMMI_GANG_TWO_HOSTS_*` env below before running
//! this test on each host — see `docs/maintainer/dev-gpu.md`'s cluster-leg
//! section for the exact steps, now the fallback the driver automates.
//! Both call the SAME [`assert_gang_checks`] helper so the two legs prove the
//! identical property (rank-ordered sum, unequal-count gather, lockstep
//! flags, barrier, not-aborted) rather than two hand-maintained copies that
//! could silently drift apart.
//!
//! # The two-host leg's env contract
//!
//! The two-host leg is a no-op with none of this env set: without it it
//! skips loudly (never `#[ignore]`, never a vacuous pass). Whoever runs it —
//! U7b-A2b's driver (`ci/scripts/runpod_gpu_cluster.sh`), or a maintainer by
//! hand as the fallback — sets:
//!
//! - `JAMMI_GANG_TWO_HOSTS_RANK` — this process's rank, `0` or `1`.
//! - `JAMMI_GANG_TWO_HOSTS_WORLD` — must be `2`; any other value is a named
//!   refusal (a panic), never a silent skip or a truncated gang — this leg
//!   proves nothing above world 2 (see "Known-unmeasured" below).
//! - `JAMMI_GANG_TWO_HOSTS_ID_FILE` — the path rank 0 mints the NCCL id to
//!   and rank 1 reads it from. The id is shipped between hosts out of band
//!   (`scp`) BEFORE rank 1's process starts; this test never touches the
//!   network to move it.
//! - `JAMMI_GANG_ARTIFACT_DIR` — where `rank-<r>.json` is written, on both
//!   the pass and the fail arm (same env name the pod leg's driver already
//!   sets — see `ci/scripts/runpod_gpu_gang.sh`).
//! - `JAMMI_REQUIRE_CUDA_TWO_HOSTS` — this leg's OWN require flag, distinct
//!   from the single-process leg's `JAMMI_REQUIRE_CUDA_GANG`: when set, a
//!   missing device OR an incomplete/malformed env is a hard FAIL, never a
//!   skip. Consulted BEFORE the availability check that would otherwise
//!   decide to skip, so a device-less cluster member fails loudly instead of
//!   the run "passing" on one rank while the other silently did nothing.
//!
//! Each rank writes `$JAMMI_GANG_ARTIFACT_DIR/rank-<r>.json`
//! ([`RankReport`]) with its rank, world, hostname, CUDA device ordinal, the
//! `NCCL_SOCKET_IFNAME` it observed (the driver pins `ens1`), a SHA-256 hex
//! digest of the reduced known vector's little-endian `f32` bytes (equal
//! across ranks on `pass`), and a `verdict`/`reason` pair — on both the pass
//! and the fail arm, so a failed run is always evidenced rather than
//! reported only as "the workflow step failed". The report NEVER carries the
//! NCCL id in any form (raw, hex, or base64): the id travels only through
//! `JAMMI_GANG_TWO_HOSTS_ID_FILE`, [`RankReport`] has no field for it, and
//! [`report_tests::rank_report_never_carries_the_id`] pins that property with
//! a non-vacuous negative control. A failed `hostname` read (or an empty
//! one) and an unset/empty `NCCL_SOCKET_IFNAME` are NEVER masked behind a
//! placeholder like `"unknown"` in a `pass` report — either is a `fail`
//! verdict naming which is missing ([`missing_report_metadata_reason`]),
//! because a `pass` report is the one artifact that must let a reviewer
//! tell a genuine two-HOST gang apart from two ranks that collapsed onto
//! one host (two placeholder-hostname reports would be indistinguishable
//! from that).
//!
//! # Known-unmeasured
//!
//! This leg proves world 2 only. Whether the NCCL pin set (`NCCL_SOCKET_IFNAME`
//! and friends) that works at world 2 still suffices at world ≥ 3 — a
//! multi-rail / multi-NIC topology a 2-host gang cannot exercise — is
//! **uncovered** here, and stated as such rather than silently assumed.

// Unconditional (not `#[cfg(feature = "cuda")]`): `harness::serial_cuda_device`
// and `harness::SerialGpu` need no `cuda` feature to name — the device
// acquisition itself always returns `None` when the CUDA backend is not
// compiled in — and `serial_cuda_device_or_require_two_hosts` below relies on
// exactly that to make its own require-gate decision correctly on EITHER
// build.
use crate::harness;
#[cfg(feature = "cuda")]
use jammi_ai::fine_tune::collective::Collective;

/// The two-host leg's four env vars, parsed and validated — see the module
/// doc's "The two-host leg's env contract".
struct TwoHostsEnv {
    rank: u32,
    world: u32,
    id_file: std::path::PathBuf,
    artifact_dir: std::path::PathBuf,
}

/// Read, validate, and hand back the two-host leg's four env vars — or a
/// hard failure under `JAMMI_REQUIRE_CUDA_TWO_HOSTS` when they are
/// incomplete, or `None` (a loud skip) when the flag is not set. Consulted
/// FIRST, before [`serial_cuda_device_or_require_two_hosts`]'s availability
/// check (module doc: "Consulted BEFORE the availability check").
///
/// Registered in `ci/kernel-oracle-helpers.txt` (KO-7): the `_` match arm's
/// `if std::env::var_os("JAMMI_REQUIRE_CUDA_TWO_HOSTS").is_some() { panic!(..)
/// }` is the canonical shape a caller's own `let Some(env) =
/// two_hosts_env_or_require(TEST) else { return; };` skip is gated against.
fn two_hosts_env(test: &str) -> TwoHostsEnv {
    let rank_s = jammi_test_resources::env("JAMMI_GANG_TWO_HOSTS_RANK");
    let world_s = jammi_test_resources::env("JAMMI_GANG_TWO_HOSTS_WORLD");
    let id_file_s = jammi_test_resources::env("JAMMI_GANG_TWO_HOSTS_ID_FILE");
    let artifact_dir_s = jammi_test_resources::env("JAMMI_GANG_ARTIFACT_DIR");

    // `JAMMI_GANG_TWO_HOSTS_WORLD` must be exactly 2: this leg proves world
    // 2 only (module doc, "Known-unmeasured"), so any other value is a named
    // refusal rather than a silently truncated or padded gang.
    let world: u32 = world_s.parse().unwrap_or_else(|_| {
        panic!("{test}: JAMMI_GANG_TWO_HOSTS_WORLD must be an integer, got {world_s:?}")
    });
    if world != 2 {
        panic!(
            "{test}: JAMMI_GANG_TWO_HOSTS_WORLD must be 2 (this leg is world-2 only), got {world}"
        );
    }
    let rank: u32 = rank_s.parse().unwrap_or_else(|_| {
        panic!("{test}: JAMMI_GANG_TWO_HOSTS_RANK must be an integer, got {rank_s:?}")
    });
    if rank >= world {
        panic!("{test}: JAMMI_GANG_TWO_HOSTS_RANK must be 0 or 1 for world 2, got {rank}");
    }

    TwoHostsEnv {
        rank,
        world,
        id_file: std::path::PathBuf::from(id_file_s),
        artifact_dir: std::path::PathBuf::from(artifact_dir_s),
    }
}

/// Run the three checks both legs prove — the rank-ordered sum of a known
/// vector (bit-for-bit vs the serial sum), an unequal-count gather including
/// the zero-row rank a remainder batch produces, and the lockstep control
/// word — plus the barrier and the not-aborted assertion, and hand back the
/// reduced known vector (check 1) for the caller's own report/digest.
///
/// ONE implementation, two bootstraps:
/// [`gang_nccl_reduces_a_known_vector_over_two_devices`] (single-process,
/// `Nccl::single_process`) and [`gang_nccl_two_hosts_reduce_a_known_vector`]
/// (two-host, `Nccl::from_rank`) each hand this their own already-joined
/// [`jammi_ai::fine_tune::collective::nccl::Nccl`] rank and device; the
/// assertions below are byte-for-byte what the single-process leg has always
/// run.
#[cfg(feature = "cuda")]
fn assert_gang_checks(
    call: &jammi_ai::fine_tune::collective::BlockingCall,
    rank: &jammi_ai::fine_tune::collective::nccl::Nccl,
    device: &candle_core::Device,
) -> Vec<f32> {
    use candle_core::Tensor;

    let r = rank.rank();
    assert_eq!(rank.world(), 2);

    // (1) The rank-ordered sum of a known vector.
    let scale = (r + 1) as f32;
    let mut tensors =
        vec![
            Tensor::from_vec(vec![1.0 * scale, 2.0 * scale, 3.0 * scale], 3, device)
                .expect("tensor"),
        ];
    rank.all_reduce_sum(call, &mut tensors)
        .expect("all_reduce_sum");
    let summed = tensors[0].to_vec1::<f32>().expect("read back");
    assert_eq!(
        summed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        vec![3.0f32.to_bits(), 6.0f32.to_bits(), 9.0f32.to_bits()],
        "rank {r}: the reduced vector must equal the serial sum bit-for-bit"
    );

    // (2) An unequal-count gather, including the zero-row rank a remainder
    // batch produces. The result is the rank-ordered concatenation of
    // exactly the contributed rows — the pad-to-max NCCL needs internally is
    // never visible here.
    let counts = [2usize, 0];
    let local = if r == 0 {
        Tensor::from_vec(vec![10.0f32, 11.0, 12.0, 13.0], (2, 2), device)
    } else {
        Tensor::from_vec(Vec::<f32>::new(), (0, 2), device)
    }
    .expect("tensor");
    let gathered = rank.all_gather(call, &local, &counts).expect("all_gather");
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

    // (2.5) `checked_gather_counts` is the ONE seam every arm shares, and it
    // refuses a 0-dim scalar before this arm's own pad/gather logic ever
    // runs — the same predicate the hermetic `Noop` and `Local` oracles pin
    // in `fine_tune::collective::tests`.
    let scalar = Tensor::new(1.0f32, device).expect("0-dim scalar");
    rank.all_gather(call, &scalar, &[0, 0])
        .expect_err("a 0-dim tensor has no row count to gather along");

    // (3) The lockstep control word: a flag set on ONE rank is seen by both.
    let mine = if r == 1 { 0b100 } else { 0 };
    assert_eq!(
        rank.all_reduce_max_flags(call, mine).expect("flags"),
        0b100,
        "rank {r}: a flag set on any rank must be seen by all"
    );

    rank.barrier(call).expect("barrier");
    assert!(!rank.is_aborted(), "rank {r} must not have aborted");

    summed
}

/// Two ranks on two devices: the rank-ordered sum of a known vector, an
/// unequal-count gather (including the zero-row rank a remainder batch
/// produces), and the lockstep control word — each compared to the value the
/// host arms are pinned to, bit-for-bit.
///
/// A single-device host cannot answer the question this asks, so it skips
/// with the reason rather than passing vacuously on a one-rank gang.
#[cfg(feature = "live-gpu-gang-tests")]
#[test]
fn gang_nccl_reduces_a_known_vector_over_two_devices() {
    #[cfg(feature = "cuda")]
    {
        use jammi_ai::fine_tune::collective::nccl::Nccl;

        // The binary's one-at-a-time device slot, held for the whole gang:
        // this test allocates on every visible device, so a sibling leg
        // measuring device memory must not run beside it.
        let slot = harness::serial_cuda_device();
        let first = slot.device().clone();
        let second = jammi_test_resources::cuda_device(1);

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
                jammi_ai::fine_tune::collective::BlockingCall::spawn_thread(move |call| {
                    assert_gang_checks(&call, &rank, &device);
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
            let refused = std::thread::scope(|scope| {
                jammi_ai::fine_tune::collective::BlockingCall::spawn_scoped(scope, |call| {
                    rank.all_reduce_max_flags(&call, 0).is_err()
                })
                .join()
                .expect("witness thread")
            });
            assert!(
                refused,
                "an aborted communicator must refuse rather than read the garbage buffer \
                 an aborted collective leaves behind"
            );
        }
        drop(ranks);
        drop(slot);
    }
}

// ─── The two-host leg's per-rank report ────────────────────────────────────

/// One rank's `$JAMMI_GANG_ARTIFACT_DIR/rank-<r>.json` report for the
/// two-host NCCL leg. Written on BOTH the pass and the fail arm (never only
/// on success), so a fail is always evidenced rather than reported only as
/// "the workflow step failed" with nothing left to inspect.
///
/// **Never carries the NCCL id.** The 128 id bytes are the capability to
/// join the gang and travel ONLY through `$JAMMI_GANG_TWO_HOSTS_ID_FILE`
/// (rank 0 writes it, rank 1 reads it); this struct has no field that could
/// hold it, and [`report_tests::rank_report_never_carries_the_id`] pins that
/// property against a synthetic report, checking the raw bytes, their hex,
/// AND their base64 form are each absent — with a non-vacuous control that
/// first proves the same check WOULD catch a genuine leak.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct RankReport {
    rank: u32,
    world: u32,
    hostname: String,
    /// The CUDA device ordinal this rank ran on — always `0` on this leg
    /// (the device is CUDA ordinal 0 on each host), kept as data rather than
    /// assumed so a future leg with more than one GPU per host has
    /// somewhere to put a different value.
    device_ordinal: i64,
    /// `NCCL_SOCKET_IFNAME` as this rank's process saw it in its own
    /// environment — `None` when the driver did not export it OR exported
    /// it empty (both refuse a `pass` verdict — see
    /// [`missing_report_metadata_reason`]). This test only READS it for the
    /// report; the driver, not this test, exports it (pinned to `ens1`).
    nccl_socket_ifname: Option<String>,
    /// SHA-256 hex digest of the reduced known vector's little-endian `f32`
    /// bytes — `Some` and equal across both ranks on a `pass`; `None` on a
    /// `fail` (a failed run may never have produced a reduced vector).
    reduced_vector_digest_sha256: Option<String>,
    /// Exactly `"pass"` or `"fail"`.
    verdict: String,
    /// Empty on `pass`; the panic/assertion text on `fail`.
    reason: String,
}

/// SHA-256 hex digest of `vector`'s little-endian `f32` bytes — the value
/// [`RankReport::reduced_vector_digest_sha256`] carries, and what a
/// `pass` verdict's equality across both ranks is checked against.
fn reduced_vector_digest_sha256(vector: &[f32]) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    for v in vector {
        hasher.update(v.to_le_bytes());
    }
    hex::encode(hasher.finalize())
}

/// This host's name, shelled out to the `hostname` command — the same
/// shell-out idiom `gguf_quantized_gpu.rs` uses for `nvidia-smi`. `Err`
/// names the problem (the command failed to run, exited non-zero, printed
/// non-UTF-8, or printed an empty name after trimming) rather than masking
/// it: the report's `hostname` field is what a reviewer uses to tell two
/// ranks on genuinely DIFFERENT hosts apart from two ranks collapsed onto
/// ONE — a silent placeholder there would make that indistinguishable
/// (closing-audit finding F4). See [`missing_report_metadata_reason`] for
/// how a bad read here turns into a `fail` verdict rather than a `pass`
/// report carrying a placeholder.
///
/// Only the two-host leg calls this today (the single-process leg writes no
/// report), so it rides the same `cuda` gate as its one call site — a
/// GPU-less build has no caller for it at all.
#[cfg(feature = "cuda")]
fn hostname() -> Result<String, String> {
    let output = std::process::Command::new("hostname")
        .output()
        .map_err(|e| format!("failed to run the `hostname` command: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "the `hostname` command exited with {}",
            output.status
        ));
    }
    let text = String::from_utf8(output.stdout)
        .map_err(|e| format!("the `hostname` command's output was not valid UTF-8: {e}"))?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err("the `hostname` command produced an empty name".to_string());
    }
    Ok(trimmed.to_string())
}

/// Whether this rank's report metadata (the hostname read and
/// `NCCL_SOCKET_IFNAME`) is fit to appear in a `pass` verdict — `Some(reason)`
/// naming WHICH is missing (one or both, `; `-joined) when not, `None` when
/// both are present and non-empty. `NEVER` a placeholder like `"unknown"`:
/// a `pass` report is the one artifact that must let a reviewer tell a
/// genuine two-HOST gang apart from two ranks that collapsed onto one host,
/// and two placeholder-hostname reports would be indistinguishable from
/// that (closing-audit finding F4).
///
/// A pure decision function, deliberately separate from [`hostname`]'s own
/// shell-out (which needs the `cuda` feature to even be reachable, since it
/// rides its one call site's gate): this is what makes the property
/// hermetically testable without a GPU — [`report_tests`] drives all three
/// bad arms (a failed hostname read, an empty-but-`Ok` hostname, and an
/// unset/empty `NCCL_SOCKET_IFNAME`) by constructing the `Result`/`Option`
/// values directly, the same shape [`hostname`]'s real call site hands it.
fn missing_report_metadata_reason(
    hostname_result: &Result<String, String>,
    nccl_socket_ifname: Option<&str>,
) -> Option<String> {
    let mut problems = Vec::new();
    match hostname_result {
        Ok(h) if !h.is_empty() => {}
        Ok(_) => problems.push("hostname: read an empty name".to_string()),
        Err(e) => problems.push(format!("hostname: {e}")),
    }
    match nccl_socket_ifname {
        Some(s) if !s.is_empty() => {}
        _ => problems
            .push("NCCL_SOCKET_IFNAME: unset or empty in this rank's own environment".to_string()),
    }
    if problems.is_empty() {
        None
    } else {
        Some(problems.join("; "))
    }
}

/// Write `report` to `<artifact_dir>/rank-<report.rank>.json`, creating
/// `artifact_dir` if it does not already exist.
fn write_rank_report(artifact_dir: &std::path::Path, report: &RankReport) -> std::io::Result<()> {
    std::fs::create_dir_all(artifact_dir)?;
    let path = artifact_dir.join(format!("rank-{}.json", report.rank));
    let body = serde_json::to_vec_pretty(report)
        .unwrap_or_else(|e| panic!("RankReport must always serialize: {e}"));
    std::fs::write(path, body)
}

/// Write the 128-byte NCCL id to `<id_file>.tmp` at mode `0600`, `fsync`,
/// then `rename` onto `id_file` — F11's atomicity: a reader can only ever
/// observe either no file or the complete 128 bytes, never a partial write.
#[cfg(feature = "cuda")]
fn write_id_file_atomically(
    id_file: &std::path::Path,
    id: jammi_ai::fine_tune::collective::nccl::NcclIdBytes,
) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;

    let tmp_name = format!(
        "{}.tmp",
        id_file
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("nccl.id")
    );
    let tmp = id_file.with_file_name(tmp_name);
    {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp)?;
        f.set_permissions(std::fs::Permissions::from_mode(0o600))?;
        f.write_all(&id)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, id_file)
}

/// Read `id_file` and refuse (typed panic naming the actual size) unless it
/// holds exactly 128 bytes — F11's reader-side half: the driver only ships
/// once `stat` reports exactly 128 bytes, and this is the defensive check on
/// the receiving rank in case that ever stops being true.
#[cfg(feature = "cuda")]
fn read_id_file_exactly_128_bytes(
    test: &str,
    id_file: &std::path::Path,
) -> jammi_ai::fine_tune::collective::nccl::NcclIdBytes {
    let bytes = std::fs::read(id_file).unwrap_or_else(|e| {
        panic!(
            "{test}: failed to read the NCCL id file {}: {e}",
            id_file.display()
        )
    });
    if bytes.len() != 128 {
        panic!(
            "{test}: the NCCL id file {} must be exactly 128 bytes, got {} — refusing to join \
             with a truncated or padded id",
            id_file.display(),
            bytes.len()
        );
    }
    let mut id = [0u8; 128];
    id.copy_from_slice(&bytes);
    id
}

/// The panic payload's message, best-effort: `catch_unwind`'s `Box<dyn Any +
/// Send>` is a `&'static str` for a `panic!("literal")` and a `String` for
/// `panic!("{fmt}", ..)` — the two shapes every panic in this module produces
/// — with a named fallback for anything else so the fail report always has
/// SOME reason rather than silently swallowing the payload.
#[cfg(feature = "cuda")]
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic payload was neither a &str nor a String".to_string()
    }
}

/// Two ranks, two HOSTS: rank 0 mints the NCCL id and ships it out of band
/// (an `scp`'d file; this test never talks to the network directly), both
/// ranks join with `Nccl::from_rank`, and both run
/// [`assert_gang_checks`] — the SAME assertions
/// [`gang_nccl_reduces_a_known_vector_over_two_devices`] runs, over a real
/// cross-host communicator instead of `ncclCommInitAll`'s single-process one.
///
/// See the module doc's "The two-host leg's env contract" for the four env
/// vars this test reads and what an incomplete/malformed one means. Without
/// them this test skips loudly (never `#[ignore]`) — it is a no-op unless
/// something sets that env: U7b-A2b's driver (`ci/scripts/runpod_gpu_cluster.sh`),
/// or a maintainer running it by hand as the fallback (module doc procedure).
#[cfg(feature = "live-gpu-cluster-tests")]
#[test]
fn gang_nccl_two_hosts_reduce_a_known_vector() {
    const TEST: &str = "gang_nccl_two_hosts_reduce_a_known_vector";

    // Consulted FIRST (module doc: "Consulted BEFORE the availability
    // check") — `env` is genuinely read on EITHER build (the `tracing::info!`
    // just below reads every field), so this is never a decorative binding.
    let env = two_hosts_env(TEST);
    tracing::info!(
        test = TEST,
        rank = env.rank,
        world = env.world,
        id_file = %env.id_file.display(),
        artifact_dir = %env.artifact_dir.display(),
        "two-host gang env parsed"
    );

    // `slot` is only consumed inside the `#[cfg(feature = "cuda")]` block
    // below — under a `cuda`-less build nothing past this point references
    // it, hence the targeted `allow` (the binding itself, and the require-
    // gate decision above it, are real on every build).
    let slot = harness::serial_cuda_device();

    #[cfg(feature = "cuda")]
    {
        use jammi_ai::fine_tune::collective::nccl::{Nccl, NcclIdBytes};

        let TwoHostsEnv {
            rank,
            world,
            id_file,
            artifact_dir,
        } = env;
        let device = slot.device().clone();
        // An empty `NCCL_SOCKET_IFNAME` is treated the same as unset — see
        // `missing_report_metadata_reason`.
        let nccl_socket_ifname = std::env::var("NCCL_SOCKET_IFNAME")
            .ok()
            .filter(|s| !s.is_empty());
        let host_result = hostname();
        // Used for BOTH report arms below; only ever a real hostname on the
        // arm that can actually reach `pass` — see the metadata check at
        // the top of the closure, which panics (and so routes through the
        // `Err` arm below, never the `Ok` one) before either bad value could
        // land in a `pass` report.
        let host = host_result.clone().unwrap_or_default();
        let device_ordinal: i64 = 0;

        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Vec<f32> {
            if let Some(reason) =
                missing_report_metadata_reason(&host_result, nccl_socket_ifname.as_deref())
            {
                panic!(
                    "{TEST}: rank {rank}: report metadata missing before the gang ran: {reason}"
                );
            }
            let id: NcclIdBytes = if rank == 0 {
                let id = Nccl::new_id()
                    .unwrap_or_else(|e| panic!("{TEST}: ncclGetUniqueId failed on rank 0: {e}"));
                write_id_file_atomically(&id_file, id).unwrap_or_else(|e| {
                    panic!(
                        "{TEST}: failed to write the NCCL id to {}: {e}",
                        id_file.display()
                    )
                });
                id
            } else {
                read_id_file_exactly_128_bytes(TEST, &id_file)
            };
            // Rank 0 blocks here until rank 1 joins with the same id — that
            // order (mint → write → join) is correct: the file must exist
            // before either side can complete `ncclCommInitRank`.
            let joined = Nccl::from_rank(&device, rank, world, id).unwrap_or_else(|e| {
                panic!("{TEST}: ncclCommInitRank failed for rank {rank} of {world}: {e}")
            });
            let reduced = std::thread::scope(|scope| {
                jammi_ai::fine_tune::collective::BlockingCall::spawn_scoped(scope, |call| {
                    assert_gang_checks(&call, &joined, &device)
                })
                .join()
                .expect("witness thread")
            });
            assert!(!joined.is_aborted(), "rank {rank} must not have aborted");
            reduced
        }));

        match outcome {
            Ok(reduced) => {
                let report = RankReport {
                    rank,
                    world,
                    hostname: host,
                    device_ordinal,
                    nccl_socket_ifname,
                    reduced_vector_digest_sha256: Some(reduced_vector_digest_sha256(&reduced)),
                    verdict: "pass".to_string(),
                    reason: String::new(),
                };
                write_rank_report(&artifact_dir, &report).unwrap_or_else(|e| {
                    panic!(
                        "{TEST}: rank {rank} passed but failed to write its report to {}: {e}",
                        artifact_dir.display()
                    )
                });
            }
            Err(payload) => {
                let reason = panic_message(&payload);
                let report = RankReport {
                    rank,
                    world,
                    hostname: host,
                    device_ordinal,
                    nccl_socket_ifname,
                    reduced_vector_digest_sha256: None,
                    verdict: "fail".to_string(),
                    reason: reason.clone(),
                };
                if let Err(e) = write_rank_report(&artifact_dir, &report) {
                    eprintln!(
                        "{TEST}: rank {rank} ALSO failed to write its fail report to {}: {e} \
                         (original failure: {reason})",
                        artifact_dir.display()
                    );
                }
                std::panic::resume_unwind(payload);
            }
        }
    }
}

// ─── Hermetic report tests (no GPU) ────────────────────────────────────────

#[cfg(test)]
mod report_tests {
    use super::{
        missing_report_metadata_reason, reduced_vector_digest_sha256, write_rank_report, RankReport,
    };

    fn fake_id() -> [u8; 128] {
        let mut id = [0u8; 128];
        for (i, b) in id.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(7).wrapping_add(3);
        }
        id
    }

    /// Both good: no reason. The baseline every bad-arm test below is
    /// contrasted against.
    #[test]
    fn missing_report_metadata_reason_is_none_when_both_are_present() {
        assert_eq!(
            missing_report_metadata_reason(&Ok("host-a".to_string()), Some("ens1")),
            None
        );
    }

    /// `NCCL_SOCKET_IFNAME` unset (`None`) is a named reason, hostname
    /// staying good.
    #[test]
    fn missing_report_metadata_reason_names_an_unset_iface() {
        let reason = missing_report_metadata_reason(&Ok("host-a".to_string()), None)
            .expect("an unset iface must be a reason, not None");
        assert!(
            reason.contains("NCCL_SOCKET_IFNAME"),
            "reason must name the iface: {reason}"
        );
        assert!(
            !reason.contains("hostname"),
            "a good hostname must not appear as a problem: {reason}"
        );
    }

    /// `NCCL_SOCKET_IFNAME` set but empty is treated identically to unset —
    /// the driver's own call site filters `""` to `None` before this
    /// function ever sees it, but the function's own `Some("")` arm is
    /// pinned directly too (defense in depth: nothing upstream of this
    /// function is trusted to have already done the filtering).
    #[test]
    fn missing_report_metadata_reason_names_an_empty_iface() {
        let reason = missing_report_metadata_reason(&Ok("host-a".to_string()), Some(""))
            .expect("an empty iface must be a reason, not None");
        assert!(
            reason.contains("NCCL_SOCKET_IFNAME"),
            "reason must name the iface: {reason}"
        );
    }

    /// A failed hostname read (the `Err` arm [`super::hostname`] takes on a
    /// missing/failing command, a non-zero exit, or non-UTF-8 output) is a
    /// named reason — this is the arm a fake hostname resolver would drive;
    /// since `hostname()` itself shells out and is `cuda`-gated, this
    /// injects the SAME `Result` shape its real call site would hand this
    /// function, covering the hostname arm through the identical decision
    /// code path without needing a GPU or the `cuda` feature.
    #[test]
    fn missing_report_metadata_reason_names_a_failed_hostname_read() {
        let reason = missing_report_metadata_reason(
            &Err("failed to run the `hostname` command: no such file or directory".to_string()),
            Some("ens1"),
        )
        .expect("a failed hostname read must be a reason, not None");
        assert!(
            reason.contains("hostname"),
            "reason must name the hostname problem: {reason}"
        );
        assert!(
            !reason.contains("NCCL_SOCKET_IFNAME"),
            "a good iface must not appear as a problem: {reason}"
        );
    }

    /// An `Ok` but empty hostname (the command ran, exited 0, and printed
    /// nothing) is ALSO a named reason — never treated as a successful read
    /// just because the `Result` was `Ok`.
    #[test]
    fn missing_report_metadata_reason_names_an_empty_ok_hostname() {
        let reason = missing_report_metadata_reason(&Ok(String::new()), Some("ens1"))
            .expect("an empty Ok(\"\") hostname must be a reason, not None");
        assert!(
            reason.contains("hostname"),
            "reason must name the hostname problem: {reason}"
        );
    }

    /// Both bad: the reason names BOTH, not just the first one found —
    /// `";"`-joined, so a reviewer sees the whole picture in one report
    /// rather than fixing one problem only to hit the other on the next run.
    #[test]
    fn missing_report_metadata_reason_names_both_when_both_are_bad() {
        let reason = missing_report_metadata_reason(&Err("boom".to_string()), None)
            .expect("both bad must be a reason, not None");
        assert!(reason.contains("hostname"), "must name hostname: {reason}");
        assert!(
            reason.contains("NCCL_SOCKET_IFNAME"),
            "must name the iface: {reason}"
        );
    }

    /// Minimal standard base64 (with `=` padding). This crate carries no
    /// `base64` dependency (nothing else in it needs one), and the negative
    /// control below needs the id's base64 form to check for — so this is a
    /// small self-contained encoder for that one check, not a general-purpose
    /// one.
    fn base64_standard(bytes: &[u8]) -> String {
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
        for chunk in bytes.chunks(3) {
            let b0 = chunk[0] as u32;
            let b1 = *chunk.get(1).unwrap_or(&0) as u32;
            let b2 = *chunk.get(2).unwrap_or(&0) as u32;
            let n = (b0 << 16) | (b1 << 8) | b2;
            out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
            out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
            out.push(if chunk.len() > 1 {
                ALPHABET[((n >> 6) & 0x3f) as usize] as char
            } else {
                '='
            });
            out.push(if chunk.len() > 2 {
                ALPHABET[(n & 0x3f) as usize] as char
            } else {
                '='
            });
        }
        out
    }

    /// Whether `bytes` contains `id` in ANY of the three encodings the
    /// report must never carry: the raw 128 bytes, lowercase hex, or
    /// standard base64 — each as a literal byte-window search over the raw
    /// buffer, never a `String` round-trip (the id is arbitrary bytes, not
    /// necessarily valid UTF-8, so decoding it lossy would corrupt the very
    /// sequence this is checking for; this mirrors the driver's own scan,
    /// `open(p, 'rb').read()` over raw bytes — see contract §8 F5). Shared
    /// by both tests below so the "it would have caught a real leak" control
    /// and the real report check run the exact same predicate.
    fn contains_id_encoding(bytes: &[u8], id: &[u8; 128]) -> bool {
        let hex_id = hex::encode(id);
        let base64_id = base64_standard(id);
        bytes.windows(id.len()).any(|w| w == id)
            || bytes.windows(hex_id.len()).any(|w| w == hex_id.as_bytes())
            || bytes
                .windows(base64_id.len())
                .any(|w| w == base64_id.as_bytes())
    }

    #[test]
    fn rank_report_round_trips_through_json_with_every_field() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let vector = [3.0f32, 6.0, 9.0];
        let report = RankReport {
            rank: 1,
            world: 2,
            hostname: "member-1".to_string(),
            device_ordinal: 0,
            nccl_socket_ifname: Some("ens1".to_string()),
            reduced_vector_digest_sha256: Some(reduced_vector_digest_sha256(&vector)),
            verdict: "pass".to_string(),
            reason: String::new(),
        };
        write_rank_report(dir.path(), &report).expect("write");
        let path = dir.path().join("rank-1.json");
        let body = std::fs::read_to_string(&path).expect("read back");

        let parsed: RankReport = serde_json::from_str(&body).expect("valid JSON");
        assert_eq!(parsed, report, "round-trip through JSON must be exact");

        // Every documented field is present under its documented name — a
        // silent rename would still round-trip (serde renames both sides
        // together) but would break the driver's parse; check the raw
        // shape, not just the round-trip.
        let value: serde_json::Value = serde_json::from_str(&body).expect("valid JSON");
        for key in [
            "rank",
            "world",
            "hostname",
            "device_ordinal",
            "nccl_socket_ifname",
            "reduced_vector_digest_sha256",
            "verdict",
            "reason",
        ] {
            assert!(
                value.get(key).is_some(),
                "report JSON must have a {key:?} field, got {value}"
            );
        }
    }

    #[test]
    fn rank_report_writes_a_fail_verdict_with_no_digest_and_the_reason() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let report = RankReport {
            rank: 0,
            world: 2,
            hostname: "member-0".to_string(),
            device_ordinal: 0,
            nccl_socket_ifname: None,
            reduced_vector_digest_sha256: None,
            verdict: "fail".to_string(),
            reason: "rank 0: the reduced vector must equal the serial sum bit-for-bit".to_string(),
        };
        write_rank_report(dir.path(), &report).expect("write");
        let parsed: RankReport =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("rank-0.json")).unwrap())
                .expect("valid JSON");
        assert_eq!(parsed.verdict, "fail");
        assert!(parsed.reduced_vector_digest_sha256.is_none());
        assert!(parsed.nccl_socket_ifname.is_none());
        assert!(!parsed.reason.is_empty());
    }

    /// The id is never written anywhere but the id file — checked with a
    /// NON-VACUOUS control: [`contains_id_encoding`] is first proven to
    /// actually catch a genuine leak in each of the three encodings (a
    /// detector that always returns `false` would pass the real check below
    /// vacuously), and only then is the real [`RankReport`] — which has no
    /// field that could hold the id at all — checked clean.
    #[test]
    fn rank_report_never_carries_the_id() {
        let id = fake_id();

        // Non-vacuous control: a hostile byte buffer that DOES embed the id
        // in each encoding must be caught. Built as raw `Vec<u8>` (never a
        // `String`) so the raw-bytes case actually carries the id's real
        // bytes rather than a UTF-8-lossy corruption of them.
        let hex_leak = format!("\"reason\":\"leaked {}\"", hex::encode(id)).into_bytes();
        assert!(
            contains_id_encoding(&hex_leak, &id),
            "the detector must catch a hex-encoded id leak — this control failing means the \
             real check below would be vacuous"
        );
        let base64_leak = format!("\"reason\":\"leaked {}\"", base64_standard(&id)).into_bytes();
        assert!(
            contains_id_encoding(&base64_leak, &id),
            "the detector must catch a base64-encoded id leak — this control failing means the \
             real check below would be vacuous"
        );
        let mut raw_leak = b"\"reason\":\"leaked ".to_vec();
        raw_leak.extend_from_slice(&id);
        raw_leak.extend_from_slice(b"\"");
        assert!(
            contains_id_encoding(&raw_leak, &id),
            "the detector must catch a raw id leak — this control failing means the real check \
             below would be vacuous"
        );

        // The real report: `RankReport` has no field that could hold the
        // id, so nothing here can leak it regardless of the values chosen.
        let vector = [1.5f32, -2.25, 3.0, f32::NAN];
        let report = RankReport {
            rank: 0,
            world: 2,
            hostname: "member-0".to_string(),
            device_ordinal: 0,
            nccl_socket_ifname: Some("ens1".to_string()),
            reduced_vector_digest_sha256: Some(reduced_vector_digest_sha256(&vector)),
            verdict: "pass".to_string(),
            reason: String::new(),
        };
        let body = serde_json::to_string(&report).expect("serialize");
        assert!(
            !contains_id_encoding(body.as_bytes(), &id),
            "RankReport JSON must never contain the NCCL id in any encoding: {body}"
        );
    }
}
