//! What every ladder leg captures the same way, whichever tier produced it.
//!
//! A leg is one run of one implementation stack of a workload: identity (what
//! two legs must agree on to be comparable), provenance (recorded, never
//! compared) and measurements. A producer emits legs and decides nothing, so
//! this module holds capture only — no ratio, no threshold, no verdict:
//!
//! * [`IterationSeries`] — the per-iteration wall-clock series after warm-up.
//!   A leg always carries the series, never only a summary of it.
//! * [`Artifact`] — an outcome file a leg hands to the next stack or to a
//!   comparison: written once, content-addressed by its sha256.
//! * [`write_keyed_vectors`] / [`keyed_vector_digest`] — the one on-disk shape
//!   and the one checksum for "one `f32` vector per key".
//! * [`leg_per_point`] — a size sweep runs each point in its own process,
//!   because peak host memory is the kernel's high-water mark
//!   ([`crate::rss::peak_rss_bytes`]) and that mark never falls: a second point
//!   measured in the same process would inherit the first's peak.
//! * [`LegReport`] — the document a leg-producing subcommand prints.

use std::ffi::OsString;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use candle_core::{Device, Tensor};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::report::{Host, Provenance};

/// One run of one rung of a workload. `identity` is what two legs must agree on
/// to be comparable; `provenance` is recorded and never compared; `measured`
/// holds the series, the memory peaks and the outcome.
#[derive(Debug, Serialize)]
pub struct Leg<I, P, M> {
    /// The workload this leg ran.
    pub workload: &'static str,
    /// The implementation stack that ran it, with the one parameter that
    /// separates it from its neighbouring rung where there is one (`plan@4`).
    pub rung: String,
    /// What a comparable leg must match.
    pub identity: I,
    /// Recorded, never compared.
    pub provenance: P,
    /// What was measured.
    pub measured: M,
}

/// The per-iteration wall-clock series of a leg: the first `warmup` recorded
/// iterations are dropped, the rest are kept in order, in seconds.
#[derive(Debug)]
pub struct IterationSeries {
    warmup: usize,
    iterations: usize,
    recorded: usize,
    seconds: Vec<f64>,
}

impl IterationSeries {
    /// A series that drops `warmup` iterations and keeps the `iterations` after.
    pub fn new(warmup: usize, iterations: usize) -> Self {
        Self {
            warmup,
            iterations,
            recorded: 0,
            seconds: Vec::with_capacity(iterations),
        }
    }

    /// How many iterations the producer must run: warm-up plus measured.
    pub fn total(&self) -> usize {
        self.warmup + self.iterations
    }

    /// Record one iteration's wall-clock, in run order.
    pub fn record(&mut self, elapsed: Duration) {
        if self.recorded >= self.warmup {
            self.seconds.push(elapsed.as_secs_f64());
        }
        self.recorded += 1;
    }

    /// The measured series, seconds per iteration, warm-up excluded.
    pub fn into_seconds(self) -> Vec<f64> {
        self.seconds
    }
}

/// One outcome file of a leg. The sha256 is of the file's own bytes, so two
/// legs that name the same artifact are provably reading the same input.
#[derive(Debug, Clone, Serialize)]
pub struct Artifact {
    /// Where the file was written.
    pub path: String,
    /// sha256 (hex) of the file's bytes.
    pub sha256: String,
    /// The file's size in bytes.
    pub bytes: u64,
}

/// Describe a file already on disk as an [`Artifact`].
pub fn artifact_of(path: &Path) -> Result<Artifact, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    Ok(Artifact {
        path: path.display().to_string(),
        sha256: hex::encode(Sha256::digest(&bytes)),
        bytes: bytes.len() as u64,
    })
}

/// Write `bytes` to `dir/name` (creating `dir`) and describe the result.
pub fn write_artifact(
    dir: &Path,
    name: &str,
    bytes: &[u8],
) -> Result<Artifact, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(name);
    std::fs::write(&path, bytes)?;
    artifact_of(&path)
}

/// Write one JSON object per line to `dir/name`, in iteration order.
pub fn write_jsonl<T: Serialize>(
    dir: &Path,
    name: &str,
    rows: impl IntoIterator<Item = T>,
) -> Result<Artifact, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    for row in rows {
        serde_json::to_writer(&mut out, &row)?;
        out.push(b'\n');
    }
    write_artifact(dir, name, &out)
}

/// The name of the tensor a keyed-vector file holds its rows under.
pub const VECTORS_TENSOR: &str = "vectors";

/// Persist `rows` — one `f32` vector per key, in the order given — as
/// `dir/<stem>.safetensors` (an `[n, d]` tensor named [`VECTORS_TENSOR`]) beside
/// `dir/<stem>.keys.txt` (one key per line, row-aligned). Returns the tensor
/// file's artifact; the keys file sits at the same stem.
pub fn write_keyed_vectors(
    dir: &Path,
    stem: &str,
    rows: &[(String, Vec<f32>)],
) -> Result<Artifact, Box<dyn std::error::Error>> {
    let dim = rows.first().map_or(0, |(_, v)| v.len());
    if let Some((key, v)) = rows.iter().find(|(_, v)| v.len() != dim) {
        return Err(format!(
            "keyed vectors are ragged: '{key}' is {}-wide, the first row is {dim}-wide",
            v.len()
        )
        .into());
    }
    let flat: Vec<f32> = rows.iter().flat_map(|(_, v)| v.iter().copied()).collect();
    let tensor = Tensor::from_vec(flat, (rows.len(), dim), &Device::Cpu)?;
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{stem}.safetensors"));
    candle_core::safetensors::save(
        &std::collections::HashMap::from([(VECTORS_TENSOR.to_string(), tensor)]),
        &path,
    )?;
    let keys: String = rows.iter().map(|(key, _)| format!("{key}\n")).collect();
    std::fs::write(dir.join(format!("{stem}.keys.txt")), keys)?;
    artifact_of(&path)
}

/// The checksum of "one `f32` vector per key": FNV-1a over the rows in the
/// order given, mixing each key's bytes, a separator, and each lane's raw
/// little-endian `f32` bits. Callers sort by key first when the digest must be
/// independent of scan order.
pub fn keyed_vector_digest(rows: &[(String, Vec<f32>)]) -> String {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    let mut mix = |byte: u8| {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    };
    for (key, vector) in rows {
        key.bytes().for_each(&mut mix);
        // A separator byte so `"ab","c"` and `"a","bc"` cannot collide.
        mix(0xff);
        vector
            .iter()
            .flat_map(|lane| lane.to_bits().to_le_bytes())
            .for_each(&mut mix);
    }
    format!("{hash:016x}")
}

/// The document a leg-producing subcommand prints: the legs it ran, under the
/// same engine version, host facts and baked build identity every harness
/// report carries.
#[derive(Debug, Serialize)]
pub struct LegReport<L> {
    /// Workspace version this binary was built from.
    pub engine_version: &'static str,
    /// Host facts that bear on the numbers.
    pub host: Host,
    /// Which subcommand produced the legs.
    pub subcommand: &'static str,
    /// This binary's baked build-time identity.
    pub provenance: Provenance,
    /// The legs, in the order their points were given.
    pub legs: Vec<L>,
}

impl<L: Serialize> LegReport<L> {
    /// A report of `legs` from `subcommand`.
    pub fn new(subcommand: &'static str, legs: Vec<L>) -> Self {
        Self {
            engine_version: env!("CARGO_PKG_VERSION"),
            host: Host::detect(),
            subcommand,
            provenance: Provenance::baked(),
            legs,
        }
    }

    /// Print the report as pretty JSON on stdout.
    pub fn emit(&self) -> Result<(), serde_json::Error> {
        println!("{}", serde_json::to_string_pretty(self)?);
        Ok(())
    }
}

/// Re-run this binary with `args` in a fresh process and return the one leg of
/// the [`LegReport`] it prints. The child's stderr is inherited so a failure
/// surfaces in the parent's log.
async fn leg_from_fresh_process(
    args: &[OsString],
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let output = tokio::process::Command::new(std::env::current_exe()?)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()
        .await?;
    if !output.status.success() {
        return Err(format!("child {args:?} exited with {}", output.status).into());
    }
    let mut report: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    match report.get_mut("legs").and_then(|l| l.as_array_mut()) {
        Some(legs) if legs.len() == 1 => Ok(legs.remove(0)),
        _ => Err(format!("child {args:?} did not print exactly one leg").into()),
    }
}

/// One leg per point, each owning its process's peak resident set. A single
/// point is measured here, by `in_process`; several points are a sweep, and
/// each runs in a fresh process — `args_for` names the invocation of this
/// binary that measures exactly that one point — so no point inherits an
/// earlier point's high-water mark.
pub async fn leg_per_point<P, L: Serialize>(
    points: &[P],
    in_process: impl std::future::Future<Output = Result<L, Box<dyn std::error::Error>>>,
    args_for: impl Fn(&P) -> Vec<OsString>,
) -> Result<Vec<serde_json::Value>, Box<dyn std::error::Error>> {
    if let [_] = points {
        return Ok(vec![serde_json::to_value(in_process.await?)?]);
    }
    let mut legs = Vec::with_capacity(points.len());
    for point in points {
        legs.push(leg_from_fresh_process(&args_for(point)).await?);
    }
    Ok(legs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn series_drops_exactly_the_warmup_and_keeps_order() {
        let mut series = IterationSeries::new(2, 3);
        assert_eq!(series.total(), 5);
        for ms in [900, 800, 3, 1, 2] {
            series.record(Duration::from_millis(ms));
        }
        assert_eq!(series.into_seconds(), vec![0.003, 0.001, 0.002]);
    }

    #[test]
    fn artifact_is_addressed_by_its_own_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let a = write_artifact(dir.path(), "a.txt", b"abc").unwrap();
        assert_eq!(a.bytes, 3);
        assert_eq!(
            a.sha256,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let b = write_jsonl(dir.path(), "b.jsonl", [serde_json::json!({"k": 1})]).unwrap();
        assert_eq!(
            std::fs::read_to_string(&b.path).unwrap(),
            "{\"k\":1}\n",
            "one object per line"
        );
    }

    #[test]
    fn keyed_vectors_round_trip_and_digest_sees_keys_and_bits() {
        let rows = vec![
            ("a".to_string(), vec![1.0f32, -2.5]),
            ("b".to_string(), vec![0.0, 3.25]),
        ];
        let dir = tempfile::tempdir().unwrap();
        let artifact = write_keyed_vectors(dir.path(), "out", &rows).unwrap();
        let loaded = candle_core::safetensors::load(&artifact.path, &Device::Cpu).unwrap();
        assert_eq!(
            loaded[VECTORS_TENSOR].to_vec2::<f32>().unwrap(),
            vec![vec![1.0, -2.5], vec![0.0, 3.25]]
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("out.keys.txt")).unwrap(),
            "a\nb\n"
        );

        let base = keyed_vector_digest(&rows);
        let mut renamed = rows.clone();
        renamed[0].0 = "c".into();
        let mut nudged = rows.clone();
        nudged[1].1[1] = f32::from_bits(3.25f32.to_bits() + 1);
        assert_ne!(base, keyed_vector_digest(&renamed));
        assert_ne!(base, keyed_vector_digest(&nudged));

        let ragged = vec![("a".to_string(), vec![1.0f32]), ("b".to_string(), vec![])];
        assert!(write_keyed_vectors(dir.path(), "ragged", &ragged).is_err());
    }
}
