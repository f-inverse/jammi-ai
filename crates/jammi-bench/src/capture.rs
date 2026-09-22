//! What every leg producer captures the same way, whichever workload it
//! serves: the per-iteration series, the files a leg hands to the next stack
//! or to the comparator, the vector rows the ladder pairs by row, the file a
//! leg is filed as, and one process per leg.
//!
//! A producer emits legs and decides nothing; the leg itself is
//! [`crate::leg::Leg`]. This module holds the capture around it.

use std::ffi::OsString;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::leg::{Leg, MutantStamp, Payload, Provenance};
use crate::report::{Report, Tiers};

/// The per-iteration wall-clock series of a leg: the first `warmup` recorded
/// iterations are dropped, the rest are kept in order, in seconds — the
/// ladder's `iter_wall_s`.
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

/// One file of a leg. The sha256 is of the file's own bytes, so two legs that
/// name the same artifact are provably reading the same input.
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
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
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

/// One `f32` vector per key, on disk as the ladder pairs rows: the rows as
/// little-endian `f32`, row-major, in the order given (`<stem>.vectors.f32`),
/// and the keys one per line beside them (`<stem>.keys.txt`).
#[derive(Debug, Clone, Serialize)]
pub struct VectorRows {
    /// The rows' file.
    pub file: Artifact,
    /// The keys' file, row-aligned.
    pub keys: Artifact,
    /// The width of every row.
    pub dim: usize,
}

/// Persist `rows` under `dir/<stem>`. Every row must have one width.
pub fn write_vector_rows(
    dir: &Path,
    stem: &str,
    rows: &[KeyedVector],
) -> Result<VectorRows, Box<dyn std::error::Error>> {
    let dim = rows.first().map_or(0, |(_, v)| v.len());
    if let Some((key, v)) = rows.iter().find(|(_, v)| v.len() != dim) {
        return Err(format!(
            "vector rows are ragged: '{key}' is {}-wide, the first row is {dim}-wide",
            v.len()
        )
        .into());
    }
    let bytes: Vec<u8> = rows
        .iter()
        .flat_map(|(_, v)| v.iter().flat_map(|x| x.to_le_bytes()))
        .collect();
    let file = write_artifact(dir, &format!("{stem}.vectors.f32"), &bytes)?;
    let keys: String = rows.iter().map(|(key, _)| format!("{key}\n")).collect();
    let keys = write_artifact(dir, &format!("{stem}.keys.txt"), keys.as_bytes())?;
    Ok(VectorRows { file, keys, dim })
}

/// One row of a vectors file: a key with its vector.
pub type KeyedVector = (String, Vec<f32>);

/// Read rows written by [`write_vector_rows`] back, keys and all — what a
/// test holds a filed leg's vectors against; the twins read them in Python.
#[cfg(test)]
pub fn read_vector_rows(
    vectors: &Path,
    dim: usize,
) -> Result<Vec<KeyedVector>, Box<dyn std::error::Error>> {
    let keys_path = vectors
        .to_str()
        .and_then(|p| p.strip_suffix(".vectors.f32"))
        .map(|stem| format!("{stem}.keys.txt"))
        .ok_or_else(|| format!("{} is not a `<stem>.vectors.f32` file", vectors.display()))?;
    let keys = std::fs::read_to_string(&keys_path)?;
    let bytes = std::fs::read(vectors)?;
    let row_bytes = dim * 4;
    if dim == 0 || bytes.len() % row_bytes != 0 {
        return Err(format!(
            "{}: {} bytes is not a whole number of {dim}-dimensional f32 rows",
            vectors.display(),
            bytes.len()
        )
        .into());
    }
    let rows = keys
        .lines()
        .zip(bytes.chunks_exact(row_bytes))
        .map(|(key, row)| {
            (
                key.to_string(),
                row.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    if rows.len() * row_bytes != bytes.len() {
        return Err(format!("{keys_path} does not hold one key per row").into());
    }
    Ok(rows)
}

/// The checksum of "one `f32` vector per key": FNV-1a over the rows in the
/// order given, mixing each key's bytes, a separator, and each lane's raw
/// little-endian `f32` bits. Callers sort by key first when the digest must be
/// independent of scan order. Equal between two runs of one stack on one box;
/// never expected equal across stacks.
pub fn vector_rows_digest(rows: &[KeyedVector]) -> String {
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

/// A suffix no earlier run of this process or another produced: the clock's
/// nanoseconds, in hex — what a name registered in a fleet's catalog, shared
/// across runs, is made unique by.
#[cfg(feature = "plane")]
pub fn unique_suffix() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    format!(
        "{:x}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    )
}

/// A leg's file stem by the ladder's contract: `<rung>__<unit>__r<take>`.
pub fn leg_stem(rung: &str, unit: &str, take: usize) -> String {
    format!("{rung}__{unit}__r{take}")
}

/// Write `report` as `<stem>.json` under `legs_dir` and return the file name.
pub fn file_leg(
    legs_dir: &Path,
    stem: &str,
    report: &Report,
) -> Result<String, Box<dyn std::error::Error>> {
    let name = format!("{stem}.json");
    std::fs::create_dir_all(legs_dir)?;
    std::fs::write(legs_dir.join(&name), serde_json::to_string_pretty(report)?)?;
    Ok(name)
}

/// The report a leg subcommand files: one leg under its tier key, with the
/// engine version, host and build identity every report carries. The leg's
/// identity fields are asserted present on the way.
pub fn leg_report<P: Payload + std::fmt::Debug>(
    subcommand: &'static str,
    leg: Leg<P>,
    place: impl FnOnce(Leg<P>) -> Tiers,
) -> Report {
    leg.to_value();
    Report::new(subcommand, place(leg))
}

/// The provenance of a leg the engine produces on the CPU: this build's
/// features and kernel arms, recorded and never compared.
pub fn cpu_provenance() -> Provenance {
    let kernels_disabled_requested = jammi_kernels::admission::disabled_ops_requested();
    Provenance {
        device_name: "cpu".to_string(),
        build_features: crate::report::build_features()
            .into_iter()
            .map(str::to_string)
            .collect(),
        flash_compiled: jammi_kernels::admission::FLASH_COMPILED,
        kernels_disabled_fired: jammi_kernels::admission::disabled_ops_fired(),
        arm: if kernels_disabled_requested.is_empty() {
            "fused"
        } else {
            "alloff"
        }
        .to_string(),
        // The graph workloads' attention is the eager composition.
        attention_arm: "eager".to_string(),
        kernels_disabled_requested,
        mutant: MutantStamp::default(),
        ran_on: None,
    }
}

/// The legs filed by a run of this binary with `args`, in a fresh process:
/// the child prints the file names it filed as a JSON array; its stderr is
/// inherited so a failure surfaces in the parent's log.
async fn legs_from_fresh_process(
    args: &[OsString],
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let output = tokio::process::Command::new(std::env::current_exe()?)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()
        .await?;
    if !output.status.success() {
        return Err(format!("child {args:?} exited with {}", output.status).into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

/// One leg per point, each owning its process's peak resident set: a single
/// point is filed here by `in_process`; several are a sweep and each runs in
/// a fresh process — `args_for` names the invocation of this binary that
/// files exactly that one point — so no point inherits an earlier point's
/// high-water mark. Returns every leg's file name.
pub async fn legs_per_point<P>(
    points: &[P],
    in_process: impl std::future::Future<Output = Result<Vec<String>, Box<dyn std::error::Error>>>,
    args_for: impl Fn(&P) -> Vec<OsString>,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    if let [_] = points {
        return in_process.await;
    }
    let mut files = Vec::with_capacity(points.len());
    for point in points {
        files.extend(legs_from_fresh_process(&args_for(point)).await?);
    }
    Ok(files)
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
        assert_eq!(std::fs::read_to_string(&b.path).unwrap(), "{\"k\":1}\n");
    }

    #[test]
    fn vector_rows_round_trip_as_little_endian_f32_and_the_digest_sees_keys_and_bits() {
        let rows = vec![
            ("a".to_string(), vec![1.0f32, -2.5]),
            ("b".to_string(), vec![0.0, 3.25]),
        ];
        let dir = tempfile::tempdir().unwrap();
        let written = write_vector_rows(dir.path(), "out", &rows).unwrap();
        assert_eq!(written.dim, 2);
        assert_eq!(
            std::fs::read(&written.file.path).unwrap(),
            [1.0f32, -2.5, 0.0, 3.25]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>()
        );
        assert_eq!(
            std::fs::read_to_string(&written.keys.path).unwrap(),
            "a\nb\n"
        );
        assert_eq!(
            read_vector_rows(Path::new(&written.file.path), 2).unwrap(),
            rows
        );

        let base = vector_rows_digest(&rows);
        let mut renamed = rows.clone();
        renamed[0].0 = "c".into();
        let mut nudged = rows.clone();
        nudged[1].1[1] = f32::from_bits(3.25f32.to_bits() + 1);
        assert_ne!(base, vector_rows_digest(&renamed));
        assert_ne!(base, vector_rows_digest(&nudged));

        let ragged = vec![("a".to_string(), vec![1.0f32]), ("b".to_string(), vec![])];
        assert!(write_vector_rows(dir.path(), "ragged", &ragged).is_err());
    }

    #[test]
    fn a_leg_stem_is_the_ladders_name() {
        assert_eq!(leg_stem("sampler", "edges64", 2), "sampler__edges64__r2");
    }
}
