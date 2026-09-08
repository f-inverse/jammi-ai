//! esc-090: a `Utf8View` path column must take the SAME arm `Utf8` takes in
//! `arrow_to_images`/`arrow_to_audio` — same whole-call-Err-on-bad-path
//! contract, same per-row null handling. `Utf8View`/`BinaryView` are Arrow's
//! "view" string/binary layouts; DataFusion's parquet reader (Arrow 57 under
//! DataFusion 52, this workspace's pinned versions) returns `Utf8View` for a
//! plain Parquet `Utf8` column by default, so a source registered with an
//! ordinary string path column hits this arm on the real, unmodified
//! end-to-end path — not just in a hand-built unit test.
//!
//! `get_string_value` (`inference::mod::get_string_value`, used by
//! `arrow_to_texts`) already handles `Utf8View`; `arrow_to_images` and
//! `arrow_to_audio` did not, so a `Utf8View` path column refused the WHOLE
//! call with "Unsupported column type" — even though every row's path was
//! perfectly valid.

use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryArray, Int64Array, StringArray, StringViewArray};
use jammi_ai::inference::{arrow_to_audio, arrow_to_images, arrow_to_texts};
use jammi_ai::session::InferenceSession;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};

use crate::common;

fn image_path(name: &str) -> PathBuf {
    common::cookbook_fixture("tiny_image_corpus").join(name)
}

fn audio_path(name: &str) -> PathBuf {
    common::cookbook_fixture("tiny_audio_corpus").join(name)
}

// =============================================================================
// arrow_to_images: Utf8 vs Utf8View parity
// =============================================================================

/// A `Utf8View` path column decodes to the SAME images, in the SAME order,
/// with the SAME null handling, as the equivalent `Utf8` column.
#[test]
fn arrow_to_images_utf8view_matches_utf8_on_valid_paths() {
    let p0 = image_path("img_circle_0.png");
    let p1 = image_path("img_hexagon_0.png");

    let utf8_col: ArrayRef = Arc::new(StringArray::from(vec![
        Some(p0.to_str().unwrap()),
        None,
        Some(p1.to_str().unwrap()),
    ]));
    let utf8view_col: ArrayRef = Arc::new(StringViewArray::from(vec![
        Some(p0.to_str().unwrap()),
        None,
        Some(p1.to_str().unwrap()),
    ]));

    let from_utf8 = arrow_to_images(&[utf8_col]).expect("Utf8 path column decodes");
    let from_utf8view = arrow_to_images(&[utf8view_col]).expect(
        "a Utf8View path column must decode exactly like the equivalent Utf8 column, \
         never refuse the whole call",
    );

    assert_eq!(from_utf8.len(), from_utf8view.len());
    for (i, (a, b)) in from_utf8.iter().zip(&from_utf8view).enumerate() {
        match (a, b) {
            (None, None) => {}
            (Some(Ok(a)), Some(Ok(b))) => {
                assert_eq!(
                    a.to_rgba8().into_raw(),
                    b.to_rgba8().into_raw(),
                    "row {i}: Utf8 and Utf8View must decode to the identical image"
                );
            }
            (Some(Err(a)), _) => panic!("row {i}: Utf8 decode failed unexpectedly: {a}"),
            (_, Some(Err(b))) => panic!("row {i}: Utf8View decode failed unexpectedly: {b}"),
            _ => panic!("row {i}: null-handling differs between Utf8 ({a:?}) and Utf8View ({b:?})"),
        }
    }
}

/// The bad-path control the spec pins: a `Utf8View` column with one invalid
/// path must fail the WHOLE call, exactly like `Utf8` does — never degrade to
/// a per-row error/None while `Utf8` stays whole-call.
#[test]
fn arrow_to_images_utf8view_bad_path_fails_whole_call_like_utf8() {
    let bad = "/nonexistent/path/does/not/exist.png";

    let utf8_col: ArrayRef = Arc::new(StringArray::from(vec![Some(bad)]));
    let utf8view_col: ArrayRef = Arc::new(StringViewArray::from(vec![Some(bad)]));

    let utf8_err =
        arrow_to_images(&[utf8_col]).expect_err("an invalid Utf8 path must fail the whole call");
    let utf8view_err = arrow_to_images(&[utf8view_col]).expect_err(
        "an invalid Utf8View path must fail the whole call — the SAME contract Utf8 has, \
         never a per-row swallow",
    );

    // Both are the SAME whole-call `Err` shape: a single `JammiError`, not a
    // `Vec` with a per-row hole. Comparing message content (not just
    // "is_err") pins that both arms hit the SAME `image::open` failure path.
    assert!(
        utf8_err.to_string().contains("Failed to read image file"),
        "unexpected Utf8 error shape: {utf8_err}"
    );
    assert!(
        utf8view_err
            .to_string()
            .contains("Failed to read image file"),
        "unexpected Utf8View error shape: {utf8view_err}"
    );
}

// =============================================================================
// arrow_to_audio: Utf8 vs Utf8View parity
// =============================================================================

/// A `Utf8View` path column decodes to the SAME audio clips as the
/// equivalent `Utf8` column, including null handling.
#[test]
fn arrow_to_audio_utf8view_matches_utf8_on_valid_paths() {
    let p0 = audio_path("clip_harmonic_0.wav");
    let p1 = audio_path("clip_noise_0.wav");

    let utf8_col: ArrayRef = Arc::new(StringArray::from(vec![
        Some(p0.to_str().unwrap()),
        None,
        Some(p1.to_str().unwrap()),
    ]));
    let utf8view_col: ArrayRef = Arc::new(StringViewArray::from(vec![
        Some(p0.to_str().unwrap()),
        None,
        Some(p1.to_str().unwrap()),
    ]));

    let from_utf8 = arrow_to_audio(&[utf8_col]).expect("Utf8 path column decodes");
    let from_utf8view = arrow_to_audio(&[utf8view_col]).expect(
        "a Utf8View path column must decode exactly like the equivalent Utf8 column, \
         never refuse the whole call",
    );

    assert_eq!(from_utf8.len(), from_utf8view.len());
    for (i, (a, b)) in from_utf8.iter().zip(&from_utf8view).enumerate() {
        match (a, b) {
            (None, None) => {}
            (Some(Ok(a)), Some(Ok(b))) => {
                assert_eq!(
                    a.samples, b.samples,
                    "row {i}: Utf8 and Utf8View must decode to identical PCM samples"
                );
                assert_eq!(
                    a.sample_rate, b.sample_rate,
                    "row {i}: Utf8 and Utf8View must decode to the identical sample rate"
                );
            }
            (Some(Err(a)), _) => panic!("row {i}: Utf8 decode failed unexpectedly: {a}"),
            (_, Some(Err(b))) => panic!("row {i}: Utf8View decode failed unexpectedly: {b}"),
            (a, b) => {
                let a_is_some = a.is_some();
                let b_is_some = b.is_some();
                panic!(
                    "row {i}: null-handling differs between Utf8 (is_some={a_is_some}) and \
                     Utf8View (is_some={b_is_some})"
                );
            }
        }
    }
}

/// The bad-path control for audio: a `Utf8View` column with one invalid path
/// must fail the whole call, exactly like `Utf8` does.
#[test]
fn arrow_to_audio_utf8view_bad_path_fails_whole_call_like_utf8() {
    let bad = "/nonexistent/path/does/not/exist.wav";

    let utf8_col: ArrayRef = Arc::new(StringArray::from(vec![Some(bad)]));
    let utf8view_col: ArrayRef = Arc::new(StringViewArray::from(vec![Some(bad)]));

    let utf8_err = match arrow_to_audio(&[utf8_col]) {
        Ok(_) => panic!("an invalid Utf8 path must fail the whole call"),
        Err(e) => e,
    };
    let utf8view_err = match arrow_to_audio(&[utf8view_col]) {
        Ok(_) => panic!(
            "an invalid Utf8View path must fail the whole call — the SAME contract Utf8 \
             has, never a per-row swallow"
        ),
        Err(e) => e,
    };

    assert!(
        utf8_err.to_string().contains("Failed to read audio file"),
        "unexpected Utf8 error shape: {utf8_err}"
    );
    assert!(
        utf8view_err
            .to_string()
            .contains("Failed to read audio file"),
        "unexpected Utf8View error shape: {utf8view_err}"
    );
}

// =============================================================================
// End-to-end: a Parquet source's Utf8 path column really does scan back as
// Utf8View through DataFusion — the real trigger, not just a hand-built
// array.
// =============================================================================

/// Writes a Parquet file with one plain `Utf8` column of image file paths.
fn write_image_path_parquet(dir: &std::path::Path) -> PathBuf {
    use arrow::array::RecordBatch;
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;

    let paths = [
        image_path("img_circle_0.png"),
        image_path("img_hexagon_0.png"),
    ];
    let path_strs: Vec<&str> = paths.iter().map(|p| p.to_str().unwrap()).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("image_path", DataType::Utf8, false),
    ]));
    let ids = Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef;
    let image_paths = Arc::new(StringArray::from(path_strs)) as ArrayRef;
    let batch = RecordBatch::try_new(schema.clone(), vec![ids, image_paths]).unwrap();

    let out = dir.join("image_paths.parquet");
    let mut writer =
        ArrowWriter::try_new(std::fs::File::create(&out).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    out
}

fn tiny_open_clip_model() -> String {
    "local:".to_string() + common::fixture("tiny_open_clip").to_str().unwrap()
}

/// The end-to-end trigger the spec calls for: a Parquet source with a plain
/// `Utf8` path column, scanned through DataFusion (which — under this
/// workspace's pinned Arrow/DataFusion versions — surfaces the column as
/// `Utf8View`, not `Utf8`) and fed straight into image-embedding generation.
/// RED at the pre-fix `arrow_to_images` (whole call refused with
/// "Unsupported column type: Utf8View"); GREEN once the `Utf8View` arm
/// mirrors `Utf8`.
#[tokio::test(flavor = "multi_thread")]
async fn parquet_utf8_path_column_scans_as_utf8view_and_embeds() {
    let dir = tempfile::TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    let parquet_path = write_image_path_parquet(dir.path());
    session
        .add_source(
            "image_paths",
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", parquet_path.display())),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Confirm the REAL trigger: the scanned batch's `image_path` column is
    // `Utf8View`, not `Utf8` — otherwise this test would not exercise the
    // arm at all.
    let batches = session
        .sql("SELECT image_path FROM image_paths.public.image_paths")
        .await
        .unwrap();
    let col = batches[0].column_by_name("image_path").unwrap();
    assert_eq!(
        col.data_type(),
        &arrow::datatypes::DataType::Utf8View,
        "expected the Parquet scan to surface Utf8View for a plain Utf8 column under this \
         workspace's pinned Arrow/DataFusion versions — if this fails, the trigger this test \
         exists to exercise no longer reproduces and the test needs a new trigger"
    );

    let (table, _outcome) = session
        .generate_image_embeddings(
            "image_paths",
            &tiny_open_clip_model(),
            "image_path",
            "id",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .expect(
            "a Utf8View image-path column (the real DataFusion scan output for a plain Utf8 \
             Parquet column) must embed successfully, exactly like Utf8 does",
        );

    let vectors = session.read_vectors(&table).await.unwrap();
    assert_eq!(
        vectors.len(),
        2,
        "both rows should have produced an embedding"
    );
    for v in &vectors {
        assert!(!v.is_empty(), "embedding vector should not be empty");
    }
}

/// Writes a Parquet file with one plain `Utf8` column of audio file paths —
/// the audio peer of [`write_image_path_parquet`].
fn write_audio_path_parquet(dir: &std::path::Path) -> PathBuf {
    use arrow::array::RecordBatch;
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;

    let paths = [
        audio_path("clip_harmonic_0.wav"),
        audio_path("clip_noise_0.wav"),
    ];
    let path_strs: Vec<&str> = paths.iter().map(|p| p.to_str().unwrap()).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("audio_path", DataType::Utf8, false),
    ]));
    let ids = Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef;
    let audio_paths = Arc::new(StringArray::from(path_strs)) as ArrayRef;
    let batch = RecordBatch::try_new(schema.clone(), vec![ids, audio_paths]).unwrap();

    let out = dir.join("audio_paths.parquet");
    let mut writer =
        ArrowWriter::try_new(std::fs::File::create(&out).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    out
}

fn tiny_htsat_clap_model() -> String {
    "local:".to_string()
        + common::cookbook_fixture("htsat_clap_tiny")
            .to_str()
            .unwrap()
}

/// The audio peer of [`parquet_utf8_path_column_scans_as_utf8view_and_embeds`]:
/// a Parquet source with a plain `Utf8` column of audio file paths, scanned
/// through DataFusion (which — under this workspace's pinned Arrow/DataFusion
/// versions — surfaces the column as `Utf8View`, not `Utf8`) and fed straight
/// into audio-embedding generation.
#[tokio::test(flavor = "multi_thread")]
async fn parquet_utf8_audio_path_column_scans_as_utf8view_and_embeds() {
    let dir = tempfile::TempDir::new().unwrap();
    let session = Arc::new(
        InferenceSession::new(common::test_config(dir.path()))
            .await
            .unwrap(),
    );
    let parquet_path = write_audio_path_parquet(dir.path());
    session
        .add_source(
            "audio_paths",
            SourceType::File,
            SourceConnection {
                url: Some(format!("file://{}", parquet_path.display())),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Confirm the REAL trigger: the scanned batch's `audio_path` column is
    // `Utf8View`, not `Utf8` — otherwise this test would not exercise the arm
    // at all.
    let batches = session
        .sql("SELECT audio_path FROM audio_paths.public.audio_paths")
        .await
        .unwrap();
    let col = batches[0].column_by_name("audio_path").unwrap();
    assert_eq!(
        col.data_type(),
        &arrow::datatypes::DataType::Utf8View,
        "expected the Parquet scan to surface Utf8View for a plain Utf8 column under this \
         workspace's pinned Arrow/DataFusion versions — if this fails, the trigger this test \
         exists to exercise no longer reproduces and the test needs a new trigger"
    );

    let (table, _outcome) = session
        .generate_audio_embeddings(
            "audio_paths",
            &tiny_htsat_clap_model(),
            "audio_path",
            "id",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .expect(
            "a Utf8View audio-path column (the real DataFusion scan output for a plain Utf8 \
             Parquet column) must embed successfully, exactly like Utf8 does",
        );

    let vectors = session.read_vectors(&table).await.unwrap();
    assert_eq!(
        vectors.len(),
        2,
        "both rows should have produced an embedding"
    );
    for v in &vectors {
        assert!(!v.is_empty(), "embedding vector should not be empty");
    }
}

// =============================================================================
// F4 (review pass on esc-090/esc-091): `arrow_to_texts` must apply the SAME
// column-type policy `fine_tune::worker::extract_string_column` already
// applies on the training path — binary families refused outright, other
// non-string types cast with a refusal on any introduced null, nulls keep
// the documented "" reading. Pre-fix, `get_string_value`'s `_ => None` arm
// let ANY non-string-like column (including raw image/audio bytes) silently
// read as "" for every row, with no error — the server would embed empty
// strings. RED without the fix.
// =============================================================================

/// A `Binary` column under a text-embedding call must refuse the WHOLE call
/// with a typed error naming the column's data type — never silently embed
/// the bytes as an empty string for every row.
#[test]
fn arrow_to_texts_binary_column_refuses_naming_the_type() {
    let col: ArrayRef = Arc::new(BinaryArray::from(vec![
        Some(b"\x89PNG\r\n\x1a\n".as_slice()),
        Some(b"RIFF....WAVEfmt ".as_slice()),
    ]));

    let err = arrow_to_texts(&[col]).expect_err(
        "a Binary column holds raw bytes, not text — this must be a typed refusal, never a \
         silent per-row empty string",
    );
    let message = err.to_string();
    assert!(
        message.contains("Binary"),
        "refusal must name the column's data type, got: {message}"
    );
}

/// An `Int64` column is not string-like, but its digits ARE honestly
/// readable as text via `arrow::compute::cast` — this must succeed and
/// match a `Utf8` column carrying the same digits as strings, never refuse
/// and never silently read as "".
#[test]
fn arrow_to_texts_int64_column_casts_and_matches_utf8_digits() {
    let int_col: ArrayRef = Arc::new(Int64Array::from(vec![Some(42), None, Some(-7)]));
    let str_col: ArrayRef = Arc::new(StringArray::from(vec![Some("42"), None, Some("-7")]));

    let from_int =
        arrow_to_texts(&[int_col]).expect("an Int64 column must cast to text, never refuse");
    let from_str = arrow_to_texts(&[str_col]).expect("Utf8 column must embed as usual");

    assert_eq!(
        from_int, from_str,
        "an Int64 column's cast-to-text reading must match the equivalent Utf8 column of the \
         same digits, row for row"
    );
    // Null rows keep the documented "" reading on both sides.
    assert_eq!(from_int[1], "", "a null row must read as the empty string");
}
