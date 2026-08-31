//! Safetensors header-parsed residency estimation (issue #431): mirrors
//! [`super::gguf::estimate_gguf_residency`]'s resolve-time, header-only,
//! conservative shape for the OTHER weight-storage format the Candle backend
//! loads.
//!
//! # The bug this closes
//!
//! `ModelResolver`'s three resolve paths (`resolver.rs`) and
//! `CandleBackend::estimate_memory` (`candle.rs`) used to cost a safetensors
//! checkpoint at the plain on-disk file-byte sum (`std::fs::metadata`), which
//! is the on-disk STORED size — but `CandleBackend::load` always materializes
//! every weight at the resolve-time `compute_dtype`
//! (`VarBuilder::from_mmaped_safetensors(&vb_weights_paths, compute_dtype,
//! &device)`, `candle.rs`), whose default is `F32`
//! (`jammi_db::config::GpuConfig`'s manual `impl Default`), REGARDLESS of the
//! dtype the checkpoint was saved at. An F16-on-disk checkpoint served under
//! that default is therefore resident at roughly 2x its file-byte sum: the
//! file-byte estimate under-reports true residency in exactly the direction
//! that over-admits a load `cache.rs`'s admission check should have refused,
//! turning a typed refusal into an OOM.
//!
//! # The fix
//!
//! [`estimate_safetensors_residency`] parses ONLY the safetensors HEADER (the
//! leading 8-byte little-endian length prefix plus the JSON object it names —
//! see [`read_safetensors_header`] — never any tensor DATA) and costs every
//! tensor at `elem_count * max(on_disk_dtype_bytes, `[`widest_compute_precision_byte_size`]`())`
//! — the SAME widest-width clamp rationale
//! [`super::gguf::estimate_gguf_residency`] already applies to every
//! densified GGUF tensor: `F32` is the widest byte width any
//! [`ComputePrecision`](jammi_numerics::ComputePrecision) variant can take
//! today, so costing at that width stays conservative (`>=` true residency)
//! under every reachable EFFECTIVE precision this workspace can select at
//! load time — including a fine-tuned adapter's own persisted
//! `backbone_dtype`, which can outlive/override the resolve-time
//! `compute_precision` this estimator has no visibility into (the same
//! adapter-backbone-dtype window `estimate_gguf_residency`'s own doc names).
//! Widening a tensor already stored WIDER than `F32` (e.g. a hypothetical
//! `F64` checkpoint) keeps its own on-disk width rather than narrowing it,
//! since candle never narrows a load past what a tensor was stored at. Each
//! file's own header-framing bytes (the 8-byte length prefix plus the JSON
//! header body itself) are folded into the total too — a small, fixed
//! per-file addend that keeps this estimate `>=` the PRE-FIX plain
//! file-byte sum even for an on-disk `F32` checkpoint, where no tensor
//! actually gets widened by the dtype clamp above.
//!
//! A header that cannot be parsed — a truncated length prefix, a length
//! prefix past the end of the file, non-UTF8/invalid-JSON header bytes, a
//! tensor entry missing `dtype`/`shape`, or a `dtype` string this workspace
//! does not recognize — is a typed resolver refusal, never a silent
//! fallback to the raw file-byte sum: every safetensors file candle's own
//! `VarBuilder::from_mmaped_safetensors` can actually load carries this same
//! header in this same format (the safetensors wire format IS the header),
//! so a header this parser cannot read is a header candle cannot load
//! either — a refusal here can never reject a checkpoint that would
//! otherwise have loaded fine.

use std::io::Read;
use std::path::{Path, PathBuf};

use jammi_db::error::{JammiError, Result};

use super::gguf::widest_compute_precision_byte_size;

fn refusal(model_id: &str, message: String) -> JammiError {
    JammiError::Model {
        model_id: model_id.to_string(),
        message,
    }
}

/// Byte width of one element of a safetensors on-disk `dtype` string, per
/// the safetensors spec's fixed dtype enum. `None` for a `dtype` string this
/// workspace does not recognize — the caller turns that into a typed
/// refusal (module doc) rather than guessing a width.
fn safetensors_dtype_byte_size(dtype: &str) -> Option<usize> {
    match dtype {
        "BOOL" | "U8" | "I8" | "F8_E5M2" | "F8_E4M3" => Some(1),
        "I16" | "U16" | "F16" | "BF16" => Some(2),
        "I32" | "U32" | "F32" => Some(4),
        "I64" | "U64" | "F64" => Some(8),
        _ => None,
    }
}

/// Parse ONLY `path`'s safetensors header: the leading 8-byte little-endian
/// `u64` header length, then that many bytes of UTF-8 JSON — the object
/// mapping each tensor name to its `dtype`/`shape`/`data_offsets`, plus an
/// optional `__metadata__` string map safetensors reserves for non-tensor
/// metadata (module doc: never tensor DATA, which starts immediately after
/// the header and this function never reads). Returns the header's own
/// on-disk framing size (`8 + header_len` — the length prefix plus the JSON
/// body [`estimate_safetensors_residency`] folds into its total so the
/// figure stays `>=` the file's own byte size even for an on-disk `F32`
/// checkpoint, where no tensor gets WIDENED) alongside the parsed object.
fn read_safetensors_header(
    path: &Path,
    model_id: &str,
) -> Result<(u128, serde_json::Map<String, serde_json::Value>)> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| refusal(model_id, format!("failed to open {path:?}: {e}")))?;
    let file_len = file
        .metadata()
        .map_err(|e| refusal(model_id, format!("failed to stat {path:?}: {e}")))?
        .len();

    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes).map_err(|e| {
        refusal(
            model_id,
            format!("failed to read safetensors header length prefix of {path:?}: {e}"),
        )
    })?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > file_len.saturating_sub(8) {
        return Err(refusal(
            model_id,
            format!(
                "safetensors header length {header_len} in {path:?} exceeds the file's own \
                 size ({file_len} bytes) — malformed header"
            ),
        ));
    }
    let header_len = usize::try_from(header_len).map_err(|_| {
        refusal(
            model_id,
            format!("safetensors header length {header_len} of {path:?} does not fit in memory"),
        )
    })?;

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes).map_err(|e| {
        refusal(
            model_id,
            format!("failed to read safetensors header body of {path:?}: {e}"),
        )
    })?;

    let header_frame_bytes = 8u128 + header_len as u128;
    let header_map = serde_json::from_slice::<serde_json::Value>(&header_bytes)
        .map_err(|e| {
            refusal(
                model_id,
                format!("failed to parse safetensors header JSON of {path:?}: {e}"),
            )
        })?
        .as_object()
        .cloned()
        .ok_or_else(|| {
            refusal(
                model_id,
                format!("safetensors header of {path:?} is not a JSON object"),
            )
        })?;
    Ok((header_frame_bytes, header_map))
}

/// RESOLVE-TIME residency estimation for a safetensors checkpoint (issue
/// #431): sums, across every file in `paths` (a sharded checkpoint carries
/// more than one), that file's own header-framing byte size (see
/// [`read_safetensors_header`]'s doc — keeps this `>=` the plain file-byte
/// sum even for an on-disk `F32` checkpoint, where no tensor is widened)
/// plus, for every tensor entry in that file's header, `elem_count` times
/// `max(on_disk_dtype_bytes, `[`widest_compute_precision_byte_size`]`())` —
/// see the module doc for the full rationale. Never reads tensor data (only
/// [`read_safetensors_header`]'s 8-byte length prefix plus the JSON header
/// it names).
pub(crate) fn estimate_safetensors_residency(paths: &[PathBuf], model_id: &str) -> Result<usize> {
    let target_dtype_bytes = widest_compute_precision_byte_size() as u128;
    let mut total: u128 = 0;

    for path in paths {
        let (header_frame_bytes, header) = read_safetensors_header(path, model_id)?;
        total = total.saturating_add(header_frame_bytes);
        for (name, entry) in &header {
            // `__metadata__` is safetensors' reserved non-tensor slot (an
            // arbitrary string->string map) — never a tensor entry, so it
            // carries no `dtype`/`shape` and must be skipped rather than
            // treated as a malformed tensor.
            if name == "__metadata__" {
                continue;
            }
            let obj = entry.as_object().ok_or_else(|| {
                refusal(
                    model_id,
                    format!(
                        "safetensors tensor '{name}' in {path:?} has a non-object header entry"
                    ),
                )
            })?;
            let dtype = obj.get("dtype").and_then(|v| v.as_str()).ok_or_else(|| {
                refusal(
                    model_id,
                    format!(
                        "safetensors tensor '{name}' in {path:?} is missing a string 'dtype' field"
                    ),
                )
            })?;
            let dtype_bytes = safetensors_dtype_byte_size(dtype).ok_or_else(|| {
                refusal(
                    model_id,
                    format!(
                        "safetensors tensor '{name}' in {path:?} has unsupported dtype '{dtype}'"
                    ),
                )
            })? as u128;
            let shape = obj.get("shape").and_then(|v| v.as_array()).ok_or_else(|| {
                refusal(
                    model_id,
                    format!(
                        "safetensors tensor '{name}' in {path:?} is missing a 'shape' array field"
                    ),
                )
            })?;
            let mut elem_count: u128 = 1;
            for dim in shape {
                let d = dim.as_u64().ok_or_else(|| {
                    refusal(
                        model_id,
                        format!(
                            "safetensors tensor '{name}' in {path:?} has a non-integer shape \
                             dimension"
                        ),
                    )
                })?;
                elem_count = elem_count.saturating_mul(d as u128);
            }
            let cost_bytes = dtype_bytes.max(target_dtype_bytes);
            total = total.saturating_add(elem_count.saturating_mul(cost_bytes));
        }
    }

    Ok(total as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a minimal safetensors file: an 8-byte LE header length prefix,
    /// the JSON header bytes, then `body_len` zero data bytes (this
    /// estimator never reads them, but a realistic file has SOME data
    /// section). Returns the written path alongside the header's own
    /// on-disk framing size (`8 + header_bytes.len()`) — the SAME per-file
    /// addend [`read_safetensors_header`]/[`estimate_safetensors_residency`]
    /// fold into their total, so a caller computing an EXACT expected
    /// estimate never hand-derives it a second, independently-drifting way.
    fn write_fixture(
        dir: &std::path::Path,
        filename: &str,
        header: &serde_json::Value,
        body_len: usize,
    ) -> (PathBuf, usize) {
        let header_bytes = serde_json::to_vec(header).unwrap();
        let header_frame_bytes = 8 + header_bytes.len();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(&header_bytes);
        buf.extend(std::iter::repeat_n(0u8, body_len));
        let path = dir.join(filename);
        std::fs::write(&path, &buf).unwrap();
        (path, header_frame_bytes)
    }

    fn tensor_entry(dtype: &str, shape: &[usize], start: usize, end: usize) -> serde_json::Value {
        serde_json::json!({
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, end],
        })
    }

    #[test]
    fn f16_on_disk_estimate_is_at_least_its_f32_resident_bytes() {
        let dir = tempfile::tempdir().unwrap();
        // 1000 F16 elements: 2000 stored bytes, but resident at F32 width
        // (4 bytes/elem) once loaded through the F32-default compute dtype
        // — 4000 bytes. The pre-fix file-byte sum would have reported
        // (roughly) the on-disk size, under-reporting true residency by
        // ~2x — the exact #431 acceptance.
        let header = serde_json::json!({
            "weight": tensor_entry("F16", &[10, 100], 0, 2000),
        });
        let (path, header_frame_bytes) =
            write_fixture(dir.path(), "model.safetensors", &header, 2000);

        let estimate =
            estimate_safetensors_residency(std::slice::from_ref(&path), "f16-model").unwrap();
        let file_bytes = std::fs::metadata(&path).unwrap().len() as usize;
        let f32_resident_bytes = header_frame_bytes + 1000 * 4;

        assert!(
            estimate >= f32_resident_bytes,
            "estimate {estimate} must be >= the true F32-resident size {f32_resident_bytes}"
        );
        // The old file-byte sum (`header_frame_bytes + 2000`) fails this
        // exact bound — sanity-check the fixture actually exercises the
        // under-estimate the fix closes, not merely a fixture too small to
        // matter.
        assert!(
            file_bytes < f32_resident_bytes,
            "fixture must actually exercise the ~2x under-estimate (file_bytes={file_bytes}, \
             f32_resident_bytes={f32_resident_bytes})"
        );
        assert_eq!(estimate, f32_resident_bytes);
    }

    #[test]
    fn f32_on_disk_estimate_is_unchanged_in_direction_from_the_file_byte_sum() {
        let dir = tempfile::tempdir().unwrap();
        // F32 on disk == F32 resident: on-disk bytes and the new estimate
        // agree exactly (both width-4), preserving the existing >=
        // file-bytes direction for the format the estimator was already
        // correct for.
        let header = serde_json::json!({
            "weight": tensor_entry("F32", &[10, 100], 0, 4000),
        });
        let (path, header_frame_bytes) =
            write_fixture(dir.path(), "model.safetensors", &header, 4000);

        let estimate =
            estimate_safetensors_residency(std::slice::from_ref(&path), "f32-model").unwrap();
        let file_bytes = std::fs::metadata(&path).unwrap().len() as usize;

        assert!(estimate >= file_bytes, "estimate must be >= raw file bytes");
        assert_eq!(estimate, header_frame_bytes + 1000 * 4);
    }

    #[test]
    fn bf16_on_disk_estimate_is_at_least_its_f32_resident_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let header = serde_json::json!({
            "weight": tensor_entry("BF16", &[4, 8], 0, 64),
        });
        let (path, header_frame_bytes) =
            write_fixture(dir.path(), "model.safetensors", &header, 64);

        let estimate = estimate_safetensors_residency(&[path], "bf16-model").unwrap();
        assert_eq!(estimate, header_frame_bytes + 32 * 4);
    }

    #[test]
    fn metadata_entry_is_skipped_not_treated_as_a_malformed_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let header = serde_json::json!({
            "__metadata__": {"format": "pt"},
            "weight": tensor_entry("F32", &[2, 2], 0, 16),
        });
        let (path, header_frame_bytes) =
            write_fixture(dir.path(), "model.safetensors", &header, 16);

        let estimate = estimate_safetensors_residency(&[path], "meta-model").unwrap();
        assert_eq!(estimate, header_frame_bytes + 4 * 4);
    }

    #[test]
    fn sharded_checkpoint_sums_every_shard_header() {
        let dir = tempfile::tempdir().unwrap();
        let header_a = serde_json::json!({ "a": tensor_entry("F16", &[10], 0, 20) });
        let header_b = serde_json::json!({ "b": tensor_entry("F16", &[10], 0, 20) });
        let (path_a, frame_a) = write_fixture(
            dir.path(),
            "model-00001-of-00002.safetensors",
            &header_a,
            20,
        );
        let (path_b, frame_b) = write_fixture(
            dir.path(),
            "model-00002-of-00002.safetensors",
            &header_b,
            20,
        );

        let estimate = estimate_safetensors_residency(&[path_a, path_b], "sharded-model").unwrap();
        // 10 elems each, F32-costed (widest): 10*4 + 10*4 = 80, plus BOTH
        // shards' own header framing.
        assert_eq!(estimate, frame_a + frame_b + 80);
    }

    #[test]
    fn unsupported_dtype_is_a_typed_refusal_not_a_silent_guess() {
        let dir = tempfile::tempdir().unwrap();
        let header = serde_json::json!({
            "weight": tensor_entry("NOT_A_REAL_DTYPE", &[2, 2], 0, 16),
        });
        let (path, _) = write_fixture(dir.path(), "model.safetensors", &header, 16);

        let err = estimate_safetensors_residency(&[path], "bad-dtype-model").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("unsupported dtype"),
            "expected an unsupported-dtype refusal, got: {msg}"
        );
    }

    #[test]
    fn header_length_past_end_of_file_is_a_typed_refusal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truncated.safetensors");
        // Claims a header of u64::MAX bytes, then provides none.
        std::fs::write(&path, u64::MAX.to_le_bytes()).unwrap();

        let err = estimate_safetensors_residency(&[path], "truncated-model").unwrap_err();
        assert!(
            err.to_string().contains("exceeds the file's own size"),
            "expected a header-length-past-EOF refusal, got: {err}"
        );
    }

    #[test]
    fn missing_shape_field_is_a_typed_refusal() {
        let dir = tempfile::tempdir().unwrap();
        let header = serde_json::json!({
            "weight": {"dtype": "F32", "data_offsets": [0, 16]},
        });
        let (path, _) = write_fixture(dir.path(), "model.safetensors", &header, 16);

        let err = estimate_safetensors_residency(&[path], "no-shape-model").unwrap_err();
        assert!(
            err.to_string().contains("missing a 'shape' array field"),
            "expected a missing-shape refusal, got: {err}"
        );
    }
}
