//! The per-version manifest of a versioned result table
//! (`{table}__v{N}.version.json`) and the version identity (K7).
//!
//! A version is a snapshot: the fragments (immutable Parquet objects) whose
//! union is the table's physical row set, the ANN segments indexing them, the
//! cumulative deletion mask that hides superseded and deleted keys, and the
//! delta descriptor + input anchors that produced it. Every fragment and
//! segment is stamped with its producing version; the mask's horizons are
//! version numbers (see [`crate::store::deletes`]). The manifest carries its
//! own `version_format` with a reject-newer guard, independent of the
//! `.materialization.json` `MANIFEST_VERSION`, whose shape is untouched.
//!
//! Identity: the base version's identity IS the table's `.materialization.json`
//! artifact hex (so publishing the base moves no downstream anchor); every
//! later version folds `parent_identity`, the table's `definition_hash`, the
//! canonical delta descriptor, every fragment digest in manifest order and the
//! deletes digest — length-prefixed and domain-separated exactly like the
//! definition hash. Counts, segments (the ANN index is never attested) and
//! `produced_by` / `produced_at` are outputs, never inputs.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{JammiError, Result};
use crate::store::manifest::{ArtifactDigest, DefinitionHash, InputAnchor, ProducingDescriptor};

/// The `.version.json` format this build writes and the newest it reads.
pub const VERSION_FORMAT: u32 = 1;

/// One data fragment of a version: a Parquet object stamped with the version
/// that wrote it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FragmentRef {
    /// The Parquet object's storage URL.
    pub url: String,
    /// The version that wrote it (the base version for the base Parquet).
    pub version: i64,
    /// Its physical row count (footer).
    pub rows: usize,
    /// SHA-256 over its bytes.
    pub digest: ArtifactDigest,
}

/// One ANN segment of a version, by catalog id, stamped with its producing
/// version (the base version for a `version IS NULL` catalog row).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentRef {
    pub segment_id: i64,
    pub version: i64,
}

/// The cumulative deletion mask of a version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeletesRef {
    /// The `{table}__v{N}.deletes.parquet` URL.
    pub url: String,
    /// The number of `(key, horizon)` entries.
    pub entries: usize,
    /// SHA-256 over its bytes.
    pub digest: ArtifactDigest,
}

/// What produced this version and over which input state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionDelta {
    /// `Embedding` at the base, `EmbeddingDelta` after a refresh,
    /// `EmbeddingCompaction` after a compaction — what `producing_descriptor`
    /// returns and `recompute` replays.
    pub descriptor: ProducingDescriptor,
    pub input_anchors: Vec<InputAnchor>,
}

/// The `{table}__v{N}.version.json` document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionManifest {
    pub version_format: u32,
    pub table: String,
    pub version: i64,
    pub parent: Option<i64>,
    pub definition_hash: DefinitionHash,
    pub delta: VersionDelta,
    pub fragments: Vec<FragmentRef>,
    pub segments: Vec<SegmentRef>,
    pub deletes: Option<DeletesRef>,
    pub live_rows: usize,
    pub masked_rows: usize,
    pub identity: String,
    pub produced_by: String,
    pub produced_at: String,
    pub engine_version: String,
}

impl VersionManifest {
    /// Serialise for the sidecar.
    pub fn to_json_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec_pretty(self).map_err(JammiError::Json)
    }

    /// Parse a sidecar, rejecting a `version_format` newer than this build
    /// reads (a typed [`JammiError::IncompatibleFormat`], the same stance as
    /// every other stamped sidecar).
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        let m: Self = serde_json::from_slice(bytes).map_err(JammiError::Json)?;
        if m.version_format > VERSION_FORMAT {
            return Err(JammiError::IncompatibleFormat {
                artifact: "version-manifest".into(),
                found: m.version_format.to_string(),
                supported: VERSION_FORMAT.to_string(),
            });
        }
        Ok(m)
    }

    /// The identity of a non-base version (see the module doc). Pure.
    pub fn compute_identity(
        parent_identity: &str,
        definition_hash: &DefinitionHash,
        descriptor: &ProducingDescriptor,
        fragments: &[FragmentRef],
        deletes: Option<&DeletesRef>,
    ) -> Result<String> {
        let descriptor_bytes = descriptor
            .canonical_bytes()
            .map_err(crate::store::manifest_to_jammi)?;
        let mut h = Sha256::new();
        h.update(b"jammi.version.identity.v1");
        let mut put = |tag: &[u8], bytes: &[u8]| {
            h.update(tag);
            h.update((bytes.len() as u64).to_le_bytes());
            h.update(bytes);
        };
        put(b"\0parent\0", parent_identity.as_bytes());
        put(b"\0definition\0", definition_hash.as_str().as_bytes());
        put(b"\0delta\0", &descriptor_bytes);
        h.update(b"\0fragments\0");
        for f in fragments {
            h.update((f.digest.as_str().len() as u64).to_le_bytes());
            h.update(f.digest.as_str().as_bytes());
        }
        let deletes_digest = deletes.map(|d| d.digest.as_str()).unwrap_or("");
        h.update(b"\0deletes\0");
        h.update((deletes_digest.len() as u64).to_le_bytes());
        h.update(deletes_digest.as_bytes());
        Ok(hex::encode(h.finalize()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_task::ModelTask;

    fn descriptor() -> ProducingDescriptor {
        ProducingDescriptor::Embedding {
            model_id: "m".into(),
            task: ModelTask::TextEmbedding,
            source_id: "s".into(),
            columns: vec!["body".into()],
            key_column: "id".into(),
            dimensions: 4,
        }
    }

    fn manifest() -> VersionManifest {
        VersionManifest {
            version_format: VERSION_FORMAT,
            table: "t".into(),
            version: 1,
            parent: Some(0),
            definition_hash: DefinitionHash("d".into()),
            delta: VersionDelta {
                descriptor: descriptor(),
                input_anchors: vec![],
            },
            fragments: vec![FragmentRef {
                url: "file:///t__v1.parquet".into(),
                version: 1,
                rows: 3,
                digest: ArtifactDigest("ab".into()),
            }],
            segments: vec![SegmentRef {
                segment_id: 1,
                version: 1,
            }],
            deletes: None,
            live_rows: 3,
            masked_rows: 0,
            identity: "x".into(),
            produced_by: "run".into(),
            produced_at: "now".into(),
            engine_version: "0".into(),
        }
    }

    #[test]
    fn round_trips_and_rejects_newer() {
        let m = manifest();
        let bytes = m.to_json_bytes().unwrap();
        assert_eq!(VersionManifest::from_json_bytes(&bytes).unwrap(), m);
        let mut newer = m.clone();
        newer.version_format = VERSION_FORMAT + 1;
        let err = VersionManifest::from_json_bytes(&newer.to_json_bytes().unwrap()).unwrap_err();
        assert!(matches!(
            err,
            JammiError::IncompatibleFormat { ref artifact, .. } if artifact == "version-manifest"
        ));
    }

    #[test]
    fn identity_folds_every_input_and_no_output() {
        let m = manifest();
        let base = VersionManifest::compute_identity(
            "p",
            &m.definition_hash,
            &m.delta.descriptor,
            &m.fragments,
            None,
        )
        .unwrap();
        // Outputs do not participate.
        let mut same = m.clone();
        same.live_rows = 99;
        same.segments.clear();
        same.produced_at = "later".into();
        assert_eq!(
            base,
            VersionManifest::compute_identity(
                "p",
                &same.definition_hash,
                &same.delta.descriptor,
                &same.fragments,
                None
            )
            .unwrap()
        );
        // Every input does.
        assert_ne!(
            base,
            VersionManifest::compute_identity(
                "q",
                &m.definition_hash,
                &m.delta.descriptor,
                &m.fragments,
                None
            )
            .unwrap()
        );
        assert_ne!(
            base,
            VersionManifest::compute_identity(
                "p",
                &DefinitionHash("e".into()),
                &m.delta.descriptor,
                &m.fragments,
                None
            )
            .unwrap()
        );
        let deletes = DeletesRef {
            url: "u".into(),
            entries: 1,
            digest: ArtifactDigest("cd".into()),
        };
        assert_ne!(
            base,
            VersionManifest::compute_identity(
                "p",
                &m.definition_hash,
                &m.delta.descriptor,
                &m.fragments,
                Some(&deletes)
            )
            .unwrap()
        );
        let mut frags = m.fragments.clone();
        frags[0].digest = ArtifactDigest("ac".into());
        assert_ne!(
            base,
            VersionManifest::compute_identity(
                "p",
                &m.definition_hash,
                &m.delta.descriptor,
                &frags,
                None
            )
            .unwrap()
        );
    }
}
