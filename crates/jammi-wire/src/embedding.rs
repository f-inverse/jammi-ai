//! `EmbeddingService` proto↔domain conversions.
//!
//! Maps the embedding compute wire enums/messages onto the engine's
//! [`Modality`], [`QueryInput`], and the result-table record. Modality and
//! input are validated at decode: an unspecified modality and a
//! text/bytes-vs-modality mismatch are rejected with `invalid_argument`.
//!
//! The source-registration and model-introspection conversions
//! (`SourceType` / `SourceConnection` / `FileFormat` / `SourceDescriptor` /
//! `Model`) live with the control-plane catalog wire surface
//! ([`super::catalog`]); only the compute verbs' shapes are here.

use jammi_db::catalog::result_repo::{ResultTableKind, ResultTableRecord};
use tonic::Status;

use crate::proto::embedding as pb;
use crate::request::{Modality, QueryInput};

/// Map the proto [`Modality`] onto the engine's [`Modality`]. An unspecified
/// modality is rejected — a request that names no tower is a client error, not
/// a silent default.
impl TryFrom<pb::Modality> for Modality {
    type Error = Status;

    fn try_from(modality: pb::Modality) -> Result<Self, Self::Error> {
        match modality {
            pb::Modality::Text => Ok(Modality::Text),
            pb::Modality::Image => Ok(Modality::Image),
            pb::Modality::Audio => Ok(Modality::Audio),
            pb::Modality::Unspecified => {
                Err(Status::invalid_argument("modality must be specified"))
            }
        }
    }
}

/// Decode the raw enum discriminant a request carries. An out-of-range value is
/// rejected with the same message an `UNSPECIFIED` modality is — the request
/// names no valid tower either way.
impl TryFrom<i32> for Modality {
    type Error = Status;

    fn try_from(modality: i32) -> Result<Self, Self::Error> {
        match pb::Modality::try_from(modality) {
            Ok(m) => Modality::try_from(m),
            Err(_) => Err(Status::invalid_argument("modality must be specified")),
        }
    }
}

/// The proto query oneof paired with its resolved [`Modality`]. The oneof alone
/// does not say which tower it feeds, so decode takes both: TEXT requires
/// `text`, IMAGE/AUDIO require `data` (raw bytes); a missing oneof or a mismatch
/// is a client error.
pub struct ProtoQueryInput {
    pub input: Option<pb::encode_query_request::Input>,
    pub modality: Modality,
}

impl TryFrom<ProtoQueryInput> for QueryInput {
    type Error = Status;

    fn try_from(value: ProtoQueryInput) -> Result<Self, Self::Error> {
        use pb::encode_query_request::Input as ProtoInput;
        let input = value
            .input
            .ok_or_else(|| Status::invalid_argument("input (text or data) is required"))?;
        match (value.modality, input) {
            (Modality::Text, ProtoInput::Text(text)) => {
                if text.is_empty() {
                    return Err(Status::invalid_argument("text is required"));
                }
                Ok(QueryInput::Text(text))
            }
            (Modality::Image | Modality::Audio, ProtoInput::Data(data)) => {
                if data.is_empty() {
                    return Err(Status::invalid_argument("data is required"));
                }
                Ok(QueryInput::Bytes(data))
            }
            (Modality::Text, ProtoInput::Data(_)) => Err(Status::invalid_argument(
                "TEXT modality requires text input, got data",
            )),
            (Modality::Image | Modality::Audio, ProtoInput::Text(_)) => Err(
                Status::invalid_argument("IMAGE/AUDIO modality requires data input, got text"),
            ),
        }
    }
}

/// Map the engine's [`ResultTableKind`] onto the wire enum — the inverse of
/// [`result_table_kind_from_proto`]. Total: every engine kind maps to a
/// concrete wire variant (the engine type has no unspecified state). Mirrors
/// [`super::model_task_to_proto`].
fn result_table_kind_to_proto(kind: ResultTableKind) -> pb::ResultTableKind {
    match kind {
        ResultTableKind::Model => pb::ResultTableKind::Model,
        ResultTableKind::NeighborGraph => pb::ResultTableKind::NeighborGraph,
        ResultTableKind::AsofJoin => pb::ResultTableKind::AsofJoin,
        ResultTableKind::TrainingSet => pb::ResultTableKind::TrainingSet,
    }
}

/// Map the wire [`pb::ResultTableKind`] discriminant onto the engine's
/// [`ResultTableKind`]. An unspecified/unknown kind is rejected — a
/// `ResultTable` that names no kind is a malformed message, not a silent
/// `Model` guess. Mirrors [`super::model_task_from_proto`].
fn result_table_kind_from_proto(kind: i32) -> Result<ResultTableKind, Status> {
    match pb::ResultTableKind::try_from(kind) {
        Ok(pb::ResultTableKind::Model) => Ok(ResultTableKind::Model),
        Ok(pb::ResultTableKind::NeighborGraph) => Ok(ResultTableKind::NeighborGraph),
        Ok(pb::ResultTableKind::AsofJoin) => Ok(ResultTableKind::AsofJoin),
        Ok(pb::ResultTableKind::TrainingSet) => Ok(ResultTableKind::TrainingSet),
        Ok(pb::ResultTableKind::Unspecified) | Err(_) => Err(Status::invalid_argument(
            "result table kind must be specified",
        )),
    }
}

/// Encode the engine's result-table record into the wire `ResultTable`. The
/// engine's optional `dimensions` is flattened to `0` for a non-embedding /
/// unset result, `row_count` widens to the wire's `u64`, and `task` rides the
/// shared [`super::model_task_to_proto`] task vocabulary. `kind` and
/// `derived_from` ride the same faithful mapping so a `DescribeSource`
/// projection tells a neighbor-graph or as-of-join row apart from a model
/// output rather than fabricating `MODEL` for every row.
///
/// The wire `ResultTable` carries its own `task` (the embedding tower), so the
/// reconstruction recovers it faithfully from the message itself — never from a
/// modality threaded in out of band, never a guess.
impl From<ResultTableRecord> for pb::ResultTable {
    fn from(record: ResultTableRecord) -> Self {
        let dimensions = record.dimensions_raw().unwrap_or(0);
        pb::ResultTable {
            table_name: record.table_name,
            source_id: record.source_id,
            model_id: record.model_id,
            dimensions,
            row_count: record.row_count as u64,
            status: record.status,
            task: super::model_task_to_proto(record.task) as i32,
            // A bare record carries no producer cache outcome (a catalog
            // projection, not a producer return) → unset, the honest "no
            // producer ran" value. A producer handler uses
            // [`result_table_with_outcome`] to carry the real outcome.
            cache_outcome: None,
            key_column: record.key_column.unwrap_or_default(),
            kind: result_table_kind_to_proto(record.kind) as i32,
            derived_from: record.derived_from,
        }
    }
}

/// Encode a producer's result-table record **with** the cache outcome it
/// returned, so reuse is observable on the wire — the shape a producer RPC
/// handler builds, distinct from the bare [`From`] projection (which has no
/// producer to attribute an outcome to). `outcome` is the wire enum value from
/// the engine's `CacheOutcome` ([`crate::cache_outcome_to_proto`]).
pub fn result_table_with_outcome(
    record: ResultTableRecord,
    outcome: crate::proto::inference::CacheOutcome,
) -> pb::ResultTable {
    pb::ResultTable {
        cache_outcome: Some(outcome),
        ..pb::ResultTable::from(record)
    }
}

/// Reconstruct the engine's result-table record from the wire `ResultTable` a
/// `GenerateEmbeddings` or `DescribeSource` response carries.
///
/// The wire message is the client-observable projection: it carries the fields
/// a client needs to locate and query the persisted embedding table
/// (`table_name`, `source_id`, `model_id`, `dimensions`, `row_count`, `status`,
/// `task`, `key_column`, `kind`, `derived_from`). The engine's server-internal
/// bookkeeping — storage/index paths, timestamps, text columns — is
/// intentionally not on the wire, so the reconstruction leaves those at their
/// "not carried" values (`String::new` / `None`). A remote consumer keys off
/// the same fields a local one reads back; the dropped fields are server-side
/// state, not result data. The message is self-describing in `task` and
/// `kind`, so an out-of-range/unspecified value in either is the faithful
/// `invalid_argument` the shared decoders build.
pub fn result_table_from_proto(table: pb::ResultTable) -> Result<ResultTableRecord, Status> {
    let task = super::model_task_from_proto(table.task)?;
    let kind = result_table_kind_from_proto(table.kind)?;
    // `ResultTableRecord::from_wire_projection` is the sole authorized
    // cross-crate constructor: `dimensions`'s privacy otherwise closes off
    // building a record field-by-field from outside `jammi-db`. Every field
    // the wire message does not carry (storage/index paths, timestamps,
    // tenant, lease, version bookkeeping) is left at its "not carried"
    // value inside that constructor — the same values this reconstruction
    // used inline before `dimensions` was privatized:
    //   - `parquet_path`, `distance_metric`, `created_at`: `String::new()`
    //   - `text_columns`, `completed_at`: `None`
    //   - `tenant_id`: server-side bookkeeping resolved from the session,
    //     not carried on the result wire.
    //   - `definition_hash`, `input_anchors_json`: server-side provenance;
    //     a remote consumer that wants the attestation reaches it through
    //     `verify_materialization`, not this reconstruction.
    //   - `storage_precision`, `oversample`: server-side bookkeeping — a
    //     remote consumer never builds/loads the index directly.
    //   - `writer_id`, `lease_expires_at`: the writer lease is server-side
    //     bookkeeping on a `building` row; a wire result is `ready`.
    //   - `current_version`, `next_version`: versioned-table bookkeeping a
    //     remote consumer never reads back through this projection.
    Ok(ResultTableRecord::from_wire_projection(
        table.table_name,
        table.source_id,
        table.model_id,
        task,
        kind,
        table.derived_from,
        table.dimensions,
        table.row_count as usize,
        table.status,
        (!table.key_column.is_empty()).then_some(table.key_column),
    ))
}

#[cfg(test)]
mod result_table_kind_tests {
    use super::{
        result_table_from_proto, result_table_kind_from_proto, result_table_kind_to_proto,
    };
    use crate::proto::embedding as pb;
    use jammi_db::catalog::result_repo::{ResultTableKind, ResultTableRecord};
    use jammi_db::ModelTask;

    /// Discriminants scanned when deriving the wire enum's value set. Well
    /// above the served range, so the scan below is a genuine enumeration of
    /// the generated `pb::ResultTableKind` rather than a restatement of the
    /// list under test.
    const SCAN_LIMIT: i32 = 64;

    /// Every engine [`ResultTableKind`], hand-enumerated and then checked
    /// complete two ways: [`kind_is_enumerated`] below has no `_` arm (a
    /// variant added to the engine enum fails to compile until it is listed
    /// here), and
    /// `engine_kinds_and_wire_kinds_are_the_same_size` pins this array's
    /// length against the discriminant scan.
    const ALL_ENGINE_KINDS: [ResultTableKind; 4] = [
        ResultTableKind::Model,
        ResultTableKind::NeighborGraph,
        ResultTableKind::AsofJoin,
        ResultTableKind::TrainingSet,
    ];

    /// Compile-time completeness witness for [`ALL_ENGINE_KINDS`]: no `_` arm,
    /// so a new engine variant reds this function, and each arm names the
    /// array slot the variant occupies.
    fn kind_is_enumerated(kind: ResultTableKind) -> bool {
        let slot = match kind {
            ResultTableKind::Model => 0,
            ResultTableKind::NeighborGraph => 1,
            ResultTableKind::AsofJoin => 2,
            ResultTableKind::TrainingSet => 3,
        };
        ALL_ENGINE_KINDS[slot] == kind
    }

    /// The wire enum's value set, derived by scanning discriminants rather
    /// than restating the list: `try_from` accepts exactly the values the
    /// generated enum declares.
    fn wire_kind_values() -> Vec<i32> {
        (0..SCAN_LIMIT)
            .filter(|v| pb::ResultTableKind::try_from(*v).is_ok())
            .collect()
    }

    /// The frozen wire numbering. `ResultTableKind` is append-only: a
    /// renumbered or dropped value is a breaking change to a served enum, and
    /// an added one is appended here in the same change that adds it to the
    /// proto.
    #[test]
    fn wire_kind_values_are_frozen_and_append_only() {
        assert_eq!(pb::ResultTableKind::Unspecified as i32, 0);
        assert_eq!(pb::ResultTableKind::Model as i32, 1);
        assert_eq!(pb::ResultTableKind::NeighborGraph as i32, 2);
        assert_eq!(pb::ResultTableKind::AsofJoin as i32, 3);
        assert_eq!(pb::ResultTableKind::TrainingSet as i32, 4);
        assert_eq!(
            wire_kind_values(),
            vec![0, 1, 2, 3, 4],
            "the served ResultTableKind values are frozen; adding one is an \
             append to this list, renumbering or removing one is breaking"
        );
    }

    /// The engine→wire half of the mirror is exhaustive and injective, and the
    /// wire→engine half inverts it: every engine kind survives a round trip as
    /// itself, and no two engine kinds share a wire value (an alias would make
    /// one of them undecodable).
    #[test]
    fn every_engine_kind_round_trips_through_the_wire_mirror() {
        let mut seen: Vec<i32> = Vec::new();
        for kind in ALL_ENGINE_KINDS {
            assert!(kind_is_enumerated(kind), "{kind:?} is not enumerated");
            let wire = result_table_kind_to_proto(kind) as i32;
            assert_ne!(
                wire,
                pb::ResultTableKind::Unspecified as i32,
                "{kind:?} must map to a concrete wire value"
            );
            assert!(
                !seen.contains(&wire),
                "{kind:?} aliases wire value {wire}, already used by another kind"
            );
            seen.push(wire);
            assert_eq!(
                result_table_kind_from_proto(wire).expect("a served kind decodes"),
                kind,
                "{kind:?} must survive the wire round trip as itself"
            );
        }
    }

    /// The wire→engine half is total over the served values: every value the
    /// generated enum declares except `UNSPECIFIED` decodes to an engine kind
    /// that re-encodes to the same value. A proto value added without a mirror
    /// arm cannot reach this test — `result_table_kind_from_proto` has no `_`
    /// arm, so it fails to compile first — but a value mirrored onto the WRONG
    /// engine kind reds here.
    #[test]
    fn every_served_wire_kind_decodes_to_the_engine_kind_it_encodes_from() {
        for value in wire_kind_values() {
            if value == pb::ResultTableKind::Unspecified as i32 {
                continue;
            }
            let kind = result_table_kind_from_proto(value)
                .unwrap_or_else(|e| panic!("served wire kind {value} must decode: {e}"));
            assert_eq!(
                result_table_kind_to_proto(kind) as i32,
                value,
                "wire kind {value} decoded to {kind:?}, which re-encodes elsewhere"
            );
        }
    }

    /// The two sides have the same cardinality, so the hand-written
    /// [`ALL_ENGINE_KINDS`] cannot silently lag a proto that grew a value.
    #[test]
    fn engine_kinds_and_wire_kinds_are_the_same_size() {
        let served = wire_kind_values().len() - 1; // less UNSPECIFIED
        assert_eq!(
            ALL_ENGINE_KINDS.len(),
            served,
            "every served wire kind mirrors exactly one engine kind"
        );
    }

    /// An unspecified or out-of-range kind is a malformed message, not a
    /// silent `Model` guess.
    #[test]
    fn unspecified_and_unknown_wire_kinds_are_rejected() {
        for value in [pb::ResultTableKind::Unspecified as i32, SCAN_LIMIT, -1] {
            let err = result_table_kind_from_proto(value)
                .expect_err("an unspecified/unknown kind must be rejected");
            assert_eq!(err.code(), tonic::Code::InvalidArgument);
            assert_eq!(err.message(), "result table kind must be specified");
        }
    }

    /// The record projection carries the kind faithfully in both directions —
    /// the path a `DescribeSource` / producer response actually takes — so a
    /// remote consumer reads the same kind an embedded one does rather than a
    /// fabricated `MODEL`.
    #[test]
    fn a_training_set_record_keeps_its_kind_across_the_projection() {
        let record = ResultTableRecord::from_wire_projection(
            "jammi_train_set_1".to_string(),
            "src-1".to_string(),
            "model-1".to_string(),
            ModelTask::TextEmbedding,
            ResultTableKind::TrainingSet,
            None,
            0,
            7,
            "ready".to_string(),
            None,
        );
        let wire = pb::ResultTable::from(record);
        assert_eq!(wire.kind, pb::ResultTableKind::TrainingSet as i32);
        let back = result_table_from_proto(wire).expect("the projection reconstructs");
        assert_eq!(back.kind, ResultTableKind::TrainingSet);
        assert_eq!(back.row_count, 7);
    }
}
