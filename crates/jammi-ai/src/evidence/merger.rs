//! Merge per-batch [`ChannelContribution`]s into result `RecordBatch`es
//! using catalog-declared channel schemas as the source of truth.
//!
//! The engine **never writes** to provenance columns — callers supply
//! contributions; this function merges them by column name into the
//! batch. Unsupplied channels yield all-null arrays of the declared
//! dtype.

use std::sync::Arc;

use arrow::array::{new_null_array, ArrayRef, ListArray, RecordBatch, StringArray};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Schema};

use jammi_engine::catalog::channel_repo::ChannelSpec;
use jammi_engine::catalog::Catalog;
use jammi_engine::error::{JammiError, Result};
use jammi_engine::ChannelId;

use super::channel::ChannelContribution;

/// Merge channel contributions into batches.
///
/// Output schema = source columns + `retrieved_by` + `annotated_by` +
/// per-channel declared columns (sorted by `(priority, ordinal)`).
///
/// Errors:
/// - `contributions.len() != batches.len()` — shape mismatch.
/// - A contribution claims a channel not in `participating`.
/// - A contribution's `columns` length disagrees with the channel's
///   declared column count.
/// - A contributed array length disagrees with the batch's row count.
/// - A contributed array's dtype disagrees with the catalog declaration.
pub async fn merge_channels(
    catalog: &Catalog,
    batches: &[RecordBatch],
    participating: &[ChannelId],
    retrieved: &[ChannelId],
    annotated: &[ChannelId],
    contributions: &[Vec<ChannelContribution>],
) -> Result<Vec<RecordBatch>> {
    if contributions.len() != batches.len() {
        return Err(JammiError::EvidenceChannel(format!(
            "contributions length {} does not match batches length {}",
            contributions.len(),
            batches.len()
        )));
    }

    // Resolve specs once, in priority order. Errors here mean a channel
    // in `participating` isn't registered.
    let mut specs: Vec<ChannelSpec> = Vec::with_capacity(participating.len());
    for id in participating {
        let spec = catalog.channels().get(id).await?.ok_or_else(|| {
            JammiError::EvidenceChannel(format!("channel '{id}': not registered"))
        })?;
        specs.push(spec);
    }
    specs.sort_by_key(|s| s.priority);

    let suffix_fields = build_suffix_fields(&specs);

    batches
        .iter()
        .enumerate()
        .map(|(i, batch)| {
            merge_one_batch(
                batch,
                &specs,
                &suffix_fields,
                retrieved,
                annotated,
                &contributions[i],
                i,
            )
        })
        .collect()
}

fn build_suffix_fields(specs: &[ChannelSpec]) -> Vec<Field> {
    let list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
    let mut fields = vec![
        Field::new("retrieved_by", list_type.clone(), false),
        Field::new("annotated_by", list_type, false),
    ];
    for spec in specs {
        for col in &spec.columns {
            fields.push(Field::new(&col.name, col.data_type.to_arrow(), true));
        }
    }
    fields
}

fn merge_one_batch(
    batch: &RecordBatch,
    specs: &[ChannelSpec],
    suffix_fields: &[Field],
    retrieved: &[ChannelId],
    annotated: &[ChannelId],
    contributions: &[ChannelContribution],
    batch_index: usize,
) -> Result<RecordBatch> {
    let row_count = batch.num_rows();

    // At most one contribution per `(batch, channel)` — surface the
    // collision rather than silently dropping the second one.
    let mut seen: std::collections::HashSet<&ChannelId> =
        std::collections::HashSet::with_capacity(contributions.len());
    for contrib in contributions {
        if !seen.insert(&contrib.channel) {
            return Err(JammiError::EvidenceChannel(format!(
                "channel '{}': duplicate contribution on batch {batch_index}",
                contrib.channel
            )));
        }
    }

    // Validate every contribution against its declared spec before
    // assembling any output array.
    for contrib in contributions {
        validate_contribution(contrib, specs, row_count, batch_index)?;
    }

    let retrieved_by = build_list_column(row_count, retrieved)?;
    let annotated_by = build_list_column(row_count, annotated)?;

    let mut suffix_columns: Vec<ArrayRef> = vec![retrieved_by, annotated_by];
    for spec in specs {
        let supplied = contributions.iter().find(|c| c.channel == spec.id);
        for (col_idx, decl) in spec.columns.iter().enumerate() {
            let arr: ArrayRef = match supplied {
                Some(c) => Arc::clone(&c.columns[col_idx]),
                None => new_null_array(&decl.data_type.to_arrow(), row_count),
            };
            suffix_columns.push(arr);
        }
    }

    let mut all_fields: Vec<Arc<Field>> = batch.schema().fields().to_vec();
    all_fields.extend(suffix_fields.iter().cloned().map(Arc::new));
    let mut all_columns: Vec<ArrayRef> = batch.columns().to_vec();
    all_columns.extend(suffix_columns);

    RecordBatch::try_new(Arc::new(Schema::new(all_fields)), all_columns).map_err(|e| {
        JammiError::EvidenceChannel(format!("batch {batch_index}: assembly failed: {e}"))
    })
}

fn validate_contribution(
    contrib: &ChannelContribution,
    specs: &[ChannelSpec],
    row_count: usize,
    batch_index: usize,
) -> Result<()> {
    let spec = specs
        .iter()
        .find(|s| s.id == contrib.channel)
        .ok_or_else(|| {
            JammiError::EvidenceChannel(format!(
            "batch {batch_index}: contribution for channel '{}' is not in the participating set",
            contrib.channel
        ))
        })?;

    if contrib.columns.len() != spec.columns.len() {
        return Err(JammiError::EvidenceChannel(format!(
            "batch {batch_index}: channel '{}' contribution has {} columns, expected {}",
            contrib.channel,
            contrib.columns.len(),
            spec.columns.len(),
        )));
    }

    for (idx, (arr, decl)) in contrib.columns.iter().zip(spec.columns.iter()).enumerate() {
        if arr.len() != row_count {
            return Err(JammiError::EvidenceChannel(format!(
                "batch {batch_index}: channel '{}' column '{}' has {} rows, expected {}",
                contrib.channel,
                decl.name,
                arr.len(),
                row_count,
            )));
        }
        let want = decl.data_type.to_arrow();
        if arr.data_type() != &want {
            return Err(JammiError::EvidenceChannel(format!(
                "batch {batch_index}: channel '{}' column '{}' has dtype {:?}, expected {:?} (column index {idx})",
                contrib.channel, decl.name, arr.data_type(), want,
            )));
        }
    }
    Ok(())
}

fn build_list_column(row_count: usize, values: &[ChannelId]) -> Result<ArrayRef> {
    let strs: Vec<&str> = values.iter().map(|c| c.as_str()).collect();
    let flat_values: Vec<&str> = (0..row_count).flat_map(|_| strs.iter().copied()).collect();
    let values_array = Arc::new(StringArray::from(flat_values));
    let offsets: Vec<i32> = (0..=row_count).map(|i| (i * strs.len()) as i32).collect();

    let list = ListArray::try_new(
        Arc::new(Field::new("item", DataType::Utf8, true)),
        OffsetBuffer::new(offsets.into()),
        values_array,
        None,
    )
    .map_err(|e| {
        JammiError::EvidenceChannel(format!("list construction for retrieved/annotated: {e}"))
    })?;

    Ok(Arc::new(list))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Float32Array, Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use jammi_engine::catalog::channel_repo::{ChannelColumn, ChannelColumnType, ChannelSpec};
    use tempfile::tempdir;

    async fn open_catalog() -> (tempfile::TempDir, Catalog) {
        let dir = tempdir().unwrap();
        let catalog = Catalog::open(dir.path()).await.unwrap();
        (dir, catalog)
    }

    fn source_batch(n: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("_row_id", DataType::Utf8, false),
            Field::new("_source_id", DataType::Utf8, false),
        ]));
        let row_id = Arc::new(StringArray::from(
            (0..n).map(|i| format!("r{i}")).collect::<Vec<_>>(),
        )) as ArrayRef;
        let src = Arc::new(StringArray::from(vec!["src"; n])) as ArrayRef;
        RecordBatch::try_new(schema, vec![row_id, src]).unwrap()
    }

    async fn register_scored_by(catalog: &Catalog) -> ChannelId {
        let id = ChannelId::new("scored_by").unwrap();
        catalog
            .channels()
            .register(&ChannelSpec {
                id: id.clone(),
                priority: 3,
                columns: vec![
                    ChannelColumn {
                        name: "ranker".into(),
                        data_type: ChannelColumnType::Utf8,
                    },
                    ChannelColumn {
                        name: "rank_score".into(),
                        data_type: ChannelColumnType::Float32,
                    },
                ],
            })
            .await
            .unwrap();
        id
    }

    #[tokio::test]
    async fn merge_adds_declared_columns_in_priority_order() {
        let (_dir, catalog) = open_catalog().await;
        let vector = ChannelId::new("vector").unwrap();
        let scored_by = register_scored_by(&catalog).await;
        let batch = source_batch(2);

        let vector_contrib = ChannelContribution::single(
            vector.clone(),
            Arc::new(Float32Array::from(vec![0.9_f32, 0.8])) as ArrayRef,
        );
        let scored_contrib = ChannelContribution {
            channel: scored_by.clone(),
            columns: vec![
                Arc::new(StringArray::from(vec!["bm25", "bm25"])) as ArrayRef,
                Arc::new(Float32Array::from(vec![1.2_f32, 0.7])) as ArrayRef,
            ],
        };

        let merged = merge_channels(
            &catalog,
            &[batch],
            &[vector.clone(), scored_by.clone()],
            &[vector.clone(), scored_by.clone()],
            &[],
            &[vec![vector_contrib, scored_contrib]],
        )
        .await
        .unwrap();

        assert_eq!(merged.len(), 1);
        let schema = merged[0].schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            names,
            vec![
                "_row_id",
                "_source_id",
                "retrieved_by",
                "annotated_by",
                "similarity",
                "ranker",
                "rank_score",
            ]
        );
        assert_eq!(merged[0].num_rows(), 2);
    }

    #[tokio::test]
    async fn unsupplied_channel_columns_become_null() {
        let (_dir, catalog) = open_catalog().await;
        let vector = ChannelId::new("vector").unwrap();
        let inference = ChannelId::new("inference").unwrap();
        let batch = source_batch(3);

        let vector_contrib = ChannelContribution::single(
            vector.clone(),
            Arc::new(Float32Array::from(vec![0.5; 3])) as ArrayRef,
        );

        let merged = merge_channels(
            &catalog,
            &[batch],
            &[vector.clone(), inference.clone()],
            &[vector.clone()],
            &[inference.clone()],
            &[vec![vector_contrib]],
        )
        .await
        .unwrap();

        // inference's three columns must be all-null.
        let m = &merged[0];
        let inf_model = m
            .column(m.schema().index_of("inference_model").unwrap())
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(inf_model.null_count(), 3);
        let inf_conf = m
            .column(m.schema().index_of("inference_confidence").unwrap())
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert_eq!(inf_conf.null_count(), 3);
    }

    #[tokio::test]
    async fn rejects_length_mismatch_between_contribution_and_batch() {
        let (_dir, catalog) = open_catalog().await;
        let vector = ChannelId::new("vector").unwrap();
        let batch = source_batch(3);
        let bad = ChannelContribution::single(
            vector.clone(),
            Arc::new(Float32Array::from(vec![0.5_f32, 0.4])) as ArrayRef,
        );
        let err = merge_channels(
            &catalog,
            &[batch],
            &[vector.clone()],
            &[vector.clone()],
            &[],
            &[vec![bad]],
        )
        .await
        .unwrap_err();
        match err {
            JammiError::EvidenceChannel(m) => {
                assert!(m.contains("similarity"));
                assert!(m.contains("2 rows"));
                assert!(m.contains("expected 3"));
            }
            other => panic!("expected EvidenceChannel(length), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rejects_wrong_arrow_dtype() {
        let (_dir, catalog) = open_catalog().await;
        let vector = ChannelId::new("vector").unwrap();
        let batch = source_batch(2);
        let bad = ChannelContribution::single(
            vector.clone(),
            Arc::new(Int32Array::from(vec![1_i32, 2])) as ArrayRef,
        );
        let err = merge_channels(
            &catalog,
            &[batch],
            &[vector.clone()],
            &[vector.clone()],
            &[],
            &[vec![bad]],
        )
        .await
        .unwrap_err();
        match err {
            JammiError::EvidenceChannel(m) => assert!(m.contains("dtype")),
            other => panic!("expected EvidenceChannel(dtype), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rejects_contribution_for_non_participating_channel() {
        let (_dir, catalog) = open_catalog().await;
        let vector = ChannelId::new("vector").unwrap();
        let inference = ChannelId::new("inference").unwrap();
        let batch = source_batch(1);
        let stranger = ChannelContribution::single(
            inference.clone(),
            Arc::new(StringArray::from(vec!["m"])) as ArrayRef,
        );
        let err = merge_channels(
            &catalog,
            &[batch],
            &[vector.clone()], // inference is NOT participating
            &[vector.clone()],
            &[],
            &[vec![stranger]],
        )
        .await
        .unwrap_err();
        match err {
            JammiError::EvidenceChannel(m) => assert!(m.contains("not in the participating set")),
            other => panic!("expected EvidenceChannel(non-participating), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rejects_unregistered_channel() {
        let (_dir, catalog) = open_catalog().await;
        let batch = source_batch(1);
        let err = merge_channels(
            &catalog,
            &[batch],
            &[ChannelId::new("nonexistent").unwrap()],
            &[],
            &[],
            &[vec![]],
        )
        .await
        .unwrap_err();
        match err {
            JammiError::EvidenceChannel(m) => assert!(m.contains("not registered")),
            other => panic!("expected EvidenceChannel(unregistered), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn batches_and_contributions_must_have_equal_length() {
        let (_dir, catalog) = open_catalog().await;
        let vector = ChannelId::new("vector").unwrap();
        let batches = vec![source_batch(1), source_batch(2)];
        let contribs: Vec<Vec<ChannelContribution>> = vec![vec![]]; // only one
        let err = merge_channels(
            &catalog,
            &batches,
            &[vector.clone()],
            &[vector.clone()],
            &[],
            &contribs,
        )
        .await
        .unwrap_err();
        match err {
            JammiError::EvidenceChannel(m) => {
                assert!(m.contains("length"));
                assert!(m.contains("does not match"));
            }
            other => panic!("expected EvidenceChannel(shape), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn preserves_source_columns_intact() {
        let (_dir, catalog) = open_catalog().await;
        let vector = ChannelId::new("vector").unwrap();
        let batch = source_batch(2);
        let contrib = ChannelContribution::single(
            vector.clone(),
            Arc::new(Float32Array::from(vec![0.1_f32, 0.2])) as ArrayRef,
        );
        let merged = merge_channels(
            &catalog,
            &[batch.clone()],
            &[vector.clone()],
            &[vector.clone()],
            &[],
            &[vec![contrib]],
        )
        .await
        .unwrap();
        // Source column data matches the input.
        let m = &merged[0];
        let row_id = m.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(row_id.value(0), "r0");
        assert_eq!(row_id.value(1), "r1");
    }

    #[tokio::test]
    async fn rejects_duplicate_contribution_for_same_channel_on_one_batch() {
        let (_dir, catalog) = open_catalog().await;
        let vector = ChannelId::new("vector").unwrap();
        let batch = source_batch(2);
        let dup_a = ChannelContribution::single(
            vector.clone(),
            Arc::new(Float32Array::from(vec![0.1_f32, 0.2])) as ArrayRef,
        );
        let dup_b = ChannelContribution::single(
            vector.clone(),
            Arc::new(Float32Array::from(vec![0.3_f32, 0.4])) as ArrayRef,
        );
        let err = merge_channels(
            &catalog,
            &[batch],
            &[vector.clone()],
            &[vector.clone()],
            &[],
            &[vec![dup_a, dup_b]],
        )
        .await
        .unwrap_err();
        match err {
            JammiError::EvidenceChannel(m) => {
                assert!(m.contains("duplicate contribution"));
                assert!(m.contains("batch 0"));
            }
            other => panic!("expected EvidenceChannel(duplicate), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn channel_columns_are_nullable_in_output_schema() {
        let (_dir, catalog) = open_catalog().await;
        let vector = ChannelId::new("vector").unwrap();
        let batch = source_batch(1);
        let contrib = ChannelContribution::single(
            vector.clone(),
            Arc::new(Float32Array::from(vec![0.5_f32])) as ArrayRef,
        );
        let merged = merge_channels(
            &catalog,
            &[batch],
            &[vector.clone()],
            &[vector.clone()],
            &[],
            &[vec![contrib]],
        )
        .await
        .unwrap();
        let schema = merged[0].schema();
        let similarity = schema.field(schema.index_of("similarity").unwrap());
        assert!(
            similarity.is_nullable(),
            "channel-declared columns are always nullable"
        );
    }
}
