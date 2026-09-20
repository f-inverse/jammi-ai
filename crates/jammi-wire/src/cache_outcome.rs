//! The one wire form of the engine's [`CacheOutcome`]: the
//! `jammi.v1.inference.CacheOutcome` message every producer response and
//! every job result carries. Encode is total; decode refuses an absent
//! message and one with no arm set, so a reuse can never be inferred from a
//! default.

use jammi_db::catalog::artifact_repo::ArtifactRef;
use jammi_db::catalog::result_repo::ResultTableName;
use jammi_db::store::{CacheOutcome, ReusedArtifact};
use tonic::Status;

use crate::proto::inference::cache_outcome::{Computed, Outcome};
use crate::proto::inference::CacheOutcome as Pb;

/// Encode the engine outcome, naming the reused artifact when there is one.
pub fn cache_outcome_to_proto(outcome: &CacheOutcome) -> Pb {
    let outcome = match outcome {
        CacheOutcome::Computed => Outcome::Computed(Computed {}),
        CacheOutcome::Reused(ReusedArtifact::Table(table)) => {
            Outcome::ReusedTable(table.table_name().to_string())
        }
        CacheOutcome::Reused(ReusedArtifact::Model(artifact)) => {
            Outcome::ReusedModelArtifact(artifact.to_string())
        }
    };
    Pb {
        outcome: Some(outcome),
    }
}

/// Decode the wire outcome a producer or a job result carries. An absent
/// message, or one with no arm set, is a loud `internal` error: the engine
/// always states which path it took, and a decoder never assumes one.
pub fn cache_outcome_from_proto(outcome: Option<Pb>) -> Result<CacheOutcome, Status> {
    match outcome.and_then(|pb| pb.outcome) {
        Some(Outcome::Computed(Computed {})) => Ok(CacheOutcome::Computed),
        Some(Outcome::ReusedTable(table)) => Ok(CacheOutcome::Reused(ReusedArtifact::Table(
            ResultTableName::new(table),
        ))),
        Some(Outcome::ReusedModelArtifact(artifact)) => ArtifactRef::parse(&artifact)
            .map(|artifact| CacheOutcome::Reused(ReusedArtifact::Model(artifact)))
            .map_err(|e| {
                Status::internal(format!(
                    "producer returned a reused model artifact that is not a reference: {e}"
                ))
            }),
        None => Err(Status::internal("producer returned no cache outcome")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_outcome_round_trips_through_the_wire_naming_what_it_reused() {
        let outcomes = [
            CacheOutcome::Computed,
            CacheOutcome::Reused(ReusedArtifact::Table(ResultTableName::new(
                "embeddings_patents_abc",
            ))),
            CacheOutcome::Reused(ReusedArtifact::Model(
                ArtifactRef::parse("file:///store/models/_global/job/w/1").unwrap(),
            )),
        ];
        for outcome in outcomes {
            let encoded = cache_outcome_to_proto(&outcome);
            assert_eq!(cache_outcome_from_proto(Some(encoded)).unwrap(), outcome);
        }
    }

    #[test]
    fn an_absent_or_empty_outcome_is_a_loud_error_never_computed() {
        for missing in [None, Some(Pb { outcome: None })] {
            let err = cache_outcome_from_proto(missing).unwrap_err();
            assert_eq!(err.code(), tonic::Code::Internal);
        }
        let err = cache_outcome_from_proto(Some(Pb {
            outcome: Some(Outcome::ReusedModelArtifact("bogus://models/x".into())),
        }))
        .unwrap_err();
        assert_eq!(err.code(), tonic::Code::Internal);
    }
}
