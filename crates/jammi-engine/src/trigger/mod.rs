//! Trigger-stream primitive: pub/sub over Arrow `RecordBatch` with SQL
//! predicate filters and a Phase-2 mutable backing table for the
//! authoritative event log. See
//! `docs/plans/cp9-substrate-primitives/SPEC-04-trigger-stream.md`.

pub mod broker;
pub mod error;
pub mod ids;
pub mod in_memory;
#[cfg(feature = "jetstream-broker")]
pub mod jetstream;
pub mod offset;
pub mod predicate;
pub mod publisher;
pub mod subscriber;
pub mod subscription;
pub mod topic;

pub use broker::{BrokerKind, TriggerBroker};
pub use error::TriggerError;
pub use ids::{SubscriptionId, TopicId};
pub use in_memory::InMemoryBroker;
#[cfg(feature = "jetstream-broker")]
pub use jetstream::JetStreamBroker;
pub use offset::Offset;
pub use predicate::Predicate;
pub use publisher::Publisher;
pub use subscriber::Subscriber;
pub use subscription::{DeliveredBatch, Subscription};
pub use topic::{TopicDefinition, OFFSET_COLUMN, PRODUCED_AT_COLUMN, ROW_INDEX_COLUMN};
