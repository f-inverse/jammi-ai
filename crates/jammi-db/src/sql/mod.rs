//! Jammi-specific SQL extensions parsed at the session boundary.
//!
//! DataFusion's parser does not know about `CREATE TOPIC` / `DROP TOPIC`;
//! [`JammiSession::sql`](crate::session::JammiSession::sql) inspects every
//! input string through [`topic_ddl::maybe_parse`] before handing it to the
//! DataFusion `SessionContext`, routing the trigger-stream DDL to the
//! engine's [`crate::catalog::topic_repo::TopicRepo`].

pub mod topic_ddl;
