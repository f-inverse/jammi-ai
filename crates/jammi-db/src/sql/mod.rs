//! SQL identifier helpers shared by the read-path query builders.

pub mod ident;

pub use ident::{quote_ident, quote_relation, source_relation};
