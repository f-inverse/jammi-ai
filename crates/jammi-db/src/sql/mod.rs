//! SQL identifier helpers shared by the read-path query builders.

pub mod ident;

pub use ident::{quote_ident, source_relation};
