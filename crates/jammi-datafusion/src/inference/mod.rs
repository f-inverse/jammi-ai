//! The forward stage: run a model over a relation as a physical stage.
//!
//! [`exec::plan_inference`] builds the one plan every model forward runs
//! in: the rows are ordered, numbered and chunked by a token budget once,
//! below every exchange ([`numbered`]); each chunk is prepared on the
//! host, admitted against its device and forwarded ([`runner`]); the
//! output carries the task's columns behind a common prefix ([`schema`],
//! [`adapter`]); and the plan is placeable on another process through its
//! wire form ([`wire`]) and the runtime that process binds ([`runtime`]).

pub mod adapter;
pub mod chunk;
pub mod columns;
pub mod exec;
pub mod key_check;
pub mod numbered;
pub mod observer;
pub mod output;
pub mod row_cost;
pub mod runner;
pub mod runtime;
pub mod schema;
pub mod spec;
pub mod wire;
