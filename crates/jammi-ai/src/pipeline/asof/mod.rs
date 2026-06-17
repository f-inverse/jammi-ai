//! As-of temporal join: match each spine row to the at-most-one fact valid as-of
//! its instant within an equality group.
//!
//! The module is four concerns behind four types: [`spec`] is the *what* (the
//! frozen [`AsofJoinSpec`] and the four pinned knobs), [`exec`] is the *plan
//! contract* (the [`AsofJoinExec`] physical operator), [`merge`] is the
//! *algorithm* (the single-pointer sort-merge core), and [`verb`] is the
//! *lifecycle* (resolve relations → plan → run → write the attested result
//! table). The verb is the engine's public surface; the operator and merge are
//! reusable by any temporal path that needs one as-of semantics.

pub mod exec;
pub mod merge;
pub mod spec;
pub mod verb;

pub use exec::AsofJoinExec;
pub use spec::{
    AsofError, AsofJoinSpec, AsofJoinSpecBuilder, AsofKey, Boundary, MatchDirection, TieBreak,
    Tolerance,
};
