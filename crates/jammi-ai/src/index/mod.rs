//! Out-of-band retrieval indexes that ride beside a result table as sidecar
//! objects.
//!
//! The ANN (USearch) sidecar lives in the `jammi-db` substrate next to the
//! storage layout it serialises through. The lexical (BM25) sidecar lives here
//! in the AI layer alongside the retrieval surfaces that fuse it with dense
//! search — it carries no substrate concern beyond the `.tantivy` extension
//! the `jammi_db::storage::sidecar_layout::SidecarKind::Lexical` registry arm
//! declares.

pub mod lexical;

pub use lexical::{Analyzer, LexicalHit, LexicalIndex};
