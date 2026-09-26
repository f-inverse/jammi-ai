//! Retrieval indexes the AI layer owns: the lexical (BM25) index, built in
//! memory from a lexical table's rows. The ANN index lives in the `jammi-db`
//! substrate beside the storage it serialises through.

pub mod lexical;

pub use lexical::{LexicalHit, LexicalIndex, LexicalIndexes};
