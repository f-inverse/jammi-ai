//! An inner doc comment for this fixture module (`//!`).

// A trailing line comment.
/// A doc comment on the next item (`///`).
fn line_and_doc_comments() -> u8 {
    1 // a trailing comment after code on the same line
}

/* A single-line block comment. */
/* A block comment
   spanning multiple
   lines. */
/* An outer /* nested */ block comment on one line. */
fn block_comments() -> u8 {
    /* /* deeply /* nested */ block */ comment, */
    2
}
