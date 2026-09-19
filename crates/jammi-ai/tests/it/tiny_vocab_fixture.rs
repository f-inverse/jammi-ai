//! `jammi_test_utils::tiny_vocab_text` against the real `tiny_bert` tokenizer:
//! the property every fixture built on it relies on.

use std::collections::HashSet;

use jammi_test_utils::{cookbook_fixture, tiny_vocab_text};

fn tiny_bert_tokenizer() -> tokenizers::Tokenizer {
    tokenizers::Tokenizer::from_file(cookbook_fixture("tiny_bert").join("tokenizer.json"))
        .expect("the tiny_bert fixture ships its tokenizer")
}

/// Every `(role, index)` is a distinct token sequence with no `[UNK]` in it —
/// across roles too, so an anchor never equals its positive.
#[test]
fn every_role_and_index_is_a_distinct_unk_free_token_sequence() {
    let tokenizer = tiny_bert_tokenizer();
    let unk = tokenizer
        .token_to_id("[UNK]")
        .expect("the vocabulary names its unknown token");
    let mut seen = HashSet::new();
    for role in ['a', 'p', 'n', '7'] {
        for index in 0..2_000 {
            let text = tiny_vocab_text(role, index);
            let encoding = tokenizer.encode(text.as_str(), false).unwrap();
            assert!(
                !encoding.get_ids().contains(&unk),
                "{text:?} contains a token the model cannot see: {:?}",
                encoding.get_tokens()
            );
            assert!(
                seen.insert(encoding.get_ids().to_vec()),
                "{text:?} tokenizes like an earlier (role, index)"
            );
        }
    }
}

/// What the helper exists to replace: an English fixture collapses. This pins
/// the premise, so a richer fixture tokenizer would be noticed here rather
/// than leaving the helper's reason for existing silently false.
#[test]
fn an_english_fixture_collapses_under_the_tiny_vocabulary() {
    let tokenizer = tiny_bert_tokenizer();
    let ids = |text: &str| tokenizer.encode(text, false).unwrap().get_ids().to_vec();
    assert_eq!(
        ids("graph_node_text_3"),
        ids("graph_node_text_5"),
        "two different node texts are one token sequence to the model"
    );
    assert_eq!(
        ids("pod leg anchor text 3"),
        ids("pod leg positive text 3"),
        "an anchor and its positive are one token sequence to the model"
    );
}
