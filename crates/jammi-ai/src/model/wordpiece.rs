//! WordPiece-vocabulary tokenizer loader.
//!
//! Constructs a `tokenizers::Tokenizer` from the `vocab.txt` that BERT-family
//! checkpoints ship in place of a converted `tokenizer.json`, configured by
//! the checkpoint's `tokenizer_config.json` when it has one. The pipeline is
//! the one `transformers`' `BertTokenizer` assembles from the same files: the
//! BERT normalizer (casing, accent stripping, CJK splitting, control
//! characters), the BERT pre-tokenizer, WordPiece with `##` continuations,
//! the special tokens matched whole in the input, and the
//! `[CLS] $A [SEP]` / `[CLS] $A [SEP] $B [SEP]` template.

use std::path::Path;

use jammi_db::error::JammiError;
use serde::Deserialize;
use tokenizers::decoders::wordpiece::WordPiece as WordPieceDecoder;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::tokenizer::{AddedToken, Model, Tokenizer};

type Result<T> = std::result::Result<T, JammiError>;

/// The `tokenizer_config.json` keys the BERT pipeline reads. A key the
/// checkpoint omits (or writes as `null`) takes `transformers`' `BertTokenizer`
/// default, which is also what a checkpoint with no config file gets.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct WordPieceConfig {
    do_lower_case: Option<bool>,
    /// `None` strips accents exactly when the text is lowercased.
    strip_accents: Option<bool>,
    tokenize_chinese_chars: Option<bool>,
    unk_token: Option<SpecialToken>,
    sep_token: Option<SpecialToken>,
    pad_token: Option<SpecialToken>,
    cls_token: Option<SpecialToken>,
    mask_token: Option<SpecialToken>,
}

/// A special token as `tokenizer_config.json` writes it: the bare string, or
/// a serialized `AddedToken` carrying it as `content`.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum SpecialToken {
    Bare(String),
    Added { content: String },
}

impl SpecialToken {
    fn content(token: Option<&Self>, default: &'static str) -> String {
        match token {
            Some(Self::Bare(content) | Self::Added { content }) => content.clone(),
            None => default.to_string(),
        }
    }
}

/// Build the tokenizer from a checkpoint's `vocab.txt` and, when it ships
/// one, its `tokenizer_config.json`.
pub fn load_wordpiece_vocab(vocab: &Path, config: Option<&Path>) -> Result<Tokenizer> {
    let config = match config {
        Some(path) => {
            let text = std::fs::read_to_string(path).map_err(|e| JammiError::Model {
                model_id: String::new(),
                message: format!("Failed to read {}: {e}", path.display()),
            })?;
            serde_json::from_str(&text).map_err(|e| JammiError::Model {
                model_id: String::new(),
                message: format!("{} is unparseable: {e}", path.display()),
            })?
        }
        None => WordPieceConfig::default(),
    };
    build_wordpiece_tokenizer(vocab, &config)
}

/// Build the tokenizer from `vocab.txt` under `config`. The vocabulary is
/// read by the `tokenizers` crate's own WordPiece reader: one token per line,
/// its id the line number.
fn build_wordpiece_tokenizer(vocab: &Path, config: &WordPieceConfig) -> Result<Tokenizer> {
    let vocab_file = vocab.to_str().ok_or_else(|| JammiError::Model {
        model_id: String::new(),
        message: format!("WordPiece vocabulary path {} is not UTF-8", vocab.display()),
    })?;
    let unk = SpecialToken::content(config.unk_token.as_ref(), "[UNK]");
    let sep = SpecialToken::content(config.sep_token.as_ref(), "[SEP]");
    let pad = SpecialToken::content(config.pad_token.as_ref(), "[PAD]");
    let cls = SpecialToken::content(config.cls_token.as_ref(), "[CLS]");
    let mask = SpecialToken::content(config.mask_token.as_ref(), "[MASK]");

    let model = WordPiece::from_file(vocab_file)
        .unk_token(unk.clone())
        .continuing_subword_prefix("##".to_string())
        .max_input_chars_per_word(100)
        .build()
        .map_err(|e| JammiError::Model {
            model_id: String::new(),
            message: format!(
                "Failed to read WordPiece vocabulary {}: {e}",
                vocab.display()
            ),
        })?;
    let id_of = |role: &str, token: &str| {
        model.token_to_id(token).ok_or_else(|| JammiError::Model {
            model_id: String::new(),
            message: format!("The WordPiece vocabulary has no {role} token `{token}`"),
        })
    };
    let (sep_id, pad_id, cls_id) = (
        id_of("sep", &sep)?,
        id_of("pad", &pad)?,
        id_of("cls", &cls)?,
    );
    id_of("unk", &unk)?;

    let mut tokenizer = Tokenizer::new(model);
    let lowercase = config.do_lower_case.unwrap_or(true);
    tokenizer.with_normalizer(Some(BertNormalizer::new(
        true,
        config.tokenize_chinese_chars.unwrap_or(true),
        config.strip_accents,
        lowercase,
    )));
    tokenizer.with_pre_tokenizer(Some(BertPreTokenizer));

    let post = TemplateProcessing::builder()
        .try_single(format!("{cls}:0 $A:0 {sep}:0"))
        .and_then(|builder| builder.try_pair(format!("{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1")))
        .map_err(|e| JammiError::Model {
            model_id: String::new(),
            message: format!("WordPiece post-processor template failed: {e}"),
        })?
        .special_tokens(vec![(cls.clone(), cls_id), (sep.clone(), sep_id)])
        .build()
        .map_err(|e| JammiError::Model {
            model_id: String::new(),
            message: format!("WordPiece post-processor build failed: {e}"),
        })?;
    tokenizer.with_post_processor(Some(post));
    tokenizer.with_decoder(Some(WordPieceDecoder::new("##".to_string(), true)));

    // A special token written in the text is matched whole, before the
    // normalizer could lowercase or split it.
    tokenizer.add_special_tokens(
        &[&pad, &unk, &cls, &sep, &mask].map(|token| AddedToken::from(token.as_str(), true)),
    );

    tokenizer.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::BatchLongest,
        pad_id,
        pad_token: pad,
        ..Default::default()
    }));

    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Deserialize)]
    struct Probe {
        text: String,
        ids: Vec<u32>,
    }

    fn variant(name: &str) -> std::path::PathBuf {
        jammi_test_utils::fixture("wordpiece_vocab").join(name)
    }

    fn expected(name: &str) -> Vec<Probe> {
        let path = jammi_test_utils::fixture("wordpiece_vocab/expected_ids.json");
        let mut all: std::collections::HashMap<String, Vec<Probe>> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        all.remove(name).unwrap()
    }

    fn ids(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
        tokenizer.encode(text, true).unwrap().get_ids().to_vec()
    }

    /// The oracle is `transformers` itself: `tests/fixtures/generate_wordpiece_vocab.py`
    /// records the ids it encodes each probe to from the same two files.
    #[test]
    fn encodes_every_probe_as_transformers_does() {
        for name in ["cased", "uncased"] {
            let dir = variant(name);
            let tokenizer = load_wordpiece_vocab(
                &dir.join("vocab.txt"),
                Some(&dir.join("tokenizer_config.json")),
            )
            .unwrap();
            for probe in expected(name) {
                assert_eq!(
                    ids(&tokenizer, &probe.text),
                    probe.ids,
                    "{name}: {:?}",
                    probe.text
                );
            }
        }
    }

    /// A checkpoint with no `tokenizer_config.json` gets `BertTokenizer`'s
    /// defaults, which are the uncased fixture's settings.
    #[test]
    fn a_missing_config_takes_the_bert_defaults() {
        let tokenizer = load_wordpiece_vocab(&variant("uncased").join("vocab.txt"), None).unwrap();
        for probe in expected("uncased") {
            assert_eq!(ids(&tokenizer, &probe.text), probe.ids, "{:?}", probe.text);
        }
    }

    fn vocab_file(dir: &tempfile::TempDir, tokens: &[&str]) -> std::path::PathBuf {
        let path = dir.path().join("vocab.txt");
        std::fs::write(&path, tokens.join("\n") + "\n").unwrap();
        path
    }

    #[test]
    fn special_tokens_in_object_form_are_read_by_content() {
        let dir = tempfile::tempdir().unwrap();
        let vocab = vocab_file(&dir, &["[PAD]", "[UNK]", "<cls>", "[SEP]", "[MASK]", "Hi"]);
        let config: WordPieceConfig = serde_json::from_str(
            r#"{"do_lower_case": false, "cls_token": {"content": "<cls>", "special": true}}"#,
        )
        .unwrap();
        let tokenizer = build_wordpiece_tokenizer(&vocab, &config).unwrap();
        assert_eq!(ids(&tokenizer, "Hi"), vec![2, 5, 3]);
    }

    #[test]
    fn a_vocabulary_without_a_special_token_is_refused_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let vocab = vocab_file(&dir, &["[PAD]", "[UNK]", "[SEP]", "word"]);
        let err = build_wordpiece_tokenizer(&vocab, &WordPieceConfig::default()).unwrap_err();
        assert!(err.to_string().contains("no cls token `[CLS]`"), "{err}");
    }
}
