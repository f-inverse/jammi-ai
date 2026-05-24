//! OpenCLIP native BPE tokenizer loader.
//!
//! Constructs a `tokenizers::Tokenizer` from the OpenCLIP-native
//! `bpe_simple_vocab_16e6.txt.gz` merges file shipped with stock OpenCLIP
//! checkpoints (no pre-converted HF `tokenizer.json` required).
//!
//! The vocabulary is built deterministically from the merges and the
//! canonical `bytes_to_unicode` byte-to-printable-codepoint mapping, with
//! `<|startoftext|>` and `<|endoftext|>` reserved at the tail of the
//! vocabulary (final indices `vocab_size - 2` and `vocab_size - 1`,
//! respectively, matching upstream OpenCLIP's tokenizer).

use std::fs::File;
use std::io::Read;
use std::path::Path;

use flate2::read::GzDecoder;
use jammi_engine::error::JammiError;
use tokenizers::models::bpe::{Merges, Vocab, BPE};
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::tokenizer::Tokenizer;
use tokenizers::SplitDelimiterBehavior;

type Result<T> = std::result::Result<T, JammiError>;

/// Total vocabulary size produced by OpenCLIP's BPE: 256 byte tokens + 256
/// byte tokens with the `</w>` end-of-word suffix + 48894 merges + 2 special
/// tokens (`<|startoftext|>`, `<|endoftext|>`).
pub const OPEN_CLIP_VOCAB_SIZE: usize = 49408;

/// Number of merge lines OpenCLIP reads from the vocab file (everything
/// after the version header up to `vocab_size - 256 - 2`).
const NUM_MERGES: usize = OPEN_CLIP_VOCAB_SIZE - 256 - 2 - 256;

/// OpenCLIP's pre-tokenization regex. Matches the special tokens, common
/// English contractions, alphabetic runs, single digits, and non-whitespace
/// non-alphanumeric runs — identical to the regex in OpenCLIP's
/// `simple_tokenizer.py`.
const CLIP_REGEX: &str =
    r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+";

const SOT_TOKEN: &str = "<|startoftext|>";
const EOT_TOKEN: &str = "<|endoftext|>";

/// Build a `tokenizers::Tokenizer` from the OpenCLIP native vocab file.
///
/// `path` points to `bpe_simple_vocab_16e6.txt.gz` (the file shipped inside
/// OpenCLIP repos and bundled with stock CLIP releases). The returned
/// tokenizer is configured with batch-longest padding and a post-processor
/// that wraps each sequence in `<|startoftext|> ... <|endoftext|>` so the
/// EOT-pool path in [`jammi_encoders::ClipText`] finds the EOT at
/// `argmax(input_ids)`.
pub fn load_open_clip_bpe(path: &Path) -> Result<Tokenizer> {
    let mut file = File::open(path).map_err(|e| JammiError::Model {
        model_id: String::new(),
        message: format!(
            "Failed to open OpenCLIP BPE vocab at {}: {e}",
            path.display()
        ),
    })?;
    let mut compressed = Vec::new();
    file.read_to_end(&mut compressed)
        .map_err(|e| JammiError::Model {
            model_id: String::new(),
            message: format!("Failed to read OpenCLIP BPE vocab: {e}"),
        })?;

    let mut decoder = GzDecoder::new(&compressed[..]);
    let mut text = String::new();
    decoder
        .read_to_string(&mut text)
        .map_err(|e| JammiError::Model {
            model_id: String::new(),
            message: format!("Failed to gunzip OpenCLIP BPE vocab: {e}"),
        })?;

    build_open_clip_tokenizer(&text)
}

/// Build the tokenizer from the in-memory contents of the
/// `bpe_simple_vocab_16e6.txt` file (header + merges separated by `\n`).
/// Factored so the unit tests can exercise the vocab construction without
/// touching the filesystem.
pub fn build_open_clip_tokenizer(text: &str) -> Result<Tokenizer> {
    let lines: Vec<&str> = text.split('\n').collect();
    if lines.len() < NUM_MERGES + 1 {
        return Err(JammiError::Model {
            model_id: String::new(),
            message: format!(
                "OpenCLIP BPE vocab has {} lines, expected at least {}",
                lines.len(),
                NUM_MERGES + 1
            ),
        });
    }

    // Line 0 is a header; lines 1..=NUM_MERGES are merge pairs.
    let merges: Merges = lines[1..=NUM_MERGES]
        .iter()
        .enumerate()
        .map(|(offset, line)| {
            let mut parts = line.split_whitespace();
            let a = parts.next().ok_or_else(|| JammiError::Model {
                model_id: String::new(),
                message: format!("OpenCLIP BPE merge line {} is empty", offset + 1),
            })?;
            let b = parts.next().ok_or_else(|| JammiError::Model {
                model_id: String::new(),
                message: format!(
                    "OpenCLIP BPE merge line {} has only one token; expected two",
                    offset + 1
                ),
            })?;
            Ok((a.to_string(), b.to_string()))
        })
        .collect::<Result<_>>()?;

    // Build the byte → printable-unicode mapping used by OpenCLIP.
    let byte_chars = bytes_to_unicode();

    // Construct vocab in the same order as OpenCLIP:
    //   [byte tokens (256)] ++ [byte tokens with </w> suffix (256)]
    //   ++ [each merge joined (48894)] ++ [<|startoftext|>, <|endoftext|>].
    let mut vocab: Vec<String> = Vec::with_capacity(OPEN_CLIP_VOCAB_SIZE);
    for ch in &byte_chars {
        vocab.push(ch.to_string());
    }
    for ch in &byte_chars {
        vocab.push(format!("{ch}</w>"));
    }
    for (a, b) in &merges {
        vocab.push(format!("{a}{b}"));
    }
    vocab.push(SOT_TOKEN.to_string());
    vocab.push(EOT_TOKEN.to_string());

    if vocab.len() != OPEN_CLIP_VOCAB_SIZE {
        return Err(JammiError::Model {
            model_id: String::new(),
            message: format!(
                "OpenCLIP vocab construction produced {} tokens, expected {}",
                vocab.len(),
                OPEN_CLIP_VOCAB_SIZE
            ),
        });
    }

    let vocab_map: Vocab = vocab
        .into_iter()
        .enumerate()
        .map(|(i, t)| (t, i as u32))
        .collect();

    let bpe = BPE::builder()
        .vocab_and_merges(vocab_map, merges)
        .end_of_word_suffix("</w>".to_string())
        .build()
        .map_err(|e| JammiError::Model {
            model_id: String::new(),
            message: format!("BPE build failed: {e}"),
        })?;

    let mut tokenizer = Tokenizer::new(bpe);

    // Pre-tokenization: split on the OpenCLIP regex, isolating matched
    // tokens (special tokens, contractions, alphabetic runs, digits,
    // punctuation runs). `Split` with `Isolated` keeps the matched runs as
    // distinct pre-tokens.
    let pre = Split::new(
        SplitPattern::Regex(CLIP_REGEX.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| JammiError::Model {
        model_id: String::new(),
        message: format!("CLIP pre-tokenizer regex failed: {e}"),
    })?;
    tokenizer.with_pre_tokenizer(Some(pre));

    // Post-processing: every sequence is wrapped as
    // `<|startoftext|> ... <|endoftext|>`, matching OpenCLIP's
    // `tokenize()` helper. This guarantees the EOT token is present at
    // `argmax(input_ids, dim=1)` for the EOT-pool used by `ClipText`.
    let sot_id = (OPEN_CLIP_VOCAB_SIZE - 2) as u32;
    let eot_id = (OPEN_CLIP_VOCAB_SIZE - 1) as u32;
    let post = TemplateProcessing::builder()
        .try_single(format!("{SOT_TOKEN}:0 $A:0 {EOT_TOKEN}:0"))
        .map_err(|e| JammiError::Model {
            model_id: String::new(),
            message: format!("CLIP post-processor single template failed: {e}"),
        })?
        .special_tokens(vec![
            (SOT_TOKEN.to_string(), sot_id),
            (EOT_TOKEN.to_string(), eot_id),
        ])
        .build()
        .map_err(|e| JammiError::Model {
            model_id: String::new(),
            message: format!("CLIP post-processor build failed: {e}"),
        })?;
    tokenizer.with_post_processor(Some(post));

    tokenizer.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::BatchLongest,
        pad_id: 0,
        pad_token: byte_chars[0].to_string(),
        ..Default::default()
    }));

    Ok(tokenizer)
}

/// Build the OpenCLIP `bytes_to_unicode` mapping: a reversible bijection
/// from the 256 byte values to printable-unicode codepoints, used as the
/// base alphabet of the BPE vocabulary.
///
/// The mapping reserves codepoints 33..=126 ("!".."~"), 161..=172, and
/// 174..=255 for direct identity-mapped bytes; remaining bytes are mapped
/// to a contiguous tail of codepoints 256..=287 (one extra codepoint per
/// "missing" byte, in ascending byte order).
fn bytes_to_unicode() -> Vec<char> {
    // Codepoints with a direct identity mapping (printable ASCII + Latin-1
    // supplement minus the control ranges).
    let mut bs: Vec<u32> = (33..=126).collect();
    bs.extend(161..=172);
    bs.extend(174..=255);

    let mut cs: Vec<u32> = bs.clone();
    let mut n: u32 = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    // Now bs is a permutation of 0..256 and cs is the corresponding
    // codepoint sequence. Build a 256-entry char vector indexed by byte.
    let mut out: Vec<char> = vec!['\0'; 256];
    for (b, c) in bs.into_iter().zip(cs.into_iter()) {
        out[b as usize] = char::from_u32(c).expect("valid printable codepoint");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_to_unicode_is_bijection() {
        let chars = bytes_to_unicode();
        assert_eq!(chars.len(), 256);
        let unique: std::collections::HashSet<char> = chars.iter().copied().collect();
        assert_eq!(
            unique.len(),
            256,
            "bytes_to_unicode must be a bijection (no duplicate codepoints)"
        );
        // Every output codepoint must be in the printable Latin-1 / extended
        // range used by OpenCLIP. The "extended" tail starts at 256 and
        // covers one codepoint per byte not in the direct-identity ranges
        // (256 - 94 - 12 - 82 = 68 extras).
        let identity_count = (126 - 33 + 1) + (172 - 161 + 1) + (255 - 174 + 1);
        let extra_count = 256 - identity_count;
        let extra_end = 256 + extra_count as u32;
        for c in &chars {
            let cp = *c as u32;
            assert!(
                (33..=126).contains(&cp)
                    || (161..=172).contains(&cp)
                    || (174..=255).contains(&cp)
                    || (256..extra_end).contains(&cp),
                "unexpected codepoint {cp:#x}"
            );
        }
    }
}
