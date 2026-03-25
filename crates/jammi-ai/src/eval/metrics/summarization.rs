//! Summarization metrics: ROUGE-L based on Longest Common Subsequence.

use serde::{Deserialize, Serialize};

/// ROUGE-L precision, recall, and F1 scores.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RougeLScores {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

pub struct SummarizationMetrics;

impl SummarizationMetrics {
    /// Compute ROUGE-L scores between a generated summary and a reference.
    ///
    /// Normalization: lowercase, replace non-alphanumeric with spaces, split on
    /// whitespace. Matches Google's rouge-score reference implementation.
    pub fn rouge_l(generated: &str, reference: &str) -> RougeLScores {
        let gen_normalized = Self::normalize(generated);
        let ref_normalized = Self::normalize(reference);
        let gen_tokens: Vec<&str> = gen_normalized.split_whitespace().collect();
        let ref_tokens: Vec<&str> = ref_normalized.split_whitespace().collect();

        let lcs_len = Self::lcs_length(&gen_tokens, &ref_tokens);

        let precision = if gen_tokens.is_empty() {
            0.0
        } else {
            lcs_len as f64 / gen_tokens.len() as f64
        };
        let recall = if ref_tokens.is_empty() {
            0.0
        } else {
            lcs_len as f64 / ref_tokens.len() as f64
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        RougeLScores {
            precision,
            recall,
            f1,
        }
    }

    /// Average ROUGE-L scores across multiple pairs.
    pub fn aggregate(scores: &[RougeLScores]) -> RougeLScores {
        let n = scores.len() as f64;
        if n == 0.0 {
            return RougeLScores::default();
        }
        RougeLScores {
            precision: scores.iter().map(|s| s.precision).sum::<f64>() / n,
            recall: scores.iter().map(|s| s.recall).sum::<f64>() / n,
            f1: scores.iter().map(|s| s.f1).sum::<f64>() / n,
        }
    }

    /// Length of the Longest Common Subsequence between two token sequences.
    fn lcs_length(a: &[&str], b: &[&str]) -> usize {
        let m = a.len();
        let n = b.len();
        let mut dp = vec![vec![0usize; n + 1]; m + 1];
        for i in 1..=m {
            for j in 1..=n {
                dp[i][j] = if a[i - 1] == b[j - 1] {
                    dp[i - 1][j - 1] + 1
                } else {
                    dp[i - 1][j].max(dp[i][j - 1])
                };
            }
        }
        dp[m][n]
    }

    /// Normalize text for ROUGE: lowercase, replace non-alphanumeric with spaces.
    fn normalize(text: &str) -> String {
        text.to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { ' ' })
            .collect()
    }
}
