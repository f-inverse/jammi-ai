//! Integration-test entry. Each encoder file is owned by its Phase B subagent;
//! the pooling/mask module is owned by the main session in Phase A.

mod aggregate;
mod batch_composition_invariance;
mod bert;
mod clip_text;
mod distilbert;
mod lora_site_names;
mod modernbert;
mod modernbert_sliding_window;
mod pooling;
