//! Jammi AI — model loading, inference execution, and output adapters.
//!
//! This crate provides the intelligence layer of the Jammi platform:
//! explicit model source resolution, Candle/ORT backends, batch inference
//! with backpressure, and typed output adapters for embedding,
//! classification, summarization, and other ML tasks.

pub mod concurrency;
pub mod inference;
pub mod model;
pub mod operator;
pub mod pipeline;
pub mod session;
