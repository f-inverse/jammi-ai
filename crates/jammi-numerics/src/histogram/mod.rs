//! Histogram building blocks shared by divergence kernels: binning, smoothing,
//! padded-range computation, and 1-D interpolation.

pub mod binning;
pub mod interpolate;

pub use binning::{bin_proportions, padded_range, smooth_and_renormalise, NUM_BINS, RANGE_PADDING};
pub use interpolate::interpolate_to;
