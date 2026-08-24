//! CUDA glue, compiled only when the `cuda` feature is active. `build.rs`
//! compiles `axpy.cu` to PTX at `OUT_DIR/axpy.ptx`; embedded here via
//! `include_str!` and loaded at runtime through candle's public
//! `CudaDevice::get_or_load_custom_func`.

pub(crate) mod axpy;
pub(crate) mod geglu;
pub(crate) mod layer_norm;
pub(crate) mod rope;
pub(crate) mod softmax;

pub(crate) const PTX_AXPY: &str = include_str!(concat!(env!("OUT_DIR"), "/axpy.ptx"));
pub(crate) const PTX_GEGLU: &str = include_str!(concat!(env!("OUT_DIR"), "/geglu.ptx"));
pub(crate) const PTX_LAYER_NORM: &str = include_str!(concat!(env!("OUT_DIR"), "/layer_norm.ptx"));
pub(crate) const PTX_ROPE: &str = include_str!(concat!(env!("OUT_DIR"), "/rope.ptx"));
pub(crate) const PTX_SOFTMAX: &str = include_str!(concat!(env!("OUT_DIR"), "/softmax.ptx"));
