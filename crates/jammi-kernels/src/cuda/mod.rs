//! CUDA glue, compiled only when the `cuda` feature is active. `build.rs`
//! compiles `axpy.cu` to PTX at `OUT_DIR/axpy.ptx`; embedded here via
//! `include_str!` and loaded at runtime through candle's public
//! `CudaDevice::get_or_load_custom_func`.

pub(crate) mod axpy;

pub(crate) const PTX_AXPY: &str = include_str!(concat!(env!("OUT_DIR"), "/axpy.ptx"));
