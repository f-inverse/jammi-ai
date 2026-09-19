pub mod gpu_scheduler;
pub use gpu_scheduler::{DeviceSchedulers, ForwardPermit, GpuPermit, GpuPriority, GpuScheduler};

/// Size the process-wide CPU pool — the rayon global pool candle's CPU math and
/// the media front ends run on — to the engine's CPU parallelism budget
/// (`[engine] execution_threads`).
///
/// For a BINARY to call once at startup, before any engine work: the pool is
/// process-global and can be sized only once, so a process that embeds the
/// engine as a library owns it instead. Calling again with the size the pool
/// already has is a no-op; a different size is refused, since the running pool
/// cannot be resized and silently keeping the old size would leave the
/// configured budget unenforced.
#[cfg(feature = "local")]
pub fn init_cpu_pool(threads: std::num::NonZeroUsize) -> jammi_db::error::Result<()> {
    match rayon::ThreadPoolBuilder::new()
        .num_threads(threads.get())
        .thread_name(|index| format!("jammi-cpu-{index}"))
        .build_global()
    {
        Ok(()) => Ok(()),
        Err(_) if rayon::current_num_threads() == threads.get() => Ok(()),
        Err(_) => Err(jammi_db::error::JammiError::Config(format!(
            "[engine] execution_threads = {threads} cannot apply: this process's CPU pool is \
             already running {} threads",
            rayon::current_num_threads()
        ))),
    }
}
