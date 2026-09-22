//! Host physical-memory reader for `[engine] memory_limit`'s `"<n>%"` form.
//!
//! [`EngineConfig::memory_limit_bytes`](super::EngineConfig::memory_limit_bytes)
//! is the ONE reader of `[engine] memory_limit`; when the configured value is
//! a percentage, it resolves against [`total_physical_memory_bytes`] here —
//! the ONE place the engine asks the OS "how much memory is there". An
//! unreadable host is a typed [`JammiError::Config`], never a silent default
//! (e.g. `0`, or some made-up byte count) that could size a
//! [`crate::memory_pool::ActiveSpillPool`] wrong without
//! anyone knowing.
//!
//! # What "total" means
//!
//! The lower of two numbers: the host's total physical memory (Linux:
//! `/proc/meminfo`'s `MemTotal`; macOS: `sysctlbyname("hw.memsize")`), and,
//! when this process runs inside a Linux cgroup with an actual ceiling set
//! (v2 `/sys/fs/cgroup/memory.max`, falling back to v1's
//! `/sys/fs/cgroup/memory/memory.limit_in_bytes`), that ceiling. A
//! containerized deployment's `"75%"` must mean 75% of what the container can
//! actually use, not 75% of the bare hardware total it can never reach — the
//! same reasoning [`crate::config::WorkerConfig::topology`]'s device-bound
//! validation applies to `local_ranks`, applied here to memory instead.
//!
//! No cgroup file, an unreadable one, or the v2 sentinel `"max"` (explicitly
//! "no ceiling") all resolve to "no cgroup bound" — the host total stands
//! alone. A cgroup v1 host with no limit set carries an enormous sentinel
//! value instead of a keyword; that is handled for free by taking the
//! minimum against the host total, which is always smaller.

use crate::error::{JammiError, Result};

/// The byte bound `[engine] memory_limit`'s `"<n>%"` form resolves against.
pub fn total_physical_memory_bytes() -> Result<u64> {
    let host = platform_total_memory_bytes()?;
    match cgroup_memory_limit_bytes() {
        Some(limit) if limit < host => Ok(limit),
        _ => Ok(host),
    }
}

#[cfg(target_os = "linux")]
fn platform_total_memory_bytes() -> Result<u64> {
    let contents = std::fs::read_to_string("/proc/meminfo").map_err(|e| {
        JammiError::Config(format!(
            "[engine] memory_limit: could not read /proc/meminfo to resolve a percentage: {e}"
        ))
    })?;
    contents
        .lines()
        .find_map(|l| l.strip_prefix("MemTotal:"))
        .and_then(|rest| rest.trim().strip_suffix("kB"))
        .and_then(|kb| kb.trim().parse::<u64>().ok())
        .map(|kb| kb.saturating_mul(1024))
        .ok_or_else(|| {
            JammiError::Config(
                "[engine] memory_limit: /proc/meminfo carries no parseable 'MemTotal' line"
                    .to_string(),
            )
        })
}

#[cfg(target_os = "linux")]
fn cgroup_memory_limit_bytes() -> Option<u64> {
    if let Ok(s) = std::fs::read_to_string("/sys/fs/cgroup/memory.max") {
        let s = s.trim();
        return if s == "max" {
            None
        } else {
            s.parse::<u64>().ok()
        };
    }
    std::fs::read_to_string("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
}

#[cfg(target_os = "macos")]
fn platform_total_memory_bytes() -> Result<u64> {
    let mut size: u64 = 0;
    let mut len = std::mem::size_of::<u64>() as libc::size_t;
    let name = std::ffi::CString::new("hw.memsize").expect("no interior NUL in a literal");
    // SAFETY: `oldp` points at a live `u64` and `oldlenp` at its true size;
    // `sysctlbyname` writes at most `len` bytes back through `oldp` and
    // updates `oldlenp` to the number written, both preconditions this call
    // satisfies. `newp`/`newlen` are null/0 — a read, never a write to the
    // MIB.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &mut size as *mut u64 as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return Err(JammiError::Config(format!(
            "[engine] memory_limit: sysctlbyname(\"hw.memsize\") failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok(size)
}

#[cfg(target_os = "macos")]
fn cgroup_memory_limit_bytes() -> Option<u64> {
    None
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn platform_total_memory_bytes() -> Result<u64> {
    Err(JammiError::Config(
        "[engine] memory_limit: host physical memory is not readable on this platform (only \
         linux and macos are supported for the '<n>%' form) -- use an absolute '<n>GB'/'<n>MB'/\
         '<n>KB'/'<n>' byte form instead"
            .to_string(),
    ))
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn cgroup_memory_limit_bytes() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one oracle every platform can run without root or a container: the
    /// host reader returns SOME positive number and never silently zero — a
    /// zero total would make every non-degenerate percentage resolve to a
    /// pool the floor then always refuses, hiding a broken reader behind a
    /// config error instead of surfacing the reader's own failure.
    #[test]
    fn total_physical_memory_bytes_is_positive_on_this_platform() {
        let total = total_physical_memory_bytes().expect("host memory must be readable in CI");
        assert!(total > 0, "a real host reports nonzero physical memory");
    }
}
