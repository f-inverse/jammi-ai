//! What a test needs from its host, acquired or refused by name.
//!
//! A test that needs something the host may lack — a CUDA or Metal device, a
//! database, a non-root process — is compiled only under the cargo feature that
//! says the host has it (`live-gpu-tests`, `live-postgres-tests`,
//! `unprivileged-tests`, …). Inside such a test, acquisition cannot fail
//! quietly: every function here returns the resource or panics naming what is
//! missing. There is no `Option` to return early on, so a test either runs its
//! assertions or fails.

use std::path::{Path, PathBuf};

/// The value of environment variable `name`.
///
/// # Panics
/// When `name` is unset or empty.
pub fn env(name: &str) -> String {
    match std::env::var(name) {
        Ok(value) if !value.is_empty() => value,
        _ => panic!("this test needs the environment variable {name}; it is unset or empty"),
    }
}

/// The path of executable `name`, resolved against `PATH`.
///
/// # Panics
/// When no directory on `PATH` holds an executable `name`.
pub fn executable(name: &str) -> PathBuf {
    std::env::var_os("PATH")
        .into_iter()
        .flat_map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
        .map(|dir| dir.join(name))
        .find(|candidate| is_executable(candidate))
        .unwrap_or_else(|| panic!("this test needs `{name}` on PATH; none was found"))
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    path.metadata()
        .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

/// Asserts that this process is refused a file its mode bits deny it.
///
/// Root bypasses permission checks, and so does a filesystem that ignores mode
/// bits; a test of how the engine handles an unreadable file means nothing in
/// either case. The check is empirical: a file set to mode `000` must not open.
///
/// # Panics
/// When the file opens anyway, or the probe itself cannot be set up.
#[cfg(unix)]
pub fn assert_permissions_enforced() {
    use std::os::unix::fs::PermissionsExt;
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let probe = std::env::temp_dir().join(format!(
        "jammi-permission-probe-{}-{nanos}",
        std::process::id()
    ));
    std::fs::write(&probe, b"x").expect("write the permission probe file");
    std::fs::set_permissions(&probe, std::fs::Permissions::from_mode(0o000))
        .expect("chmod the permission probe file to 000");
    let opened = std::fs::File::open(&probe).is_ok();
    std::fs::remove_file(&probe).expect("remove the permission probe file");
    assert!(
        !opened,
        "this test needs a process that file permissions apply to, but a mode-000 file \
         opened: run it as a non-root user on a filesystem that enforces mode bits"
    );
}

/// CUDA device `ordinal`.
///
/// # Panics
/// When the device cannot be opened.
#[cfg(feature = "cuda")]
pub fn cuda_device(ordinal: usize) -> candle_core::Device {
    candle_core::Device::new_cuda(ordinal)
        .unwrap_or_else(|e| panic!("this test needs CUDA device {ordinal}: {e}"))
}

/// The Metal device.
///
/// # Panics
/// When no Metal device can be opened. `Device::new_metal` itself panics on a
/// host with no Metal support; that panic is reported the same way.
#[cfg(feature = "metal")]
pub fn metal_device() -> candle_core::Device {
    let opened = std::panic::catch_unwind(|| candle_core::Device::new_metal(0));
    match opened {
        Ok(Ok(device)) => device,
        Ok(Err(e)) => panic!("this test needs a Metal device: {e}"),
        Err(payload) => {
            let reason = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "Device::new_metal panicked".to_owned());
            panic!("this test needs a Metal device: {reason}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "JAMMI_TEST_RESOURCES_SURELY_UNSET")]
    fn env_names_the_missing_variable() {
        env("JAMMI_TEST_RESOURCES_SURELY_UNSET");
    }

    #[test]
    fn env_returns_a_set_variable() {
        // PATH is set for every process that can run cargo.
        assert!(!env("PATH").is_empty());
    }

    #[test]
    #[should_panic(expected = "jammi-no-such-executable")]
    fn executable_names_a_missing_program() {
        executable("jammi-no-such-executable");
    }

    #[cfg(unix)]
    #[test]
    fn executable_resolves_a_program_on_path() {
        assert!(executable("sh").is_absolute());
    }
}
