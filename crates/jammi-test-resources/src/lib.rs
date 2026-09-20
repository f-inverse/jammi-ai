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

/// A command that runs test `path` (`module::test_name`) of the current test
/// binary, alone, in a fresh process.
///
/// A test that must run in its own process (a clean environment, a process-wide
/// setting read once, a crash to survive) is split in two: the parent test that
/// spawns and checks it, and the child body, marked
/// `#[ignore = "child process of <parent>"]` so the harness reports it as ignored
/// rather than running it in place. This runs exactly that one test with
/// `--ignored --exact`; add the child's environment to the returned command.
pub fn child_test(path: &str) -> std::process::Command {
    let exe = std::env::current_exe().expect("the running test binary's path");
    let mut cmd = std::process::Command::new(exe);
    cmd.args([path, "--exact", "--ignored", "--nocapture"]);
    cmd
}

/// Runs `cmd` (from [`child_test`]) to completion and returns its stdout.
///
/// # Panics
/// When the child fails, or when it did not run exactly one test: a filter that
/// matches nothing exits successfully having run nothing.
pub fn child_test_stdout(cmd: &mut std::process::Command) -> String {
    let output = cmd.output().expect("spawn the child test process");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "the child test failed ({}):\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    assert!(
        stdout.contains("test result: ok. 1 passed;"),
        "the child test process did not run exactly one test:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    stdout
}

/// CUDA device `ordinal`.
///
/// # Panics
/// When the device cannot be opened — including in a build without candle's
/// CUDA backend.
#[cfg(feature = "candle")]
pub fn cuda_device(ordinal: usize) -> candle_core::Device {
    candle_core::Device::new_cuda(ordinal)
        .unwrap_or_else(|e| panic!("this test needs CUDA device {ordinal}: {e}"))
}

/// CUDA device `ordinal`'s backend handle, for code that launches kernels on it
/// directly.
///
/// # Panics
/// As [`cuda_device`].
#[cfg(feature = "candle")]
pub fn cuda_backend(ordinal: usize) -> candle_core::CudaDevice {
    cuda_device(ordinal)
        .as_cuda_device()
        .unwrap_or_else(|e| panic!("CUDA device {ordinal} has no CUDA backend: {e}"))
        .clone()
}

/// The Metal device.
///
/// # Panics
/// When no Metal device can be opened — including in a build without candle's
/// Metal backend. `Device::new_metal` itself panics on some hosts with no Metal
/// support; that panic is reported the same way.
#[cfg(feature = "candle")]
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

    #[test]
    #[ignore = "child process of child_test_runs_exactly_the_named_child"]
    fn child_body_prints_a_marker() {
        println!("CHILD_MARKER={}", std::env::var("CHILD_INPUT").unwrap());
    }

    #[test]
    fn child_test_runs_exactly_the_named_child() {
        let mut cmd = child_test("tests::child_body_prints_a_marker");
        cmd.env("CHILD_INPUT", "from-the-parent");
        assert!(child_test_stdout(&mut cmd).contains("CHILD_MARKER=from-the-parent"));
    }

    #[test]
    #[should_panic(expected = "did not run exactly one test")]
    fn child_test_refuses_a_filter_that_matches_nothing() {
        child_test_stdout(&mut child_test("tests::no_such_test"));
    }
}
