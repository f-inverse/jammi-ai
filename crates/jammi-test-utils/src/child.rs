//! A drained child-process driver — the esc-078 mechanism fix.
//!
//! `jammi-db`'s `esc_073_foreign_sqlite_library.rs` harness (and the sibling
//! `sqlite_single_process_seam.rs`) pipe a child's stdout/stderr with
//! [`std::process::Stdio::piped`] and only read them **after** the child has
//! either exited or been killed at a ceiling, via `wait_with_output`. A pipe's
//! kernel buffer is small (on Linux, ~64 KiB); a child whose `stderr` is
//! chattier than that fills the buffer and parks in `write(2)` until a reader
//! drains it. Since nothing reads while the child runs, a sufficiently
//! chatty-but-otherwise-healthy child is indistinguishable from a genuinely
//! wedged one: both sit past the ceiling and get killed, and because the log
//! is read only on the exit/kill path (never concurrently), the kill path
//! discards it outright — the one outcome the harness's own module doc calls
//! "a FAILURE of the same weight as a crash" is the one outcome that leaves
//! no evidence at all.
//!
//! [`DrainedChild`] fixes the mechanism: [`DrainedChild::spawn`] pipes both
//! streams and starts one reader thread per stream that drains it into an
//! in-memory, retention-capped buffer **while the child runs**, so a chatty
//! child never fills the pipe and a killed child's progress up to the kill is
//! still on hand. [`DrainedChild::wait_bounded`] polls for exit against a
//! ceiling measured from either [`Epoch::Spawn`] or [`Epoch::Call`], and on
//! either the exit path or the kill path performs a short, bounded "settle"
//! (waiting for the reader threads to observe EOF) before snapshotting the
//! buffers into a [`Capture`].
//!
//! # Retention cap
//!
//! Draining removes the pipe's implicit backpressure, so an unbounded buffer
//! would let a fast-erroring, long-lived child (say, 120 s of chatter) yield a
//! multi-gigabyte in-memory log. Each stream is capped independently at
//! [`DEFAULT_HEAD_CAP`] bytes of head plus [`DEFAULT_TAIL_CAP`] bytes of tail
//! (6 MiB + 2 MiB by default — far above anything this module's own tests
//! produce, so the oracle test retains every byte of its 4 MiB flood); bytes
//! in between are dropped and counted in `Capture::stdout_truncated` /
//! `Capture::stderr_truncated`. The raw `Capture::stdout`/`stderr` fields are
//! always exactly the spliced head-then-tail bytes with **no** marker
//! inserted — [`render`] is the only place a
//! `[... N bytes truncated ...]` line gets inserted, at the head/tail seam,
//! and only when the stream's truncated count is nonzero. Comparators and
//! byte-count assertions must run on the raw fields, never on `render`'s
//! output.
//!
//! # Prior art
//!
//! `jammi-cli`'s `tests/it/server_harness.rs` (around its `drain_pipe`
//! helper) already drains a child's pipes on background threads while it
//! runs — the same mechanism this module generalizes — but reads
//! line-by-line through `BufReader::read_line`, which silently truncates a
//! stream at its first invalid-UTF-8 byte. This module works on raw bytes
//! throughout and never decodes, so it has no such failure mode; the two
//! implementations are not consolidated here (that's `jammi-cli`-domain
//! debt, out of scope for this fix).
//!
//! # Precondition for a complete log
//!
//! A drained child's log is complete only if nothing else holds the pipe's
//! write end open. A child that spawns a grandchild without redirecting the
//! grandchild's stdio inherits the pipe, and if the child exits before the
//! grandchild does, the reader thread will not see EOF until the grandchild
//! also closes its end. `wait_bounded`'s settle is itself bounded (about 1 s),
//! so this case is never a hang: `wait_bounded` still returns promptly with
//! `hung == false`, but the returned log can be truncated relative to what
//! the grandchild eventually writes. Callers that spawn fd-inheriting
//! grandchildren must document that residual themselves.
//!
//! # Revert recipe
//!
//! To reproduce the pre-fix (undrained) driver exactly: swap the body of
//! `spawn` for `spawn_undrained`'s, and swap `wait_bounded`'s two
//! drained-finish branches for the undrained-finish branches (no reader
//! threads; the exit path reads via `wait_with_output`; the kill path returns
//! empty buffers without ever reading the pipe). Under that swap,
//! `flood_child_is_drained_and_its_evidence_is_retained` goes RED (the child
//! hangs at the harness's own ceiling instead of finishing). The standing
//! differential `flood_is_a_hang_on_the_undrained_driver` keeps that RED
//! shape alive permanently, without a manual swap, by exercising the
//! undrained constructor directly against the same flood child.

use std::collections::VecDeque;
use std::fmt;
use std::io::{self, Read};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Which instant a [`DrainedChild::wait_bounded`] ceiling (and the returned
/// [`Capture::elapsed`]) is measured from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Epoch {
    /// From the moment [`DrainedChild::spawn`] (or `spawn_undrained`)
    /// returned.
    Spawn,
    /// From the moment `wait_bounded` itself is called.
    Call,
}

/// Default per-stream head-retention cap (see the module doc's "Retention
/// cap" section). Bytes beyond this, up to [`DEFAULT_TAIL_CAP`] bytes from
/// the end, are dropped and counted in `Capture::stdout_truncated` /
/// `Capture::stderr_truncated`.
pub const DEFAULT_HEAD_CAP: usize = 6 * 1024 * 1024;

/// Default per-stream tail-retention cap. See [`DEFAULT_HEAD_CAP`].
pub const DEFAULT_TAIL_CAP: usize = 2 * 1024 * 1024;

fn lock_or_recover<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A stream's retained bytes: the first `head_cap` bytes seen plus the last
/// `tail_cap` bytes seen, with everything in between dropped and counted.
///
/// Every byte pushed goes either into `head` (only while it has not yet
/// reached `head_cap`) or into `rest`, a ring bounded at `tail_cap`. When
/// `total <= head_cap + tail_cap` this never drops anything — `rest` never
/// exceeds its cap, so `head` followed by `rest` is exactly the original
/// stream and `truncated()` is `0`. Only once `total` exceeds the combined
/// cap does `rest` start evicting its oldest bytes, at which point `head`
/// followed by `rest` is exactly "first `head_cap` bytes, last `tail_cap`
/// bytes", with `truncated()` reporting the dropped middle.
struct RetainedBuf {
    head: Vec<u8>,
    rest: VecDeque<u8>,
    head_cap: usize,
    tail_cap: usize,
    total: u64,
}

impl RetainedBuf {
    fn new(head_cap: usize, tail_cap: usize) -> Self {
        Self {
            head: Vec::new(),
            rest: VecDeque::new(),
            head_cap,
            tail_cap,
            total: 0,
        }
    }

    fn push(&mut self, mut data: &[u8]) {
        self.total += data.len() as u64;
        if self.head.len() < self.head_cap {
            let take = (self.head_cap - self.head.len()).min(data.len());
            self.head.extend_from_slice(&data[..take]);
            data = &data[take..];
        }
        if !data.is_empty() {
            self.rest.extend(data.iter().copied());
            let len = self.rest.len();
            if len > self.tail_cap {
                let excess = len - self.tail_cap;
                self.rest.drain(..excess);
            }
        }
    }

    /// Bytes dropped from the middle, `0` while `total <= head_cap + tail_cap`.
    fn truncated(&self) -> u64 {
        self.total
            .saturating_sub((self.head_cap + self.tail_cap) as u64)
    }

    /// The raw spliced buffer: `head` followed by `rest`. Never carries a
    /// truncation marker — see [`render`].
    fn raw(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.head.len() + self.rest.len());
        out.extend_from_slice(&self.head);
        out.extend(self.rest.iter().copied());
        out
    }
}

/// Read `pipe` in an 8 KiB stack buffer until EOF, appending every chunk read
/// into `buf` (locked only for the append) and, when `last_byte` is `Some`,
/// stamping it with an absolute [`Instant`] after every non-empty read.
fn spawn_reader<R: Read + Send + 'static>(
    mut pipe: R,
    buf: Arc<Mutex<RetainedBuf>>,
    last_byte: Option<Arc<Mutex<Option<Instant>>>>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut chunk = [0u8; 8192];
        loop {
            match pipe.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    lock_or_recover(&buf).push(&chunk[..n]);
                    if let Some(lb) = &last_byte {
                        *lock_or_recover(lb) = Some(Instant::now());
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
        }
    })
}

struct Readers {
    stdout: Arc<Mutex<RetainedBuf>>,
    stderr: Arc<Mutex<RetainedBuf>>,
    /// Absolute stamp of the last successful `stderr` `read()`, updated by the
    /// stderr reader thread. Only `stderr` is stamped: `Capture::silence`
    /// exists to answer "how long has this child been quiet on its
    /// diagnostic stream", not to track stdout.
    stderr_last_byte: Arc<Mutex<Option<Instant>>>,
    stdout_handle: JoinHandle<()>,
    stderr_handle: JoinHandle<()>,
}

/// A child process whose stdout/stderr are drained on background threads
/// while it runs. See the module doc for the mechanism and rationale.
pub struct DrainedChild {
    child: Child,
    spawned_at: Instant,
    /// `None` only for the `cfg(test)` undrained constructor (the standing
    /// pre-fix differential); a `DrainedChild` built via [`DrainedChild::spawn`]
    /// always has readers.
    readers: Option<Readers>,
}

impl DrainedChild {
    /// Spawn `cmd` with both stdout and stderr piped and drained on
    /// background reader threads, using [`DEFAULT_HEAD_CAP`] /
    /// [`DEFAULT_TAIL_CAP`] as the per-stream retention cap.
    pub fn spawn(cmd: &mut Command) -> io::Result<Self> {
        Self::spawn_inner(cmd, DEFAULT_HEAD_CAP, DEFAULT_TAIL_CAP)
    }

    /// Test-only: identical to [`DrainedChild::spawn`], but with the
    /// per-stream retention cap injected explicitly. This exists so the
    /// retention cap has its own oracle (`flood_over_cap`) without needing a
    /// multi-hundred-megabyte flood to exercise the default cap.
    #[cfg(test)]
    pub(crate) fn spawn_with_caps(
        cmd: &mut Command,
        head_cap: usize,
        tail_cap: usize,
    ) -> io::Result<Self> {
        Self::spawn_inner(cmd, head_cap, tail_cap)
    }

    fn spawn_inner(cmd: &mut Command, head_cap: usize, tail_cap: usize) -> io::Result<Self> {
        let mut child = cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
        let stdout_pipe = child
            .stdout
            .take()
            .expect("spawn_inner always sets Stdio::piped() for stdout");
        let stderr_pipe = child
            .stderr
            .take()
            .expect("spawn_inner always sets Stdio::piped() for stderr");
        let stdout_buf = Arc::new(Mutex::new(RetainedBuf::new(head_cap, tail_cap)));
        let stderr_buf = Arc::new(Mutex::new(RetainedBuf::new(head_cap, tail_cap)));
        let stderr_last_byte = Arc::new(Mutex::new(None));
        let stdout_handle = spawn_reader(stdout_pipe, Arc::clone(&stdout_buf), None);
        let stderr_handle = spawn_reader(
            stderr_pipe,
            Arc::clone(&stderr_buf),
            Some(Arc::clone(&stderr_last_byte)),
        );
        Ok(Self {
            child,
            spawned_at: Instant::now(),
            readers: Some(Readers {
                stdout: stdout_buf,
                stderr: stderr_buf,
                stderr_last_byte,
                stdout_handle,
                stderr_handle,
            }),
        })
    }

    /// Test-only: the pre-fix shape — both streams are still piped (so a
    /// well-behaved exit can still be read via `wait_with_output`), but
    /// nothing drains them while the child runs, and a kill path never reads
    /// the pipe at all. This is the standing differential that keeps
    /// `flood_is_a_hang_on_the_undrained_driver` demonstrating the exact bug
    /// esc-078 reported, without needing to hand-swap `spawn`'s body.
    #[cfg(test)]
    pub(crate) fn spawn_undrained(cmd: &mut Command) -> io::Result<Self> {
        let child = cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
        Ok(Self {
            child,
            spawned_at: Instant::now(),
            readers: None,
        })
    }

    /// Non-blocking poll for exit, forwarding to
    /// [`std::process::Child::try_wait`].
    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        self.child.try_wait()
    }

    /// A point-in-time copy of the raw spliced `(stdout, stderr)` buffers.
    /// Safe to call concurrently with the reader threads (each stream's lock
    /// is held only for the copy) and before the child has exited.
    pub fn snapshot(&self) -> (Vec<u8>, Vec<u8>) {
        match &self.readers {
            Some(r) => {
                let stdout = lock_or_recover(&r.stdout).raw();
                let stderr = lock_or_recover(&r.stderr).raw();
                (stdout, stderr)
            }
            None => (Vec::new(), Vec::new()),
        }
    }

    /// Wait for the child to exit, bounded by `ceiling` measured from
    /// `epoch`. Polls `try_wait` every 20 ms, checking the deadline **before**
    /// sleeping (so `ceiling = Duration::ZERO` is well-defined: it goes
    /// straight to the kill path on the first iteration without ever
    /// sleeping).
    ///
    /// - On exit: settles (waits up to ~1 s for both reader threads to
    ///   observe EOF), snapshots, and returns with `hung: false`.
    /// - On ceiling: kills the child, reaps by polling `try_wait` for up to
    ///   ~1 s (a `SIGKILL`ed child stuck in uninterruptible sleep can outlive
    ///   this poll — `wait_bounded` still returns, with `Capture::status ==
    ///   None`), then performs the same settle, snapshots, and returns with
    ///   `hung: true`.
    ///
    /// Bounded on both paths: at most `ceiling` plus roughly 2 s (reap +
    /// settle). See the module doc's "Precondition for a complete log"
    /// section for the one case (an fd-inheriting grandchild) where the
    /// settle can expire before the log is complete.
    pub fn wait_bounded(mut self, ceiling: Duration, epoch: Epoch) -> Capture {
        let epoch_base = match epoch {
            Epoch::Spawn => self.spawned_at,
            Epoch::Call => Instant::now(),
        };
        let deadline = epoch_base + ceiling;
        let drained = self.readers.is_some();
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => {
                    return if drained {
                        self.finish_drained(Some(status), false, epoch_base)
                    } else {
                        self.finish_undrained_exit(status, epoch_base)
                    };
                }
                Ok(None) => {
                    if Instant::now() >= deadline {
                        let _ = self.child.kill();
                        let reaped = reap(&mut self.child, Duration::from_secs(1));
                        return if drained {
                            self.finish_drained(reaped, true, epoch_base)
                        } else {
                            self.finish_undrained_hung(reaped, epoch_base)
                        };
                    }
                    thread::sleep(Duration::from_millis(20));
                }
                Err(_) => {
                    return if drained {
                        self.finish_drained(None, false, epoch_base)
                    } else {
                        self.finish_undrained_hung(None, epoch_base)
                    };
                }
            }
        }
    }

    fn finish_drained(
        self,
        status: Option<ExitStatus>,
        hung: bool,
        epoch_base: Instant,
    ) -> Capture {
        let readers = self
            .readers
            .expect("finish_drained is only called when self.readers is Some");
        let Readers {
            stdout,
            stderr,
            stderr_last_byte,
            stdout_handle,
            stderr_handle,
        } = readers;

        let settle_deadline = Instant::now() + Duration::from_secs(1);
        loop {
            if stdout_handle.is_finished() && stderr_handle.is_finished() {
                break;
            }
            if Instant::now() >= settle_deadline {
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
        // Only join a thread that has already observed EOF: joining an
        // unfinished reader (the fd-inheriting-grandchild case) would block
        // past the settle bound above.
        if stdout_handle.is_finished() {
            let _ = stdout_handle.join();
        }
        if stderr_handle.is_finished() {
            let _ = stderr_handle.join();
        }

        let (stdout_bytes, stdout_truncated) = {
            let g = lock_or_recover(&stdout);
            (g.raw(), g.truncated())
        };
        let (stderr_bytes, stderr_truncated) = {
            let g = lock_or_recover(&stderr);
            (g.raw(), g.truncated())
        };
        let last_byte_instant = *lock_or_recover(&stderr_last_byte);
        let returned_at = Instant::now();
        Capture {
            status,
            hung,
            elapsed: returned_at.saturating_duration_since(epoch_base),
            stdout: stdout_bytes,
            stderr: stderr_bytes,
            stdout_truncated,
            stderr_truncated,
            last_byte_instant,
            returned_at,
        }
    }

    /// Undrained exit path: the child is known to have exited already (via
    /// `try_wait`); `wait_with_output` reads whatever remains in the pipes to
    /// completion, matching the pre-fix shape's own only read.
    fn finish_undrained_exit(self, status: ExitStatus, epoch_base: Instant) -> Capture {
        let output = self
            .child
            .wait_with_output()
            .unwrap_or(std::process::Output {
                status,
                stdout: Vec::new(),
                stderr: Vec::new(),
            });
        let returned_at = Instant::now();
        Capture {
            status: Some(output.status),
            hung: false,
            elapsed: returned_at.saturating_duration_since(epoch_base),
            stdout: output.stdout,
            stderr: output.stderr,
            stdout_truncated: 0,
            stderr_truncated: 0,
            last_byte_instant: None,
            returned_at,
        }
    }

    /// Undrained kill path: deliberately does **not** read the pipe (reading
    /// here would return whatever the ~64 KiB pipe still holds and break
    /// fidelity to the pre-fix shape, which discarded the log outright on a
    /// hang) — this is the esc-078 bug, reproduced on demand.
    fn finish_undrained_hung(self, status: Option<ExitStatus>, epoch_base: Instant) -> Capture {
        let returned_at = Instant::now();
        Capture {
            status,
            hung: true,
            elapsed: returned_at.saturating_duration_since(epoch_base),
            stdout: Vec::new(),
            stderr: Vec::new(),
            stdout_truncated: 0,
            stderr_truncated: 0,
            last_byte_instant: None,
            returned_at,
        }
    }
}

/// Poll `try_wait` on an already-killed child for up to `ceiling`, returning
/// the reaped status or `None` if it never reaped within the bound (a
/// documented caveat: a `SIGKILL`ed process in uninterruptible sleep can
/// outlive this poll).
fn reap(child: &mut Child, ceiling: Duration) -> Option<ExitStatus> {
    let deadline = Instant::now() + ceiling;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status),
            Ok(None) => {
                if Instant::now() >= deadline {
                    return None;
                }
                thread::sleep(Duration::from_millis(20));
            }
            Err(_) => return None,
        }
    }
}

/// The outcome of a [`DrainedChild::wait_bounded`] call.
///
/// `stdout`/`stderr` are always the raw spliced buffers (first-`head_cap`
/// bytes followed by last-`tail_cap` bytes, with no marker inserted even when
/// `stdout_truncated`/`stderr_truncated` are nonzero) — see [`render`] for
/// rendering a truncation marker for human consumption.
pub struct Capture {
    /// The child's exit status, or `None` if it was killed and never reaped
    /// within the reap bound (see [`DrainedChild::wait_bounded`]).
    pub status: Option<ExitStatus>,
    /// `true` iff the child was killed at the ceiling rather than exiting on
    /// its own.
    pub hung: bool,
    /// Time from the call's [`Epoch`] to this `Capture` being returned
    /// (post-settle) — not the time of exit detection.
    pub elapsed: Duration,
    /// Raw spliced stdout bytes (see the struct doc).
    pub stdout: Vec<u8>,
    /// Raw spliced stderr bytes (see the struct doc).
    pub stderr: Vec<u8>,
    /// Bytes dropped from the middle of stdout by the retention cap, `0` if
    /// none were dropped.
    pub stdout_truncated: u64,
    /// Bytes dropped from the middle of stderr by the retention cap, `0` if
    /// none were dropped.
    pub stderr_truncated: u64,
    /// Absolute stamp of the last successful `stderr` `read()`, or `None` if
    /// the child never wrote to stderr (or this `Capture` came from the
    /// undrained driver).
    pub last_byte_instant: Option<Instant>,
    /// Absolute stamp of when this `Capture` was constructed (post-settle).
    pub returned_at: Instant,
}

impl Capture {
    /// How long stderr has been quiet as of `returned_at`: `returned_at -
    /// last_byte_instant`, or `None` if stderr never produced a byte.
    /// Computed from two absolute [`Instant`]s, so it stays correct
    /// regardless of which [`Epoch`] the call used.
    pub fn silence(&self) -> Option<Duration> {
        self.last_byte_instant
            .map(|t| self.returned_at.saturating_duration_since(t))
    }
}

impl fmt::Debug for Capture {
    /// Hand-written: reports lengths and a short lossy-UTF-8 tail per stream,
    /// never the full buffers, and reports `silence()`/`elapsed` rather than
    /// the raw `Instant` fields (an `Instant` prints as an opaque, useless
    /// value; the durations derived from it are what a panic message needs).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const DEBUG_TAIL: usize = 256;
        fn tail_lossy(buf: &[u8], n: usize) -> String {
            let start = buf.len().saturating_sub(n);
            String::from_utf8_lossy(&buf[start..]).into_owned()
        }
        f.debug_struct("Capture")
            .field("status", &self.status)
            .field("hung", &self.hung)
            .field("elapsed", &self.elapsed)
            .field("silence", &self.silence())
            .field("stdout_len", &self.stdout.len())
            .field("stdout_truncated", &self.stdout_truncated)
            .field("stdout_tail", &tail_lossy(&self.stdout, DEBUG_TAIL))
            .field("stderr_len", &self.stderr.len())
            .field("stderr_truncated", &self.stderr_truncated)
            .field("stderr_tail", &tail_lossy(&self.stderr, DEBUG_TAIL))
            .finish()
    }
}

/// Render raw captured bytes as lossy UTF-8 for a panic message or test
/// failure, inserting exactly one `[... N bytes truncated ...]` line at the
/// head/tail seam when `truncated > 0`, and nothing when it is `0`.
/// `head_cap` must be the head-retention cap the capturing `DrainedChild` was
/// built with ([`DEFAULT_HEAD_CAP`] unless a `cfg(test)`-injected cap was
/// used). Comparators and byte-count assertions must run on `Capture::stdout`
/// / `Capture::stderr` directly — never on this rendered form.
pub fn render(buf: &[u8], truncated: u64, head_cap: usize) -> String {
    if truncated == 0 {
        return String::from_utf8_lossy(buf).into_owned();
    }
    let seam = head_cap.min(buf.len());
    let mut out = String::from_utf8_lossy(&buf[..seam]).into_owned();
    out.push_str(&format!("\n[... {truncated} bytes truncated ...]\n"));
    out.push_str(&String::from_utf8_lossy(&buf[seam..]));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::io::Write;

    /// Env var a spawning test sets on its self-exec'd child; read back as
    /// the very first statement of every spawning test (the dispatch guard).
    const CHILD_MODE_ENV: &str = "JAMMI_CHILD_MODE";
    /// Env var carrying the spawning test's own `--exact` path, so a child
    /// running in `"grandchild"` mode can reuse it for its own grandchild
    /// spawn (with `CHILD_MODE_ENV` overridden to `"sleeper"`).
    const SELF_EXACT_ENV: &str = "JAMMI_SELF_EXACT";

    /// Build a `Command` that re-execs this very test binary, selecting only
    /// `test_path` (a fully-qualified `module::path::of::the_test_fn`) and
    /// running it uncaptured on a single thread — both flags are load-bearing
    /// (measured, plan round 5): without `--nocapture`, libtest swallows the
    /// child's `eprintln!` output and the parent observes zero bytes (a
    /// vacuous oracle); without `--exact`, every test whose name is a prefix
    /// match would enter its own dispatch guard concurrently, producing
    /// nondeterministic byte counts.
    fn self_exec(test_path: &str, mode: &str) -> Command {
        let mut cmd = Command::new(env::current_exe().expect("current_exe for self-exec"));
        cmd.args(["--exact", test_path, "--nocapture", "--test-threads=1"]);
        cmd.env(CHILD_MODE_ENV, mode);
        cmd.env(SELF_EXACT_ENV, test_path);
        cmd
    }

    /// One 64-byte flood line: `flood <012-digit n>` right-padded with spaces
    /// to 63 bytes, plus a trailing newline.
    fn flood_line(n: u32) -> [u8; 64] {
        let mut line = [b' '; 64];
        let text = format!("flood {n:012}");
        line[..text.len()].copy_from_slice(text.as_bytes());
        line[63] = b'\n';
        line
    }

    /// Independently reconstructs the exact byte stream `flood_child` writes
    /// for `lines` lines, used by the parent test as the oracle for
    /// head/tail retention rather than trusting the child's own write.
    fn expected_flood_stream(lines: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(lines as usize * 64 + 32);
        for n in 0..lines {
            out.extend_from_slice(&flood_line(n));
        }
        out.extend_from_slice(flood_sentinel(lines).as_bytes());
        out
    }

    fn flood_sentinel(lines: u32) -> String {
        format!("FLOOD-DONE bytes={}\n", lines as u64 * 64)
    }

    /// Dispatches a self-exec'd child by `JAMMI_CHILD_MODE`. Never returns.
    fn run_child(mode: &str) -> ! {
        match mode {
            "flood" => flood_child(),
            "wedge" => wedge_child(),
            "silent" => std::process::exit(0),
            "grandchild" => grandchild_child(),
            "sleeper" => sleeper_child(),
            other => panic!("child.rs test harness: unknown {CHILD_MODE_ENV}={other:?}"),
        }
    }

    /// Writes `JAMMI_FLOOD_LINES` (default 65536 => 4 MiB) fixed-width lines
    /// to stderr, then a `FLOOD-DONE bytes=<n>` sentinel, and exits 0.
    fn flood_child() -> ! {
        let lines: u32 = env::var("JAMMI_FLOOD_LINES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(65536);
        let mut out = io::stderr();
        for n in 0..lines {
            out.write_all(&flood_line(n)).expect("write flood line");
        }
        out.write_all(flood_sentinel(lines).as_bytes())
            .expect("write flood sentinel");
        out.flush().expect("flush flood stderr");
        std::process::exit(0);
    }

    /// Prints two phase markers then loops forever — a child that never
    /// exits on its own and must be killed at the ceiling.
    fn wedge_child() -> ! {
        eprintln!("phase: a");
        eprintln!("phase: b");
        loop {
            thread::sleep(Duration::from_secs(1));
        }
    }

    /// Spawns `current_exe()` in `"sleeper"` mode, reusing this test's own
    /// `--exact` path (`SELF_EXACT_ENV`, set by `self_exec` on this very
    /// process) but overriding `CHILD_MODE_ENV` to `"sleeper"` — reusing the
    /// inherited mode would fork-bomb. The grandchild is spawned without
    /// stdio redirection, so it inherits this process's stderr (the pipe the
    /// grandparent `DrainedChild` is draining). Exits 0 immediately,
    /// deliberately not waiting on the grandchild.
    fn grandchild_child() -> ! {
        let exact = env::var(SELF_EXACT_ENV)
            .expect("grandchild mode requires SELF_EXACT_ENV set by self_exec");
        let mut cmd = Command::new(env::current_exe().expect("current_exe for grandchild spawn"));
        cmd.args(["--exact", &exact, "--nocapture", "--test-threads=1"]);
        cmd.env(CHILD_MODE_ENV, "sleeper");
        let _child = cmd.spawn().expect("spawn sleeper grandchild");
        std::process::exit(0);
    }

    /// Sleeps 3 s (holding whatever stdio it inherited open) then exits 0.
    fn sleeper_child() -> ! {
        thread::sleep(Duration::from_secs(3));
        std::process::exit(0);
    }

    /// Builds the `--exact` path libtest expects for `name`, a `#[test]` fn
    /// defined directly in this module. `module_path!()` (expanded here, in
    /// this module) includes the crate name as its first segment
    /// (`jammi_test_utils::child::tests`), but libtest's own test
    /// identifiers — what `--exact`/`--list` actually match against — omit
    /// it (`child::tests::<name>`), so that leading segment is stripped.
    fn test_exact_path(name: &str) -> String {
        let full = module_path!();
        let stripped = full.split_once("::").map_or(full, |(_, rest)| rest);
        format!("{stripped}::{name}")
    }

    /// closes_escape: esc-078
    ///
    /// The oracle for the fix itself: a child that writes exactly 4 MiB
    /// (65536 fixed-width lines) to stderr, well past the ~64 KiB an
    /// undrained pipe can hold, must finish and have every byte retained.
    /// Pre-fix (see the module doc's "Revert recipe"), this test hangs at the
    /// harness's own ceiling and loses the log entirely — the exact esc-078
    /// symptom.
    #[test]
    fn flood_child_is_drained_and_its_evidence_is_retained() {
        if let Ok(mode) = env::var(CHILD_MODE_ENV) {
            run_child(&mode);
        }
        let exact = test_exact_path("flood_child_is_drained_and_its_evidence_is_retained");
        let mut cmd = self_exec(&exact, "flood");
        let child = DrainedChild::spawn(&mut cmd).expect("spawn flood child");
        let cap = child.wait_bounded(Duration::from_secs(30), Epoch::Spawn);

        assert!(!cap.hung, "flood child should not hang: {cap:?}");
        assert_eq!(
            cap.status.and_then(|s| s.code()),
            Some(0),
            "flood child should exit 0: {cap:?}"
        );
        let sentinel = flood_sentinel(65536);
        assert_eq!(
            cap.stderr.len(),
            4_194_304 + sentinel.len(),
            "stderr byte count: {cap:?}"
        );
        assert!(
            cap.stderr.ends_with(sentinel.as_bytes()),
            "stderr should end with the sentinel: {cap:?}"
        );
        assert_eq!(cap.stderr_truncated, 0, "well under the cap: {cap:?}");
        assert!(
            cap.elapsed < Duration::from_secs(10),
            "elapsed={:?}: {cap:?}",
            cap.elapsed
        );
        eprintln!(
            "flood_child_is_drained_and_its_evidence_is_retained: measured elapsed={:?}",
            cap.elapsed
        );
    }

    /// Standing differential for esc-078 (kept live, not hand-run): replays
    /// the exact flood child above through [`DrainedChild::spawn_undrained`]
    /// (the pre-fix, undrained shape) at a 5 s ceiling. The undrained driver
    /// never reads the pipe while the child runs, so the 4 MiB flood parks
    /// the child in `write(2)` and the ceiling kills it before it can
    /// finish — `hung == true` with an **empty** stderr, reproducing exactly
    /// what esc-078 reported the pre-fix harness losing. If this test ever
    /// stops hanging (e.g. on a platform with a much larger pipe buffer), it
    /// is this test that reds, marking the oracle above as vacuous rather
    /// than silently losing coverage.
    #[test]
    fn flood_is_a_hang_on_the_undrained_driver() {
        if let Ok(mode) = env::var(CHILD_MODE_ENV) {
            run_child(&mode);
        }
        let exact = test_exact_path("flood_is_a_hang_on_the_undrained_driver");
        let mut cmd = self_exec(&exact, "flood");
        let child = DrainedChild::spawn_undrained(&mut cmd).expect("spawn undrained flood child");
        let cap = child.wait_bounded(Duration::from_secs(5), Epoch::Spawn);

        assert!(
            cap.hung,
            "expected the undrained driver to hang on a 4 MiB flood: {cap:?}"
        );
        assert!(
            cap.stderr.is_empty(),
            "undrained kill path must not read the pipe: {cap:?}"
        );
    }

    /// esc-078 control (1): a child killed at the ceiling must still yield
    /// its progress (`phase: b`) and an accurate `silence()`. Its undrained
    /// twin demonstrates the same kill path losing the log regardless of
    /// whether the pipe was ever full — the pre-fix bug was in the kill path
    /// itself, not only in pipe backpressure.
    #[test]
    fn wedged_child_is_killed_at_the_ceiling_with_its_evidence() {
        if let Ok(mode) = env::var(CHILD_MODE_ENV) {
            run_child(&mode);
        }
        let exact = test_exact_path("wedged_child_is_killed_at_the_ceiling_with_its_evidence");

        let mut cmd = self_exec(&exact, "wedge");
        let child = DrainedChild::spawn(&mut cmd).expect("spawn wedge child");
        let cap = child.wait_bounded(Duration::from_secs(5), Epoch::Spawn);

        assert!(cap.hung, "wedge child never exits: {cap:?}");
        assert!(
            cap.elapsed >= Duration::from_secs(5),
            "elapsed={:?}: {cap:?}",
            cap.elapsed
        );
        let text = String::from_utf8_lossy(&cap.stderr);
        assert!(text.contains("phase: b"), "stderr={text:?}: {cap:?}");
        let silence = cap
            .silence()
            .expect("stderr produced bytes, so silence() must be Some");
        assert!(
            silence >= Duration::from_secs(4),
            "silence={silence:?}: {cap:?}"
        );

        let mut undrained_cmd = self_exec(&exact, "wedge");
        let undrained_child =
            DrainedChild::spawn_undrained(&mut undrained_cmd).expect("spawn undrained wedge child");
        let undrained_cap = undrained_child.wait_bounded(Duration::from_secs(5), Epoch::Spawn);
        assert!(undrained_cap.hung, "{undrained_cap:?}");
        assert!(
            undrained_cap.stderr.is_empty(),
            "undrained kill path must not read the pipe: {undrained_cap:?}"
        );
    }

    /// Driver-level control (5): a no-output, exit-0 child is not a hang, and
    /// carries no stale silence signal.
    #[test]
    fn silent_child_is_not_a_hang() {
        if let Ok(mode) = env::var(CHILD_MODE_ENV) {
            run_child(&mode);
        }
        let exact = test_exact_path("silent_child_is_not_a_hang");
        let mut cmd = self_exec(&exact, "silent");
        let child = DrainedChild::spawn(&mut cmd).expect("spawn silent child");
        let cap = child.wait_bounded(Duration::from_secs(10), Epoch::Spawn);

        assert!(!cap.hung, "{cap:?}");
        assert_eq!(cap.status.and_then(|s| s.code()), Some(0), "{cap:?}");
        assert!(cap.stderr.is_empty(), "{cap:?}");
        assert!(cap.last_byte_instant.is_none(), "{cap:?}");
        assert!(cap.silence().is_none(), "{cap:?}");
    }

    /// The settle is bounded even when a grandchild inherits the pipe: the
    /// child here spawns a `"sleeper"` grandchild (which inherits stderr and
    /// sleeps 3 s) and exits immediately, so `wait_bounded` must return
    /// within about a second of the *child's* exit, not the grandchild's —
    /// with `hung == false`, and nothing left running once the grandchild
    /// self-terminates.
    #[test]
    fn settle_returns_within_bound_when_a_grandchild_holds_the_pipe() {
        if let Ok(mode) = env::var(CHILD_MODE_ENV) {
            run_child(&mode);
        }
        let exact = test_exact_path("settle_returns_within_bound_when_a_grandchild_holds_the_pipe");
        let mut cmd = self_exec(&exact, "grandchild");
        let started = Instant::now();
        let child = DrainedChild::spawn(&mut cmd).expect("spawn grandchild-spawning child");
        let cap = child.wait_bounded(Duration::from_secs(10), Epoch::Spawn);
        let wall = started.elapsed();

        assert!(!cap.hung, "{cap:?}");
        assert!(
            wall < Duration::from_secs(2),
            "settle should return within ~1s of the child's near-immediate exit, got {wall:?}: {cap:?}"
        );
        eprintln!(
            "settle_returns_within_bound_when_a_grandchild_holds_the_pipe: measured wall={wall:?} cap.elapsed={:?}",
            cap.elapsed
        );
    }

    /// The retention cap's own oracle: with the cap injected to head 1 MiB +
    /// tail 256 KiB, a 2 MiB flood must retain exactly the first 1 MiB and
    /// exactly the last 256 KiB, report the exact dropped middle, and render
    /// exactly one truncation marker; a flood well under the combined cap
    /// must render no marker at all.
    #[test]
    fn flood_over_cap() {
        if let Ok(mode) = env::var(CHILD_MODE_ENV) {
            run_child(&mode);
        }
        let exact = test_exact_path("flood_over_cap");

        const HEAD_CAP: usize = 1024 * 1024; // 1 MiB
        const TAIL_CAP: usize = 256 * 1024; // 256 KiB
        const OVER_CAP_LINES: u32 = 32768; // 32768 * 64 = 2 MiB
        const SUB_CAP_LINES: u32 = 4096; // 4096 * 64 = 256 KiB, well under 1.25 MiB

        // Over cap: the 2 MiB stream exceeds the 1.25 MiB combined cap.
        let mut cmd = self_exec(&exact, "flood");
        cmd.env("JAMMI_FLOOD_LINES", OVER_CAP_LINES.to_string());
        let child = DrainedChild::spawn_with_caps(&mut cmd, HEAD_CAP, TAIL_CAP)
            .expect("spawn over-cap flood child");
        let cap = child.wait_bounded(Duration::from_secs(30), Epoch::Spawn);
        assert!(!cap.hung, "{cap:?}");
        assert_eq!(cap.status.and_then(|s| s.code()), Some(0), "{cap:?}");

        let full = expected_flood_stream(OVER_CAP_LINES);
        let expected_truncated = full.len() as u64 - (HEAD_CAP + TAIL_CAP) as u64;
        assert_eq!(cap.stderr_truncated, expected_truncated, "{cap:?}");
        assert_eq!(cap.stderr.len(), HEAD_CAP + TAIL_CAP, "{cap:?}");
        assert_eq!(
            &cap.stderr[..HEAD_CAP],
            &full[..HEAD_CAP],
            "retained head must equal the exact first 1 MiB"
        );
        assert_eq!(
            &cap.stderr[HEAD_CAP..],
            &full[full.len() - TAIL_CAP..],
            "retained tail must equal the exact last 256 KiB"
        );
        assert!(
            cap.stderr
                .ends_with(flood_sentinel(OVER_CAP_LINES).as_bytes()),
            "the sentinel must survive in the tail: {cap:?}"
        );

        let rendered = render(&cap.stderr, cap.stderr_truncated, HEAD_CAP);
        let marker = format!("[... {expected_truncated} bytes truncated ...]");
        assert_eq!(
            rendered.matches("bytes truncated").count(),
            1,
            "exactly one marker: {rendered}"
        );
        assert!(
            rendered.contains(&marker),
            "marker with the right N: {rendered}"
        );

        // Sub cap: well under the 1.25 MiB combined cap renders no marker.
        let mut sub_cmd = self_exec(&exact, "flood");
        sub_cmd.env("JAMMI_FLOOD_LINES", SUB_CAP_LINES.to_string());
        let sub_child = DrainedChild::spawn_with_caps(&mut sub_cmd, HEAD_CAP, TAIL_CAP)
            .expect("spawn sub-cap flood child");
        let sub_cap = sub_child.wait_bounded(Duration::from_secs(30), Epoch::Spawn);
        assert!(!sub_cap.hung, "{sub_cap:?}");
        assert_eq!(sub_cap.stderr_truncated, 0, "{sub_cap:?}");
        let sub_expected = expected_flood_stream(SUB_CAP_LINES);
        assert_eq!(sub_cap.stderr, sub_expected, "{sub_cap:?}");
        let sub_rendered = render(&sub_cap.stderr, sub_cap.stderr_truncated, HEAD_CAP);
        assert!(
            !sub_rendered.contains("truncated"),
            "sub-cap flood must render no marker: {sub_rendered}"
        );
    }
}
