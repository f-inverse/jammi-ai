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
//! inserted — [`render`] (or the safer `Capture::render_stdout`/
//! `render_stderr`, which cannot be called with the wrong `head_cap`) is the
//! only place a `[... N bytes truncated ...]` line gets inserted, at the
//! head/tail seam, and only when the stream's truncated count is nonzero.
//! Comparators and byte-count assertions must run on the raw fields, never on
//! the rendered output.
//!
//! # `hung` vs `killed` vs `complete`
//!
//! These are three separate axes, not one:
//!
//! - `Capture::killed` — did `wait_bounded` ever call `kill()`? True whenever
//!   the ceiling was reached (or an OS error forced a defensive kill),
//!   independent of whether the child turns out to have needed it.
//! - `Capture::hung` — computed from the reaped status by the pure
//!   `disposition` function: a signal death, or a reap give-up (`status:
//!   None` after a kill), is `true`; a normal exit code is `false` **even
//!   when `killed` is also `true`** — the ceiling and a fast, self-
//!   terminating exit can race, and a child that finishes on its own in that
//!   window is not a hang just because the kill was already in flight.
//! - `Capture::complete` ([`Completeness`]) — did the drained reader threads
//!   actually reach EOF before the ~1 s settle bound expired? A `Capture` can
//!   be `hung: false` and still `SettleExpired` (an fd-inheriting grandchild
//!   holding the pipe past the settle bound; see "Precondition for a
//!   complete log"), or `hung: true` and still `Completeness::Complete` (a
//!   killed child whose own readers finish the instant its pipe closes).
//!
//! Any OS-level error encountered while producing a `Capture` (`try_wait`,
//! `kill`, the reap poll, or the undrained driver's `wait_with_output`) is
//! recorded in `Capture::wait_error` rather than silently collapsing into a
//! clean-looking result.
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
//! `hung == false`, but `Capture::complete` reports
//! `Completeness::SettleExpired` and the returned log can be truncated
//! relative to what the grandchild eventually writes. Callers that spawn
//! fd-inheriting grandchildren must document that residual themselves.
//!
//! # Revert recipe
//!
//! To reproduce the pre-fix (undrained) driver exactly: swap the body of
//! `spawn` for `spawn_undrained`'s (`readers: None`). `wait_bounded` already
//! dispatches internally between the drained and undrained finish paths
//! based on whether readers were installed, so that one swap is sufficient —
//! nothing else needs to change. Under it,
//! `flood_child_is_drained_and_its_evidence_is_retained` goes RED (the child
//! hangs at the harness's own ceiling instead of finishing, with its log
//! discarded). The standing differential `flood_is_a_hang_on_the_undrained_driver`
//! keeps that RED shape alive permanently, without a manual swap, by
//! exercising the undrained constructor directly against the same flood
//! child.

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

/// How a reader thread's loop ended, recorded into a shared slot by the
/// thread itself right before it returns (never inferred after the fact from
/// silence). Used to compute [`Completeness`].
#[derive(Debug, Clone)]
enum ReaderOutcome {
    /// `read()` returned `Ok(0)`: the pipe's last writer closed it.
    Eof,
    /// `read()` returned a non-interrupted error; the loop broke without
    /// reaching EOF.
    Error(String),
}

/// Read `pipe` in an 8 KiB stack buffer until EOF, appending every chunk read
/// into `buf` (locked only for the append) and, when `last_byte` is `Some`,
/// stamping it with an absolute [`Instant`] after every non-empty read.
/// Records its terminal reason into `outcome` (never a silent break) so
/// [`Completeness`] can distinguish a clean EOF from a read error.
fn spawn_reader<R: Read + Send + 'static>(
    mut pipe: R,
    buf: Arc<Mutex<RetainedBuf>>,
    last_byte: Option<Arc<Mutex<Option<Instant>>>>,
    outcome: Arc<Mutex<Option<ReaderOutcome>>>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut chunk = [0u8; 8192];
        loop {
            match pipe.read(&mut chunk) {
                Ok(0) => {
                    *lock_or_recover(&outcome) = Some(ReaderOutcome::Eof);
                    break;
                }
                Ok(n) => {
                    lock_or_recover(&buf).push(&chunk[..n]);
                    if let Some(lb) = &last_byte {
                        *lock_or_recover(lb) = Some(Instant::now());
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    *lock_or_recover(&outcome) = Some(ReaderOutcome::Error(e.to_string()));
                    break;
                }
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
    stdout_outcome: Arc<Mutex<Option<ReaderOutcome>>>,
    stderr_outcome: Arc<Mutex<Option<ReaderOutcome>>>,
    stdout_handle: JoinHandle<()>,
    stderr_handle: JoinHandle<()>,
}

/// A child process whose stdout/stderr are drained on background threads
/// while it runs. See the module doc for the mechanism and rationale.
pub struct DrainedChild {
    child: Child,
    spawned_at: Instant,
    head_cap: usize,
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
        let stdout_outcome = Arc::new(Mutex::new(None));
        let stderr_outcome = Arc::new(Mutex::new(None));
        let stdout_handle = spawn_reader(
            stdout_pipe,
            Arc::clone(&stdout_buf),
            None,
            Arc::clone(&stdout_outcome),
        );
        let stderr_handle = spawn_reader(
            stderr_pipe,
            Arc::clone(&stderr_buf),
            Some(Arc::clone(&stderr_last_byte)),
            Arc::clone(&stderr_outcome),
        );
        Ok(Self {
            child,
            spawned_at: Instant::now(),
            head_cap,
            readers: Some(Readers {
                stdout: stdout_buf,
                stderr: stderr_buf,
                stderr_last_byte,
                stdout_outcome,
                stderr_outcome,
                stdout_handle,
                stderr_handle,
            }),
        })
    }

    /// Test-only: the pre-fix shape — both streams are still piped (so a
    /// well-behaved exit can still be read via `wait_with_output`), but
    /// nothing drains them while the child runs, and a kill path never reads
    /// the pipe at all. This is the standing differential that keeps
    /// `flood_is_a_hang_on_the_undrained_driver` demonstrating the undrained
    /// shape's own hang, without needing to hand-swap `spawn`'s body.
    #[cfg(test)]
    pub(crate) fn spawn_undrained(cmd: &mut Command) -> io::Result<Self> {
        let child = cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
        Ok(Self {
            child,
            spawned_at: Instant::now(),
            head_cap: DEFAULT_HEAD_CAP,
            readers: None,
        })
    }

    /// Non-blocking poll for exit, forwarding to
    /// [`std::process::Child::try_wait`]. Safe to call before
    /// [`DrainedChild::wait_bounded`] (e.g. to confirm a child is still
    /// running before inspecting [`DrainedChild::snapshot`]).
    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        self.child.try_wait()
    }

    /// A point-in-time copy of the raw spliced `(stdout, stderr)` buffers,
    /// reflecting only whatever bytes the reader threads have read *so far*
    /// — not necessarily the child's final output. Safe to call concurrently
    /// with the reader threads (each stream's lock is held only for the
    /// copy) and before the child has exited.
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
    /// sleeping). The deadline is computed with a saturating `checked_add`
    /// (`saturating_deadline`) so a `ceiling` extreme enough to overflow
    /// `Instant`'s internal representation cannot panic.
    ///
    /// - On a normal exit, `disposition` reports `hung: false` — even if a
    ///   kill was already issued for this same call (see "hung vs killed vs
    ///   complete" in the module doc). Settles (waits up to ~1 s for both
    ///   reader threads to observe EOF), snapshots, and returns.
    /// - On the ceiling (or a `try_wait` error, which is treated the same
    ///   way rather than silently reported as a clean exit): kills the child
    ///   (`Capture::killed = true`) and reaps by polling `try_wait` for up to
    ///   ~1 s. A signal-terminated status, or a reap give-up (`status: None`
    ///   — a `SIGKILL`ed child stuck in uninterruptible sleep can outlive
    ///   this poll), is `hung: true`; a normal exit code reaped here (the
    ///   ceiling and the exit raced) is `hung: false`. Any OS error is
    ///   recorded in `Capture::wait_error`. Then performs the same settle and
    ///   snapshots.
    ///
    /// Bounded on both paths **for the drained driver**: at most `ceiling`
    /// plus roughly 2 s (reap + settle) — see [`Completeness`] for what the
    /// settle bound can leave incomplete. The `cfg(test)` undrained driver's
    /// exit path is *not* bounded this way: it calls `wait_with_output`,
    /// which itself blocks until EOF, so an fd-inheriting grandchild can hold
    /// it open indefinitely.
    pub fn wait_bounded(mut self, ceiling: Duration, epoch: Epoch) -> Capture {
        let epoch_base = match epoch {
            Epoch::Spawn => self.spawned_at,
            Epoch::Call => Instant::now(),
        };
        let deadline = saturating_deadline(epoch_base, ceiling);
        let drained = self.readers.is_some();
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => {
                    let hung = disposition(Some(status), false);
                    return if drained {
                        self.finish_drained(Some(status), hung, false, None, epoch_base)
                    } else {
                        self.finish_undrained_exit(status, epoch_base)
                    };
                }
                Ok(None) => {
                    if Instant::now() >= deadline {
                        let (reaped, wait_error) =
                            kill_and_reap(&mut self.child, Duration::from_secs(1));
                        let hung = disposition(reaped, true);
                        return if drained {
                            self.finish_drained(reaped, hung, true, wait_error, epoch_base)
                        } else {
                            self.finish_undrained_hung(reaped, hung, wait_error, epoch_base)
                        };
                    }
                    thread::sleep(Duration::from_millis(20));
                }
                Err(e) => {
                    let (reaped, kill_err) = kill_and_reap(&mut self.child, Duration::from_secs(1));
                    let wait_error = Some(match kill_err {
                        Some(k) => format!("try_wait: {e}; {k}"),
                        None => format!("try_wait: {e}"),
                    });
                    let hung = disposition(reaped, true);
                    return if drained {
                        self.finish_drained(reaped, hung, true, wait_error, epoch_base)
                    } else {
                        self.finish_undrained_hung(reaped, hung, wait_error, epoch_base)
                    };
                }
            }
        }
    }

    fn finish_drained(
        self,
        status: Option<ExitStatus>,
        hung: bool,
        killed: bool,
        wait_error: Option<String>,
        epoch_base: Instant,
    ) -> Capture {
        let head_cap = self.head_cap;
        let readers = self
            .readers
            .expect("finish_drained is only called when self.readers is Some");
        let Readers {
            stdout,
            stderr,
            stderr_last_byte,
            stdout_outcome,
            stderr_outcome,
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

        let stdout_done = stdout_handle.is_finished();
        let stderr_done = stderr_handle.is_finished();
        // Unfinished readers are detached by necessity: joining one that has
        // not observed EOF would block past the settle bound just enforced
        // above (the fd-inheriting-grandchild case). They keep running
        // against their shared Arc<Mutex<..>> buffers, harmlessly, until the
        // pipe's last writer eventually closes it.
        let stdout_panicked = join_if_finished(stdout_done, stdout_handle);
        let stderr_panicked = join_if_finished(stderr_done, stderr_handle);

        let stdout_outcome_val = lock_or_recover(&stdout_outcome).clone();
        let stderr_outcome_val = lock_or_recover(&stderr_outcome).clone();
        let stdout_reader_error = match &stdout_outcome_val {
            Some(ReaderOutcome::Error(e)) => Some(format!("stdout reader: {e}")),
            _ => None,
        };
        let stderr_reader_error = match &stderr_outcome_val {
            Some(ReaderOutcome::Error(e)) => Some(format!("stderr reader: {e}")),
            _ => None,
        };
        let complete = if !(stdout_done && stderr_done) {
            Completeness::SettleExpired
        } else if stdout_panicked
            || stderr_panicked
            || stdout_reader_error.is_some()
            || stderr_reader_error.is_some()
        {
            Completeness::ReaderFailed
        } else {
            Completeness::Complete
        };
        // Reader read-errors are folded into `wait_error` too (rather than
        // only driving `complete`), so a panic message that only prints
        // `wait_error` still names what went wrong.
        let wait_error = {
            let mut parts: Vec<String> = wait_error.into_iter().collect();
            parts.extend(stdout_reader_error);
            parts.extend(stderr_reader_error);
            joined_or_none(parts)
        };

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
            killed,
            elapsed: returned_at.saturating_duration_since(epoch_base),
            stdout: stdout_bytes,
            stderr: stderr_bytes,
            stdout_truncated,
            stderr_truncated,
            last_byte_instant,
            returned_at,
            complete,
            wait_error,
            head_cap,
        }
    }

    /// Undrained exit path: the child is known to have exited already (via
    /// `try_wait`); `wait_with_output` reads whatever remains in the pipes to
    /// completion, matching the pre-fix shape's own only read. Any error is
    /// recorded in `Capture::wait_error` rather than silently swallowed.
    fn finish_undrained_exit(self, status: ExitStatus, epoch_base: Instant) -> Capture {
        let head_cap = self.head_cap;
        let (out_status, stdout, stderr, wait_error) = match self.child.wait_with_output() {
            Ok(output) => (output.status, output.stdout, output.stderr, None),
            Err(e) => (
                status,
                Vec::new(),
                Vec::new(),
                Some(format!("wait_with_output: {e}")),
            ),
        };
        let returned_at = Instant::now();
        Capture {
            status: Some(out_status),
            hung: false,
            killed: false,
            elapsed: returned_at.saturating_duration_since(epoch_base),
            stdout,
            stderr,
            stdout_truncated: 0,
            stderr_truncated: 0,
            last_byte_instant: None,
            returned_at,
            complete: Completeness::Undrained,
            wait_error,
            head_cap,
        }
    }

    /// Undrained kill path: deliberately does **not** read the pipe (reading
    /// here would return whatever the ~64 KiB pipe still holds and break
    /// fidelity to the pre-fix shape, which discarded the log outright on a
    /// kill) — this is a literal reproduction of the undrained driver's
    /// pre-fix shape, not a byte-for-byte replay of any specific esc-078 CI
    /// incident (that log is gone; that loss is the defect).
    fn finish_undrained_hung(
        self,
        status: Option<ExitStatus>,
        hung: bool,
        wait_error: Option<String>,
        epoch_base: Instant,
    ) -> Capture {
        let head_cap = self.head_cap;
        let returned_at = Instant::now();
        Capture {
            status,
            hung,
            killed: true,
            elapsed: returned_at.saturating_duration_since(epoch_base),
            stdout: Vec::new(),
            stderr: Vec::new(),
            stdout_truncated: 0,
            stderr_truncated: 0,
            last_byte_instant: None,
            returned_at,
            complete: Completeness::Undrained,
            wait_error,
            head_cap,
        }
    }
}

/// `epoch_base + ceiling`, saturating instead of panicking if the addition
/// would overflow `Instant`'s internal representation — a `ceiling` extreme
/// enough to trigger this is unreachable in practice (nobody passes a
/// multi-decade ceiling), but the fallback keeps the contract total: a
/// deadline far enough out that it never fires spuriously, rather than
/// panicking or silently treating the ceiling as already expired.
fn saturating_deadline(epoch_base: Instant, ceiling: Duration) -> Instant {
    epoch_base
        .checked_add(ceiling)
        .unwrap_or_else(|| Instant::now() + Duration::from_secs(365 * 24 * 3600))
}

/// Pure function turning a reaped status plus whether a kill was issued into
/// `hung`. A normal exit code — even one reaped just after `kill()` was
/// called — means the child finished on its own (the ceiling and the exit
/// merely raced) and is **not** hung; only a signal death (the kill actually
/// took a still-running process down) or a reap give-up (`reaped == None`
/// after a kill) counts as hung. Unit-tested directly below with
/// hand-constructed `ExitStatus`es, independent of any real subprocess.
fn disposition(reaped: Option<ExitStatus>, killed: bool) -> bool {
    match reaped {
        Some(status) => killed && is_signal_death(status),
        None => killed,
    }
}

#[cfg(unix)]
fn is_signal_death(status: ExitStatus) -> bool {
    use std::os::unix::process::ExitStatusExt;
    status.signal().is_some()
}

#[cfg(not(unix))]
fn is_signal_death(_status: ExitStatus) -> bool {
    false
}

/// If `done`, joins `handle` (cheap and non-blocking, since it has already
/// finished) and reports whether it panicked; otherwise drops `handle`
/// without joining, leaving the thread detached rather than blocking on a
/// reader that has not observed EOF.
fn join_if_finished(done: bool, handle: JoinHandle<()>) -> bool {
    done && handle.join().is_err()
}

/// Kill `child` and reap it by polling `try_wait` for up to `ceiling`.
/// Returns the reaped status (`None` if it never reaped within the bound — a
/// documented caveat: a `SIGKILL`ed process in uninterruptible sleep can
/// outlive this poll, and that alone is not an error) and any `kill`/reap
/// error text, joined together and never silently folded into a clean-
/// looking `None`.
fn kill_and_reap(child: &mut Child, ceiling: Duration) -> (Option<ExitStatus>, Option<String>) {
    let mut errors = Vec::new();
    if let Err(e) = child.kill() {
        errors.push(format!("kill: {e}"));
    }
    let deadline = Instant::now() + ceiling;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return (Some(status), joined_or_none(errors)),
            Ok(None) => {
                if Instant::now() >= deadline {
                    return (None, joined_or_none(errors));
                }
                thread::sleep(Duration::from_millis(20));
            }
            Err(e) => {
                errors.push(format!("reap try_wait: {e}"));
                return (None, joined_or_none(errors));
            }
        }
    }
}

fn joined_or_none(errors: Vec<String>) -> Option<String> {
    if errors.is_empty() {
        None
    } else {
        Some(errors.join("; "))
    }
}

/// How complete a [`Capture`]'s captured streams are — a separate axis from
/// `hung`/`killed`; see the module doc's "hung vs killed vs complete"
/// section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Completeness {
    /// Both reader threads observed EOF before the ~1 s settle bound expired.
    Complete,
    /// At least one reader thread had not observed EOF when the settle bound
    /// expired — most commonly an fd-inheriting grandchild still holding the
    /// pipe open (see the module doc's "Precondition for a complete log").
    /// The unfinished thread(s) are left running, detached by necessity,
    /// until the pipe's last writer eventually closes it.
    SettleExpired,
    /// Both reader threads finished, but at least one ended on a read error
    /// or panicked rather than a clean EOF.
    ReaderFailed,
    /// Always used for a `Capture` produced by the `cfg(test)` undrained
    /// driver, which has no reader threads to track.
    Undrained,
}

/// The outcome of a [`DrainedChild::wait_bounded`] call.
///
/// `stdout`/`stderr` are always the raw spliced buffers (first-`head_cap`
/// bytes followed by last-`tail_cap` bytes, with no marker inserted even when
/// `stdout_truncated`/`stderr_truncated` are nonzero) — see [`render`] (or
/// `render_stdout`/`render_stderr`) for rendering a truncation marker for
/// human consumption.
pub struct Capture {
    /// The child's exit status, or `None` if it was killed and never reaped
    /// within the reap bound (see [`DrainedChild::wait_bounded`]).
    pub status: Option<ExitStatus>,
    /// `true` iff the reaped status represents a genuine hang (a signal
    /// death or a reap give-up) — see the module doc's "hung vs killed vs
    /// complete" section and `disposition`.
    pub hung: bool,
    /// `true` iff `wait_bounded` issued a `kill()` for this call, regardless
    /// of whether the reaped status turned out to be a hang or a raced
    /// normal exit.
    pub killed: bool,
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
    /// Whether the drained reader threads actually reached EOF before the
    /// settle bound expired. `Completeness::Undrained` for a `Capture` from
    /// the `cfg(test)` undrained driver.
    pub complete: Completeness,
    /// Text of any `try_wait`/`kill`/reap/`wait_with_output` error
    /// encountered while producing this `Capture`; `None` if none occurred.
    /// A reap give-up (`status: None` after a kill, with no error) is *not*
    /// a `wait_error` — that is the documented SIGKILL-in-uninterruptible-
    /// sleep caveat, distinguished here from a genuine OS error.
    pub wait_error: Option<String>,
    head_cap: usize,
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

    /// Safe form of the free [`render`] function for `stdout`: always uses
    /// the head-retention cap this `Capture` was actually built with, so it
    /// cannot be called with a mismatched cap.
    pub fn render_stdout(&self) -> String {
        render(&self.stdout, self.stdout_truncated, self.head_cap)
    }

    /// Safe form of the free [`render`] function for `stderr`. See
    /// [`Capture::render_stdout`].
    pub fn render_stderr(&self) -> String {
        render(&self.stderr, self.stderr_truncated, self.head_cap)
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
            .field("killed", &self.killed)
            .field("complete", &self.complete)
            .field("wait_error", &self.wait_error)
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
/// used) — prefer `Capture::render_stdout`/`render_stderr`, which cannot be
/// called with a mismatched cap. Comparators and byte-count assertions must
/// run on `Capture::stdout`/`Capture::stderr` directly — never on this
/// rendered form.
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
    /// the very first statement of every spawning test (the dispatch guard,
    /// via `dispatch_if_child`).
    const CHILD_MODE_ENV: &str = "JAMMI_CHILD_MODE";
    /// Env var carrying the spawning test's own `--exact` path, so a child
    /// running in `ChildMode::Grandchild` can reuse it for its own
    /// grandchild spawn (with `CHILD_MODE_ENV` overridden to
    /// `ChildMode::Sleeper`).
    const SELF_EXACT_ENV: &str = "JAMMI_SELF_EXACT";

    /// The self-exec'd child's behavior, selected by `JAMMI_CHILD_MODE`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ChildMode {
        Flood,
        Wedge,
        Silent,
        Grandchild,
        Sleeper,
        PrintSleep,
    }

    impl ChildMode {
        fn as_str(self) -> &'static str {
            match self {
                ChildMode::Flood => "flood",
                ChildMode::Wedge => "wedge",
                ChildMode::Silent => "silent",
                ChildMode::Grandchild => "grandchild",
                ChildMode::Sleeper => "sleeper",
                ChildMode::PrintSleep => "printsleep",
            }
        }

        fn parse(raw: &str) -> Option<Self> {
            Some(match raw {
                "flood" => ChildMode::Flood,
                "wedge" => ChildMode::Wedge,
                "silent" => ChildMode::Silent,
                "grandchild" => ChildMode::Grandchild,
                "sleeper" => ChildMode::Sleeper,
                "printsleep" => ChildMode::PrintSleep,
                _ => return None,
            })
        }
    }

    /// The dispatch guard: every spawning test's first statement. When this
    /// process was re-exec'd as a child (`JAMMI_CHILD_MODE` set), runs the
    /// child body and never returns; otherwise a no-op, so the test proceeds
    /// as the parent.
    fn dispatch_if_child() {
        if let Ok(raw) = env::var(CHILD_MODE_ENV) {
            let mode = ChildMode::parse(&raw).unwrap_or_else(|| {
                panic!("child.rs test harness: unknown {CHILD_MODE_ENV}={raw:?}")
            });
            run_child(mode);
        }
    }

    /// Dispatches a self-exec'd child by [`ChildMode`]. Never returns.
    fn run_child(mode: ChildMode) -> ! {
        match mode {
            ChildMode::Flood => flood_child(),
            ChildMode::Wedge => wedge_child(),
            ChildMode::Silent => std::process::exit(0),
            ChildMode::Grandchild => grandchild_child(),
            ChildMode::Sleeper => sleeper_child(),
            ChildMode::PrintSleep => printsleep_child(),
        }
    }

    /// Build a `Command` that re-execs this very test binary, selecting only
    /// `test_path` (a fully-qualified `module::path::of::the_test_fn`) and
    /// running it uncaptured on a single thread — both flags are load-bearing
    /// (measured, plan round 5): without `--nocapture`, libtest swallows the
    /// child's `eprintln!` output and the parent observes zero bytes (a
    /// vacuous oracle); without `--exact`, every test whose name is a prefix
    /// match would enter its own dispatch guard concurrently, producing
    /// nondeterministic byte counts.
    fn self_exec(test_path: &str, mode: ChildMode) -> Command {
        let mut cmd = Command::new(env::current_exe().expect("current_exe for self-exec"));
        cmd.args(["--exact", test_path, "--nocapture", "--test-threads=1"]);
        cmd.env(CHILD_MODE_ENV, mode.as_str());
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

    /// Spawns `current_exe()` in `ChildMode::Sleeper` mode, reusing this
    /// test's own `--exact` path (`SELF_EXACT_ENV`, set by `self_exec` on
    /// this very process) but overriding `CHILD_MODE_ENV` to
    /// `ChildMode::Sleeper` — reusing the inherited mode would fork-bomb. The
    /// grandchild is spawned without stdio redirection, so it inherits this
    /// process's stdout/stderr (the pipes the grandparent `DrainedChild` is
    /// draining). Exits 0 immediately, deliberately not waiting on the
    /// grandchild.
    fn grandchild_child() -> ! {
        let exact = env::var(SELF_EXACT_ENV)
            .expect("grandchild mode requires SELF_EXACT_ENV set by self_exec");
        let mut cmd = Command::new(env::current_exe().expect("current_exe for grandchild spawn"));
        cmd.args(["--exact", &exact, "--nocapture", "--test-threads=1"]);
        cmd.env(CHILD_MODE_ENV, ChildMode::Sleeper.as_str());
        let _child = cmd.spawn().expect("spawn sleeper grandchild");
        std::process::exit(0);
    }

    /// Sleeps 3 s (holding whatever stdio it inherited open) then exits 0.
    fn sleeper_child() -> ! {
        thread::sleep(Duration::from_secs(3));
        std::process::exit(0);
    }

    /// Prints one line to stderr, then sleeps 2 s before exiting 0 — used to
    /// give `snapshot()`/`try_wait()` something to observe mid-run.
    fn printsleep_child() -> ! {
        eprintln!("hello-mid-run");
        let _ = io::stderr().flush();
        thread::sleep(Duration::from_secs(2));
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

    // ---- disposition: pure, no subprocess involved -------------------

    #[cfg(unix)]
    fn exit_status_code(code: i32) -> ExitStatus {
        use std::os::unix::process::ExitStatusExt;
        ExitStatus::from_raw(code << 8)
    }

    #[cfg(unix)]
    fn exit_status_signal(signal: i32) -> ExitStatus {
        use std::os::unix::process::ExitStatusExt;
        ExitStatus::from_raw(signal)
    }

    #[cfg(unix)]
    #[test]
    fn disposition_signal_death_after_kill_is_hung() {
        let status = exit_status_signal(9); // SIGKILL
        assert!(
            disposition(Some(status), true),
            "a signal death after a kill is the classic hang path"
        );
    }

    #[cfg(unix)]
    #[test]
    fn disposition_normal_exit_after_kill_is_not_hung() {
        let status = exit_status_code(0);
        assert!(
            !disposition(Some(status), true),
            "a normal exit reaped after a kill raced the ceiling, not a hang"
        );
        let status_nonzero = exit_status_code(7);
        assert!(
            !disposition(Some(status_nonzero), true),
            "a nonzero-but-not-signaled exit is still not a hang"
        );
    }

    #[test]
    fn disposition_reap_giveup_after_kill_is_hung() {
        assert!(
            disposition(None, true),
            "a reap give-up after a kill is the SIGKILL-in-D-state caveat, still a hang"
        );
    }

    #[test]
    fn disposition_without_a_kill_is_never_hung() {
        assert!(!disposition(None, false));
    }

    // ---- the esc-078 oracle and its differentials ---------------------

    /// closes_escape: esc-078
    ///
    /// The oracle for the fix itself: a child that writes exactly 4 MiB
    /// (65536 fixed-width lines) to stderr, well past the ~64 KiB an
    /// undrained pipe can hold, must finish, have every byte retained, and
    /// have both reader threads reach a clean EOF. Pre-fix (see the module
    /// doc's "Revert recipe"), this test hangs at the harness's own ceiling
    /// and loses the log entirely — the exact esc-078 symptom.
    #[test]
    fn flood_child_is_drained_and_its_evidence_is_retained() {
        dispatch_if_child();
        let exact = test_exact_path("flood_child_is_drained_and_its_evidence_is_retained");
        let mut cmd = self_exec(&exact, ChildMode::Flood);
        let child = DrainedChild::spawn(&mut cmd).expect("spawn flood child");
        let cap = child.wait_bounded(Duration::from_secs(30), Epoch::Spawn);

        assert!(!cap.hung, "flood child should not hang: {cap:?}");
        assert!(!cap.killed, "flood child exits on its own: {cap:?}");
        assert_eq!(cap.complete, Completeness::Complete, "{cap:?}");
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

        // stdout carries libtest's own banner (this child process is itself
        // a `--exact --nocapture` libtest invocation) — asserted so the
        // stdout reader thread's own retention/no-truncation path is
        // exercised by something other than an always-empty stream.
        let stdout_text = String::from_utf8_lossy(&cap.stdout);
        assert!(
            stdout_text.contains("running 1 test"),
            "stdout should carry libtest's banner: {stdout_text:?}"
        );
        assert_eq!(cap.stdout_truncated, 0, "{cap:?}");

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
    /// finish — `hung == true` with an **empty** stderr. That empty-stderr
    /// assert is a literal check of the undrained driver's own pre-fix shape
    /// (`finish_undrained_hung` never reads the pipe on the kill path), not a
    /// claim that it reproduces the exact bytes any specific esc-078 CI
    /// incident lost — that original log is gone; the loss itself is the
    /// defect this differential keeps demonstrating. If this test ever stops
    /// hanging (e.g. on a platform with a much larger pipe buffer), it is
    /// this test that reds, marking the oracle above as vacuous rather than
    /// silently losing coverage.
    #[test]
    fn flood_is_a_hang_on_the_undrained_driver() {
        dispatch_if_child();
        let exact = test_exact_path("flood_is_a_hang_on_the_undrained_driver");
        let mut cmd = self_exec(&exact, ChildMode::Flood);
        let child = DrainedChild::spawn_undrained(&mut cmd).expect("spawn undrained flood child");
        let cap = child.wait_bounded(Duration::from_secs(5), Epoch::Spawn);

        assert!(
            cap.hung,
            "expected the undrained driver to hang on a 4 MiB flood: {cap:?}"
        );
        assert!(cap.killed, "{cap:?}");
        assert!(
            cap.stderr.is_empty(),
            "undrained kill path must not read the pipe: {cap:?}"
        );
    }

    /// esc-078 control (1): a child killed at the ceiling must still yield
    /// its progress (`phase: b`) and an accurate `silence()`. Its undrained
    /// twin is the same literal pre-fix-shape check as the flood
    /// differential above — the kill path never reads the pipe — showing
    /// that the bug was in the kill path itself, not only in pipe
    /// backpressure (the wedge child's own two lines never come close to
    /// filling a pipe).
    #[test]
    fn wedged_child_is_killed_at_the_ceiling_with_its_evidence() {
        dispatch_if_child();
        let exact = test_exact_path("wedged_child_is_killed_at_the_ceiling_with_its_evidence");

        let mut cmd = self_exec(&exact, ChildMode::Wedge);
        let child = DrainedChild::spawn(&mut cmd).expect("spawn wedge child");
        let cap = child.wait_bounded(Duration::from_secs(5), Epoch::Spawn);

        assert!(cap.hung, "wedge child never exits: {cap:?}");
        assert!(cap.killed, "{cap:?}");
        assert_eq!(cap.complete, Completeness::Complete, "{cap:?}");
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
        // The wedge's last write is near-instant and the ceiling is 5s, so
        // silence should be close to 5s; the >= 3s bound leaves ~2s of slack
        // for scheduling jitter rather than a brittle near-equality check.
        assert!(
            silence >= Duration::from_secs(3),
            "silence={silence:?}: {cap:?}"
        );

        let mut undrained_cmd = self_exec(&exact, ChildMode::Wedge);
        let undrained_child =
            DrainedChild::spawn_undrained(&mut undrained_cmd).expect("spawn undrained wedge child");
        let undrained_cap = undrained_child.wait_bounded(Duration::from_secs(5), Epoch::Spawn);
        assert!(undrained_cap.hung, "{undrained_cap:?}");
        assert!(undrained_cap.killed, "{undrained_cap:?}");
        assert!(
            undrained_cap.stderr.is_empty(),
            "undrained kill path must not read the pipe: {undrained_cap:?}"
        );
    }

    /// Driver-level control (5): a no-output, exit-0 child is not a hang, and
    /// carries no stale silence signal.
    #[test]
    fn silent_child_is_not_a_hang() {
        dispatch_if_child();
        let exact = test_exact_path("silent_child_is_not_a_hang");
        let mut cmd = self_exec(&exact, ChildMode::Silent);
        let child = DrainedChild::spawn(&mut cmd).expect("spawn silent child");
        let cap = child.wait_bounded(Duration::from_secs(10), Epoch::Spawn);

        assert!(!cap.hung, "{cap:?}");
        assert!(!cap.killed, "{cap:?}");
        assert_eq!(cap.status.and_then(|s| s.code()), Some(0), "{cap:?}");
        assert!(cap.stderr.is_empty(), "{cap:?}");
        assert!(cap.last_byte_instant.is_none(), "{cap:?}");
        assert!(cap.silence().is_none(), "{cap:?}");
    }

    /// The settle is bounded even when a grandchild inherits the pipe: the
    /// child here spawns a `ChildMode::Sleeper` grandchild (which inherits
    /// stdout/stderr and sleeps 3 s) and exits immediately, so `wait_bounded`
    /// must return well short of the grandchild's 3 s sleep — with
    /// `hung == false`, `killed == false`, `complete ==
    /// Completeness::SettleExpired` (the settle bound, ~1 s, expired before
    /// either reader saw EOF), and nothing left running once the grandchild
    /// self-terminates. The 2.5 s bound (vs the 1 s hard settle) still proves
    /// the wait was capped well short of the grandchild's 3 s sleep — it is
    /// not a near-equality check, just wide enough to absorb scheduling
    /// jitter without ever reaching 3 s.
    #[test]
    fn settle_returns_within_bound_when_a_grandchild_holds_the_pipe() {
        dispatch_if_child();
        let exact = test_exact_path("settle_returns_within_bound_when_a_grandchild_holds_the_pipe");
        let mut cmd = self_exec(&exact, ChildMode::Grandchild);
        let started = Instant::now();
        let child = DrainedChild::spawn(&mut cmd).expect("spawn grandchild-spawning child");
        let cap = child.wait_bounded(Duration::from_secs(10), Epoch::Spawn);
        let wall = started.elapsed();

        assert!(!cap.hung, "{cap:?}");
        assert!(!cap.killed, "{cap:?}");
        assert_eq!(cap.complete, Completeness::SettleExpired, "{cap:?}");
        assert!(
            wall < Duration::from_millis(2500),
            "settle should return well short of the grandchild's 3s sleep, got {wall:?}: {cap:?}"
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
        dispatch_if_child();
        let exact = test_exact_path("flood_over_cap");

        const HEAD_CAP: usize = 1024 * 1024; // 1 MiB
        const TAIL_CAP: usize = 256 * 1024; // 256 KiB
        const OVER_CAP_LINES: u32 = 32768; // 32768 * 64 = 2 MiB
        const SUB_CAP_LINES: u32 = 4096; // 4096 * 64 = 256 KiB, well under 1.25 MiB

        // Over cap: the 2 MiB stream exceeds the 1.25 MiB combined cap.
        let mut cmd = self_exec(&exact, ChildMode::Flood);
        cmd.env("JAMMI_FLOOD_LINES", OVER_CAP_LINES.to_string());
        let child = DrainedChild::spawn_with_caps(&mut cmd, HEAD_CAP, TAIL_CAP)
            .expect("spawn over-cap flood child");
        let cap = child.wait_bounded(Duration::from_secs(30), Epoch::Spawn);
        assert!(!cap.hung, "{cap:?}");
        assert!(!cap.killed, "{cap:?}");
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

        let rendered = cap.render_stderr();
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
        let mut sub_cmd = self_exec(&exact, ChildMode::Flood);
        sub_cmd.env("JAMMI_FLOOD_LINES", SUB_CAP_LINES.to_string());
        let sub_child = DrainedChild::spawn_with_caps(&mut sub_cmd, HEAD_CAP, TAIL_CAP)
            .expect("spawn sub-cap flood child");
        let sub_cap = sub_child.wait_bounded(Duration::from_secs(30), Epoch::Spawn);
        assert!(!sub_cap.hung, "{sub_cap:?}");
        assert_eq!(sub_cap.stderr_truncated, 0, "{sub_cap:?}");
        let sub_expected = expected_flood_stream(SUB_CAP_LINES);
        assert_eq!(sub_cap.stderr, sub_expected, "{sub_cap:?}");
        let sub_rendered = sub_cap.render_stderr();
        assert!(
            !sub_rendered.contains("truncated"),
            "sub-cap flood must render no marker: {sub_rendered}"
        );
    }

    // ---- round W1.1: Epoch::Call, Duration::ZERO, snapshot(), try_wait() --

    /// A zero ceiling measured from `Epoch::Call` against a still-running
    /// child must kill immediately: the deadline check precedes any sleep,
    /// so the very first `try_wait` poll (finding the sleeper still running)
    /// is immediately followed by a kill, never a 20 ms sleep. The bound
    /// (2s) matches `wait_bounded`'s own documented kill-path worst case
    /// ("ceiling plus roughly 2s", ceiling here being zero) rather than a
    /// tighter number: under heavy in-binary parallelism (this module's
    /// other 12+ tests, several themselves spawning subprocesses,
    /// contending for the host's CPUs) a tighter bound was observed to
    /// flake even on the *other* new test below, purely from scheduling
    /// delay, not from `wait_bounded` failing to kill promptly. It is still
    /// a real assertion: it rules out waiting anywhere near the sleeper's
    /// own natural 3s lifetime.
    #[test]
    fn zero_ceiling_with_epoch_call_kills_immediately() {
        dispatch_if_child();
        let exact = test_exact_path("zero_ceiling_with_epoch_call_kills_immediately");
        let mut cmd = self_exec(&exact, ChildMode::Sleeper);
        let started = Instant::now();
        let child = DrainedChild::spawn(&mut cmd).expect("spawn sleeper child");
        let cap = child.wait_bounded(Duration::ZERO, Epoch::Call);
        let wall = started.elapsed();

        assert!(cap.killed, "{cap:?}");
        assert!(
            cap.hung,
            "a zero ceiling against a still-sleeping child must kill it: {cap:?}"
        );
        assert!(wall < Duration::from_secs(2), "wall={wall:?}: {cap:?}");
        assert!(
            cap.elapsed < Duration::from_secs(2),
            "elapsed={:?}: {cap:?}",
            cap.elapsed
        );
    }

    /// `Epoch::Call` measures from the call, not from spawn: spawning a
    /// fast-exiting child, then sleeping 300 ms *before* calling
    /// `wait_bounded(.., Epoch::Call)`, must NOT fold that 300 ms into
    /// `elapsed`. Asserted as a *relative* comparison against the actual
    /// measured spawn-to-return wall (which necessarily includes the 300 ms
    /// sleep plus everything `elapsed` covers) rather than a fixed absolute
    /// threshold: `cap.elapsed` is, by construction, a strict sub-interval
    /// of `wall` starting only once the 300 ms sleep has already elapsed, so
    /// this holds regardless of how fast or contended the host is — under
    /// `Epoch::Spawn` instead, `elapsed` and `wall` would be measuring
    /// (almost) the same interval, not a comfortably shorter one.
    #[test]
    fn epoch_call_measures_from_the_call() {
        dispatch_if_child();
        let exact = test_exact_path("epoch_call_measures_from_the_call");
        let mut cmd = self_exec(&exact, ChildMode::Silent);
        let started = Instant::now();
        let child = DrainedChild::spawn(&mut cmd).expect("spawn silent child");
        thread::sleep(Duration::from_millis(300));
        let cap = child.wait_bounded(Duration::from_secs(10), Epoch::Call);
        let wall = started.elapsed();

        assert!(!cap.hung, "{cap:?}");
        assert!(
            cap.elapsed + Duration::from_millis(150) < wall,
            "Epoch::Call's elapsed ({:?}) should be measured from the call, so \
             it must fall meaningfully short of the full spawn-to-return wall \
             ({wall:?}), which additionally covers the 300ms pre-call sleep: {cap:?}",
            cap.elapsed
        );
    }

    /// `snapshot()` and `try_wait()` both work before the child has exited:
    /// a child that prints one line and then sleeps 2s must still be
    /// reported running by `try_wait()`, and `snapshot()` must already
    /// contain the line it printed, before `wait_bounded` is ever called.
    /// Polls (up to 3s) instead of a single fixed sleep before checking, so
    /// this tolerates slow child-process startup under heavy in-binary
    /// parallelism rather than assuming the print happens within some fixed
    /// number of milliseconds: it succeeds the instant the line appears, and
    /// separately confirms the child had not yet exited at that same moment
    /// (it always still has ~2s left to sleep at that point).
    #[test]
    fn snapshot_mid_run_returns_bytes_so_far() {
        dispatch_if_child();
        let exact = test_exact_path("snapshot_mid_run_returns_bytes_so_far");
        let mut cmd = self_exec(&exact, ChildMode::PrintSleep);
        let mut child = DrainedChild::spawn(&mut cmd).expect("spawn printsleep child");

        let poll_deadline = Instant::now() + Duration::from_secs(3);
        let mut seen = false;
        let mut still_running = false;
        while Instant::now() < poll_deadline {
            let running = child
                .try_wait()
                .expect("try_wait should not error on a live child")
                .is_none();
            let (_, stderr_so_far) = child.snapshot();
            if String::from_utf8_lossy(&stderr_so_far).contains("hello-mid-run") {
                seen = true;
                still_running = running;
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
        assert!(seen, "snapshot() never observed the mid-run line within 3s");
        assert!(
            still_running,
            "the line appears immediately after the child starts, well before its \
             own 2s sleep completes, so try_wait() should still report it running \
             at that same moment"
        );

        let cap = child.wait_bounded(Duration::from_secs(10), Epoch::Spawn);
        assert!(!cap.hung, "{cap:?}");
        assert!(
            String::from_utf8_lossy(&cap.stderr).contains("hello-mid-run"),
            "{cap:?}"
        );
    }
}
