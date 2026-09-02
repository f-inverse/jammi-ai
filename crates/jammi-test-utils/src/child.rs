//! [`DrainedChild`] is a drained, bounded child-process output-capture
//! primitive. [`DrainedChild::spawn`] pipes a child's stdout/stderr and
//! starts one reader thread per stream that drains it into an in-memory,
//! retention-capped buffer **while the child runs** — capture never depends
//! on the pipe's own kernel buffer surviving a chatty or long-lived child.
//! [`DrainedChild::wait_bounded`] then waits for exit against a ceiling
//! measured from either [`Epoch::Spawn`] or [`Epoch::Call`], performs a
//! short, bounded "settle" (waiting for the reader threads to observe EOF,
//! but never indefinitely — see "Precondition for a complete log"), and
//! reports the outcome as a [`Capture`].
//!
//! A `Capture` separates its outcome into four independent axes a caller
//! inspects on their own terms, rather than one collapsed verdict:
//! `Capture::killed` (was the child forcibly terminated?), `Capture::hung`
//! (was that termination a genuine hang, as opposed to racing a normal
//! exit?), `Capture::complete` ([`Completeness`]; did the reader threads
//! themselves reach a clean EOF before the settle bound?), and per-stream
//! truncation (`Capture::stdout_truncated`/`stderr_truncated`; did the
//! retention cap drop bytes?). [`Capture::is_trustworthy`] is the single
//! predicate that ANDs completeness and truncation together for a caller
//! that just wants to know "is this evidence whole" — see "`hung` vs
//! `killed` vs `complete` vs trustworthy", below, for how the four axes
//! combine.
//!
//! # Why this exists
//!
//! This module exists to fix a specific mechanism defect (esc-078):
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
//! no evidence at all. Draining both streams while the child runs (above)
//! fixes the mechanism: a chatty child never fills the pipe, and a killed
//! child's progress up to the kill is still on hand.
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
//! **The cap is a second, independent incompleteness source from the reader
//! threads themselves**: a capture whose reader threads both reached a clean
//! EOF (`Completeness::Complete`) can still be silently missing bytes the cap
//! dropped from the middle of a long-running child's output.
//! `Capture::complete` alone does NOT mean "trust this evidence as whole" —
//! see `Capture::is_trustworthy`, below, for the single predicate that does.
//!
//! # `hung` vs `killed` vs `complete` vs trustworthy
//!
//! Four separate axes, not one:
//!
//! - `Capture::killed` — did `wait_bounded` ever call `kill()`? True whenever
//!   the ceiling was reached (or an OS error forced a defensive kill),
//!   independent of whether the child turns out to have needed it.
//! - `Capture::hung` — computed from `killed` and the reaped status by the
//!   pure `disposition` function: `true` **iff a kill was issued** and the
//!   reaped status is a signal death or a reap give-up (`status: None`
//!   after that kill). A normal exit code is `false` even when `killed` is
//!   also `true` — the ceiling and a fast, self-terminating exit can race,
//!   and a child that finishes on its own in that window is not a hang just
//!   because the kill was already in flight. A signal death `wait_bounded`
//!   never issued a kill for (a genuine self-inflicted crash, e.g. a
//!   `SIGSEGV`) is also `false` — that is a live signal exit for the caller
//!   to inspect via `Capture::status` directly, not a hang this driver
//!   detected and terminated.
//! - `Capture::complete` ([`Completeness`]) — the READER-THREAD axis only:
//!   did both reader threads conclude `Eof` before the ~1 s settle bound
//!   expired? On unix `Eof` is not necessarily a literal closed pipe: once
//!   `wait_bounded` has reaped the child, a reader that finds the pipe idle
//!   on one 50 ms poll concludes `Eof` for THAT CHILD's bytes, regardless
//!   of whether some other process still holds the write end open (see
//!   "Precondition for a complete log") — a foreign holder only produces
//!   `SettleExpired` if it keeps writing with gaps shorter than that 50 ms
//!   window; a foreign grandchild's own later bytes, in either case, are
//!   never part of this child's evidence. `hung: false` and
//!   `SettleExpired` can both be true together (a fast-writing foreign
//!   holder), and so can `hung: true` and `Completeness::Complete` (a
//!   killed child whose own readers conclude `Eof` the moment the confirmed
//!   exit's next idle poll lands). It says nothing about the retention cap
//!   — see the previous section.
//! - `Capture::is_trustworthy()` — the single predicate a consumer should
//!   gate a "this evidence is whole" decision on: `complete ==
//!   Completeness::Complete` AND no `wait_error` AND neither stream was
//!   truncated by the cap. This is the AND of the reader axis and the cap
//!   axis; `complete` by itself is not sufficient.
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
//! A reaped child cannot write again: process exit is strictly ordered
//! before `try_wait`/`waitpid` can observe it, so every byte the child will
//! ever produce is already sitting in the pipe by the time `wait_bounded`
//! confirms the exit. On unix this is the completeness mechanism itself,
//! not an afterthought: each reader thread polls its fd (`libc::poll`, 50 ms
//! timeout) rather than blocking in `read()`, and once `wait_bounded` has
//! reaped the child, an empty poll means the reader concludes `Eof`
//! immediately — it does NOT wait for the pipe to actually close, so
//! **something else merely holding the write end open (without writing to
//! it) no longer delays completeness at all**, regardless of why that other
//! holder has the fd. Two distinct things can put another fd on the same
//! pipe:
//!
//! 1. **An fd-inheriting grandchild.** A child that spawns a grandchild
//!    without redirecting the grandchild's stdio inherits the pipe.
//! 2. **Non-atomic `CLOEXEC` on Apple platforms.** On Linux, `std` creates a
//!    `Stdio::piped()` pipe with `pipe2(O_CLOEXEC)`, one atomic syscall. On
//!    macOS/iOS, `std` instead calls `pipe()` then `fcntl(F_SETFD,
//!    FD_CLOEXEC)` as two separate syscalls, so there is a real (if narrow)
//!    window where the new pipe's write end is open and NOT yet
//!    close-on-exec; a concurrent `posix_spawn` on another OS thread of this
//!    SAME process can `fork` inside that window and inherit it. Every spawn
//!    this module itself performs is serialized across its own
//!    pipe-creation-to-exec window by a process-wide lock, cutting this off
//!    for driver-to-driver races — cheap defense in depth, but a
//!    **mitigation**, not the fix: it cannot help against a non-driver
//!    `Command::spawn` racing in the same process (anything that does not go
//!    through this module), and Linux never needs it (`pipe2(O_CLOEXEC)` is
//!    atomic there).
//!
//! What actually keeps completeness correct regardless of either cause is
//! the poll-based reader above: a silent fd-holder from either cause is
//! invisible to it (no `POLLIN` ever arrives from it, so the first empty
//! poll after the confirmed exit ends the reader). The ONE case that still
//! produces `Completeness::SettleExpired` is a holder that keeps *writing*,
//! with gaps shorter than the reader's 50 ms poll timeout, after the target
//! child exits — a live grandchild in a tight write loop, say — because that
//! keeps `POLLIN` arriving before the reader ever sees an idle poll, so it
//! keeps draining; a holder with LONGER gaps between writes (say, one line
//! every 500 ms) instead presents an idle poll during one of those gaps and
//! is indistinguishable from a finished stream, so it is `Complete` too, not
//! `SettleExpired` — the mechanism cannot tell "paused" from "done" on any
//! single poll, only "still actively streaming" from "not". For a genuinely
//! tight writer, `finish_drained`'s settle bound (about 1 s) is what
//! eventually gives up, a rarely-hit fallback now rather than the primary
//! mechanism. Either way `wait_bounded` still returns promptly
//! with `hung == false` (never a hang), but `Capture::complete` reports
//! `Completeness::SettleExpired` and the returned log can be missing
//! whatever the other writer produces after the settle bound — honest,
//! never `Completeness::Complete`. The non-unix fallback reader has none of
//! this: it blocks in `read()` until an actual EOF, so both causes above
//! still delay it there, exactly as on unix pre-fix.
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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::io::AsRawFd;

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

/// Serializes every driver-initiated `Command::spawn` in this process across
/// its pipe-creation-to-exec window (see the module doc's "Precondition for
/// a complete log", cause 2). Held only across the `spawn()` call itself —
/// never across any of the slower work afterward (reader-thread setup,
/// `wait_bounded`'s poll loop) — so it adds no measurable serialization
/// beyond the syscall window it exists to protect.
///
/// Defensive, with no local oracle: `concurrent_spawns_do_not_race_the_pipe_cloexec_window`
/// cannot demonstrate this lock's necessity by itself, because the fd a
/// racing `fork` would mis-inherit lands at an arbitrary, a-priori-unknown
/// descriptor number in the *other* process's fd table — nothing in that
/// process's own code (e.g. `eprintln!`, hardcoded to fd 2) ever writes to
/// it, so the poll-based reader's "an idle fd is done" rule (see
/// `spawn_reader`) already treats a mis-inherited-but-never-written-to fd as
/// harmless regardless of whether this lock ran. A test that could write to
/// that specific, unpredictable fd number would need to enumerate the
/// process's own open descriptors, a materially larger fixture than this
/// round's scope covers; tracked as a residual, not fixed here.
static SPAWN_LOCK: Mutex<()> = Mutex::new(());

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

/// Read `pipe` into `buf` (locked only for each append), stamping `last_byte`
/// (when `Some`) with an absolute [`Instant`] after every non-empty read, and
/// recording its terminal reason into `outcome` (never a silent break) so
/// [`Completeness`] can distinguish a clean EOF from a read error.
///
/// On unix, this polls `pipe`'s fd with a 50 ms timeout rather than blocking
/// in `read()`: a reaped child cannot write again (process exit precedes
/// `waitpid`/`try_wait` observing it, and all of the child's writes are
/// already resident in the pipe by then), so once `child_exited` is set
/// (by `DrainedChild::wait_bounded`, the instant it reaps the child) an empty
/// poll means every byte this child will ever produce has already been read
/// — the reader concludes `Eof` without waiting for an actual close-on-exec
/// EOF, which an unrelated process holding an inherited pipe end (see the
/// module doc's "Precondition for a complete log", cause 2) could delay
/// indefinitely. A grandchild that keeps writing keeps `POLLIN` arriving, so
/// this reader keeps draining it and never falsely concludes `Eof` — that
/// case still relies on `finish_drained`'s bounded settle, now a rarely-hit
/// fallback rather than the primary mechanism.
///
/// `child_exited` is loaded **before** calling `poll`, not after it returns
/// — loading it afterward would leave the interval between "poll observed
/// nothing" and "we read the flag" unprotected: a child that writes its
/// final line, exits, and is reaped inside that interval would have that
/// poll's empty result (computed strictly before the write) paired with a
/// newly-true flag, and the reader would conclude `Eof` having never
/// actually re-checked the pipe after the write landed — silently dropping
/// it. Loading the flag first instead guarantees that if it reads `true`,
/// the `poll` call that follows runs strictly after the confirmed exit (and
/// therefore strictly after every byte the child will ever write), so an
/// empty result from *that* `poll` is genuinely final.
#[cfg(unix)]
fn spawn_reader<R: Read + AsRawFd + Send + 'static>(
    mut pipe: R,
    buf: Arc<Mutex<RetainedBuf>>,
    last_byte: Option<Arc<Mutex<Option<Instant>>>>,
    outcome: Arc<Mutex<Option<ReaderOutcome>>>,
    child_exited: Arc<AtomicBool>,
    // Test-only hook, invoked exactly once, right after `poll` returns 0
    // and BEFORE `exited_before_poll` is consulted -- lets a test pin the
    // exact TOCTOU interleaving the fn doc above describes (child writes
    // its final line, exits, and is reaped) deterministically instead of
    // hoping wall-clock timing lands in that interval. Does not exist in
    // non-test builds: zero cost, no public API, and it cannot affect the
    // shipped driver's behavior even in a `cfg(test)` build of a consuming
    // crate, since it is private to this module and always `None` at both
    // of `spawn_reader`'s call sites in `spawn_inner`.
    #[cfg(test)] mut ready_hook: Option<Box<dyn FnOnce() + Send>>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let fd = pipe.as_raw_fd();
        let mut chunk = [0u8; 8192];
        loop {
            // See the fn doc: this MUST be read before `poll`, not after.
            let exited_before_poll = child_exited.load(Ordering::SeqCst);
            let mut pollfd = libc::pollfd {
                fd,
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: `pollfd` is a single, stack-local, correctly
            // initialized `libc::pollfd` and `nfds == 1` matches the buffer
            // handed to `poll(2)`.
            let ret = unsafe { libc::poll(&mut pollfd, 1, 50) };
            if ret < 0 {
                let err = io::Error::last_os_error();
                if err.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                *lock_or_recover(&outcome) = Some(ReaderOutcome::Error(err.to_string()));
                break;
            }
            if ret == 0 {
                #[cfg(test)]
                if let Some(hook) = ready_hook.take() {
                    hook();
                }
                // Timed out: nothing available right now. Safe to conclude
                // `Eof` only because the exit was already confirmed BEFORE
                // this specific `poll` call ran (see the fn doc).
                if exited_before_poll {
                    *lock_or_recover(&outcome) = Some(ReaderOutcome::Eof);
                    break;
                }
                continue;
            }
            let revents = pollfd.revents;
            if revents & libc::POLLIN != 0 {
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
            } else if revents & libc::POLLERR != 0 {
                *lock_or_recover(&outcome) = Some(ReaderOutcome::Error(
                    "poll reported POLLERR on the pipe fd".to_string(),
                ));
                break;
            } else if revents & libc::POLLHUP != 0 {
                // Peer closed with nothing left buffered (POLLIN would also
                // be set if there were still bytes to drain) -- a real EOF.
                *lock_or_recover(&outcome) = Some(ReaderOutcome::Eof);
                break;
            }
            // Any other spurious wakeup: loop and poll again.
        }
    })
}

/// Non-unix fallback: plain blocking `read()` to EOF, with no `poll`-based
/// short-circuit on `child_exited` (that mechanism is unix-`poll`-specific).
/// This platform relies solely on `finish_drained`'s bounded settle for the
/// cases the module doc's "Precondition for a complete log" describes.
#[cfg(not(unix))]
fn spawn_reader<R: Read + Send + 'static>(
    mut pipe: R,
    buf: Arc<Mutex<RetainedBuf>>,
    last_byte: Option<Arc<Mutex<Option<Instant>>>>,
    outcome: Arc<Mutex<Option<ReaderOutcome>>>,
    _child_exited: Arc<AtomicBool>,
    // See the unix `spawn_reader`'s doc -- this mechanism is unix-`poll`-
    // specific, so the hook is accepted for call-site parity but unused.
    #[cfg(test)] _ready_hook: Option<Box<dyn FnOnce() + Send>>,
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
    /// Set by `wait_bounded` the instant it reaps the child (`try_wait`
    /// returns `Some`), read by both reader threads on unix to conclude
    /// `Eof` as soon as an empty poll follows a confirmed exit, rather than
    /// waiting for an actual pipe close that an unrelated fd-holder could
    /// delay indefinitely. See `spawn_reader`.
    child_exited: Arc<AtomicBool>,
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
        let mut child = {
            let _spawn_guard = lock_or_recover(&SPAWN_LOCK);
            cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?
        };
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
        let child_exited = Arc::new(AtomicBool::new(false));
        let stdout_handle = spawn_reader(
            stdout_pipe,
            Arc::clone(&stdout_buf),
            None,
            Arc::clone(&stdout_outcome),
            Arc::clone(&child_exited),
            #[cfg(test)]
            None,
        );
        let stderr_handle = spawn_reader(
            stderr_pipe,
            Arc::clone(&stderr_buf),
            Some(Arc::clone(&stderr_last_byte)),
            Arc::clone(&stderr_outcome),
            Arc::clone(&child_exited),
            #[cfg(test)]
            None,
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
                child_exited,
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
        let child = {
            let _spawn_guard = lock_or_recover(&SPAWN_LOCK);
            cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?
        };
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
        // A clone of the shared flag the reader threads poll on unix to
        // conclude `Eof` as soon as this child is confirmed reaped, rather
        // than waiting for an actual pipe close (see `spawn_reader`).
        let child_exited = self.readers.as_ref().map(|r| Arc::clone(&r.child_exited));
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => {
                    if let Some(flag) = &child_exited {
                        flag.store(true, Ordering::SeqCst);
                    }
                    let hung = disposition(Some(status), false);
                    return if drained {
                        self.finish_drained(Some(status), hung, false, None, epoch_base)
                    } else {
                        self.finish_undrained_exit(status, epoch_base)
                    };
                }
                Ok(None) => {
                    if Instant::now() >= deadline {
                        let (reaped, wait_error) = kill_and_reap(
                            &mut self.child,
                            Duration::from_secs(1),
                            child_exited.as_ref(),
                        );
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
                    let (reaped, kill_err) = kill_and_reap(
                        &mut self.child,
                        Duration::from_secs(1),
                        child_exited.as_ref(),
                    );
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
            child_exited: _,
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
        let stdout_panic = join_if_finished(stdout_done, stdout_handle);
        let stderr_panic = join_if_finished(stderr_done, stderr_handle);

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
        let complete = completeness(
            stdout_done,
            stderr_done,
            &stdout_panic,
            &stderr_panic,
            &stdout_reader_error,
            &stderr_reader_error,
        );
        // Reader panics/read-errors are folded into `wait_error` too (rather
        // than only driving `complete`), so a panic message that only
        // prints `wait_error` still names what went wrong.
        let wait_error = {
            let mut parts: Vec<String> = wait_error.into_iter().collect();
            if let Some(p) = stdout_panic {
                parts.push(format!("stdout reader panicked: {p}"));
            }
            if let Some(p) = stderr_panic {
                parts.push(format!("stderr reader panicked: {p}"));
            }
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

/// `epoch_base + ceiling`, clamped instead of panicking if the addition
/// would overflow `Instant`'s internal representation — a `ceiling` extreme
/// enough to trigger this is unreachable in practice (nobody passes a
/// multi-decade ceiling), but the fallback keeps the contract total. Every
/// step uses `checked_add`, never the panicking `+`, including the fallback
/// itself: clamp to `now() + 1 year` (via `checked_add`, not `+`); if even
/// that overflows (only possible if `now()` is itself already absurdly
/// close to `Instant`'s maximum representable value), clamp to `now()`
/// directly. Either fallback yields a deadline that never fires
/// spuriously in any realistic run, without ever risking a panic.
fn saturating_deadline(epoch_base: Instant, ceiling: Duration) -> Instant {
    epoch_base.checked_add(ceiling).unwrap_or_else(|| {
        let now = Instant::now();
        now.checked_add(Duration::from_secs(365 * 24 * 3600))
            .unwrap_or(now)
    })
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
/// finished) and, if it panicked, returns the panic payload rendered as a
/// human-readable string (never discarded — it is what tells a caller
/// *why* `Completeness::ReaderFailed`); otherwise (not done, or finished
/// cleanly) returns `None`. A reader that has not observed EOF is left
/// running rather than joined (joining it would block past the settle bound
/// already enforced by the caller).
fn join_if_finished(done: bool, handle: JoinHandle<()>) -> Option<String> {
    if !done {
        return None;
    }
    handle.join().err().map(|payload| panic_message(&*payload))
}

/// Render a `std::thread` panic payload as a human-readable string. Panics
/// raised via `panic!("...")`/`assert!`/`unwrap` carry a `&str` or `String`
/// payload; anything else (a custom `panic_any` payload) falls back to a
/// fixed message rather than failing to report at all.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "reader thread panicked with a non-string payload".to_string()
    }
}

/// Pure classification of the drained reader threads' terminal state into
/// [`Completeness`], factored out of `finish_drained` so it can be
/// unit-tested directly (a cheap, deterministic `Completeness::ReaderFailed`
/// oracle, without needing to actually inject a panic into a live reader
/// thread — see the `completeness_*` tests below).
fn completeness(
    stdout_done: bool,
    stderr_done: bool,
    stdout_panic: &Option<String>,
    stderr_panic: &Option<String>,
    stdout_reader_error: &Option<String>,
    stderr_reader_error: &Option<String>,
) -> Completeness {
    if !(stdout_done && stderr_done) {
        Completeness::SettleExpired
    } else if stdout_panic.is_some()
        || stderr_panic.is_some()
        || stdout_reader_error.is_some()
        || stderr_reader_error.is_some()
    {
        Completeness::ReaderFailed
    } else {
        Completeness::Complete
    }
}

/// Kill `child` and reap it by polling `try_wait` for up to `ceiling`.
/// Returns the reaped status (`None` if it never reaped within the bound — a
/// documented caveat: a `SIGKILL`ed process in uninterruptible sleep can
/// outlive this poll, and that alone is not an error) and any `kill`/reap
/// error text, joined together and never silently folded into a clean-
/// looking `None`.
fn kill_and_reap(
    child: &mut Child,
    ceiling: Duration,
    child_exited: Option<&Arc<AtomicBool>>,
) -> (Option<ExitStatus>, Option<String>) {
    let mut errors = Vec::new();
    if let Err(e) = child.kill() {
        errors.push(format!("kill: {e}"));
    }
    let deadline = Instant::now() + ceiling;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if let Some(flag) = child_exited {
                    flag.store(true, Ordering::SeqCst);
                }
                return (Some(status), joined_or_none(errors));
            }
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
    /// Both reader threads concluded `Eof` before the ~1 s settle bound
    /// expired. On unix this is not necessarily a literal closed pipe: once
    /// the child is confirmed reaped, an idle 50 ms poll is treated as
    /// `Eof` for that child's own bytes (a reaped child cannot write
    /// again). A foreign process that still holds the write end open, but
    /// is not itself writing faster than that 50 ms window, is
    /// indistinguishable from "done" on any single poll and lands here too
    /// — see the module doc's "Precondition for a complete log".
    Complete,
    /// At least one reader thread had not concluded `Eof` when the settle
    /// bound expired — on unix this means it kept observing `POLLIN` (new
    /// data) with gaps shorter than the 50 ms poll timeout even after the
    /// target child was confirmed reaped: a foreign process (e.g. an
    /// fd-inheriting grandchild) still ACTIVELY WRITING, not merely holding
    /// the pipe open — a silent holder is folded into `Complete` instead
    /// (see the module doc's "Precondition for a complete log"). The
    /// unfinished thread(s) are left running, detached by necessity, until
    /// the pipe's last writer eventually closes it.
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
    /// `true` iff a kill was issued (`killed == true`) *and* the reaped
    /// status is a signal death or a reap give-up — see the module doc's
    /// "hung vs killed vs complete" section and `disposition`. A signal
    /// death `wait_bounded` never killed for (a self-inflicted crash) is
    /// `false`: it is a live signal exit visible via `status`, not a hang.
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
    /// The READER-THREAD completeness axis only: whether both reader
    /// threads concluded `Eof` before the settle bound expired. On unix
    /// this is synthesized from an idle poll after the child is confirmed
    /// reaped, not a literal observed pipe close — see
    /// [`Completeness::Complete`] for the exact boundary (a foreign holder
    /// that writes slower than the 50 ms poll timeout is folded in here
    /// too). `Completeness::Undrained` for a `Capture` from the `cfg(test)`
    /// undrained driver. This does NOT account for the retention cap — a
    /// `Complete` capture can still be missing bytes the cap dropped; use
    /// [`Capture::is_trustworthy`] to gate on both axes at once.
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

    /// The single predicate a consumer should gate a "this evidence is
    /// whole" decision on: both reader threads concluded `Eof`
    /// (`complete == Completeness::Complete` — on unix, an idle 50 ms poll
    /// after the child is confirmed reaped, not necessarily a literal
    /// closed pipe; see [`Completeness::Complete`]), no OS-level error
    /// occurred while producing this `Capture` (`wait_error.is_none()`),
    /// and the retention cap did not drop any bytes from either stream
    /// (`stdout_truncated == 0 && stderr_truncated == 0`). `complete` alone
    /// is only the reader-thread axis (see the module doc's "hung vs killed
    /// vs complete vs trustworthy" section) — a `Complete` capture can
    /// still be silently missing bytes the cap dropped from the middle of a
    /// long-running child's output, so a caller that only checks `complete`
    /// can be fooled into trusting a truncated log. A live foreign writer
    /// with gaps under 50 ms keeps a capture `SettleExpired` (never falsely
    /// `Complete`), but a foreign grandchild's OWN later bytes are never
    /// part of THIS child's evidence in the first place —
    /// `is_trustworthy()` says nothing about data anyone else produces on a
    /// shared fd.
    pub fn is_trustworthy(&self) -> bool {
        self.complete == Completeness::Complete
            && self.wait_error.is_none()
            && self.stdout_truncated == 0
            && self.stderr_truncated == 0
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
    /// running in `ChildMode::Grandchild`/`GrandchildWriting` can reuse it
    /// for its own grandchild spawn (with `CHILD_MODE_ENV` overridden to
    /// `ChildMode::Sleeper`/`WritingSleeper` respectively).
    const SELF_EXACT_ENV: &str = "JAMMI_SELF_EXACT";

    /// The self-exec'd child's behavior, selected by `JAMMI_CHILD_MODE`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ChildMode {
        Flood,
        Wedge,
        Silent,
        Grandchild,
        GrandchildWriting,
        Sleeper,
        WritingSleeper,
        PrintSleep,
    }

    impl ChildMode {
        fn as_str(self) -> &'static str {
            match self {
                ChildMode::Flood => "flood",
                ChildMode::Wedge => "wedge",
                ChildMode::Silent => "silent",
                ChildMode::Grandchild => "grandchild",
                ChildMode::GrandchildWriting => "grandchild-writing",
                ChildMode::Sleeper => "sleeper",
                ChildMode::WritingSleeper => "writing-sleeper",
                ChildMode::PrintSleep => "printsleep",
            }
        }

        fn parse(raw: &str) -> Option<Self> {
            Some(match raw {
                "flood" => ChildMode::Flood,
                "wedge" => ChildMode::Wedge,
                "silent" => ChildMode::Silent,
                "grandchild" => ChildMode::Grandchild,
                "grandchild-writing" => ChildMode::GrandchildWriting,
                "sleeper" => ChildMode::Sleeper,
                "writing-sleeper" => ChildMode::WritingSleeper,
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
            ChildMode::Grandchild => grandchild_child(ChildMode::Sleeper),
            ChildMode::GrandchildWriting => grandchild_child(ChildMode::WritingSleeper),
            ChildMode::Sleeper => sleeper_child(),
            ChildMode::WritingSleeper => writing_sleeper_child(),
            ChildMode::PrintSleep => printsleep_child(),
        }
    }

    /// Build a `Command` that re-execs this very test binary, selecting only
    /// `test_path` (a fully-qualified `module::path::of::the_test_fn`) and
    /// running it uncaptured on a single thread — both flags are load-bearing,
    /// confirmed by running the suite with each one dropped in turn: without
    /// `--nocapture`, libtest swallows the child's `eprintln!` output and the
    /// parent observes zero bytes (a vacuous oracle); without `--exact`,
    /// every test whose name is a prefix match would enter its own dispatch
    /// guard concurrently, producing nondeterministic byte counts.
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

    /// Spawns `current_exe()` in `grandchild_mode`, reusing this test's own
    /// `--exact` path (`SELF_EXACT_ENV`, set by `self_exec` on this very
    /// process) but overriding `CHILD_MODE_ENV` to `grandchild_mode` —
    /// reusing the inherited mode would fork-bomb. The grandchild is spawned
    /// without stdio redirection, so it inherits this process's
    /// stdout/stderr (the pipes the grandparent `DrainedChild` is draining).
    /// Exits 0 immediately, deliberately not waiting on the grandchild.
    fn grandchild_child(grandchild_mode: ChildMode) -> ! {
        let exact = env::var(SELF_EXACT_ENV)
            .expect("grandchild mode requires SELF_EXACT_ENV set by self_exec");
        let mut cmd = Command::new(env::current_exe().expect("current_exe for grandchild spawn"));
        cmd.args(["--exact", &exact, "--nocapture", "--test-threads=1"]);
        cmd.env(CHILD_MODE_ENV, grandchild_mode.as_str());
        let _child = cmd.spawn().expect("spawn grandchild");
        std::process::exit(0);
    }

    /// Sleeps 3 s (holding whatever stdio it inherited open, but never
    /// writing to it) then exits 0. With the poll-based unix reader, merely
    /// inheriting the pipe no longer delays completeness — see
    /// `writing_sleeper_child` for the shape that still does.
    fn sleeper_child() -> ! {
        thread::sleep(Duration::from_secs(3));
        std::process::exit(0);
    }

    /// Writes a short line every 10 ms for 2 s (200 lines), then exits —
    /// unlike `sleeper_child`, this one actively produces new data on the
    /// inherited pipe well past when its parent (the `grandchild`-mode
    /// child) has already exited. The 10 ms gap is deliberately well under
    /// `spawn_reader`'s 50 ms poll timeout: the poll-based unix reader only
    /// concludes `Eof` on an EMPTY poll once the target child has exited, so
    /// a writer with gaps shorter than the poll timeout never presents that
    /// empty poll and the reader keeps draining it — a slower, intermittent
    /// writer (e.g. one line every 500 ms) would instead let a single quiet
    /// gap after the target's exit look exactly like a finished stream,
    /// which is NOT the shape this test exists to pin.
    fn writing_sleeper_child() -> ! {
        for _ in 0..200 {
            eprintln!("hb");
            let _ = io::stderr().flush();
            thread::sleep(Duration::from_millis(10));
        }
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

    // ---- saturating_deadline: pure, no subprocess involved -----------

    #[test]
    fn saturating_deadline_does_not_panic_on_an_overflowing_ceiling() {
        let base = Instant::now();
        // Duration::from_secs(u64::MAX) is a valid Duration but adding it to
        // any real Instant overflows the internal representation on every
        // platform this crate builds for -- this exercises the
        // checked_add-returns-None fallback for real, not just in theory.
        let deadline = saturating_deadline(base, Duration::from_secs(u64::MAX));
        assert!(
            deadline >= base,
            "an overflowing ceiling must still yield a valid, non-panicking \
             Instant at or after epoch_base, deadline={deadline:?} base={base:?}"
        );
    }

    #[test]
    fn saturating_deadline_is_exact_for_an_ordinary_ceiling() {
        let base = Instant::now();
        let deadline = saturating_deadline(base, Duration::from_secs(5));
        assert_eq!(deadline, base + Duration::from_secs(5));
    }

    // ---- completeness: pure, no subprocess involved (the cheap
    // Completeness::ReaderFailed oracle: rather than injecting a real panic
    // into a live reader thread -- which would need a test-only hook into
    // spawn_reader's generic Read type, widening spawn_inner's signature
    // just for this -- the classification logic itself is factored into
    // this pure function and exercised directly here, the same way
    // `disposition` covers `hung` without a live subprocess) -------------

    #[test]
    fn completeness_settle_expired_when_either_reader_is_unfinished() {
        assert_eq!(
            completeness(false, true, &None, &None, &None, &None),
            Completeness::SettleExpired
        );
        assert_eq!(
            completeness(true, false, &None, &None, &None, &None),
            Completeness::SettleExpired
        );
        assert_eq!(
            completeness(false, false, &None, &None, &None, &None),
            Completeness::SettleExpired
        );
    }

    #[test]
    fn completeness_reader_failed_on_a_panic_or_a_read_error() {
        let panic = Some("boom".to_string());
        let err = Some("read error".to_string());
        assert_eq!(
            completeness(true, true, &panic, &None, &None, &None),
            Completeness::ReaderFailed,
            "stdout panic"
        );
        assert_eq!(
            completeness(true, true, &None, &panic, &None, &None),
            Completeness::ReaderFailed,
            "stderr panic"
        );
        assert_eq!(
            completeness(true, true, &None, &None, &err, &None),
            Completeness::ReaderFailed,
            "stdout read error"
        );
        assert_eq!(
            completeness(true, true, &None, &None, &None, &err),
            Completeness::ReaderFailed,
            "stderr read error"
        );
        // SettleExpired takes precedence: an unfinished reader beats a
        // failure recorded on the other, already-finished one.
        assert_eq!(
            completeness(false, true, &None, &panic, &None, &None),
            Completeness::SettleExpired
        );
    }

    #[test]
    fn completeness_complete_when_both_readers_finish_cleanly() {
        assert_eq!(
            completeness(true, true, &None, &None, &None, &None),
            Completeness::Complete
        );
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

    #[cfg(unix)]
    #[test]
    fn disposition_signal_death_without_a_kill_is_not_hung() {
        // A self-inflicted crash (e.g. SIGSEGV) that `wait_bounded` never
        // killed is not a hang -- it's a live signal exit the caller sees
        // via `Capture::status` directly, not something this driver
        // detected and terminated.
        let status = exit_status_signal(11); // SIGSEGV
        assert!(
            !disposition(Some(status), false),
            "a signal death without an issued kill is not `hung`"
        );
    }

    #[cfg(unix)]
    #[test]
    fn disposition_normal_exit_without_a_kill_is_not_hung() {
        let status = exit_status_code(0);
        assert!(!disposition(Some(status), false));
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
            cap.is_trustworthy(),
            "a complete, untruncated, error-free capture must be trustworthy: {cap:?}"
        );

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

    /// A grandchild that merely INHERITS the pipe (never writes to it) no
    /// longer delays completeness on unix: the child here spawns a
    /// `ChildMode::Sleeper` grandchild (inherits stdout/stderr, sleeps 3 s,
    /// writes nothing) and exits immediately. Once `wait_bounded` reaps the
    /// child, the poll-based reader sees an empty pipe and concludes `Eof`
    /// on its own -- it does not wait for the grandchild to actually close
    /// the fd -- so the capture is `Completeness::Complete` and
    /// `is_trustworthy()`, not `SettleExpired`. The non-unix fallback reader
    /// has no such short-circuit (it blocks in `read()` until an actual
    /// EOF), so it still shows the pre-fix `SettleExpired` shape there. See
    /// `settle_expires_only_when_the_grandchild_keeps_writing` for the one
    /// shape that still produces `SettleExpired` on unix too. The 2.5 s wall
    /// bound (vs the grandchild's 3 s sleep) is generous enough to hold on
    /// both platforms; on unix this returns in tens of ms in practice, not
    /// near that bound.
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
        #[cfg(unix)]
        {
            assert_eq!(
                cap.complete,
                Completeness::Complete,
                "a silent grandchild must not block completeness under the poll-based unix reader: {cap:?}"
            );
            assert!(cap.is_trustworthy(), "{cap:?}");
        }
        #[cfg(not(unix))]
        {
            assert_eq!(cap.complete, Completeness::SettleExpired, "{cap:?}");
        }
        assert!(
            wall < Duration::from_millis(2500),
            "settle should return well short of the grandchild's 3s sleep, got {wall:?}: {cap:?}"
        );
        eprintln!(
            "settle_returns_within_bound_when_a_grandchild_holds_the_pipe: measured wall={wall:?} cap.elapsed={:?}",
            cap.elapsed
        );
    }

    /// `Completeness::SettleExpired` still has a real oracle: a grandchild
    /// that keeps WRITING after its parent (the `grandchild-writing`-mode
    /// child) has already exited keeps `POLLIN` arriving, so the poll-based
    /// unix reader keeps draining it and never falsely concludes `Eof` --
    /// only `finish_drained`'s bounded settle (about 1 s) ends the wait,
    /// leaving `Completeness::SettleExpired`. This is the one case the fix
    /// does not (and should not) short-circuit: those bytes are genuinely
    /// still arriving, so calling it anything but incomplete would be
    /// dishonest.
    #[test]
    fn settle_expires_only_when_the_grandchild_keeps_writing() {
        dispatch_if_child();
        let exact = test_exact_path("settle_expires_only_when_the_grandchild_keeps_writing");
        let mut cmd = self_exec(&exact, ChildMode::GrandchildWriting);
        let started = Instant::now();
        let child = DrainedChild::spawn(&mut cmd).expect("spawn grandchild-writing-spawning child");
        let cap = child.wait_bounded(Duration::from_secs(10), Epoch::Spawn);
        let wall = started.elapsed();

        assert!(!cap.hung, "{cap:?}");
        assert!(!cap.killed, "{cap:?}");
        assert_eq!(
            cap.complete,
            Completeness::SettleExpired,
            "a grandchild that keeps writing must still show up as incomplete: {cap:?}"
        );
        assert!(!cap.is_trustworthy(), "{cap:?}");
        assert!(
            wall < Duration::from_millis(2500),
            "the settle bound must still cap the wait well short of the writing grandchild's 3s lifetime, got {wall:?}: {cap:?}"
        );
    }

    /// The retention cap's own oracle: with the cap injected to head 1 MiB +
    /// tail 256 KiB, a 2 MiB flood must retain exactly the first 1 MiB and
    /// exactly the last 256 KiB, report the exact dropped middle, and render
    /// exactly one truncation marker; a flood well under the combined cap
    /// must render no marker at all. Also pins the two-axis distinction the
    /// module doc draws: the over-cap capture's reader threads still reach a
    /// clean EOF (`complete == Completeness::Complete`) even though bytes
    /// were dropped, so `complete` alone would wrongly look like "trust
    /// this" — `is_trustworthy()` must be `false` here specifically because
    /// of the truncation, not because of `complete`.
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
        assert_eq!(
            cap.complete,
            Completeness::Complete,
            "the reader threads still reach a clean EOF despite the cap: {cap:?}"
        );

        let full = expected_flood_stream(OVER_CAP_LINES);
        let expected_truncated = full.len() as u64 - (HEAD_CAP + TAIL_CAP) as u64;
        assert_eq!(cap.stderr_truncated, expected_truncated, "{cap:?}");
        assert!(
            !cap.is_trustworthy(),
            "complete==Complete must NOT imply trustworthy once the cap has \
             dropped bytes: {cap:?}"
        );
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
        assert!(
            sub_cap.is_trustworthy(),
            "well under the cap, with no truncation, must be trustworthy: {sub_cap:?}"
        );
        let sub_expected = expected_flood_stream(SUB_CAP_LINES);
        assert_eq!(sub_cap.stderr, sub_expected, "{sub_cap:?}");
        let sub_rendered = sub_cap.render_stderr();
        assert!(
            !sub_rendered.contains("truncated"),
            "sub-cap flood must render no marker: {sub_rendered}"
        );
    }

    // ---- Epoch::Call, Duration::ZERO, snapshot(), try_wait() ------------

    /// A zero ceiling measured from `Epoch::Call` against a still-running
    /// child is well-defined and kills: the deadline check precedes any
    /// sleep (see `wait_bounded`'s doc), so `Duration::ZERO` does not
    /// silently behave as an unbounded wait. This wall-clock oracle cannot
    /// observe *whether* a single 20 ms poll sleep happened before the kill
    /// (that is an implementation detail the black-box timing here has no
    /// way to distinguish from noise) — what it does prove is that the wait
    /// ends nowhere near the sleeper's own natural 3s lifetime. The bound
    /// (2s) matches `wait_bounded`'s own documented kill-path worst case
    /// ("ceiling plus roughly 2s", ceiling here being zero) rather than a
    /// tighter number: under heavy in-binary parallelism (this module's
    /// other 12+ tests, several themselves spawning subprocesses,
    /// contending for the host's CPUs) a tighter bound was observed to
    /// flake even on the *other* new test below, purely from scheduling
    /// delay, not from `wait_bounded` failing to kill promptly.
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

    // ---- the pipe-CLOEXEC-race oracle -----------------------------------

    /// What this test ACTUALLY exercises: two OS threads of this SAME
    /// process each call `DrainedChild::spawn` as close together as a
    /// `Barrier` can make them, one for a long-lived silent sibling and one
    /// for the flood child, and the flood capture completes and reports
    /// `Completeness::Complete` / `is_trustworthy()`. It is a shape-level
    /// sanity check for concurrent spawning from independent threads under
    /// ORDINARY (non-racing) conditions -- it does not discriminate any one
    /// mechanism. It cannot pin `SPAWN_LOCK`'s necessity (see that static's
    /// own doc): the fd a racing `fork` would mis-inherit lands at an
    /// arbitrary descriptor number in the sibling's own fd table that the
    /// sibling's code never targets, so it is only ever silently held, not
    /// written to, regardless of whether the lock ran. Nor does it
    /// discriminate the poll-based reader's immunity to a silently-held
    /// foreign fd (confirmed by reverting the reader to a plain blocking
    /// `read()` and rerunning this same test unchanged: still green, 30-100
    /// runs) -- the CLOEXEC race this test tries to hit is itself too rare
    /// to reproduce reliably here, on either mechanism. The real oracle for
    /// "a foreign process silently holding an inherited pipe end does not
    /// block completeness" is `settle_returns_within_bound_when_a_grandchild_holds_the_pipe`,
    /// which constructs that holding deterministically (via direct,
    /// unredirected fd inheritance, not an accidental race) and does go red
    /// against a reverted (blocking-read) reader.
    #[test]
    fn concurrent_spawns_do_not_race_the_pipe_cloexec_window() {
        dispatch_if_child();
        let exact = test_exact_path("concurrent_spawns_do_not_race_the_pipe_cloexec_window");

        let barrier = Arc::new(std::sync::Barrier::new(2));
        let sibling_barrier = Arc::clone(&barrier);
        let sibling_exact = exact.clone();
        let sibling = thread::spawn(move || {
            let mut cmd = self_exec(&sibling_exact, ChildMode::Sleeper);
            sibling_barrier.wait();
            let child = DrainedChild::spawn(&mut cmd).expect("spawn sibling sleeper");
            child.wait_bounded(Duration::from_secs(10), Epoch::Spawn)
        });

        let mut flood_cmd = self_exec(&exact, ChildMode::Flood);
        barrier.wait();
        let flood_child = DrainedChild::spawn(&mut flood_cmd).expect("spawn flood child");
        let flood_cap = flood_child.wait_bounded(Duration::from_secs(30), Epoch::Spawn);

        let sibling_cap = sibling.join().expect("sibling thread must not panic");

        assert!(!flood_cap.hung, "{flood_cap:?}");
        assert!(!sibling_cap.hung, "{sibling_cap:?}");
        assert_eq!(
            flood_cap.complete,
            Completeness::Complete,
            "the flood child's pipe must not be held open by the concurrently spawned sibling sleeper: {flood_cap:?}"
        );
        assert!(flood_cap.is_trustworthy(), "{flood_cap:?}");
    }

    /// The oracle for the reader's exit-observation ordering (F1), pinned
    /// deterministically rather than hoped for: drives `spawn_reader`
    /// directly (bypassing `DrainedChild`/subprocesses entirely) with a
    /// hand-controlled real pipe, using the `ready_hook` parameter to pause
    /// the reader thread's loop EXACTLY between `poll` returning 0 and the
    /// `exited_before_poll` decision. The hook then writes the final line,
    /// closes the write end, and sets `child_exited` -- reproducing "the
    /// child writes its final line, exits, and is reaped inside the
    /// interval between poll-returned-0 and the flag read" on demand, on
    /// every run, rather than relying on wall-clock alignment (a live
    /// subprocess reproduction of this exact shape did not naturally occur
    /// in 580 attempts across two earlier probing sessions on a real box --
    /// the true window is narrower than timing alone can reliably hit, which
    /// is exactly why this test drives the mechanism directly instead).
    ///
    /// Pre-fix ordering (load `child_exited` AFTER `poll`, reproduced by
    /// moving the load in `spawn_reader` below the `poll` call and rerunning
    /// this same test): RED, deterministically, every time --
    /// `complete_outcome=Some(Eof) has_final=false stderr=""`, byte-for-byte
    /// the shape this fix closes. Post-fix (the committed ordering): GREEN,
    /// deterministically -- the flag was already `false` when sampled
    /// before this `poll` call, so the reader `continue`s and re-polls,
    /// correctly observing the data the hook just wrote.
    #[cfg(unix)]
    #[test]
    fn exit_observed_then_the_final_bytes_are_still_read() {
        use std::io::Write as _;
        use std::os::unix::io::FromRawFd;

        let mut fds = [0i32; 2];
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0, "pipe(2) failed");
        // SAFETY: `fds[0]`/`fds[1]` are freshly created, valid, distinct fds
        // from the `pipe(2)` call just above; each is taken exactly once.
        let read_file = unsafe { std::fs::File::from_raw_fd(fds[0]) };
        let write_fd = fds[1];

        let buf = Arc::new(Mutex::new(RetainedBuf::new(
            DEFAULT_HEAD_CAP,
            DEFAULT_TAIL_CAP,
        )));
        let outcome = Arc::new(Mutex::new(None));
        let child_exited = Arc::new(AtomicBool::new(false));

        let hook_child_exited = Arc::clone(&child_exited);
        let ready_hook: Box<dyn FnOnce() + Send> = Box::new(move || {
            // SAFETY: `write_fd` is the write end from the `pipe(2)` call
            // above, not yet taken by anything else at this point.
            let mut write_file = unsafe { std::fs::File::from_raw_fd(write_fd) };
            write_file
                .write_all(
                    b"final-line
",
                )
                .expect("write the final line");
            drop(write_file); // close the write end
            hook_child_exited.store(true, Ordering::SeqCst);
        });

        let handle = spawn_reader(
            read_file,
            Arc::clone(&buf),
            None,
            Arc::clone(&outcome),
            Arc::clone(&child_exited),
            Some(ready_hook),
        );
        handle.join().expect("reader thread must not panic");

        let raw = lock_or_recover(&buf).raw();
        let text = String::from_utf8_lossy(&raw);
        let outcome_val = lock_or_recover(&outcome).clone();
        assert!(
            text.contains("final-line"),
            "exit observed then the final bytes must still be read: \
             complete_outcome={outcome_val:?} has_final=false stderr={text:?}"
        );
    }
}
