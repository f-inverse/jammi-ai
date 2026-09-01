//! The CUDA launch-domain facts every op's CUDA glue shares: the
//! element-count ceiling each dispatch refuses above, and the grid-stride
//! launch geometry `crate::cuda::geglu` launches with.
//!
//! ## Why these live in `ops`, not in `crate::cuda`
//!
//! `mod cuda` is `#[cfg(feature = "cuda")]`. A unit test of these rules
//! placed there would only ever COMPILE on a CUDA-feature build — i.e.
//! never on the CPU lane that runs every day, which is exactly the lane
//! that can prove them (they are pure arithmetic and domain facts: no
//! device, no PTX, no launch). So they live here, where they are compiled
//! and TESTED unconditionally, and `crate::cuda` re-exports them. One
//! definition, not a second copy that could drift (`cuda::mod`'s
//! `pub(crate) use` is a re-export, not a fork).
//!
//! ## The INDEXING CONTRACT (campaign #446, finding 4)
//!
//! `geglu_f16.cu`'s grid-stride loop used to carry a 32-bit induction
//! variable (`for (unsigned int idx = ...; idx < n_out; idx += blockDim.x
//! * gridDim.x)`); the f32/bf16 twins and the three `layer_norm` `dgamma`
//! column loops carried the identical pattern. The stride is
//! [`GEGLU_BLOCK`] * [`GEGLU_MAX_GRID`] = 16_776_960, so a 32-bit lane
//! walks `idx = (idx + stride) mod 2^32` and can only ever hold values
//! congruent to its own start modulo `gcd(stride, 2^32) = 256`. Two
//! distinct failures follow, both re-derived (never restated) by the
//! model in `tests` below:
//!
//! * **Hang.** The loop exits only on a value in `[n_out, 2^32)`. That
//!   window is `2^32 - n_out` wide, so once `n_out > u32::MAX - 255` it
//!   cannot hold one value of every residue class, and every lane whose
//!   class is missing spins forever. At the largest count the guard
//!   admits (`n_out == u32::MAX`) the window is a single value, so 255 of
//!   every 256 threads never terminate.
//! * **Re-processing.** Just below that window the loop does escape — but
//!   only after re-walking its whole orbit: at `n_out == u32::MAX - 255`
//!   lane 0 visits 65_537 indices where 257 was the job. For `geglu` the
//!   repeated writes are idempotent (each output element is a pure
//!   function of its own index), so this one costs 255x the work rather
//!   than a wrong number — but the same loop shape in an ACCUMULATING
//!   kernel would double-count.
//!
//! Both halves of the fix are stated here because they are one contract,
//! and each half is only sound given the other:
//!
//! 1. **In-kernel INDEX arithmetic is 64-bit.** Every grid-stride loop in
//!    `src/cuda/*.cu` declares its induction variable `size_t` (or
//!    `unsigned long long`) and casts the `blockIdx.x * blockDim.x` /
//!    `blockDim.x * gridDim.x` products to that width BEFORE multiplying.
//!    A 64-bit induction variable cannot wrap for ANY `n` this crate can
//!    admit, so the hang is structurally impossible rather than merely
//!    unreached. `tests::every_grid_stride_loop_in_a_cuda_source_is_64_bit`
//!    below re-checks this against the `.cu` sources themselves, so a NEW
//!    kernel written with the old pattern reds the CPU lane.
//!
//! 2. **Kernel scalar PARAMETERS stay 32-bit** (`const unsigned int
//!    n_out`, `const unsigned int hidden`, ...), and so does the launch
//!    grid, which every dispatch builds from `n as u32`. This is the
//!    DELIBERATE half of the contract, not an oversight: the Rust glue
//!    pushes those arguments by value (`builder.arg(&n_out_u32)`), and a
//!    kernel parameter widened on the `.cu` side WITHOUT widening the
//!    pushed Rust value in the same edit makes the kernel read 8 bytes
//!    where 4 were pushed — a garbage upper half, out-of-bounds writes,
//!    and no small-shape test can catch it. Keeping the parameter width
//!    pinned at 32 bits keeps [`check_elem_count_fits_u32`] the single
//!    place that bounds it.
//!
//! So the ceiling below is a fact about the PARAMETER/GRID width, never
//! about the index width — the index width is what makes every `n` under
//! that ceiling safe to walk.

use candle_core::{Error, Result};

/// Refuse an element count the CUDA launch cannot address.
///
/// `n > u32::MAX` truncates silently: every dispatch builds its launch
/// grid from `n as u32` and pushes the element count to the kernel as a
/// 32-bit `unsigned int` scalar parameter (see this module's own doc for
/// why the parameter width is pinned at 32 bits and only the kernels'
/// INDEX arithmetic is 64-bit). A truncated count under-launches, leaving
/// the output allocation's tail uninitialized — a confident wrong answer,
/// not a crash (family D / K2), so it is refused here instead.
///
/// This is the ONE place that fact lives. Every op's own domain check
/// calls it: `cuda::geglu::cuda_fwd`/`cuda_bwd_dwi_out` directly,
/// `cuda::axpy`/`cast_scale`/`scaled_cast_add`/`adamw_step` directly, and
/// `layer_norm`/`rope`/`rope_positions`'s combined `check_cuda_domain`
/// plus `softmax`'s combined `check_last_and_n` for the `u32::MAX` half
/// of their check, alongside their own op-specific ceiling
/// (`MAX_HIDDEN`/`MAX_HEAD_DIM`/`MAX_LAST_DIM`). `cuda::dropout` is the
/// one deliberate non-caller — its kernels are 64-bit end to end,
/// PARAMETER included, and its own source carries the reviewed
/// `ELEM-COUNT-GUARD-WAIVER:` marker that
/// `tests::every_cuda_dispatch_bounds_its_element_count` requires.
pub(crate) fn check_elem_count_fits_u32(op: &'static str, n: usize) -> Result<()> {
    if n > u32::MAX as usize {
        return Err(Error::Msg(format!(
            "{op}: {n} elements exceeds u32::MAX ({}); this op's CUDA launch grid is built \
             from `n as u32` and its kernel takes the element count as a 32-bit `unsigned \
             int` parameter, so a larger count would silently truncate. (The kernels' own \
             INDEX arithmetic is 64-bit — that is what makes every count at or below this \
             ceiling safe to walk without a grid-stride wrap.)",
            u32::MAX
        )));
    }
    Ok(())
}

/// Grid-stride block size for `geglu.cu`/`geglu_f16.cu` — must match the
/// launch `crate::cuda::geglu::launch_config` builds (which reads THIS
/// constant, so the two cannot drift).
pub(crate) const GEGLU_BLOCK: u32 = 256;

/// A conservative 1-D grid cap for the same kernels. The kernels' own
/// grid-stride loop covers any `n_out` beyond `GEGLU_BLOCK *
/// GEGLU_MAX_GRID` correctly — unlike `Axpy`'s single-pass `if (i < n)`
/// kernel, these do not need the grid to cover `n_out` in one pass.
/// `GEGLU_BLOCK * GEGLU_MAX_GRID` is the grid-stride STRIDE, i.e. exactly
/// the quantity that used to overflow a 32-bit induction variable (this
/// module's doc).
pub(crate) const GEGLU_MAX_GRID: u32 = 65_535;

/// The 1-D grid `crate::cuda::geglu::launch_config` launches for `n_out`
/// output elements. Split out from the `LaunchConfig` construction itself
/// so the CPU lane can test the geometry (and model the loop it implies)
/// without the `cuda` feature.
pub(crate) fn geglu_grid_blocks(n_out: u32) -> u32 {
    n_out.div_ceil(GEGLU_BLOCK).clamp(1, GEGLU_MAX_GRID)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::path::{Path, PathBuf};

    fn cuda_src_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/cuda")
    }

    /// Every file in `src/cuda`, by extension, sorted — fail-closed: a
    /// missing directory or an empty result panics rather than silently
    /// vacating every scan below.
    fn cuda_sources(extensions: &[&str]) -> Vec<(String, String)> {
        let dir = cuda_src_dir();
        let entries = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
        let mut out: Vec<(String, String)> = Vec::new();
        for entry in entries {
            let path = entry.expect("dir entry").path();
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if !extensions.contains(&ext) {
                continue;
            }
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .expect("utf-8 file name")
                .to_string();
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
            out.push((name, text));
        }
        out.sort();
        assert!(
            !out.is_empty(),
            "no {extensions:?} source found under {} — this scan would be vacuous, so it \
             fails closed instead",
            dir.display()
        );
        out
    }

    /// Blank out `//`-line and `/* */`-block comment bodies (keeping
    /// newlines) so a pattern quoted in prose is never mistaken for code.
    fn strip_comments(src: &str) -> String {
        let bytes: Vec<char> = src.chars().collect();
        let mut out = String::with_capacity(src.len());
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == '/' && i + 1 < bytes.len() && bytes[i + 1] == '/' {
                while i < bytes.len() && bytes[i] != '\n' {
                    out.push(' ');
                    i += 1;
                }
            } else if bytes[i] == '/' && i + 1 < bytes.len() && bytes[i + 1] == '*' {
                out.push_str("  ");
                i += 2;
                while i < bytes.len()
                    && !(bytes[i] == '*' && i + 1 < bytes.len() && bytes[i + 1] == '/')
                {
                    out.push(if bytes[i] == '\n' { '\n' } else { ' ' });
                    i += 1;
                }
                if i < bytes.len() {
                    out.push_str("  ");
                    i += 2;
                }
            } else {
                out.push(bytes[i]);
                i += 1;
            }
        }
        out
    }

    /// The text of every `for (...)` header in `src`, paren-balanced (so a
    /// multi-line header, e.g. `dropout.cu`'s, is returned whole).
    fn for_headers(src: &str) -> Vec<String> {
        let mut headers = Vec::new();
        let chars: Vec<char> = src.chars().collect();
        let mut i = 0;
        while i + 3 < chars.len() {
            let is_for = chars[i] == 'f'
                && chars[i + 1] == 'o'
                && chars[i + 2] == 'r'
                && (i == 0 || !chars[i - 1].is_alphanumeric() && chars[i - 1] != '_');
            if !is_for {
                i += 1;
                continue;
            }
            let mut j = i + 3;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j >= chars.len() || chars[j] != '(' {
                i += 1;
                continue;
            }
            let start = j;
            let mut depth = 0usize;
            while j < chars.len() {
                match chars[j] {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            assert!(
                j < chars.len(),
                "unbalanced `for (` header starting at char {i} — the scan cannot be \
                 computed, so it fails closed"
            );
            headers.push(chars[start + 1..j].iter().collect::<String>());
            i = j + 1;
        }
        headers
    }

    /// F4, kernel half of the indexing contract: no `.cu`/`.cuh` under
    /// `src/cuda` may walk a grid-stride loop (one whose increment or
    /// condition mentions `gridDim`) with a 32-bit induction variable. A
    /// new kernel copying the old `unsigned int idx` pattern reds HERE, on
    /// the CPU lane, rather than hanging a GPU at a shape no test can
    /// afford to allocate.
    #[test]
    fn every_grid_stride_loop_in_a_cuda_source_is_64_bit() {
        let mut checked = 0usize;
        let mut offenders: Vec<String> = Vec::new();
        for (name, text) in cuda_sources(&["cu", "cuh"]) {
            let stripped = strip_comments(&text);
            for header in for_headers(&stripped) {
                if !header.contains("gridDim") {
                    continue;
                }
                checked += 1;
                let init = header.split(';').next().unwrap_or("").trim().to_string();
                let declared_type = match init.find('=') {
                    Some(eq) => init[..eq].trim().to_string(),
                    None => init.clone(),
                };
                // Drop the induction variable's own name: everything up to
                // the last whitespace-separated token is the type.
                let ty = match declared_type.rsplit_once(char::is_whitespace) {
                    Some((ty, _var)) => ty.trim().to_string(),
                    None => declared_type.clone(),
                };
                let ty = ty.replace('*', " ");
                let normalized = ty.split_whitespace().collect::<Vec<_>>().join(" ");
                if normalized != "size_t" && normalized != "unsigned long long" {
                    offenders.push(format!(
                        "{name}: grid-stride loop induction variable declared \
                         `{normalized}` (init clause `{init}`)"
                    ));
                }
            }
        }
        assert!(
            checked >= 6,
            "expected at least the six geglu grid-stride loops to be found; found only \
             {checked} — the scan is not reaching the sources it claims to check"
        );
        assert!(
            offenders.is_empty(),
            "grid-stride loops with a 32-bit induction variable wrap past `u32::MAX` and \
             never terminate (campaign #446 finding 4); every one must be `size_t` or \
             `unsigned long long`: {offenders:#?}"
        );
    }

    /// F4, host half of the indexing contract: every CUDA dispatch module
    /// that launches one of this crate's own kernels must bound its
    /// element count through [`check_elem_count_fits_u32`] — or carry the
    /// reviewed `ELEM-COUNT-GUARD-WAIVER:` marker stating, at the site,
    /// why its kernels need no ceiling. A NEW op's glue that forgets the
    /// guard reds here; it cannot ship silently unbounded.
    #[test]
    fn every_cuda_dispatch_bounds_its_element_count() {
        const GUARD: &str = "check_elem_count_fits_u32";
        const WAIVER: &str = "ELEM-COUNT-GUARD-WAIVER:";
        let mut launching = 0usize;
        let mut unbounded: Vec<String> = Vec::new();
        for (name, text) in cuda_sources(&["rs"]) {
            let stripped = strip_comments(&text);
            if !stripped.contains(".launch(") {
                continue;
            }
            launching += 1;
            // The waiver is prose, so it is looked for in the ORIGINAL
            // text; the guard call is code, so it is looked for in the
            // comment-stripped text (a guard "called" only inside a
            // comment must not count).
            if stripped.contains(GUARD) || text.contains(WAIVER) {
                continue;
            }
            unbounded.push(name);
        }
        assert!(
            launching >= 8,
            "expected at least eight kernel-launching dispatch modules under src/cuda; \
             found {launching} — the scan is not reaching the sources it claims to check"
        );
        assert!(
            unbounded.is_empty(),
            "these CUDA dispatch modules launch a kernel without bounding the element \
             count through `{GUARD}` and without a reviewed `{WAIVER}` marker: \
             {unbounded:#?}"
        );
    }

    /// Every op name the guard is actually invoked with, ENUMERATED FROM
    /// THE SOURCE (never a hand list that a new op could be added without
    /// appearing in): each `const OP: &str = "..."` and each string
    /// literal passed directly to the guard, in a file that calls it.
    fn guarded_op_names() -> Vec<String> {
        let mut names: HashSet<String> = HashSet::new();
        for (_name, text) in cuda_sources(&["rs"]) {
            let stripped = strip_comments(&text);
            if !stripped.contains("check_elem_count_fits_u32") {
                continue;
            }
            for (marker, offset) in [
                ("const OP: &str = \"", "const OP: &str = \"".len()),
                (
                    "check_elem_count_fits_u32(\"",
                    "check_elem_count_fits_u32(\"".len(),
                ),
            ] {
                let mut from = 0usize;
                while let Some(at) = stripped[from..].find(marker) {
                    let start = from + at + offset;
                    let end = start
                        + stripped[start..]
                            .find('"')
                            .expect("unterminated op-name string literal");
                    names.insert(stripped[start..end].to_string());
                    from = end;
                }
            }
        }
        let mut out: Vec<String> = names.into_iter().collect();
        out.sort();
        out
    }

    /// The guard's own boundary oracle, run for EVERY op enumerated from
    /// the source: the largest legal count is admitted, and the three
    /// counts at and past the ceiling (including `u32::MAX - 255`, the
    /// value at which the pre-fix 32-bit grid-stride loop wrapped, which
    /// is LEGAL and must stay legal now that the loop is 64-bit) are
    /// dispositioned correctly, with the op's own name in the refusal.
    #[test]
    fn the_element_count_guard_refuses_exactly_past_u32_max_for_every_op() {
        let ops = guarded_op_names();
        assert!(
            ops.len() >= 10,
            "expected at least ten guarded op names enumerated from src/cuda; found \
             {ops:?} — the enumeration is not reaching the sources it claims to read"
        );
        for op in &ops {
            // `check_elem_count_fits_u32` takes `&'static str`; the names
            // are read at runtime, so leak one static per op (a handful of
            // short strings, in a test process).
            let op_static: &'static str = Box::leak(op.clone().into_boxed_str());

            // Admitted: the pre-fix wrap point and the largest legal count.
            // Both are inside the domain the 64-bit index arithmetic makes
            // walkable, so refusing either would be a false negative.
            for legal in [u32::MAX as usize - 255, u32::MAX as usize] {
                assert!(
                    check_elem_count_fits_u32(op_static, legal).is_ok(),
                    "{op}: {legal} <= u32::MAX must be ADMITTED (the grid-stride loops are \
                     64-bit, so this count is walkable)"
                );
            }

            // Refused, with the op named: the first count past the ceiling
            // and a decisively larger one.
            for illegal in [u32::MAX as usize + 1, u64::from(u32::MAX) as usize * 4] {
                let err = check_elem_count_fits_u32(op_static, illegal)
                    .expect_err(&format!("{op}: {illegal} > u32::MAX must be REFUSED"));
                let msg = err.to_string();
                assert!(
                    msg.contains(op.as_str()),
                    "{op}: the refusal must name the op ({msg})"
                );
                assert!(
                    msg.contains(&illegal.to_string()),
                    "{op}: the refusal must name the offending count ({msg})"
                );
            }
        }
    }

    /// Degenerate/boundary counts on the SMALL end (family D): zero and
    /// one are ordinary, admitted counts — the guard is a ceiling, never a
    /// floor.
    #[test]
    fn the_element_count_guard_admits_the_degenerate_small_counts() {
        for n in [0usize, 1, 2] {
            assert!(
                check_elem_count_fits_u32("boundary_probe", n).is_ok(),
                "n = {n} must be admitted; the guard is a ceiling, not a floor"
            );
        }
    }

    /// One grid-stride lane's walk, modelled exactly as the kernel writes
    /// it: `for (idx = start; idx < n; idx += stride)`, with `idx`
    /// arithmetic taken modulo `index_modulus` when the induction variable
    /// is narrower than the values it must hold.
    #[derive(Debug, PartialEq, Eq)]
    enum LaneWalk {
        /// The loop exited; `visits` is how many indices it processed.
        Terminates { visits: u64 },
        /// The walk returned to its own starting index without ever
        /// meeting the exit condition. `idx -> (idx + stride) mod m` is a
        /// BIJECTION, so its orbit is a single cycle: coming back to
        /// `start` proves the loop repeats that cycle forever. Detected
        /// this way, rather than with a set of every visited index,
        /// because the orbit here holds 2^24 elements.
        NeverTerminates { cycle_length: u64 },
    }

    /// `index_modulus = Some(1 << 32)` models a `unsigned int` induction
    /// variable (the pre-fix kernels); `None` models `size_t` (the fixed
    /// ones). Nothing else differs between the two models — that is the
    /// point.
    fn walk_grid_stride_lane(
        start: u64,
        stride: u64,
        n: u64,
        index_modulus: Option<u64>,
    ) -> LaneWalk {
        let mut idx = start;
        let mut visits = 0u64;
        while idx < n {
            visits += 1;
            idx += stride;
            if let Some(m) = index_modulus {
                idx %= m;
                if idx == start {
                    return LaneWalk::NeverTerminates {
                        cycle_length: visits,
                    };
                }
            }
        }
        LaneWalk::Terminates { visits }
    }

    /// `gcd(a, b)`, Euclid — used to state the orbit structure below
    /// independently of the simulation.
    fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            let t = a % b;
            a = b;
            b = t;
        }
        a
    }

    /// `a^-1 mod m` by the extended Euclidean algorithm (panics unless
    /// `gcd(a, m) == 1`) — an INDEPENDENT closed form for "how many steps
    /// until the wrapped walk lands on its exit value", never a second run
    /// of the simulation it is checking.
    fn inv_mod(a: u64, m: u64) -> u64 {
        let (mut old_r, mut r) = (a as i128, m as i128);
        let (mut old_s, mut s) = (1i128, 0i128);
        while r != 0 {
            let q = old_r / r;
            (old_r, r) = (r, old_r - q * r);
            (old_s, s) = (s, old_s - q * s);
        }
        assert_eq!(old_r, 1, "{a} is not invertible modulo {m}");
        old_s.rem_euclid(m as i128) as u64
    }

    /// F4's mechanism-level RED, which no GPU run can afford to reproduce:
    /// triggering it needs ~4.29e9 elements (≈26 GB for the f16 forward's
    /// input alone, ≈42 GB for the backward), so a hardware oracle for
    /// this finding would never run on any device this project tests on.
    /// This models the loop instead, at the EXACT stride the launch code
    /// builds ([`GEGLU_BLOCK`] * [`geglu_grid_blocks`], read from the same
    /// constants `crate::cuda::geglu::launch_config` reads — not re-typed
    /// literals).
    ///
    /// THE HANG. With a 32-bit induction variable the walk is
    /// `idx = (idx + stride) mod 2^32`, so lane `start` can only ever hold
    /// values `≡ start (mod g)` where `g = gcd(stride, 2^32) = 256`. The
    /// loop exits only on a value in `[n, 2^32)`. That window holds
    /// `2^32 - n` values, so when `2^32 - n < g` it cannot contain one of
    /// every residue class — and every lane whose residue is missing spins
    /// forever. `2^32 - n < 256` is exactly `n > u32::MAX - 255`: the
    /// finding's own window, re-derived here from the arithmetic rather
    /// than restated. At `n = u32::MAX` the window is the single value
    /// `2^32 - 1 ≡ 255 (mod 256)`, so 255 of every 256 threads hang.
    #[test]
    fn the_32_bit_geglu_grid_stride_loop_hangs_within_255_of_u32_max() {
        let stride = u64::from(GEGLU_BLOCK) * u64::from(geglu_grid_blocks(u32::MAX));
        assert_eq!(
            geglu_grid_blocks(u32::MAX),
            GEGLU_MAX_GRID,
            "at this element count the grid is capped, so the stride is the full \
             GEGLU_BLOCK * GEGLU_MAX_GRID span"
        );
        assert_eq!(
            stride, 16_776_960,
            "grid-stride span, from the launch constants themselves"
        );
        let modulus = 1u64 << 32;
        let g = gcd(stride, modulus);
        assert_eq!(
            g, 256,
            "the orbit of a 32-bit lane steps in multiples of gcd"
        );

        // `u32::MAX` (the largest count the guard admits) and the FIRST
        // count inside the hang window — both must hang lane 0.
        for n_out in [u32::MAX, u32::MAX - 254] {
            let n = u64::from(n_out);
            assert!(
                modulus - n < g,
                "fixture invariant: n = {n_out} must sit inside the hang window \
                 (2^32 - n < {g})"
            );
            // Independent reason, not just the simulation's verdict: no
            // value in the exit window shares lane 0's residue class.
            assert!(
                !(n..modulus).any(|v| v % g == 0),
                "fixture invariant: lane 0 must have NO reachable exit value at \
                 n = {n_out}"
            );
            match walk_grid_stride_lane(0, stride, n, Some(modulus)) {
                LaneWalk::NeverTerminates { cycle_length } => assert_eq!(
                    cycle_length,
                    modulus / g,
                    "the cycle must be the WHOLE orbit of lane 0 under +stride \
                     (2^32 / gcd), i.e. the loop walks every multiple of {g} below \
                     2^32 and comes back, forever, at n = {n_out}"
                ),
                LaneWalk::Terminates { visits } => panic!(
                    "NEGATIVE CONTROL FAILED: the 32-bit model must NOT terminate at \
                     n_out = {n_out} (it terminated after {visits} visits) — if this \
                     ever passes, the model no longer reproduces the finding it stands \
                     in for"
                ),
            }
            // The fix, same n, same stride, only the index width changed.
            assert_eq!(
                walk_grid_stride_lane(0, stride, n, None),
                LaneWalk::Terminates {
                    visits: n.div_ceil(stride)
                },
                "the 64-bit model must terminate after exactly ceil(n / stride) visits"
            );
        }
    }

    /// The second, quieter half of the same defect, one element BELOW the
    /// hang window (`n = u32::MAX - 255`, where the exit window is exactly
    /// `g = 256` wide so every residue class does have an exit): the
    /// 32-bit loop still wraps — it just eventually escapes, after
    /// re-walking the buffer 255 extra times. For `geglu` the redundant
    /// writes are idempotent (every output element is a pure function of
    /// its own index), so the damage is a silent 255x work blowup rather
    /// than a wrong number — but the same shape in a kernel that
    /// accumulates would double-count.
    ///
    /// The expected visit count is derived in CLOSED FORM, not read back
    /// off the simulation: lane 0's orbit is `{k·s mod 2^32}`, its only
    /// member in `[n, 2^32)` is `n` itself, so the loop exits at the first
    /// `k` with `k·s ≡ n (mod 2^32)` — i.e. `k ≡ (n/g)·(s/g)^-1 (mod
    /// 2^32/g)` — and it visits exactly `k` indices on the way.
    #[test]
    fn the_32_bit_geglu_grid_stride_loop_reprocesses_the_buffer_just_below_the_hang_window() {
        let n_out = u32::MAX - 255;
        let n = u64::from(n_out);
        let stride = u64::from(GEGLU_BLOCK) * u64::from(geglu_grid_blocks(n_out));
        let modulus = 1u64 << 32;
        let g = gcd(stride, modulus);
        assert_eq!(
            modulus - n,
            g,
            "fixture invariant: this is the widest exit window that still covers every \
             residue class, i.e. the last count that does NOT hang"
        );

        let derived_visits = (n / g) * inv_mod(stride / g, modulus / g) % (modulus / g);
        let honest_visits = n.div_ceil(stride);
        assert_eq!(
            derived_visits, 65_537,
            "closed-form check of the closed form: s/g = 65535 = 2^16-1 and \
             (2^16-1)(2^16+1) ≡ -1 (mod 2^24), so k ≡ 2^16+1"
        );
        assert_eq!(honest_visits, 257, "what the loop is supposed to do");

        assert_eq!(
            walk_grid_stride_lane(0, stride, n, Some(modulus)),
            LaneWalk::Terminates {
                visits: derived_visits
            },
            "the 32-bit model must re-walk the buffer exactly as the closed form predicts"
        );
        assert_eq!(
            walk_grid_stride_lane(0, stride, n, None),
            LaneWalk::Terminates {
                visits: honest_visits
            },
            "the 64-bit model must visit each of its own indices exactly once"
        );
        assert_eq!(
            derived_visits - honest_visits,
            65_280,
            "the redundant-visit count the 32-bit loop pays: 255 extra passes"
        );
    }

    /// Non-vacuity of the model itself: at a count well inside the safe
    /// region BOTH models terminate identically, so the divergence above
    /// is attributable to the wrap and not to the two models simply being
    /// different functions.
    #[test]
    fn both_grid_stride_models_agree_below_the_wrap_point() {
        let n_out = 10_000_000u32;
        let stride = u64::from(GEGLU_BLOCK) * u64::from(geglu_grid_blocks(n_out));
        let n = u64::from(n_out);
        let narrow = walk_grid_stride_lane(0, stride, n, Some(1 << 32));
        let wide = walk_grid_stride_lane(0, stride, n, None);
        assert_eq!(
            narrow, wide,
            "below the wrap point the two models must agree"
        );
        assert_eq!(wide, LaneWalk::Terminates { visits: 1 });
    }

    /// Coverage (family D, the "did the fixed loop still visit everything
    /// exactly once" half): on a small synthetic geometry, walking EVERY
    /// lane of the 64-bit model visits each of `n` indices exactly once —
    /// including the tail, where `n` is not a multiple of the stride.
    #[test]
    fn the_64_bit_grid_stride_walk_covers_every_index_exactly_once() {
        let block = 4u64;
        let blocks = 3u64;
        let stride = block * blocks;
        for n in [1u64, 11, 12, 13, 37, 120] {
            let mut visited: Vec<u64> = Vec::new();
            for lane in 0..stride {
                match walk_grid_stride_lane(lane, stride, n, None) {
                    LaneWalk::Terminates { .. } => {}
                    LaneWalk::NeverTerminates { cycle_length } => {
                        panic!("64-bit lane {lane} must terminate (cycled after {cycle_length})")
                    }
                }
                let mut idx = lane;
                while idx < n {
                    visited.push(idx);
                    idx += stride;
                }
            }
            visited.sort_unstable();
            assert_eq!(
                visited,
                (0..n).collect::<Vec<u64>>(),
                "n = {n}: every index in 0..n must be visited exactly once"
            );
        }
    }

    /// The launch geometry itself, at its own boundaries (family D): one
    /// block minimum (never a zero-block, illegal launch), exact-multiple
    /// and tail shapes below the cap, and saturation at the cap.
    #[test]
    fn the_geglu_grid_geometry_holds_at_its_boundaries() {
        assert_eq!(
            geglu_grid_blocks(0),
            1,
            "a zero-block grid is an illegal launch"
        );
        assert_eq!(geglu_grid_blocks(1), 1);
        assert_eq!(
            geglu_grid_blocks(GEGLU_BLOCK),
            1,
            "exact multiple of the block"
        );
        assert_eq!(geglu_grid_blocks(GEGLU_BLOCK + 1), 2, "one-element tail");
        let at_cap = GEGLU_BLOCK * GEGLU_MAX_GRID;
        assert_eq!(geglu_grid_blocks(at_cap), GEGLU_MAX_GRID);
        assert_eq!(
            geglu_grid_blocks(at_cap + 1),
            GEGLU_MAX_GRID,
            "beyond the cap the grid saturates and the kernel's own grid-stride loop \
             covers the rest"
        );
        assert_eq!(geglu_grid_blocks(u32::MAX), GEGLU_MAX_GRID);
    }
}
