//! [`MemoryLimit`] — the one grammar every memory bound in the configuration
//! is written in: `[engine] memory_limit` bounds the host-side query pool,
//! `[gpu] memory_limit` bounds each device's model residency. Both say the
//! same thing — "this much of that memory" — at two scales, so they parse
//! through one type and differ only in the total a percentage resolves
//! against.

use std::fmt;
use std::num::NonZeroU64;
use std::str::FromStr;

use serde::Deserialize;

/// A percentage in `1..=100`: the share of a memory total a bound allows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Percent(u8);

impl Percent {
    /// `n` percent, or `None` outside `1..=100`.
    pub const fn new(n: u8) -> Option<Self> {
        match n {
            1..=100 => Some(Self(n)),
            _ => None,
        }
    }

    /// This share of `total` bytes, rounded down.
    pub fn of(self, total: u64) -> u64 {
        // Widened so a total near `u64::MAX` cannot overflow the product.
        (u128::from(total) * u128::from(self.0) / 100) as u64
    }
}

/// A configured memory bound: a share of the memory it bounds, or an
/// absolute size.
///
/// # Grammar
///
/// - `"<n>%"`, `1 <= n <= 100`: that share of the bounded memory's total.
/// - `"<n>GB"` / `"<n>MB"` / `"<n>KB"`: `n` binary (1024-based) units.
/// - `"<n>"`: `n` bytes, unadorned.
///
/// Anything else — an empty string, a decimal, a unit with no digits, an
/// unrecognised or lower-case suffix, a negative number, a zero — is refused
/// when the configuration is parsed, so no reader ever holds an unparsed
/// bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(try_from = "String")]
pub enum MemoryLimit {
    /// A share of the bounded memory's total, resolved by the reader that
    /// knows which total that is.
    Share(Percent),
    /// An absolute size in bytes.
    Bytes(NonZeroU64),
}

impl MemoryLimit {
    /// The bound in bytes. `total` is asked for only when the bound is a
    /// share, so an absolute bound never probes the memory it bounds.
    pub fn resolve<E>(self, total: impl FnOnce() -> Result<u64, E>) -> Result<u64, E> {
        match self {
            Self::Share(percent) => Ok(percent.of(total()?)),
            Self::Bytes(bytes) => Ok(bytes.get()),
        }
    }

    const UNITS: [(&'static str, u64); 3] = [
        ("GB", 1024 * 1024 * 1024),
        ("MB", 1024 * 1024),
        ("KB", 1024),
    ];
}

/// Why a string is not a [`MemoryLimit`]. Its message names the grammar;
/// the configuration loader prefixes the key it was written under.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryLimitGrammarError(String);

impl fmt::Display for MemoryLimitGrammarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?} is not a memory limit: use \"<n>%\" (1-100), \"<n>GB\"/\"<n>MB\"/\"<n>KB\", \
             or \"<n>\" (bytes), with n > 0",
            self.0
        )
    }
}

impl std::error::Error for MemoryLimitGrammarError {}

impl FromStr for MemoryLimit {
    type Err = MemoryLimitGrammarError;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let refused = || MemoryLimitGrammarError(raw.to_string());
        let trimmed = raw.trim();
        // `u64`'s parser accepts a leading `+`; the grammar is digits only.
        let digits = |s: &str| {
            s.bytes()
                .all(|b| b.is_ascii_digit())
                .then(|| s.parse::<u64>().ok())
                .flatten()
                .ok_or_else(refused)
        };
        if let Some(pct) = trimmed.strip_suffix('%') {
            return u8::try_from(digits(pct)?)
                .ok()
                .and_then(Percent::new)
                .map(Self::Share)
                .ok_or_else(refused);
        }
        let (count, unit) = Self::UNITS
            .iter()
            .find_map(|(suffix, unit)| trimmed.strip_suffix(suffix).map(|n| (n, *unit)))
            .unwrap_or((trimmed, 1));
        digits(count)?
            .checked_mul(unit)
            .and_then(NonZeroU64::new)
            .map(Self::Bytes)
            .ok_or_else(refused)
    }
}

impl TryFrom<String> for MemoryLimit {
    type Error = MemoryLimitGrammarError;

    fn try_from(raw: String) -> Result<Self, Self::Error> {
        raw.parse()
    }
}

/// Renders the bound in the grammar it parses from, in the largest unit
/// that states it exactly.
impl fmt::Display for MemoryLimit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Share(Percent(n)) => write!(f, "{n}%"),
            Self::Bytes(bytes) => {
                let bytes = bytes.get();
                match Self::UNITS.iter().find(|(_, unit)| bytes % unit == 0) {
                    Some((suffix, unit)) => write!(f, "{}{suffix}", bytes / unit),
                    None => write!(f, "{bytes}"),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn bytes(n: u64) -> MemoryLimit {
        MemoryLimit::Bytes(NonZeroU64::new(n).unwrap())
    }

    #[test]
    fn every_form_parses_to_its_bound() {
        assert_eq!("134217728".parse(), Ok(bytes(128 * 1024 * 1024)));
        assert_eq!("2GB".parse(), Ok(bytes(2 * GIB)));
        assert_eq!("128MB".parse(), Ok(bytes(128 * 1024 * 1024)));
        assert_eq!("131072KB".parse(), Ok(bytes(128 * 1024 * 1024)));
        assert_eq!(" 90% ".parse(), Ok(MemoryLimit::Share(Percent(90))));
        assert_eq!("007".parse(), Ok(bytes(7)));
    }

    #[test]
    fn every_malformed_form_is_refused() {
        for raw in [
            "", "0", "0%", "101%", "300%", "4.5GB", "-1", "+5", "GB", "4gb", "auto", "0GB",
        ] {
            assert!(
                raw.parse::<MemoryLimit>().is_err(),
                "{raw:?} must be refused"
            );
        }
    }

    #[test]
    fn a_share_resolves_against_the_total_and_an_absolute_bound_never_asks() {
        let share: MemoryLimit = "90%".parse().unwrap();
        assert_eq!(share.resolve(|| Ok::<_, ()>(GIB)), Ok(GIB * 9 / 10));
        let absolute: MemoryLimit = "512MB".parse().unwrap();
        assert_eq!(
            absolute.resolve(|| -> Result<u64, ()> { panic!("an absolute bound probed") }),
            Ok(512 * 1024 * 1024)
        );
        assert_eq!(Percent::new(100).unwrap().of(u64::MAX), u64::MAX);
    }

    #[test]
    fn display_round_trips_through_the_grammar() {
        for raw in ["75%", "2GB", "3MB", "5KB", "1025"] {
            let limit: MemoryLimit = raw.parse().unwrap();
            assert_eq!(limit.to_string(), raw);
            assert_eq!(limit.to_string().parse(), Ok(limit));
        }
    }
}
