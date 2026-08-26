fn plain_and_escaped_strings() -> &'static str {
    "a plain string with \"escaped quotes\" and a \\backslash inside"
}

fn strings_containing_comment_syntax() -> (&'static str, &'static str) {
    (
        "a string with // inside it, not a real comment",
        "a string with /* inside it, not a real comment either */ still string",
    )
}

fn raw_strings() -> (&'static str, &'static str, &'static str) {
    (
        r"a raw string with a \backslash and no escapes",
        r#"a raw string with a "quote" inside, needs one hash"#,
        r##"a raw string needing two hashes because it contains "# inside"##,
    )
}

fn byte_strings() -> (&'static [u8], &'static [u8]) {
    (
        b"a plain byte string",
        br#"a raw byte string with a "quote" inside"#,
    )
}

#[cfg(target_os = "linux")]
fn cfg_gated_only_on_linux() -> bool {
    true
}

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn cfg_gated_via_a_nested_predicate_string() -> bool {
    false
}
