fn char_literals() -> (char, char, char, char, char) {
    ('a', '\'', '"', '\n', '\u{1F600}')
}

fn a_lifetime_is_not_a_char_literal<'a>(x: &'a str) -> &'a str {
    x
}

fn byte_char_literals() -> (u8, u8) {
    (b'x', b'\'')
}
