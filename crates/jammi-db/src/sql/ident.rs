//! Identifier quoting for generated SQL.
//!
//! A table, source, schema, or column name that the engine interpolates into a
//! generated SQL string is an *identifier*, not data: it must be ANSI
//! double-quoted (and any embedded `"` doubled) so a name carrying a hyphen,
//! mixed case, a space, or a reserved spelling resolves verbatim instead of
//! being re-parsed as an operator or keyword. Unquoted `patents-2024` parses as
//! `patents` minus `2024`; quoted `"patents-2024"` is the one identifier.
//!
//! Every site that builds SQL by string interpolation routes its identifiers
//! through here so the quoting rule lives in one place.

/// Quote a single SQL identifier with ANSI double-quotes, doubling any embedded
/// `"`. Accepted by DataFusion, SQLite, and Postgres alike.
///
/// ```
/// use jammi_db::sql::quote_ident;
/// assert_eq!(quote_ident("patents-2024"), r#""patents-2024""#);
/// assert_eq!(quote_ident(r#"a"b"#), r#""a""b""#);
/// ```
pub fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// Quote each dot-separated part of a multi-part relation reference, joining
/// the quoted parts back with `.`. A reference like `catalog.schema.table`
/// becomes `"catalog"."schema"."table"` — each part is one identifier, so a
/// hyphenated, mixed-case, or reserved part resolves verbatim while the dots
/// stay structural. Quoting the whole string would instead yield one
/// identifier carrying literal dots (`"catalog.schema.table"`), which is the
/// wrong relation. A single-part reference is just `quote_ident`.
///
/// ```
/// use jammi_db::sql::quote_relation;
/// assert_eq!(quote_relation("catalog.schema.table"), r#""catalog"."schema"."table""#);
/// assert_eq!(quote_relation("papers"), r#""papers""#);
/// assert_eq!(quote_relation("my-cat.papers-2024"), r#""my-cat"."papers-2024""#);
/// ```
pub fn quote_relation(rel: &str) -> String {
    rel.split('.')
        .map(quote_ident)
        .collect::<Vec<_>>()
        .join(".")
}

/// Quote a source-scoped table as the three-part relation a registered source's
/// data is addressed by: `"<source>".public."<table>"`. Both the source name
/// and the table name are quoted identifiers, so a hyphenated source (a user's
/// `patents-2024`) or table resolves verbatim.
pub fn source_relation(source_id: &str, table_name: &str) -> String {
    format!(
        "{}.public.{}",
        quote_ident(source_id),
        quote_ident(table_name)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quotes_plain_identifier() {
        assert_eq!(quote_ident("papers"), r#""papers""#);
    }

    #[test]
    fn quotes_hyphenated_identifier() {
        // The #15 distributed-lane bug: an unquoted hyphen parses as a minus
        // operator. Quoting makes the hyphen part of the one identifier.
        assert_eq!(quote_ident("patents-2024"), r#""patents-2024""#);
    }

    #[test]
    fn doubles_embedded_quote() {
        assert_eq!(quote_ident(r#"a"b"#), r#""a""b""#);
    }

    #[test]
    fn quote_relation_quotes_each_multipart_segment() {
        assert_eq!(
            quote_relation("catalog.schema.table"),
            r#""catalog"."schema"."table""#
        );
    }

    #[test]
    fn quote_relation_single_part_is_one_identifier() {
        assert_eq!(quote_relation("papers"), r#""papers""#);
    }

    #[test]
    fn quote_relation_quotes_hyphenated_parts() {
        // Each hyphenated part is one identifier; the dots stay structural.
        assert_eq!(
            quote_relation("my-cat.papers-2024"),
            r#""my-cat"."papers-2024""#
        );
    }

    #[test]
    fn quote_relation_doubles_embedded_quote_per_part() {
        assert_eq!(quote_relation(r#"cat.a"b"#), r#""cat"."a""b""#);
    }

    #[test]
    fn source_relation_quotes_both_parts() {
        assert_eq!(
            source_relation("my-source-2024", "papers-v2"),
            r#""my-source-2024".public."papers-v2""#
        );
    }
}
