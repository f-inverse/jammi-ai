//! The workspace `serde_json` declaration (`Cargo.toml` `[workspace.dependencies]`,
//! `features = [.., "preserve_order"]`) reached this crate's unit graph.
//!
//! With `preserve_order` a `serde_json::Map` keeps insertion order; without it
//! keys sort. A flip of this assertion means the feature is no longer uniform
//! across build scopes, and every `serde_json::Map`/`json!` producer whose bytes
//! are persisted must be re-swept for order sensitivity.

#[test]
fn serde_json_map_keeps_insertion_order() {
    let mut map = serde_json::Map::new();
    map.insert("b".to_string(), serde_json::Value::from(1));
    map.insert("a".to_string(), serde_json::Value::from(2));
    let text = serde_json::Value::Object(map).to_string();
    assert_eq!(
        text, r#"{"b":1,"a":2}"#,
        "the workspace `preserve_order` declaration did not reach this crate's unit graph"
    );
}
