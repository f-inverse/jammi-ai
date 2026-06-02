//! Codegen for the gRPC wire surface.
//!
//! Gated behind the default-off `wire` feature so a default `jammi-ai` build
//! (and every embedded / PyO3 consumer) invokes no protoc, reads no `.proto`,
//! and links none of the tonic build-deps. Cargo compiles this whole script in
//! every build, so the `tonic_prost_build` reference is itself behind
//! `#[cfg(feature = "wire")]` — without the feature the script is an empty
//! `main`.

fn main() {
    #[cfg(feature = "wire")]
    generate();
}

#[cfg(feature = "wire")]
fn generate() {
    use std::path::PathBuf;

    let proto_root = PathBuf::from("proto");
    let proto_files = vec![
        proto_root.join("jammi/v1/error.proto"),
        proto_root.join("jammi/v1/session.proto"),
        proto_root.join("jammi/v1/trigger.proto"),
        proto_root.join("jammi/v1/embedding.proto"),
        proto_root.join("jammi/v1/inference.proto"),
        proto_root.join("jammi/v1/eval.proto"),
        proto_root.join("jammi/v1/fine_tune.proto"),
        proto_root.join("jammi/v1/mutable_table.proto"),
        proto_root.join("jammi/v1/channel.proto"),
        proto_root.join("jammi/v1/audit.proto"),
    ];

    for f in &proto_files {
        println!("cargo:rerun-if-changed={}", f.display());
    }
    println!("cargo:rerun-if-changed=proto");

    tonic_prost_build::configure()
        // Both client and server stubs are built: the server stubs back
        // `jammi-server`'s service impls; the client stubs back the integration-
        // test harness (crates/jammi-server/tests/it/*) that drives an in-process
        // server, and a future `RemoteSession` in this crate.
        .build_client(true)
        .build_server(true)
        .compile_protos(
            &proto_files
                .iter()
                .map(|p| p.to_str().unwrap())
                .collect::<Vec<_>>(),
            &[proto_root.to_str().unwrap()],
        )
        .expect("failed to compile proto files");
}
