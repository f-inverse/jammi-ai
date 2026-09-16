//! Codegen for the `jammi.ballista.v1` operator-descriptor messages
//! `JammiCodec` carries across a scheduler/executor boundary. Same shape as
//! `crates/jammi-wire/build.rs`'s proto compile, minus the service stubs —
//! this package defines messages only.

fn main() {
    use std::path::PathBuf;

    // Source builds have no guarantee of a system protoc on PATH — point
    // prost at the vendored binary unless the environment already names one.
    if std::env::var_os("PROTOC").is_none() {
        let protoc = protoc_bin_vendored::protoc_bin_path()
            .expect("vendored protoc binary unavailable for this host target");
        std::env::set_var("PROTOC", protoc);
    }

    let proto_root = PathBuf::from("proto");
    let proto_files = vec![proto_root.join("jammi/ballista/v1/plan.proto")];

    for f in &proto_files {
        println!("cargo:rerun-if-changed={}", f.display());
    }
    println!("cargo:rerun-if-changed=proto");

    tonic_prost_build::configure()
        // Messages only — no service is defined in the .proto, and the
        // codec speaks these bytes directly (`Message::encode`/`decode`)
        // rather than over a gRPC call.
        .build_client(false)
        .build_server(false)
        .compile_protos(
            &proto_files
                .iter()
                .map(|p| p.to_str().unwrap())
                .collect::<Vec<_>>(),
            &[proto_root.to_str().unwrap()],
        )
        .expect("failed to compile proto files");
}
