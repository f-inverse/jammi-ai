//! Codegen for the gRPC wire surface.
//!
//! `jammi-wire` is the candle-free wire substrate; codegen always runs (there is
//! no feature gate), so every consumer that depends on this crate gets the
//! generated `jammi.v1` tonic stubs (client + server) without naming a feature.

fn main() {
    use std::env;
    use std::path::PathBuf;

    // Source builds (`cargo install jammi-cli`) have no guarantee of a system
    // protoc on PATH. Point prost at the vendored binary when the env does not
    // already name one — an explicit `PROTOC` (CI, dev shells) still wins, so
    // those keep using the system toolchain.
    if std::env::var_os("PROTOC").is_none() {
        let protoc = protoc_bin_vendored::protoc_bin_path()
            .expect("vendored protoc binary unavailable for this host target");
        std::env::set_var("PROTOC", protoc);
    }

    let proto_root = PathBuf::from("proto");
    let proto_files = vec![
        proto_root.join("jammi/v1/error.proto"),
        proto_root.join("jammi/v1/catalog.proto"),
        proto_root.join("jammi/v1/trigger.proto"),
        proto_root.join("jammi/v1/embedding.proto"),
        proto_root.join("jammi/v1/inference.proto"),
        proto_root.join("jammi/v1/eval.proto"),
        proto_root.join("jammi/v1/pipeline.proto"),
        proto_root.join("jammi/v1/training.proto"),
        proto_root.join("jammi/v1/audit.proto"),
    ];

    for f in &proto_files {
        println!("cargo:rerun-if-changed={}", f.display());
    }
    println!("cargo:rerun-if-changed=proto");

    // Generate `prost::Name` for every message, with the standard
    // `type.googleapis.com` domain. The error-wire path packs a typed detail
    // into a `google.rpc.Status.details` `Any`, whose `type_url` must be the
    // canonical `type.googleapis.com/<full.name>` a gRPC-web client (and the
    // gRPC rich-error spec) expects; `Name::type_url()` supplies exactly that.
    // These are `prost_build::Config` knobs, so they ride a `compile_with_config`
    // rather than the tonic `Builder` surface.
    let mut config = tonic_prost_build::Config::new();
    config.enable_type_names();
    config.type_name_domain(["."], "type.googleapis.com");

    // Export the compiled `FileDescriptorSet` so consumers can decode the
    // service surface from the binary itself. The tenant-isolation oracle
    // derives the live `jammi.v1` rpc list from this descriptor rather than a
    // hand-maintained constant, so a new rpc is structurally forced into
    // coverage. `OUT_DIR` is always set for build scripts.
    let descriptor_path =
        PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set for build scripts"))
            .join("jammi_descriptor.bin");

    tonic_prost_build::configure()
        // Both client and server stubs are built: the server stubs back
        // `jammi-server`'s service impls; the client stubs back `jammi-admin` /
        // `jammi-client` and the integration-test harness that drives an
        // in-process server.
        .build_client(true)
        .build_server(true)
        .file_descriptor_set_path(&descriptor_path)
        .compile_with_config(
            config,
            &proto_files
                .iter()
                .map(|p| p.to_str().unwrap())
                .collect::<Vec<_>>(),
            &[proto_root.to_str().unwrap()],
        )
        .expect("failed to compile proto files");
}
