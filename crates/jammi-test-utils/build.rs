//! Codegen for the test-only gRPC fixtures: the echo service tests mount on
//! the engine's gRPC chain the way a downstream mounts its own services.

fn main() {
    // A fresh checkout has no guarantee of a system protoc on PATH; an
    // explicit `PROTOC` still wins.
    if std::env::var_os("PROTOC").is_none() {
        let protoc = protoc_bin_vendored::protoc_bin_path()
            .expect("vendored protoc binary unavailable for this host target");
        std::env::set_var("PROTOC", protoc);
    }
    println!("cargo:rerun-if-changed=proto");
    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        .compile_protos(&["proto/jammi/test/echo.proto"], &["proto"])
        .expect("compile the echo fixture proto");
}
