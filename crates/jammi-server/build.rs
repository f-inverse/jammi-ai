use std::path::PathBuf;

fn main() {
    let proto_root = PathBuf::from("proto");
    let proto_files = vec![
        proto_root.join("jammi/v1/session.proto"),
        proto_root.join("jammi/v1/trigger.proto"),
        proto_root.join("jammi/v1/embedding.proto"),
    ];

    for f in &proto_files {
        println!("cargo:rerun-if-changed={}", f.display());
    }
    println!("cargo:rerun-if-changed=proto");

    tonic_prost_build::configure()
        // Client stubs are used by the integration-test harness
        // (crates/jammi-server/tests/it/{flight_tenant,grpc_session}.rs)
        // to drive `SessionService.SetTenant` and `TriggerService.ListTopics`
        // against an in-process server. Other callers of the gRPC surface
        // (jammi-cli, future SDK crates) build their own clients on top of
        // these generated stubs.
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
