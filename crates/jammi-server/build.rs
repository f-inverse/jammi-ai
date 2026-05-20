use std::path::PathBuf;

fn main() {
    let proto_root = PathBuf::from("proto");
    let proto_files = vec![
        proto_root.join("jammi/v1/session.proto"),
        proto_root.join("jammi/v1/trigger.proto"),
    ];

    for f in &proto_files {
        println!("cargo:rerun-if-changed={}", f.display());
    }
    println!("cargo:rerun-if-changed=proto");

    tonic_prost_build::configure()
        .build_client(false)
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
