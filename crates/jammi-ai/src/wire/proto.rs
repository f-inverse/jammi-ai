//! Proto-generated types for the `jammi.v1` API surface.
//!
//! `build.rs` emits one file per proto package under `OUT_DIR`; each module
//! below mounts the matching package. Both client and server stubs are built
//! (`build.rs` sets `.build_client(true).build_server(true)`): the server stubs
//! back `jammi-server`'s service impls, the client stubs back the integration-
//! test harness and a future `RemoteSession`.

pub mod error {
    tonic::include_proto!("jammi.v1.error");
}
pub mod session {
    tonic::include_proto!("jammi.v1.session");
}
pub mod trigger {
    tonic::include_proto!("jammi.v1.trigger");
}
pub mod embedding {
    tonic::include_proto!("jammi.v1.embedding");
}
pub mod inference {
    tonic::include_proto!("jammi.v1.inference");
}
pub mod eval {
    tonic::include_proto!("jammi.v1.eval");
}
pub mod training {
    tonic::include_proto!("jammi.v1.training");
}
pub mod mutable_table {
    tonic::include_proto!("jammi.v1.mutable_table");
}
pub mod channel {
    tonic::include_proto!("jammi.v1.channel");
}
pub mod audit {
    tonic::include_proto!("jammi.v1.audit");
}
