//! Proto-generated types for the `jammi.v1` API surface.
//!
//! `build.rs` emits one file per proto package under `OUT_DIR`; each module
//! below mounts the matching package. Both client and server stubs are built
//! (`build.rs` sets `.build_client(true).build_server(true)`): the server stubs
//! back `jammi-server`'s service impls, the client stubs back the integration-
//! test harness and a future `remote client`.

pub mod error {
    tonic::include_proto!("jammi.v1.error");
}
pub mod catalog {
    tonic::include_proto!("jammi.v1.catalog");
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
pub mod pipeline {
    tonic::include_proto!("jammi.v1.pipeline");
}
pub mod training {
    tonic::include_proto!("jammi.v1.training");
}
pub mod audit {
    tonic::include_proto!("jammi.v1.audit");
}
