//! Shared helpers for the jammi-server integration tests. The submodules
//! exist so that multiple test files can share infrastructure without the
//! DRY violation of three near-identical copies (see CLAUDE.md §DRY).

pub mod grpc;
