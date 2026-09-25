//! A `cuda` build of the extension links the CUDA runtime libraries
//! dynamically (cudarc's `dynamic-linking`), and the loader resolves them when
//! Python imports the extension — before any code of ours runs. The
//! `jammi-ai-native-cu12` wheel does not carry them: it depends on the
//! `nvidia-*-cu12` wheels, which pip installs beside it in the same
//! `site-packages`, each under `nvidia/<component>/lib`. So the extension's
//! RUNPATH names each of those directories relative to its own location
//! (`site-packages/jammi_native/`).
//!
//! The component list is the one `packaging/server-cu12/jammi_server/_entry.py`
//! puts on the server's loader path, and the one `verify_link_set.py` checks
//! the link set against; `ci/scripts/test_cu12_component_contract.py` keeps the
//! statements in agreement.

const CUDA_COMPONENTS: [&str; 5] = ["cuda_runtime", "cublas", "curand", "cuda_nvrtc", "nccl"];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let cuda = std::env::var_os("CARGO_FEATURE_CUDA").is_some();
    let linux = std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "linux");
    if cuda && linux {
        let runpath = CUDA_COMPONENTS
            .map(|component| format!("$ORIGIN/../nvidia/{component}/lib"))
            .join(":");
        println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,{runpath}");
    }
}
