//! Distributional divergence kernels (Jensen-Shannon, PSI, 1D Wasserstein,
//! cosine-similarity shift) and the random-projection helper that lets a
//! high-dimensional vector population be summarised as a 1-D histogram for
//! these tests.

pub mod cosine_shift;
pub mod jensen_shannon;
pub mod projection;
pub mod psi;
pub mod wasserstein;

pub use cosine_shift::cosine_similarity_shift;
pub use jensen_shannon::jensen_shannon;
pub use projection::{
    build_projection_vector, mean_pairwise_cosine, project, project_all, PROJECTION_SEED,
};
pub use psi::{psi, PSI_EPSILON};
pub use wasserstein::wasserstein_1d;
