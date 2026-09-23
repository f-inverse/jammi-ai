//! The seam between the operators and a model: what the runner calls, and
//! nothing else. A consumer implements [`ModelRuntime`] over its own model
//! cache — binding a model to a process — and [`BoundModel`] over the
//! bound model: what it describes about itself, how it costs and prepares
//! rows on the host, how its device admits a forward, and the forward.

use std::any::Any;
use std::sync::Arc;

use arrow::array::ArrayRef;
use async_trait::async_trait;
use jammi_numerics::ShapeLadder;

use crate::error::{Error, Result};
use crate::inference::adapter::DistributionForm;
use crate::inference::observer::InferenceObserver;
use crate::inference::output::BackendOutput;
use crate::source::ModelSource;
use crate::task::ModelTask;

/// A chunk prepared for the device by the runtime that will forward it:
/// opaque to the operators, which only carry it from
/// [`BoundModel::prepare`] to [`BoundModel::forward`].
pub type Prepared = Box<dyn Any + Send>;

/// The admission a device grants one forward: held for the forward call and
/// released on drop. What it holds is the runtime's own (a semaphore
/// permit, a budget reservation); the operators only hold and drop it.
pub struct ForwardPermit {
    /// Held for the permit's lifetime and dropped with it: the runtime's
    /// own reservation, never read here.
    _held: Box<dyn Any + Send>,
}

impl ForwardPermit {
    /// An admission over `held`, released when the permit drops.
    pub fn new(held: impl Any + Send) -> Self {
        Self {
            _held: Box::new(held),
        }
    }
}

impl std::fmt::Debug for ForwardPermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ForwardPermit")
    }
}

/// How a forward failed — classified by the runtime, which knows its device,
/// never by the operators over an error's text. An out-of-memory failure is
/// the one the runner recovers from, by halving the chunk and retrying the
/// same rows; every other failure is systemic and propagates.
#[derive(Debug)]
pub enum ForwardError {
    /// The device ran out of memory on this chunk.
    OutOfMemory(String),
    /// Any other failure.
    Other(Error),
}

impl From<ForwardError> for Error {
    fn from(error: ForwardError) -> Self {
        match error {
            ForwardError::OutOfMemory(message) => {
                Error::Inference(format!("out of memory at the minimum chunk: {message}"))
            }
            ForwardError::Other(error) => error,
        }
    }
}

/// A model bound to this process, ready to run: what the operators ask of
/// it, and nothing else.
#[async_trait]
pub trait BoundModel: Send + Sync {
    /// Output dimensionality of the model's embedding head.
    fn embedding_dim(&self) -> usize;

    /// The persisted predictive-distribution form of a regression head, or
    /// `None` for a model that is not one.
    fn regression_form(&self) -> Option<&DistributionForm>;

    /// The scale a regression head's served σ is multiplied by to leave the
    /// standardized space it was trained in, or `None` when it serves σ as
    /// is.
    fn regression_std_scale(&self) -> Option<f32>;

    /// The cost of every row of `content` under `task`: its length along the
    /// axis a forward pads (its token count for text; one for a fixed-shape
    /// input). An empty or null row, which never reaches the device, costs
    /// zero.
    fn row_costs(&self, content: &[ArrayRef], task: ModelTask) -> Result<Vec<u32>>;

    /// The ladder a forward under `task` pads its rows on.
    fn shape_ladder(&self, task: ModelTask) -> Result<ShapeLadder>;

    /// The host half of a forward: prepare `content` for the device.
    fn prepare(&self, content: &[ArrayRef], task: ModelTask) -> Result<Prepared>;

    /// Admit one forward on this model's device, waiting for a slot. The
    /// permit is held for the forward call and released on drop.
    async fn admit_forward(&self) -> Result<ForwardPermit>;

    /// The device half of a forward: run the model over a prepared chunk.
    fn forward(&self, prepared: Prepared) -> std::result::Result<BackendOutput, ForwardError>;
}

/// What binds a model to a process: the model cache a consumer runs, asked
/// for a model by source and task when a node first needs it.
#[async_trait]
pub trait ModelRuntime: Send + Sync {
    /// Bind `source` for `task` in this process, loading it if it is not
    /// resident. The returned model stays bound for as long as it is held.
    async fn bind(&self, source: &ModelSource, task: ModelTask) -> Result<Arc<dyn BoundModel>>;
}

/// The process-local handles an [`InferenceExec`](crate::InferenceExec)
/// runs against. Never serialized: a node rebuilt in another process binds
/// to that process's own.
#[derive(Clone)]
pub struct InferenceRuntime {
    /// Where the node's model is bound.
    pub model: Arc<dyn ModelRuntime>,
    /// Observes every output batch.
    pub observer: Option<Arc<dyn InferenceObserver>>,
}

/// A runtime with no model behind it, for the operators' own tests: a
/// text row costs its words plus two special tokens, a forward answers
/// with whatever the test's closure says, and admission is a semaphore
/// the test sizes.
#[cfg(test)]
pub(crate) mod stub {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use arrow::array::{Array, ArrayRef, StringArray};
    use async_trait::async_trait;
    use jammi_numerics::ShapeLadder;

    use super::{
        BoundModel, ForwardError, ForwardPermit, InferenceRuntime, ModelRuntime, Prepared,
    };
    use crate::error::{Error, Result};
    use crate::inference::adapter::DistributionForm;
    use crate::inference::output::BackendOutput;
    use crate::source::ModelSource;
    use crate::task::ModelTask;

    /// What a forward answers, given the rows it was handed.
    pub(crate) type Forward =
        Box<dyn Fn(usize) -> std::result::Result<BackendOutput, ForwardError> + Send + Sync>;

    pub(crate) struct StubModel {
        pub(crate) embedding_dim: usize,
        /// The widest row a forward pads to: the ladder runs in powers of
        /// two up to it.
        pub(crate) max_width: usize,
        pub(crate) admission: Arc<tokio::sync::Semaphore>,
        pub(crate) forward: Forward,
    }

    impl StubModel {
        /// A one-wide embedding model with unbounded admission whose
        /// forward answers every row with `1.0`.
        pub(crate) fn embedding() -> Self {
            Self {
                embedding_dim: 1,
                max_width: 128,
                admission: Arc::new(tokio::sync::Semaphore::new(
                    tokio::sync::Semaphore::MAX_PERMITS,
                )),
                forward: Box::new(|len| Ok(ones(len))),
            }
        }
    }

    /// A one-head output of `len` rows of `1.0`.
    pub(crate) fn ones(len: usize) -> BackendOutput {
        BackendOutput {
            float_outputs: vec![vec![1.0; len]],
            string_outputs: vec![],
            row_status: vec![true; len],
            row_errors: vec![String::new(); len],
            shapes: vec![(len, 1)],
        }
    }

    #[async_trait]
    impl BoundModel for StubModel {
        fn embedding_dim(&self) -> usize {
            self.embedding_dim
        }

        fn regression_form(&self) -> Option<&DistributionForm> {
            None
        }

        fn regression_std_scale(&self) -> Option<f32> {
            None
        }

        /// A row's words plus the two special tokens a text encoder adds;
        /// an empty or null row costs nothing.
        fn row_costs(&self, content: &[ArrayRef], _task: ModelTask) -> Result<Vec<u32>> {
            let texts = content[0]
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| Error::Inference("the stub costs text".into()))?;
            Ok(texts
                .iter()
                .map(|text| match text {
                    Some(text) if !text.trim().is_empty() => {
                        text.split_whitespace().count() as u32 + 2
                    }
                    _ => 0,
                })
                .collect())
        }

        fn shape_ladder(&self, _task: ModelTask) -> Result<ShapeLadder> {
            Ok(ShapeLadder::new(self.max_width))
        }

        fn prepare(&self, content: &[ArrayRef], _task: ModelTask) -> Result<Prepared> {
            Ok(Box::new(content[0].len()))
        }

        async fn admit_forward(&self) -> Result<ForwardPermit> {
            let permit = Arc::clone(&self.admission)
                .acquire_owned()
                .await
                .map_err(|_| Error::Inference("admission closed".into()))?;
            Ok(ForwardPermit::new(permit))
        }

        fn forward(&self, prepared: Prepared) -> std::result::Result<BackendOutput, ForwardError> {
            let len = *prepared
                .downcast::<usize>()
                .expect("the stub prepared this chunk");
            (self.forward)(len)
        }
    }

    /// Binds one [`StubModel`] to every request, counting the binds.
    pub(crate) struct StubRuntime {
        model: Arc<StubModel>,
        pub(crate) binds: AtomicUsize,
    }

    impl StubRuntime {
        pub(crate) fn binds(&self) -> usize {
            self.binds.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl ModelRuntime for StubRuntime {
        async fn bind(
            &self,
            _source: &ModelSource,
            _task: ModelTask,
        ) -> Result<Arc<dyn BoundModel>> {
            self.binds.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::clone(&self.model) as Arc<dyn BoundModel>)
        }
    }

    /// An [`InferenceRuntime`] over `model`, and the handle that counts its
    /// binds.
    pub(crate) fn runtime(model: StubModel) -> (InferenceRuntime, Arc<StubRuntime>) {
        let stub = Arc::new(StubRuntime {
            model: Arc::new(model),
            binds: AtomicUsize::new(0),
        });
        (
            InferenceRuntime {
                model: Arc::clone(&stub) as Arc<dyn ModelRuntime>,
                observer: None,
            },
            stub,
        )
    }
}
