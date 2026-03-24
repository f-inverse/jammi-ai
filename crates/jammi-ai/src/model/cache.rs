use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use jammi_engine::catalog::model_repo::RegisterModelParams;
use jammi_engine::error::{JammiError, Result};
use tokio::sync::RwLock;

use super::backend::candle::CandleBackend;
use super::backend::ort::OrtBackend;
use super::backend::{DeviceConfig, ModelBackend};
use super::resolver::ModelResolver;
use super::{BackendType, LoadedModel, ModelGuard, ModelId, ModelSource, ModelTask};
use crate::concurrency::{GpuPermit, GpuScheduler};

/// Where a cached model currently resides.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelResidency {
    /// Weights are loaded on GPU memory.
    Gpu,
    /// Weights are loaded in CPU memory.
    Cpu,
    /// Model has been evicted and is no longer in memory.
    Unloaded,
}

struct CacheEntry {
    model: Arc<LoadedModel>,
    ref_count: Arc<AtomicUsize>,
    memory_bytes: usize,
    _residency: ModelResidency,
    _gpu_permit: GpuPermit,
}

struct CacheInner {
    entries: HashMap<ModelId, CacheEntry>,
    lru_order: VecDeque<ModelId>,
    in_flight: HashMap<ModelId, Arc<tokio::sync::Notify>>,
}

struct Backends {
    candle: CandleBackend,
    ort: OrtBackend,
}

/// LRU cache of loaded models with GPU memory tracking and single-flight loading.
pub struct ModelCache {
    inner: Arc<RwLock<CacheInner>>,
    resolver: ModelResolver,
    backends: Backends,
    device_config: DeviceConfig,
    gpu_scheduler: Arc<GpuScheduler>,
}

impl ModelCache {
    /// Create a cache backed by the given resolver, device config, and GPU scheduler.
    pub fn new(
        resolver: ModelResolver,
        device_config: DeviceConfig,
        gpu_scheduler: Arc<GpuScheduler>,
    ) -> Self {
        Self {
            inner: Arc::new(RwLock::new(CacheInner {
                entries: HashMap::new(),
                lru_order: VecDeque::new(),
                in_flight: HashMap::new(),
            })),
            resolver,
            backends: Backends {
                candle: CandleBackend,
                ort: OrtBackend,
            },
            device_config,
            gpu_scheduler,
        }
    }

    /// Get or load a model. Returns a guard that keeps the model alive.
    pub async fn get_or_load(
        &self,
        source: &ModelSource,
        task: ModelTask,
        backend_hint: Option<BackendType>,
    ) -> Result<ModelGuard> {
        let id = ModelId::from(source);

        loop {
            let mut cache = self.inner.write().await;

            // Fast path: already loaded
            if let Some(entry) = cache.entries.get(&id) {
                entry.ref_count.fetch_add(1, Ordering::Acquire);
                let guard = ModelGuard {
                    model: Arc::clone(&entry.model),
                    ref_count: Arc::clone(&entry.ref_count),
                };
                cache.touch_lru(&id);
                return Ok(guard);
            }

            // Single-flight: wait if another task is loading this model
            if let Some(notify) = cache.in_flight.get(&id) {
                let notify = Arc::clone(notify);
                drop(cache);
                notify.notified().await;
                continue;
            }

            // We are the loader
            let notify = Arc::new(tokio::sync::Notify::new());
            cache.in_flight.insert(id.clone(), Arc::clone(&notify));
            drop(cache);

            let result = self.do_load(&id, source, task, backend_hint).await;

            let mut cache = self.inner.write().await;
            cache.in_flight.remove(&id);

            match result {
                Ok(guard) => {
                    drop(cache);
                    notify.notify_waiters();
                    return Ok(guard);
                }
                Err(e) => {
                    drop(cache);
                    notify.notify_waiters();
                    return Err(e);
                }
            }
        }
    }

    async fn do_load(
        &self,
        id: &ModelId,
        source: &ModelSource,
        task: ModelTask,
        backend_hint: Option<BackendType>,
    ) -> Result<ModelGuard> {
        let resolved = self.resolver.resolve(source, task, backend_hint)?;
        let source_str = source.to_string();
        let backend: &dyn ModelBackend = match resolved.backend {
            BackendType::Candle => &self.backends.candle,
            BackendType::Ort => &self.backends.ort,
            other => {
                return Err(JammiError::Model {
                    model_id: source_str,
                    message: format!("Backend {other:?} not available"),
                })
            }
        };
        let memory_bytes = backend.estimate_memory(&resolved);

        let gpu_permit = loop {
            if let Some(permit) = self.gpu_scheduler.try_acquire(memory_bytes) {
                break permit;
            }
            let mut cache = self.inner.write().await;
            if !cache.evict_one() {
                return Err(JammiError::Model {
                    model_id: source_str,
                    message: "Cannot acquire GPU memory: nothing to evict".into(),
                });
            }
        };

        let loaded = backend.load(&resolved, &self.device_config)?;

        // Register model in catalog (idempotent — ignores if already registered)
        let backend_str = format!("{:?}", resolved.backend).to_lowercase();
        let task_str = format!("{task:?}").to_lowercase();
        let model_type = match source {
            ModelSource::HuggingFace(_) => "huggingface",
            ModelSource::Local(_) => "local",
        };
        if let Err(e) = self.resolver.catalog().register_model(RegisterModelParams {
            model_id: &source_str,
            version: 1,
            model_type,
            backend: &backend_str,
            task: &task_str,
            artifact_path: resolved.weights_paths.first().and_then(|p| p.to_str()),
            config_json: None,
            ..Default::default()
        }) {
            tracing::warn!(model_id = %source_str, "Failed to register model in catalog: {e}");
        }

        let mut cache = self.inner.write().await;
        let ref_count = Arc::new(AtomicUsize::new(1));
        let model = Arc::new(loaded);
        cache.entries.insert(
            id.clone(),
            CacheEntry {
                model: Arc::clone(&model),
                ref_count: Arc::clone(&ref_count),
                memory_bytes,
                _residency: ModelResidency::Gpu,
                _gpu_permit: gpu_permit,
            },
        );
        cache.lru_order.push_back(id.clone());

        Ok(ModelGuard { model, ref_count })
    }

    /// Preload a model without running inference.
    pub async fn preload(
        &self,
        source: &ModelSource,
        task: ModelTask,
        backend_hint: Option<BackendType>,
    ) -> Result<()> {
        let guard = self.get_or_load(source, task, backend_hint).await?;
        drop(guard);
        Ok(())
    }
}

impl CacheInner {
    fn touch_lru(&mut self, id: &ModelId) {
        if let Some(pos) = self.lru_order.iter().position(|x| x == id) {
            self.lru_order.remove(pos);
        }
        self.lru_order.push_back(id.clone());
    }

    fn evict_one(&mut self) -> bool {
        let evict_id = self
            .lru_order
            .iter()
            .find(|id| {
                self.entries
                    .get(*id)
                    .is_some_and(|e| e.ref_count.load(Ordering::Relaxed) == 0)
            })
            .cloned();

        if let Some(id) = evict_id {
            if let Some(entry) = self.entries.remove(&id) {
                self.lru_order.retain(|x| x != &id);
                tracing::info!(
                    model_id = %id.0,
                    bytes = entry.memory_bytes,
                    "Evicted model from cache"
                );
            }
            true
        } else {
            false
        }
    }
}
