use std::path::PathBuf;

/// Where a model is loaded from — the user declares it, no fallback.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModelSource {
    /// A Hugging Face Hub repository (e.g. `"sentence-transformers/all-MiniLM-L6-v2"`).
    HuggingFace(String),
    /// A local directory containing model files (`config.json` + weights).
    Local(PathBuf),
    /// A model served at a remote endpoint, by the name its deployment
    /// declares it under (`[models.remote.<name>]`).
    Remote(String),
}

impl ModelSource {
    /// A Hugging Face Hub source.
    pub fn hf(repo_id: impl Into<String>) -> Self {
        Self::HuggingFace(repo_id.into())
    }

    /// A local filesystem source.
    pub fn local(path: impl Into<PathBuf>) -> Self {
        Self::Local(path.into())
    }

    /// A remote source, by its declared name.
    pub fn remote(name: impl Into<String>) -> Self {
        Self::Remote(name.into())
    }

    /// Parse a user-provided model id string.
    ///
    /// Local filesystem forms follow the convention of a source URL: a
    /// `file://` URI or a filesystem path is local; a bare `owner/repo` is a
    /// Hub id.
    ///
    /// - `"local:/path/to/model"` or `"file:///path/to/model"` → `Local(path)`
    /// - a filesystem path — `"/abs/model"`, `"./model"`, `"../model"` → `Local(path)`
    /// - `"hf://owner/repo"` → `HuggingFace("owner/repo")` (strips `hf://`)
    /// - `"remote:name"` → `Remote("name")`
    /// - `"owner/repo"` → `HuggingFace("owner/repo")`
    ///
    /// A remote model is named explicitly, never matched by an unprefixed
    /// id, so a declared remote name can never shadow a Hub repository.
    ///
    /// A local path is resolved against the filesystem of the host running
    /// the model (the server, for a remote client), so it must exist there.
    pub fn parse(id: &str) -> Self {
        if let Some(path) = id.strip_prefix("local:") {
            Self::Local(PathBuf::from(path))
        } else if let Some(path) = id.strip_prefix("file://") {
            Self::Local(PathBuf::from(path))
        } else if let Some(repo_id) = id.strip_prefix("hf://") {
            Self::HuggingFace(repo_id.to_string())
        } else if let Some(name) = id.strip_prefix(Self::REMOTE_PREFIX) {
            Self::Remote(name.to_string())
        } else if id.starts_with('/') || id.starts_with("./") || id.starts_with("../") {
            Self::Local(PathBuf::from(id))
        } else {
            Self::HuggingFace(id.to_string())
        }
    }

    /// The prefix naming a remote source.
    const REMOTE_PREFIX: &'static str = "remote:";

    /// Reconstruct a source from a canonical name (as a catalog records
    /// it): a `remote:` name is remote, an absolute path that exists on disk
    /// is local, everything else a Hub id.
    pub fn from_canonical(canonical_name: &str) -> Self {
        let path = std::path::Path::new(canonical_name);
        if let Some(name) = canonical_name.strip_prefix(Self::REMOTE_PREFIX) {
            Self::Remote(name.to_string())
        } else if path.is_absolute() && path.exists() {
            Self::Local(path.to_path_buf())
        } else {
            Self::HuggingFace(canonical_name.to_string())
        }
    }
}

impl std::fmt::Display for ModelSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HuggingFace(repo_id) => write!(f, "{repo_id}"),
            Self::Local(path) => write!(f, "{}", path.display()),
            Self::Remote(name) => write!(f, "{}{name}", Self::REMOTE_PREFIX),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hub_ids() {
        assert_eq!(
            ModelSource::parse("sentence-transformers/all-MiniLM-L6-v2"),
            ModelSource::HuggingFace("sentence-transformers/all-MiniLM-L6-v2".into())
        );
        assert_eq!(
            ModelSource::parse("hf://owner/repo"),
            ModelSource::HuggingFace("owner/repo".into())
        );
        // A bare name and a bare relative `a/b` stay Hub ids (ambiguous with
        // `owner/repo`); use `./` or `file://` to force a local relative path.
        assert_eq!(
            ModelSource::parse("bert-base-uncased"),
            ModelSource::HuggingFace("bert-base-uncased".into())
        );
        assert_eq!(
            ModelSource::parse("models/bert"),
            ModelSource::HuggingFace("models/bert".into())
        );
    }

    #[test]
    fn a_remote_source_parses_from_and_renders_to_its_prefixed_name() {
        let source = ModelSource::parse("remote:hosted-encoder");
        assert_eq!(source, ModelSource::Remote("hosted-encoder".into()));
        assert_eq!(source.to_string(), "remote:hosted-encoder");
        assert_eq!(ModelSource::from_canonical(&source.to_string()), source);
        assert_eq!(
            ModelSource::parse("hosted-encoder"),
            ModelSource::HuggingFace("hosted-encoder".into()),
            "an unprefixed name is never a remote model"
        );
    }

    #[test]
    fn parse_local_paths() {
        let cases = [
            ("local:/opt/models/bert", "/opt/models/bert"),
            ("file:///opt/models/bert", "/opt/models/bert"),
            ("/opt/models/bert", "/opt/models/bert"),
            ("./models/bert", "./models/bert"),
            ("../models/bert", "../models/bert"),
        ];
        for (input, expected) in cases {
            assert_eq!(
                ModelSource::parse(input),
                ModelSource::Local(PathBuf::from(expected)),
                "parsing {input:?}"
            );
        }
    }
}
