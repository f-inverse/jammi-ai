"""The encoders the chapters run.

Every one is a real, pretrained checkpoint on the Hugging Face Hub, fetched on
first use and cached, so every number a chapter shows is a real model's. Text
has one encoder per scale: ``small`` runs a compact sentence encoder that
embeds and fine-tunes in minutes on a CPU, ``full`` the larger one the book's
at-scale findings are about. The image and audio encoders are each one
checkpoint at both scales.
"""

from __future__ import annotations

from .scale import Scale

# Both text encoders are embedding-trained: a masked-LM backbone's pooled
# states (ModernBERT-base's own) retrieve poorly, and graph propagation has
# nothing to denoise in them.
_TEXT = {
    Scale.SMALL: "sentence-transformers/all-MiniLM-L6-v2",
    Scale.FULL: "Alibaba-NLP/gte-modernbert-base",
}

#: One OpenCLIP checkpoint, carrying both an image and a text tower.
IMAGE = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

#: A CLAP checkpoint, carrying both an audio and a text tower.
AUDIO = "laion/clap-htsat-fused"


def text(scale: Scale) -> str:
    """The text encoder's model id at ``scale``."""
    return _TEXT[scale]


def training_dtype(scale: Scale) -> str:
    """The backbone precision a chapter fine-tunes at: `bf16` at `full` — the
    tensor-core precision of the sm_80+ GPU a full run needs, at half f32's
    memory — and `f32` at `small`, on a CPU, where `bf16` is refused."""
    return "bf16" if scale is Scale.FULL else "f32"
