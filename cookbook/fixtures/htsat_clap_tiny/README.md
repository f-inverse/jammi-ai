# `htsat_clap_tiny` — HTSAT-Swin CLAP audio-tower golden fixture

A tiny real HuggingFace `transformers` `ClapAudioModelWithProjection` plus
per-boundary golden activations, used to parity-test the Rust HTSAT-Swin audio
tower in `jammi-encoders`. The model carries the exact
`audio_model.audio_encoder.*` + `audio_projection.*` key layout of
`laion/clap-htsat-fused`, so the Rust port is verified against real key names and
real activations rather than a fabricated layout.

## Files

| File | Contents |
|---|---|
| `config.json` | tiny `ClapAudioConfig` (`model_type = "clap_audio_model"`) |
| `model.safetensors` | tiny real-HTSAT weights (~1.6 MB) |
| `pinned_input.safetensors` | the deterministic `input_features` `[2, 4, 512, 32]` the tower is tested on, independent of the audio front-end |
| `goldens.safetensors` | per-boundary activations (see naming below) |
| `golden_manifest.json` | `name -> {shape, dtype}` index of the goldens |

## Golden boundary names

`mel_in`, `post_batch_norm`, `post_reshape_mel2img`, `patch_embed_out`,
`stage{0..3}.block{0,1}_out`, `stage{0..2}.downsample_out`, `final_norm_out`,
`pre_pool`, `pooler_out`, `projected_unnormalized`, `projected_normalized`.

The fixture is `enable_fusion=True` (to match the real key layout) but the pinned
input is `is_longer=False`, so the forward exercises only the standard global
patch-embed → Swin path — the path the real fused checkpoint runs for normal
clips, and the only path the Rust tower ports. The AFF fusion submodules
(`fusion_model.*`, `mel_conv2d`) are present in the weights but never executed.

## Regenerate

Requires torch + transformers (run on Linux; x86_64 macOS caps torch too low for
the current `ClapModel`). **CI needs no torch** — these artifacts are committed
and the Rust tests assert against them.

```bash
pip install "torch==2.8.0" "transformers==4.57.6" "safetensors"
python tests/fixtures/generate_htsat_clap.py            # this fixture (hermetic)
python tests/fixtures/generate_htsat_clap.py --real     # + ../htsat_clap_real (downloads laion/clap-htsat-fused)
```

## Run the parity harness

```bash
cargo test -p jammi-encoders --features golden-parity --test golden_parity
```
