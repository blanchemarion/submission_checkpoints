# Sequential neural forecasting checkpoints

This bundle contains release-ready weights for every model found in the
repository's `checkpoints_*` directories. Portable weights use the
non-executable [safetensors](https://github.com/huggingface/safetensors) format,
with one `config.json` and the required normalization statistics beside each
model. The original PyTorch checkpoints and training artifacts are retained
under `checkpoints/` for provenance.

## Contents

```text
README.md
LICENSE
requirements.txt
.gitattributes
MODEL_INDEX.json
MANIFEST.json
metadata/
  widefield_data_metadata.json
modeling/
  loader.py
  transformer.py
  gru_ar.py
  cdmm_ssm.py
  linear_var.py
  mlp_2p.py
models/
  tf/seed{101,102,103}/
  tf-qtl-kl/seed{101,102,103}/
  one-step/seed{101,102,103}/
  ar-kv/seed{101,102,103}/
  gru-ar/seed{101,102,103}/
  cdmm-ssm/seed{101,102,103}/
  var/seed{101,102,103}/
  mlp-2p/seed42/
checkpoints/
  ... original .pt files and training artifacts
scripts/
  validate_release.py
```

Every model directory contains:

- `model.safetensors`: validation-selected `best_model.pt` weights only;
- `config.json`: architecture, seed, training configuration, selection metrics,
  parameter count, and source provenance;
- `normalization_mean.npy`, `normalization_std.npy`, and
  `normalization_meta.json` for widefield models.

`MODEL_INDEX.json` is the machine-readable model catalog. `MANIFEST.json`
contains the size and SHA-256 digest of every release file other than the
manifest itself.

## Model families

| Directory | Architecture | Context | Native output | Seeds |
|---|---|---:|---:|---|
| `tf` | Decoder-only Transformer | 90 × 16 | 90 × 16 | 101–103 |
| `tf-qtl-kl` | Decoder-only Transformer | 90 × 16 | 90 × 16 | 101–103 |
| `one-step` | Decoder-only Transformer | 90 × 16 | 1 × 16 | 101–103 |
| `ar-kv` | KV-cached autoregressive Transformer | 90 × 16 | 90 × 16 | 101–103 |
| `gru-ar` | Eight-layer GRU | 90 × 16 | 90 × 16 | 101–103 |
| `cdmm-ssm` | Conditional deep Markov state-space model | 90 × 16 | 90 × 16 | 101–103 |
| `var` | Linear autoregressive baseline | 90 × 16 | 90 × 16 | 101–103 |
| `mlp-2p` | Population-conditioned MLP | 90 × 88 | 1 × 88 | 42 |

The `tf-qtl-kl` name records quantile-tail and KL objective weights of 0.08 and
0.02. Those weights were encoded in the legacy source directory name but not
serialized in its checkpoint config; each release config records that
provenance explicitly.

The legacy `one-step` seed 102 and 103 checkpoints record `mode="TF"`. They are
classified as one-step models because their source directories and architecture
configs unambiguously specify `T_out=1`. This discrepancy is preserved in each
config's notes rather than silently rewritten.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# POSIX:   source .venv/bin/activate
python -m pip install -r requirements.txt
```

Choose the appropriate PyTorch build for your CPU or CUDA runtime if the
default wheel is unsuitable.

## Loading a model

Run this example from the release root:

```python
from pathlib import Path

import torch

from modeling import forecast, load_model, load_normalization

model_dir = Path("models/tf/seed101")
model, config = load_model(model_dir, device="cpu")

# Raw widefield context: [batch, time=90, regions=16]
raw_context = torch.zeros(1, 90, 16)
mean, std = load_normalization(model_dir)
context = (raw_context - torch.from_numpy(mean)) / torch.from_numpy(std)

with torch.inference_mode():
    normalized_prediction = forecast(model, config, context)

prediction = (
    normalized_prediction * torch.from_numpy(std)
    + torch.from_numpy(mean)
)
print(prediction.shape)  # [1, 90, 16]
```

The one-step model returns one native step. Repeated one-step rollout must append
each generated prediction to the context before the next call. The Transformer,
GRU, cDMM-SSM, and VAR 90-step models can likewise be called blockwise for
longer horizons.

MLP-2P consumes the original nonnegative two-photon trace scale and therefore
does not have the widefield z-score sidecars:

```python
import torch

from modeling import forecast, load_model

model, config = load_model("models/mlp-2p/seed42")
context = torch.zeros(1, 90, 88)
with torch.inference_mode():
    prediction = forecast(model, config, context)
print(prediction.shape)  # [1, 1, 88]
```

## Validation

Validate all file hashes, strictly load all 22 safe-tensor models, verify every
normalization sidecar, and smoke-test one seed per family:

```bash
python scripts/validate_release.py
```

To run inference for all seeds:

```bash
python scripts/validate_release.py --smoke-all
```

## Portable weights versus original checkpoints

Use `models/**/model.safetensors` for distribution and inference. Safetensors
does not execute Python during loading and contains only model tensors.

Files under `checkpoints/` are exact copies of the training outputs. They may
include optimizer state, scheduler state, loss histories, and Python pickle
payloads. Only load `.pt` files from a trusted source and use
`torch.load(..., weights_only=False)` only when the non-weight metadata is
required.

## Normalization and channel order

Widefield models were trained with per-region statistics computed from the
training split's input window `[0:90)`. The release loader reshapes stored NCT
statistics to broadcast-ready `[1, 1, 16]` arrays for BTC model inputs.

Channel names and order are recorded in
`metadata/widefield_data_metadata.json`. Do not reorder, add, or remove channels
without adapting and retraining the model.

## Rebuilding

From the source repository root:

```bash
python scripts/build_checkpoint_release.py
```

Pass `--without-original-checkpoints` to create a smaller bundle containing
portable weights only.