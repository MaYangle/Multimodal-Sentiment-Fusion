# CMC pseudo-label five-class port

This is a source-attributed CMC adaptation for the JevAffect canonical cache. It preserves the author's temporal modality encoders, PFM consensus attention, MCR top-k expert router, per-ID pseudo-label updates, EMA encoders, CE branch losses, and polarity-based supervised contrastive term. The output heads and pseudo distributions are five-class.

The production text input follows CMC's `text_bert` triplet contract `[N,3,T]` (token IDs, attention mask, segment IDs). Supply a local Chinese BERT model directory to `LocalBertTokenEncoder.from_local`; the port never downloads model weights. If the cache contains only contextual text sequences, `ContextualTextEncoder` is an explicit frozen-text fallback and `finetune_text=True` is rejected.

Basic usage uses only train/valid:

```python
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from cmc_port_impl.cache_adapter import load_cmc_train_valid
from cmc_port_impl.builders import build_pseudo_pretraining_bundle, build_fusion_model, build_fusion_optimizer
from cmc_port_impl.models import SupConLoss
from cmc_port_impl.training import (
    PseudoLabelBank, train_pseudo_pretrain_epoch,
    evaluate_pseudo_pretrain_epoch, train_fusion_epoch, evaluate_fusion,
)

train, valid, identity = load_cmc_train_valid("/path/to/feature-cache")
device = torch.device("cpu")  # Set to the approved device at the caller.
model_config = {"embed_dim": 32, "nhead": 2, "transformer_layers": 4,
                "out_dropout": 0.3, "top_k": 3, "temperature": 0.07}
```

Build the three source-style modality models and their EMA copies with
`build_pseudo_pretraining_bundle`, initialize `PseudoLabelBank` from the
train/valid `labels` arrays, then call `train_pseudo_pretrain_epoch` and
`evaluate_pseudo_pretrain_epoch`. The latter uses only the valid pseudo bank.
After pretraining, pass its EMA models to `build_fusion_model`; that applies the
author `transfer_models` mapping from unimodal encoders/heads into the three
CMC experts. `train_fusion_epoch` applies main five-class CE, three pseudo
branch CE losses, and CMC polarity SupCon. `evaluate_fusion` reports valid
main-head accuracy/NLL and uncalibrated probabilities.

`raw_label` remains in each cache sample for provenance. The inspected CMC
`--is_pseudo` source path uses categorical CE and SupCon after binning raw
scores; it does not include an MSE/L1 regression auxiliary, so the port adds no
such loss. It never requests a test split. See [SOURCE.md](SOURCE.md),
[manifest.json](manifest.json), and [source_manifest.json](source_manifest.json)
for provenance and exact adaptations.

CPU fixtures build a synthetic canonical cache, read only train/valid through
`jevaffect.cache.load_cached_split`, and run actual CMC pseudo-pretraining plus
fusion forward/backward passes. They are implementation checks, not experiment
results.
