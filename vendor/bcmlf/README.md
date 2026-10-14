# Vendored BC-MLF source closure

Source: [BC-MLF](https://github.com/sunyulin0421/BC-MLF), commit `428b973be580fd3cf09dd51bb61c448f2a26b2c9` (MIT; see `LICENSE`). This subset contains the original MSALM, BI_ENC, AV encoder/cross-modal GPT model, and the code dependencies required to instantiate them. Local package imports replace the original absolute `MMSA.*` imports.

The upstream BC-MLF README acknowledges use of model-related code from [DeepMLF](https://github.com/efthymisgeo/deepmlf); this acknowledgment is retained in the parent port notes. No DeepMLF files are vendored here.

This is a CH-SIMS v2 five-class protocol adapter, not the authors' published old-SIMS result. It preserves the scalar regression path with `task_out=1`; use `task_out=5` plus `vendor.bcmlf.adapters.five_class` for the five-class path. Resolve `${BCMLF_*_ROOT}` placeholders with `vendor.bcmlf.adapters.config_paths.resolve_bcmlf_paths` before model creation. Runtime requires local data, a local Chinese GPT-2 tokenizer/checkpoint, and the trained BI_ENC checkpoint. No weights or datasets are included.

`AV_Enc.from_pretrained` loads the BI_ENC state dict strictly with `weights_only=True` and CPU mapping. Every target tensor must have a matching `Model.<key>` checkpoint tensor of the same shape; missing/mismatched weights fail. No random fallback is permitted.
