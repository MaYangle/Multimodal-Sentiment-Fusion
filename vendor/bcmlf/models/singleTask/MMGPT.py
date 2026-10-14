"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from typing import List
from random import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
# import torch.nn.MultiheadAttention as MultiheadAttention
from torch.nn import functional as F
from ...transformations.pool.mmaug import SoftPerm_Fast
from ..subNets.transformers_encoder.transformer import SinusoidalPositionalEmbedding
from ..subNets.transformers_encoder.multihead_attention import MultiheadAttention

# __all__ = ['MMGPT']

HF_MODELS = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "gpt2-xl": "gpt2-xl",
    "gpt2-chinese-cluecorpussmall": "uer/gpt2-chinese-cluecorpussmall"
}

class LoRALinear(nn.Module):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Actual trainable parameters
        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(in_features, r))
            self.lora_B = nn.Parameter(torch.zeros(r, out_features))
            self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        # x: (B, L, D_in) --> (B, L, D_out)
        # debugging
        x = self.lora_dropout(x)
        x = x @ self.lora_A
        x = x @ self.lora_B
        x = x * self.scaling
        # result = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        # return result
        return x

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MultiheadCrossAttention(nn.Module):
    ''' Multi-Head Cross Attention module '''

    def __init__(self, n_head, d_model, d_k, p_drop=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model # q-dim=d_model
        self.d_k = d_k # k,v dim != d_model

        # Q: d_model --> d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        # K,V: d_k --> d_model
        self.W_kv = nn.Linear(d_k, 2*d_model, bias=False)
        # self.W_v = nn.Linear(d_k, d_model, bias=False)
        # projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(p_drop)
        self.resid_dropout = nn.Dropout(p_drop)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_scores = \
            torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        output = torch.matmul(attn_probs, v)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        # (B, nh, L, hs)
        return x.view(batch_size, seq_length, self.n_head, self.d_model // self.n_head).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, x_q, context, mask=None):
        x_q = self.split_heads(self.W_q(x_q))
        x_k, x_v = self.W_kv(context).split(self.d_model, dim=2)
        x_k = self.split_heads(x_k)
        x_v = self.split_heads(x_v)
        attn_output = self.scaled_dot_product_attention(x_q, x_k, x_v, mask)
        output = self.resid_dropout(self.W_o(self.combine_heads(attn_output)))
        return output

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LoRA_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = \
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.lora_c_fc = LoRALinear(
            config.n_embd,
            4 * config.n_embd,
            config.lora.r,
            config.lora.lora_alpha,
            config.lora.lora_dropout,
        )
        self.gelu = nn.GELU()
        self.c_proj = \
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.lora_c_proj = LoRALinear(
            4 * config.n_embd,
            config.n_embd,
            config.lora.r,
            config.lora.lora_alpha,
            config.lora.lora_dropout,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x) + self.lora_c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x) + self.lora_c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MMBlock(nn.Module):
    """extension of the Block class to multimodal gated
    cross attention
    """
    def __init__(self, config, idx=-1):
        super().__init__()
        self.idx = idx
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.kdim = config.get("kv_dim", config.n_embd)
        self.attn = MultiheadCrossAttention(
            config.n_head,
            config.n_embd,  # the same as query size
            self.kdim,  # multimodal dimension, k, v
            config.dropout,
        )
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        if config.get("use_lora", False):
            self.mlp = LoRA_MLP(config)
        else:
            self.mlp = MLP(config)
        print(f"Ongoing with ----- {config.gating} ----- gating")
        if config.gating == "tanh":
            self.gate_1 = nn.Tanh()
            self.gate_2 = nn.Tanh()
            # init gate at zero
            self.alpha_1 = nn.Parameter(torch.zeros(1))
            self.alpha_2 = nn.Parameter(torch.zeros(1))
        elif config.gating == "sigmoid":
            self.gate_1 = nn.Sigmoid()
            self.gate_2 = nn.Sigmoid()
            # init gate at ~0
            init_value = config.get("init_gate", 0)
            if config.get("init_gate_2", None):
                init_value_2 = config.get("init_gate_2")
            else:
                init_value_2 = init_value
            if self.idx == -1:
                self.alpha_1 = nn.Parameter(init_value * torch.ones(1))
                # self.alpha_1 = nn.Parameter(torch.zeros(1))
                self.alpha_2 = nn.Parameter(init_value_2 * torch.ones(1))
                # self.alpha_2 = nn.Parameter(torch.zeros(1))
            else:
                self.alpha_1 = nn.Parameter(init_value[idx] * torch.ones(1))
                self.alpha_2 = nn.Parameter(init_value_2[idx] * torch.ones(1))
        else:
            # no gating applied gate at 1
            self.gate_1 = nn.Identity()
            self.gate_2 = nn.Identity()
            self.alpha_1 = 1
            self.alpha_2 = 1

    def forward(self, x_q, x_kv):
        """
        x_q: the text-modality queries [B, L, D]
        x_k, x_v: the encoder's keys and values (context)
        """
        # (B,L,D)
        norm_x_q = self.ln_1(x_q)
        # cross attention with the context x_k, x_v
        # x_ca, _ = self.attn(norm_x_q, x_k, x_v)
        x_ca = self.attn(norm_x_q, x_kv)
        x = x_q + self.gate_1(self.alpha_1) * x_ca
        x = x + self.gate_2(self.alpha_2) * self.mlp(self.ln_2(x))
        return x


class EncBlock(nn.Module):
    """transformer encoder block
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.d_enc, bias=config.bias)
        self.attn = MultiheadAttention(
            config.d_enc, config.n_head,
            config.enc_attn_dropout, config.enc_res_dropout,
        )
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x_kvq):
        """
        x_q: the text-modality queries [B, L, D]
        x_k, x_v: the encoder's keys and values
        """
        # (B,L,D) -> (L,B,D)
        norm_x = self.ln_1(x_kvq).permute(1, 0, 2)
        # self attention with the context x_k, x_v
        x_ca, _ = self.attn(
            norm_x,
            norm_x,
            norm_x,
        )
        x = x_kvq + x_ca.permute(1, 0, 2)
        x = x + self.mlp(self.ln_2(x))
        return x


class M3(nn.Module):
    """Multimodal Masking with dropping dead timesteps
    and padding to max active len in the batch
    """
    def __init__(self, p):
        super(M3, self).__init__()
        assert 0 <= p <= 1, "Probability p must be between 0 and 1"
        self.p = p

    def forward(self, x1, x2):
        if self.training:
            batch_size, L1, D1 = x1.shape
            _, L2, D2 = x2.shape

            # Create masks for each sample in the batch
            mask1 = torch.rand(batch_size, L1, device=x1.device) > self.p
            mask2 = torch.rand(batch_size, L2, device=x2.device) > self.p

            x1 = x1 * mask1.unsqueeze_(2)
            x2 = x2 * mask2.unsqueeze_(2)
        return x1, x2


class AV_Enc(nn.Module):
    def __init__(self, args):
        super(AV_Enc, self).__init__()
        _, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_a = self.d_v = args["av_enc"]["d_enc"]
        self.d_enc_out = args["av_enc"]["d_enc_out"]
        self.layers = args["av_enc"]["nlevels"]
        self.maxlen = args["av_enc"]["maxlen"]  # text's maximum length
        self.a_pdrop = args["av_enc"]["enc_dropout"]
        self.v_pdrop = args["av_enc"]["enc_dropout"]
        self.d_enc = args["av_enc"]["d_enc"]
        self.n_head = args["av_enc"]["n_head"]
        self.p_mask = args["av_enc"]["p_mask"]
        self.nlevels = args["av_enc"]["nlevels"]
        self.use_sperm = args["av_enc"]["use_softperm"]
        self.p_perm = args["av_enc"]["p_perm"]
        self.use_bn = args["av_enc"].get("use_bn", False)
        # Batch Normalization 
        if isinstance(self.use_bn, dict):
            self.use_bn = True
            self.use_bn_a = args["av_enc"]["use_bn"].get("use_bn_a", False)
            self.use_bn_v = args["av_enc"]["use_bn"].get("use_bn_v", False)
        elif self.use_bn:
            self.use_bn = self.use_bn_a = self.use_bn_v = True 
        else:
            self.use_bn = False
        combined_dim = self.d_a + self.d_v
        # positional encodings
        self.embed_scale_a = math.sqrt(self.d_a)
        self.embed_positions_a = SinusoidalPositionalEmbedding(self.d_a)
        self.embed_scale_v = math.sqrt(self.d_v)
        self.embed_positions_v = SinusoidalPositionalEmbedding(self.d_v)

        if self.use_bn:
            if self.use_bn_a:
                self.BN_a = nn.BatchNorm1d(self.orig_d_a)
            else:
                self.BN_a = nn.Identity()
            if self.use_bn_v:
                self.BN_v = nn.BatchNorm1d(self.orig_d_v)
            else:
                self.BN_v = nn.Identity()
            print(f"Using BN_a") if self.use_bn_a else None    
            print(f"Using BN_v") if self.use_bn_v else None
        else:
            print("No normalization is used")


        # Masking as feture augmentation
        # self.m3_drop = M3(p=self.p_mask)
        if self.use_sperm:
            l_t, l_a, l_v = args["seq_lens"]
            # for mosei it seems that l_v=500
            self.a_sperm = SoftPerm_Fast(
                p_feat=self.p_perm,
                maxlen=l_a
            )
            if args['dataset_name'] == 'mosei':
                self.v_sperm = SoftPerm_Fast(
                    p_feat=self.p_perm,
                    maxlen=l_a  # for mosei
                )
            else:
                self.v_sperm = SoftPerm_Fast(
                    p_feat=self.p_perm,
                    maxlen=l_v  # for mosi/sims
                )

        # Projection Layers
        self.proj_a = nn.Linear(self.orig_d_a, self.d_a, bias=False)
        self.proj_v = nn.Linear(self.orig_d_v, self.d_v, bias=False)

        # Encoder Layers
        self.enc_a = \
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.d_enc,
                    nhead=self.n_head,
                    dropout=self.a_pdrop,
                    dim_feedforward=4*self.d_enc,
                    activation="gelu",
                    batch_first=True, norm_first=True
                    ),
                num_layers=self.layers
            )
        self.enc_v = \
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.d_enc,
                    nhead=self.n_head,
                    dropout=self.v_pdrop,
                    dim_feedforward=4*self.d_enc,
                    activation="gelu",
                    batch_first=True, norm_first=True
                    ),
                num_layers=self.layers
            )

        # Fusion Module
        self.fusion = nn.Linear(combined_dim, self.d_enc_out)

        # Clf - not actually used here but for pretraining compatibility
        self.clf = nn.Linear(self.d_enc_out, 1)

    def align(self, x):
        raw_seq_len = x.size(1)
        if raw_seq_len == self.maxlen:
            return x
        if raw_seq_len // self.maxlen == raw_seq_len / self.maxlen:
            pad_len = 0
            pool_size = raw_seq_len // self.maxlen
        else:
            pad_len = self.maxlen - raw_seq_len % self.maxlen
            pool_size = raw_seq_len // self.maxlen + 1
        pad_x = x[:, -1, :].unsqueeze(1).expand([x.size(0), pad_len, x.size(-1)])
        x = torch.cat([x, pad_x], dim=1).view(x.size(0), pool_size, self.maxlen, -1)
        x = x.mean(dim=1)

        return x

    @classmethod
    def from_pretrained(cls, path, config):
        model = cls(config)
        print(f"----------------------- Loading AV encoder from {path}")
        sd_pretr = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(sd_pretr, dict) or not sd_pretr:
            raise TypeError(
                f"BI_ENC checkpoint {path!r} must be a non-empty state-dict mapping"
            )

        target_state = model.state_dict()
        source_for_target = {key: f"Model.{key}" for key in target_state}
        missing = [
            source_key
            for target_key, source_key in source_for_target.items()
            if source_key not in sd_pretr
        ]
        mismatched = []
        loaded_state = {}
        for target_key, source_key in source_for_target.items():
            if source_key not in sd_pretr:
                continue
            source_tensor = sd_pretr[source_key]
            target_tensor = target_state[target_key]
            if not torch.is_tensor(source_tensor):
                mismatched.append(
                    f"{source_key}: expected tensor, got {type(source_tensor).__name__}"
                )
                continue
            if tuple(source_tensor.shape) != tuple(target_tensor.shape):
                mismatched.append(
                    f"{source_key}: checkpoint shape {tuple(source_tensor.shape)} "
                    f"!= target shape {tuple(target_tensor.shape)}"
                )
                continue
            loaded_state[target_key] = source_tensor

        expected_source_keys = set(source_for_target.values())
        unexpected = [
            key for key in sd_pretr
            if key not in expected_source_keys and not key.startswith("Model.clf.")
        ]
        if missing or mismatched or unexpected:
            details = []
            if missing:
                details.append("missing Model-prefixed target keys: " + ", ".join(missing))
            if mismatched:
                details.append("shape/type mismatches: " + "; ".join(mismatched))
            if unexpected:
                details.append("unexpected checkpoint keys: " + ", ".join(unexpected))
            raise RuntimeError(
                f"BI_ENC checkpoint {path!r} is incompatible with this AV_Enc: "
                + " | ".join(details)
            )

        # Strict loading prevents a partially random encoder from being used.
        model.load_state_dict(loaded_state, strict=True)
        return model

    def encode_modalities(self, x_a, x_v):
        """x_a, x_v: (B, L, D)
        """
        # mask input tokens
        if self.training and self.use_sperm:
            # x_a, x_v = self.m3_drop(x_a, x_v)
            x_a = self.a_sperm(x_a)
            x_v = self.v_sperm(x_v)

        # import pdb; pdb.set_trace()
        # avg pool to maxlen
        # apply BN across each dimension for all timesteps
        if self.use_bn:
            x_a = self.BN_a(self.align(x_a).transpose(1,2)) # (B,L,D) -> (B,D,L)
            x_v = self.BN_v(self.align(x_v).transpose(1,2)) # (B,D,L)
            #  (B,D,L)-->(B,L,D)
            x_a = x_a.transpose(1,2)
            x_v = x_v.transpose(1,2)
        else:
            # just align
            x_a = self.align(x_a)
            x_v = self.align(x_v)
        # project_in, rescaling adopted from fairseq repo
        x_a = self.proj_a(x_a) * self.embed_scale_a
        x_v = self.proj_v(x_v) * self.embed_scale_v
        # positional ambeddings
        x_a += self.embed_positions_a(x_a[:, :, 0])
        x_a = F.dropout(x_a, p=self.a_pdrop, training=self.training)
        x_v += self.embed_positions_v(x_v[:, :, 0])
        x_v = F.dropout(x_v, p=self.v_pdrop, training=self.training)
        # encode
        x_a = self.enc_a(x_a)
        x_v = self.enc_v(x_v)
        return x_a, x_v

    def forward(self, x_a, x_v):
        """x_a, x_v: (B, L, D)
        """
        x_a, x_v = self.encode_modalities(x_a, x_v)
        # fusion: [B,L,D] --> [B,L,2D]
        x_f = torch.cat((x_a, x_v), dim=2)
        x_f = self.fusion(x_f)
        # # # fusion
        # # x_f = torch.cat((x_a, x_v), dim=2)[:, -1, :]
        # # x_f = torch.mean(torch.cat((x_a, x_v), dim=2), dim=1)
        # x_f = self.clf(F.dropout(self.fusion(x_f), p=0.0, training=self.training))
        return x_f

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


@dataclass
class MMGPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # multimodal-only params
    mm_layer: List[int] = field(default_factory=lambda: [5, 7, 9, 11])  # 0: first layer, 11: last layer
    mm_dropout: float = 0.1  # internal dropout for multimodal layers
    layer_dropout: float = 0.2  # text-only layer dropout, https://arxiv.org/pdf/1909.11556.pdf, slows down training, does it work in AR setups??
    dense: bool = False  # loss type
    tie_ffn: bool = True  # tie ffn in text-only and multimodal


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),   # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class MMGPT(nn.Module):
    """a multimodal GPT extension
    """
    def __init__(self, config):
        super().__init__()
        assert config["gpt"].vocab_size is not None
        assert config["gpt"].block_size is not None
        self.gpt_config = config["gpt"]
        self.mmgpt_config = config["mmgpt"]
        self.ca_reg = self.mmgpt_config.get("ca_reg", False)
        self.ca_reg_ln = self.mmgpt_config.get("ca_reg_ln", False) # use separate ln for careg
        self.w_ca_reg = self.mmgpt_config.get("w_ca_reg", [])
        # softperm regularization
        self.use_sperm = self.mmgpt_config.get("use_softperm", False)
        if self.use_sperm:
            # apply once on average across layers
            self.p_sperm = self.mmgpt_config["p_apply"]
            self.sperm = SoftPerm_Fast(
                p_feat=self.mmgpt_config["p_perm"]
            )
        self.mm_ldrop = self.mmgpt_config.get("mm_ldrop", -1)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.gpt_config.vocab_size, self.gpt_config.n_embd),
            wpe = nn.Embedding(self.gpt_config.block_size, self.gpt_config.n_embd),
            drop = nn.Dropout(self.gpt_config.dropout),
            h = nn.ModuleList([Block(self.gpt_config) for _ in range(self.gpt_config.n_layer)]),
            h_mm = nn.ModuleList([MMBlock(self.mmgpt_config, l_mm) for l_mm in range(len(self.mmgpt_config.mm_layer))]),
            ln_f = LayerNorm(self.gpt_config.n_embd, bias=self.gpt_config.bias),
        ))
        self.lm_head = nn.Linear(self.gpt_config.n_embd, self.gpt_config.vocab_size, bias=False)
        self.lm_task_head = nn.Linear(self.gpt_config.n_embd, config.task_out)
        self.dense = config.dense
        if self.use_sperm and self.ca_reg_ln: # design where the task gets separate layer norm
            self.ln_task = LayerNorm(self.gpt_config.n_embd, bias=self.gpt_config.bias)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.gpt_config.n_layer))

        # report number of parameters
        print("number of total parameters: %.2fM" % (self.get_num_params()/1e6,))
        print("number of mutlimodal parameters: %.2fM" % (self.get_num_params(mm=True)/1e6,))

    def get_num_params(self, non_embedding=True, mm=False):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        if mm:
            n_params = []
            for n, p in self.named_parameters():
                if "h_mm" in n:
                    n_params.append(p.numel())
                    print(f"{n} has {n_params[-1]} params")
                all_params = sum(n_params)
                if self.mmgpt_config["tie_ffn"]:
                    # tied_params = ["mlp", "ln_2"]
                    for n, p in self.transformer.h_mm.named_parameters():
                        if ("mlp" in n) or ("ln_2" in n):
                            all_params -= p.numel()
            return all_params
        else:
            n_params = sum(p.numel() for p in self.parameters())
            if non_embedding:
                n_params -= self.transformer.wpe.weight.numel()
            return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        else:
            #TODO
            # print(f"Handle initialization for parameter {module}")
            pass

    def forward(self, idx, context=None):
        device = idx.device
        # (B, L)
        b, t = idx.size()
        # assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        if self.use_sperm and self.training and self.p_sperm > random():
            tok_emb = self.sperm(tok_emb)
        else:
            pass
        x = self.transformer.drop(tok_emb + pos_emb)
        counter = 0
        for l, block in enumerate(self.transformer.h):
            # apply text-only layer dropout
            if self.training:
                ### gpt-layer drop
                ## not used in the final paper
                if self.mmgpt_config.layer_dropout < random():
                    # enters here with 1-p
                    x = block(x)
                else:
                    # drops layer with p
                    pass

                ### gpt-layer soft-permutation
                # if self.use_sperm and self.p_sperm > random():
                #     x = self.sperm(x)
                # else:
                #     pass
                #     # print("no sperm")
            else:
                # always during inference
                x = block(x)

            if l in self.mmgpt_config.mm_layer:
                if isinstance(context, list):
                    layer_context = context[counter]
                    counter += 1
                else:
                    # print("single context")
                    layer_context = context
                # print(f"Inserting the mm-block {l}")
                mm_idx = self.mmgpt_config.mm_layer.index(l)
                if self.training:
                    if self.mm_ldrop == -1:
                        x = self.transformer.h_mm[mm_idx](x, layer_context)
                    elif (self.mm_ldrop > 0) and (self.mm_ldrop < random()):
                        # x_prev = x
                        if context is not None:
                            # adding multimodal context
                            print("should not be here ----")
                            x = self.transformer.h_mm[mm_idx](x, layer_context)
                        else:
                            print("Should have not been here")
                            x = self.transformer.h_mm[mm_idx](x, x)
                    else:
                        # drop layer with p prob
                        pass
                else:
                    # inference
                    if layer_context is not None:
                        # adding multimodal context
                        x = self.transformer.h_mm[mm_idx](x, layer_context)
                    else:
                        print("Should have not been here")
                        x = self.transformer.h_mm[mm_idx](x, x)
        
        x = self.transformer.ln_f(x)
        # clm loss
        text_logits = self.lm_head(x)
        # task loss
        task_logits = self.lm_task_head(x)

        return text_logits, task_logits

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {
            'gpt2',
            'gpt2-medium',
            'gpt2-large',
            'gpt2-xl',
            'gpt2-chinese-cluecorpussmall'
        }
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        # assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config = override_args
        # # n_layer, n_head and n_embd are determined from model_type
        # config_args = {
        #     'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),   # 124M params
        #     'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
        #     'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
        #     'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        # }[model_type]
        model = MMGPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(HF_MODELS[model_type])
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        # assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
                print(f"Copying param --- {k} ---")
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
                print(f"Copying param --- {k} ---")

        # explicitly define which layers are going to be copied
        ffn_names = [
            "ln_2.weight", "ln_2.bias",
            "mlp.c_fc.weight", "mlp.c_fc.bias",
            "mlp.c_proj.weight", "mlp.c_proj.bias",
        ]

        if config["mmgpt"]["tie_ffn"]:
            for idx, n in enumerate(config["mmgpt"]["mm_layer"]):
                new_pref = f"transformer.h_mm.{idx}."
                hf_pref = f"transformer.h.{n}."
                for n_ffn in ffn_names:
                    new_full = f"{new_pref}{n_ffn}"
                    hf_full = f"{hf_pref}{n_ffn}"
                    print(f"Tying {hf_full} with {new_full}")
                    if any(n_ffn.endswith(w) for w in transposed):
                        # special treatment for the Conv1D weights we need to transpose
                        assert sd_hf[hf_full].shape[::-1] == sd[new_full].shape
                        with torch.no_grad():
                            sd[new_full].copy_(sd_hf[hf_full].t())
                    else:
                        # vanilla copy over the other parameters
                        assert sd_hf[hf_full].shape == sd[new_full].shape
                        with torch.no_grad():
                            sd[new_full].copy_(sd_hf[hf_full])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class MMSeq2Seq(nn.Module):
    def __init__(self, config) -> None:
        super(MMSeq2Seq, self).__init__()
        if config.lm.startswith('gpt2'):
            self.mm_decoder = MMGPT.from_pretrained(
                config.lm,
                config
            )
        self.av_encoder = AV_Enc(config)
        
        # load pretrained av-encoder
        if config["av_enc"].get("from_pretrained", False):
            print("----------------->>> Pretrained AudioVisual Encoder")
            # old implementation simply calls self.av_encoder.from_pretrained
            self.av_encoder = self.av_encoder.from_pretrained(
                config["av_enc"]["path_to_pretrained"],
                config
            )
        else:
            print("FromScratch AudioVisual Encoder")
            self.av_encoder = AV_Enc(config)
        
        # add layer normalization
        self.use_lnorm = config.get("use_lnorm", False)
        if self.use_lnorm:
            print("------------------ Adding LNorm")
            d_av = config["av_enc"]["d_enc_out"]
            self.LN = nn.LayerNorm(d_av)
        
        # add layer normalization
        self.rescale = config.get("rescale", False)
        if self.rescale:
            self.rescaler = config.get("rescaler", "lin")
            if self.rescaler == "lin":
                print("------------------ Adding Inverse Linear Rescaling after Encoder")
                self.scaler = 1 / len(config["mmgpt"]["mm_layer"])
            elif self.rescaler == "sqrt":
                print("------------------ Adding Inverse Linear Rescaling after Encoder")
                self.scaler = 1 / math.sqrt(len(config["mmgpt"]["mm_layer"]))
            elif self.rescaler == "magn":
                print("------------------ Adding Magnifier Sqrt Rescaling after Encoder")
                self.scaler = math.sqrt(12/len(config["mmgpt"]["mm_layer"]))
            else:
                print(f"-------------------- No scaler")
                raise NotImplementedError()
        
        # add transformator net
        self.use_tf = False
        self.use_layer_cond = False
        if config["av_enc"].get("transformator", False):
            print(f"Adding TRANSFORMATOR")
            self.tf_cfg = config["av_enc"]["transformator_cfg"]
            self.tf = \
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=self.tf_cfg["d_enc"],
                        nhead=self.tf_cfg["n_head"],
                        dropout=self.tf_cfg["dropout"],
                        dim_feedforward=4*self.tf_cfg["d_enc"],
                        activation="gelu",
                        batch_first=True, norm_first=True
                        ),
                    num_layers=config["av_enc"]["transformator_layers"]
                )
            self.use_tf = True
            if config["av_enc"]["layer_cond"]:
                self.use_layer_cond = True
                # Define the embedding layer
                self.n_layers = len(config["mmgpt"]["mm_layer"])
                self.layer_idx = \
                    nn.Parameter(torch.arange(
                        0,
                        self.n_layers,
                        dtype=torch.long
                    ), requires_grad=False)
                self.embedding = \
                    nn.Embedding(
                        num_embeddings=self.n_layers,
                        embedding_dim=config["av_enc"]["layer_embd"]
                    )
                # Define a linear layer to feed the concatenated embeddings
                self.cond = \
                    nn.Linear(
                        in_features=config["av_enc"]["layer_embd"] \
                            + self.tf_cfg["d_enc"],
                        out_features=self.tf_cfg["d_enc"],
                        bias=False
                    )
        self.av_distil = config.get("av_distil", False)
        if self.av_distil:
            n_layers = config.get("av_distil_layers", 2)
            d_av = config["av_enc"]["d_enc_out"]
            layers = []
            # Adding n_layers of Linear, BN, ReLU
            for _ in range(n_layers):
                layers.append(nn.Linear(d_av, d_av))
                layers.append(nn.BatchNorm1d(d_av))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(d_av, 1))
            self.av_dec = nn.Sequential(*layers)

    def forward(self, x_l, x_a, x_v):
        """ x_l: (B, Lt, 1), seq of id's for language
            x_m: (B, Lm, Dm), seq of features
        """
        # encode a, v first in a fused representation
        h_f = self.av_encoder(x_a, x_v)
        bsz, l_av, d_av = h_f.size()
        if self.use_tf:
            print("dead code remove in the future")
            if self.use_layer_cond:
                h_f_list = []
                all_layer_embed = \
                    self.embedding(self.layer_idx)
                for k in range(self.n_layers):
                    layer_embed = \
                        all_layer_embed[k].view(1, 1, -1).expand(bsz, l_av, -1)
                    h_f_tmp = torch.cat((h_f, layer_embed), dim=2)
                    h_f_tmp = self.tf(self.cond(h_f_tmp))
                    h_f_list.append(h_f_tmp)
                h_f = h_f_list
            else:
                h_f = self.tf(h_f)
        # use layer normalization
        if self.use_lnorm:
            h_f = self.LN(h_f)
        if self.rescale:
            h_f = h_f * self.scaler
        # lm_logits, task_logits, h_gpt = self.mm_decoder(x_l, h_f)
        lm_logits, task_logits = self.mm_decoder(x_l, h_f)
        if self.av_distil:
            if self.use_layer_cond:
                print("dead code remove in the future")
                av_logits = []
                for k in range(len(h_f)):
                    h_f[k] = torch.mean(h_f[k], dim=1)
                    av_logits.append(self.av_dec(h_f[k]))
            else:
                h_f = torch.mean(h_f, dim=1)
                av_logits = self.av_dec(h_f)
            return lm_logits, task_logits, av_logits
        else:
            return lm_logits, task_logits
