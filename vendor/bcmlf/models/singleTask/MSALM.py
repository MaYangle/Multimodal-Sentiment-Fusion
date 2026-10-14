import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
import torch.distributions as D
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from .MMGPT import (AV_Enc,
                    MultiheadCrossAttention,
                    LoRA_MLP, LayerNorm, MLP)

###################################################################################################
# Helper Code
###################################################################################################
class GaussianNoise(nn.Module):
    def __init__(self, stddev, mean=0.0):
        """
        Additive Gaussian Noise layer
        Args:
            stddev (float): the standard deviation of the distribution
            mean (float): the mean of the distribution
        """
        super().__init__()
        self.stddev = stddev
        self.mean = mean

    def forward(self, x):
        if self.training:
            noise = torch.normal(mean=self.mean, std=self.stddev, size=x.size(), device=x.device)
            return x + noise
        return x

    def __repr__(self):
        return '{} (mean={}, stddev={})'.format(self.__class__.__name__,
                                                str(self.mean),
                                                str(self.stddev))

def extend_instance(obj, mixin):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(
        base_cls_name, (mixin, base_cls), {}
    )  # mixin needs to go first for our forward() logic to work

def _infer_decoder_layers_attr_name(model):
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(
        f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
    )


__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
    "smolLM": "model.layers",
    "gptneoxforcausallm": "gpt_neox.layers",
    "mpt": "transformer.blocks",
    "mosaicgpt": "transformer.blocks",
}

def getattr_recursive(obj, att):
    """
    Return nested attribute of obj
    Example: getattr_recursive(obj, 'a.b.c') is equivalent to obj.a.b.c
    """
    if att == "":
        return obj
    i = att.find(".")
    if i < 0:
        return getattr(obj, att)
    else:
        return getattr_recursive(getattr(obj, att[:i]), att[i + 1 :])


def setattr_recursive(obj, att, val):
    """
    Set nested attribute of obj
    Example: setattr_recursive(obj, 'a.b.c', val) is equivalent to obj.a.b.c = val
    """
    if "." in att:
        obj = getattr_recursive(obj, ".".join(att.split(".")[:-1]))
    setattr(obj, att.split(".")[-1], val)


###################################################################################################
# Core Layers
###################################################################################################
class SoftPerm(nn.Module):
    def __init__(
        self,
        p_apply: float = 0.5,
        p_feat: float = 0.2,
        maxlen: Optional[int] = 50,
        n_bn: Optional[int] = 16,
    ):
        """
        SoftPerm_Fast: SoftPermutation (Fast), randomly samples a Bernoulli distribution
        with hyperparam $p$. By sampling, it gets the `copy_area`
        (float), which denotes the proportion (=area) of feature dimensions
        that are going to be resampled (i.e how many features are going to be
        permuted).

        Args:
            p_t_mod (List[float]): Resample probabilities for each modality across timesteps.
                default=1's which means that we resample at every timestep
            alpha (List[float]): used in sampling uniform distribution for each modality.
                Encodes the amount of features to be resampled. In order to resample the same
                distribution one should give a list of multiple variables as input in [0,1).
            maxlen (int): must be given as input and refers to the maximum sequence length
                In this implementation we also assume that the sequences are already zero-padded
        """
        super().__init__()
        self.p_apply = p_apply
        self.p_feat = p_feat  # resampled features
        self.maxlen = maxlen
        self.n_bn = n_bn
        
        # weights
        self.w = torch.tensor([1/maxlen]*maxlen, device="cuda")
        # probs --> '1', non-resampled features
        self.bern = D.bernoulli.Bernoulli(probs=1 - torch.tensor(self.p_feat, device="cuda"))

    def forward(self, x):
        """
        Fast SoftPerm forward implementation
        Args:
            mods (varargs torch.Tensor): [B, L, D_m] Modality representations
            m_ra (int): repeated augmentation index, default = 1
            real_len (torch.Tensor): [B] tensor of ints with the unpadded seq len
        Returns:
            (List[torch.Tensor]): The modality representations. Some of them are dropped
        """
        # List of [B, L, D] and T,A,V order is assumed
        if self.training and self.p_feat > 0 and self.p_apply > random.random():
            bsz, seqlen, d = x.size()
            # w_perm = self.w.repeat(bsz, 1)
            # import pdb; pdb.set_trace()
            permutation = torch.multinomial(self.w, seqlen - self.n_bn)
            x_lang = x[:, :(seqlen-self.n_bn), :]
            x_perm = x_lang[:, permutation, :]
            # import pdb; pdb.set_trace()
            area_mask = self.bern.sample((bsz, d))
            # (B,D) --> (B, 1, D)
            area_mask = area_mask.unsqueeze(1)
            x_lang = area_mask * x_lang + (1 - area_mask) * x_perm
            x[:, :(seqlen-self.n_bn), :] = x_lang

        return x
    

class _LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class _LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.n_embd
        self.intermediate_size = config.get("n_up", 4 * self.hidden_size)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        # if self.config.pretraining_tp > 1:
        #     slice = self.intermediate_size // self.config.pretraining_tp
        #     gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
        #     up_proj_slices = self.up_proj.weight.split(slice, dim=0)
        #     down_proj_slices = self.down_proj.weight.split(slice, dim=1)

        #     gate_proj = torch.cat(
        #         [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
        #     )
        #     up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

        #     intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
        #     down_proj = [
        #         F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
        #     ]
        #     down_proj = sum(down_proj)
        # else:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj
    
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
        x = self.lora_dropout(x)
        x = x @ self.lora_A
        x = x @ self.lora_B
        x = x * self.scaling
        return x


class _LlamaMLP_LoRA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.n_embd
        self.intermediate_size = config.get("n_up", 4 * self.hidden_size)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

        #Lora params
        self.lora_gate_proj = LoRALinear(
            self.hidden_size,
            self.intermediate_size,
            self.config.lora.r,
            self.config.lora.lora_alpha,
            self.config.lora.lora_dropout,
        )
        self.lora_up_proj = LoRALinear(
            self.hidden_size,
            self.intermediate_size,
            self.config.lora.r,
            self.config.lora.lora_alpha,
            self.config.lora.lora_dropout,
        )
        self.lora_down_proj = LoRALinear(
            self.intermediate_size,
            self.hidden_size,
            self.config.lora.r,
            self.config.lora.lora_alpha,
            self.config.lora.lora_dropout,
        )

    def forward(self, x):
        x_gate = self.act_fn(self.gate_proj(x) + self.lora_gate_proj(x))
        x_up = self.up_proj(x) + self.lora_up_proj(x)
        x_down = x_gate + x_up
        x_down = self.down_proj(x_down) + self.lora_down_proj(x_down)
        return x_down



class MMBlock(nn.Module):
    """extension of the Block class to multimodal gated cross attention
    """
    def __init__(self, config, idx=-1):
        super().__init__()
        self.idx = idx
        self.lm_flavor = config.get("type", "llama")
        if "gpt" in self.lm_flavor:
            self.kdim = config.get("kv_dim", config.n_embd)
            self.ln_1 = LayerNorm(config.n_embd, config.bias)
            self.ln_2 = LayerNorm(config.n_embd, config.bias)
            self.attn = MultiheadCrossAttention(
                config.n_head,
                config.n_embd,  # the same as query size
                self.kdim,  # multimodal dimension, k, v
                config.dropout,
            )
            if config.use_lora:
                self.mlp = LoRA_MLP(config)
            else:
                self.mlp = MLP(config)
        else: 
            self.ln_1 = _LlamaRMSNorm(config.n_embd)
            # self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.kdim = config.get("kv_dim", config.n_embd)
            self.attn = MultiheadCrossAttention(
                config.n_head,
                config.n_embd,  # the same as query size
                self.kdim,  # multimodal dimension, k, v
                config.dropout,
            )
            self.ln_2 = _LlamaRMSNorm(config.n_embd)
            # self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
            # if config.get("use_lora", False):
            #     self.mlp = LoRA_MLP(config)
            # else:
            self.mlp = _LlamaMLP_LoRA(config) # adopt this to initialize at the same
        
        ###########################################################################################
        # common parameters
        print(f"Ongoing with ----- {config.gating} ----- gating")
        self.gate_1 = nn.Sigmoid()
        self.gate_2 = nn.Sigmoid()
        # self.gate_1 = nn.Tanh()
        # self.gate_2 = nn.Tanh()
        # # init gate at ~0.5
        init_value = config.get("init_gate", 0)
        print(f"idx is {idx}")
        print(f"{init_value}")
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

        # combined cross-attention
        self.combine = config.get("combine", False)
        
    def forward(self, x_q, x_kv, x_prev=None):
        """
        x_q: the text-modality queries [B, L, D]
        x_k, x_v: the encoder's keys and values (context)
        """
        # (B,L,D)
        # for use_cache = True
        # x_q, cache = x_q[0], x_q[1]
        x_q = x_q[0]
        norm_x_q = self.ln_1(x_q)
        # cross attention with the context x_k, x_v
        # x_ca, _ = self.attn(norm_x_q, x_k, x_v)
        x_ca = self.attn(norm_x_q, x_kv)
        # PAPER: gated cross-attention version
        x = x_q + self.gate_1(self.alpha_1) * x_ca
        # ABLATION: no-gate version
        # x = x_q + x_ca
        # x = x_q + x_ca
        if self.combine:
            x_comb = torch.cat((x_prev, x), dim=1)
            # PAPER: gated cross-attention version
            x = x_comb + self.gate_2(self.alpha_2) * self.mlp(self.ln_2(x_comb))
            # ablation: no-gate version
            # x = x_comb + self.mlp(self.ln_2(x_comb))
            # x = x_comb + self.mlp(self.ln_2(x_comb))
        else:
            # vanilla cross-attention implementation
            x = x + self.gate_2(self.alpha_2) * self.mlp(self.ln_2(x))
        # x = (x, cache)
        x = (x,)
        return x


class msaLMLayer(nn.Module):
    """
    msaLM layer is a wrapper around the MMBlock and native decoder LMLayer.
    """

    def __init__(
        self, ca_layer, decoder_layer, n_bn_fusion=-1, max_len=50,
        combine=False, lm_flavor="llama", ldrop=0.0,
    ):
        super().__init__()
        self.ca_layer = ca_layer
        self.decoder_layer = decoder_layer
        self.n_bn_fusion = n_bn_fusion
        self.max_len = max_len
        self.combine = combine
        self.av_x = None
        self.lm_flavor = lm_flavor
        # self.ldrop = ldrop
        self.tie_ffw()
    
    def tie_ffw(self):
        if self.ca_layer is not None:
            print("COpying---------------------------------")
            if "gpt" in self.lm_flavor:
                # print(self.decoder_layer)
                assert self.ca_layer.mlp.c_fc.weight.shape == self.decoder_layer.mlp.c_fc.weight.shape[::-1]
                with torch.no_grad():
                    self.ca_layer.mlp.c_fc.weight.copy_(
                        self.decoder_layer.mlp.c_fc.weight.data.t()
                    )
                assert self.ca_layer.mlp.c_proj.weight.shape == self.decoder_layer.mlp.c_proj.weight.shape[::-1]
                with torch.no_grad():
                    self.ca_layer.mlp.c_proj.weight.copy_(
                        self.decoder_layer.mlp.c_proj.weight.data.t()
                    )
            else:
                # llama flavor
                # Assuming all the weight attributes are PyTorch tensors or parameters
                self.ca_layer.mlp.gate_proj.weight.data.copy_(
                    self.decoder_layer.mlp.gate_proj.weight.data
                )

                self.ca_layer.mlp.up_proj.weight.data.copy_(
                    self.decoder_layer.mlp.up_proj.weight.data
                )

                self.ca_layer.mlp.down_proj.weight.data.copy_(
                    self.decoder_layer.mlp.down_proj.weight.data
                )
            
    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.av_x is not None

    # Used this great idea from this implementation of Flamingo (https://github.com/dhansmair/flamingo-mini/)
    def condition_av_x(self, av_x):
        self.av_x = av_x

    def forward(
        self,
        lang_x,
        attention_mask=None,
        **decoder_layer_kwargs,
    ):
        # Normal decoder layer
        lang_x = self.decoder_layer(
            lang_x, attention_mask=attention_mask, **decoder_layer_kwargs
        )
        
        # Cross attention
        if self.ca_layer is not None:
            if self.av_x is None:
                raise ValueError("av_x must be conditioned before forward pass")

            if self.n_bn_fusion > 0:
                # BN fusion
                # Extract the language part and the BN part
                _lang_x = lang_x[0]
                lang_part = _lang_x[:, :self.max_len, :]
                bn_part = _lang_x[:, self.max_len:, :]

                
                if self.combine:
                    # combine always
                    updated_lang_x = self.ca_layer((bn_part,), self.av_x, lang_part)[0]
                    # Apply cross-attention to the BN par
                else:
                    bn_x = self.ca_layer((bn_part,), self.av_x)
                    # Concatenate the language part with the updated BN part
                    updated_lang_x = torch.cat((lang_part, bn_x[0]), dim=1)
                
                lang_x = (updated_lang_x,)
            else:
                lang_x = self.ca_layer(
                    lang_x,
                    self.av_x,
                )
        return lang_x


class BNEmbedding(nn.Module):
    def __init__(self, n_bn_fusion=-1, d=1024, config=None):
        super().__init__()
        self.n_bn_fusion = n_bn_fusion
        self.bn_embedding = nn.Parameter(
            torch.zeros(n_bn_fusion, d)
        ) # ()
        # init
        torch.nn.init.normal_(
            self.bn_embedding,
            mean=0.0, std=0.02
        )
    
    def forward(
            self,
            lang_x,
            attention_mask=None,
            **decoder_layer_kwargs
        ):
        # (n, D) --. (1, n, D)
        bn_x = self.bn_embedding.unsqueeze(0)
        # (1, n, D) --> (B,n,D)
        bn_x = bn_x.repeat(lang_x.size(0), 1, 1)
        # (B, L+n, D)
        lang_x = torch.cat(
            (lang_x, bn_x), dim=1
        )
        return lang_x


    
class msaLMEmbedding(nn.Module):
    """
    msaLM layer is a wrapper around the MMBlock and native decoder LMLayer.
    """

    def __init__(
        self, embedding, seqaug=None,
        n_bn_fusion=-1,
        d_embd=1024,
        lm_flavor="llama",
        max_len=50,
        config=None,
    ):
        super().__init__()
        self.embedding = embedding
        # self.positional = positional
        self.seqaug = seqaug
        self.n_bn_fusion = n_bn_fusion
        self.lm_flavor = lm_flavor
        self.max_len = max_len
        if self.n_bn_fusion > 0:
            self.bn_embedding = BNEmbedding(
                self.n_bn_fusion,
                d_embd,
                config
            )

    def forward(
        self,
        lang_x,
        attention_mask=None,
        **decoder_layer_kwargs,
    ):
        # Normal embedding layer
        try:
            lang_x = self.embedding(
                lang_x,
                attention_mask=attention_mask,
                **decoder_layer_kwargs
            )
        except TypeError:
            lang_x = self.embedding(
                lang_x, **decoder_layer_kwargs
            )
        
        # sqaug layer
        if self.seqaug is not None and self.training:
            lang_x = self.seqaug(lang_x)

        # bn layer
        if self.n_bn_fusion > 0:
            if 'gpt' in self.lm_flavor:
                true_lang_x = lang_x[:, :self.max_len, :]
                _lang_x = self.bn_embedding(true_lang_x)
            else:
                _lang_x = self.bn_embedding(lang_x)
            lang_x = _lang_x
        
        # # positional embeddings - gpt case
        # handled automatically via gpt2
        # if self.positional is not None:
        #     _, t, _ = lang_x.size()
        #     device = lang_x.device
        #     pos = torch.arange(
        #         0, t, dtype=torch.long, device=device
        #     )
        #     pos_emb = self.positional(pos)
        #     lang_x = lang_x + pos_emb
        
        return lang_x

class msaLMPositional(nn.Module):
    """
    msaLM layer is a wrapper around the MMBlock and native positional encoding LMLayer.
    """

    def __init__(
        self, positional, n_fusion=8
    ):
        super().__init__()
        self.positional = positional
        self.n_fusion = n_fusion

    def forward(
        self,
        pos_id,
        attention_mask=None,
        **decoder_layer_kwargs,
    ):
        
        # positional embeddings - gpt case
        _, t = pos_id.size()
        device = pos_id.device
        pos = torch.arange(
            0, t, dtype=torch.long, device=device
        )
        pos_emb = self.positional(pos)
        return pos_emb


class msaLMMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name="model.layers"):
        self.decoder_layers_attr_name = decoder_layers_attr_name
    
    def set_embedding_attr_name(self, embedding_attr_name="model.embed_tokens"):
        self.embedding_attr_name = embedding_attr_name

    def set_positional_attr_name(self, positional_attr_name="transformer.wpe"):
        self.positional_attr_name = positional_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _get_embedding_layers(self):
        return getattr_recursive(self, self.embedding_attr_name)

    def _get_positional_layers(self):
        return getattr_recursive(self, self.positional_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)
    
    def _set_embedding_layers(self, value):
        setattr_recursive(self, self.embedding_attr_name, value)

    def _set_positional_layers(self, value):
        setattr_recursive(self, self.positional_attr_name, value)

    def init_msalm(
        self,
        ca_list,
        config
    ):
        """
        Initialize Flamingo by adding a new gated cross attn to the decoder. Store the media token id for computing the media locations.
        """
        self.msa_config = config
        self.set_embedding_attr_name(
            self.msa_config.get("embedding_attr_name", "model.embed_tokens")
        )
        self.set_decoder_layers_attr_name(
            self.msa_config.get("decoder_layers_attr_name", "model.layers")
        )
        self.lm_flavor = self.msa_config["mmgpt"].get("type", "llama")
        if "gpt" in self.lm_flavor:
            self.set_positional_attr_name()
        print(f"ca list is: {ca_list}")
        self.init_embedding_layers()
        if "gpt" in self.lm_flavor:
            self.init_positional_layers()
        self.init_msalm_layers(ca_list)
        self.initialized_msalm = True
        self._use_cached_av_x = False
        self.n_bn_fusion = self.msa_config.get("n_bn_fusion", -1)

    def init_msalm_layers(self, ca_list):
        """
        Re initializes the FlamingoLayers.
        Propagates any changes made to self.gated_corss_attn_layers or self.old_decoder_blocks
        """
        old_decoder_blocks = self._get_decoder_layers()
        ca_layers = nn.ModuleList(
            [
                MMBlock(
                    self.msa_config["mmgpt"], layer_idx
                )
                for layer_idx in range(len(ca_list))
            ]
        )
        
        all_layers = []
        for idx in range(len(old_decoder_blocks)):
            print(f"Parsing decoder block: {idx}")
            if idx in ca_list:
                # get the corresponding index in ca_list
                ca_idx = ca_list.index(idx)
                all_layers.append(
                    msaLMLayer(
                        ca_layers[ca_idx],
                        old_decoder_blocks[idx],
                        self.msa_config["n_bn_fusion"],
                        self.msa_config["max_token_len"],
                        self.msa_config["mmgpt"].get("combine", False),
                        self.msa_config["mmgpt"].get("type", "llama")
                    )
                )
            else:
                all_layers.append(
                    msaLMLayer(None, old_decoder_blocks[idx])
                )

        self._set_decoder_layers(nn.ModuleList(all_layers))
    
    def init_embedding_layers(self):
        """
        Re initializes the FlamingoLayers.
        Propagates any changes made to self.gated_corss_attn_layers or self.old_decoder_blocks
        """
        # get pretrained embedding block
        embedding_block = self._get_embedding_layers()
        positional_block = None
        # if "gpt" in self.lm_flavor:
        #     positional_block = self._get_positional_layers()
        if self.msa_config.get("use_seqaug", False):
            print("initializing SoftPerm")
            seqaug_layer = SoftPerm(
                p_apply=self.msa_config["mmgpt"]["p_apply"],
                p_feat=self.msa_config["mmgpt"]["p_perm"],
                n_bn=self.msa_config["n_bn_fusion"],
                maxlen=self.msa_config["max_token_len"]
            )
            # #### ABLATION
            # if self.msa_config["mmgpt"]["ablation"] == "dropout":
            #     # Dropout
            #     print("initializing Embedding Dropout")
            #     seqaug_layer = nn.Dropout(p=self.msa_config["mmgpt"]["p_ablation"])
            # elif self.msa_config["mmgpt"]["ablation"] == "noise":
            #     # White Noise
            #     print("initializing GaussianNoise")
            #     seqaug_layer = GaussianNoise(stddev=self.msa_config["mmgpt"]["p_ablation"])
            # else:
            #     print('No ablation ------------ SHOULD NOT BE HERE --------------')

        else:
            seqaug_layer = None
        
        new_embedding = msaLMEmbedding(
            embedding_block,
            # positional_block,
            seqaug_layer,
            self.msa_config.get("n_bn_fusion", -1),
            self.msa_config["mmgpt"]["n_embd"],
            self.msa_config["mmgpt"].get("type", 'llama'),
            self.msa_config["max_token_len"],
            self.msa_config,
        )
        self._set_embedding_layers(nn.Sequential(new_embedding))
    
    def init_positional_layers(self):
        # get pretrained embedding block
        positional_block = self._get_positional_layers()
        
        new_positional = msaLMPositional(
            positional_block
        )
        self._set_positional_layers(nn.Sequential(new_positional))
    
    def forward(self, input_ids, attention_mask, **kwargs):
        """Condition the Flamingo layers on the media locations before forward()"""
        if not self.initialized_msalm:
            raise ValueError(
                "Flamingo layers are not initialized. Please call `init_flamingo` first."
            )

        # package arguments for the other parent's forward. make them all kwargs
        if 'gpt' in self.lm_flavor:
            # gpt2
            b, _ = input_ids.size()
            zero_ids = torch.zeros(
                (b, self.n_bn_fusion),
                dtype=torch.long,
                device=input_ids.device
            )
            input_ids = torch.cat(
                (input_ids, zero_ids), dim=1
            )
            kwargs["input_ids"] = input_ids
            kwargs["output_attentions"] = False
        else:
            kwargs["input_ids"] = input_ids
            kwargs["output_attentions"] = False

        # append 1s to attention_mask
        if (attention_mask is not None) and (self.n_bn_fusion > 0):
            # Create a tensor of ones with shape (B, n)
            ones_mask = torch.ones(
                input_ids.size(0),
                self.n_bn_fusion,
                device=attention_mask.device,
                dtype=attention_mask.dtype
            )
            # Concatenate the original attention_mask with the ones_mask
            attention_mask = torch.cat(
                (attention_mask, ones_mask), dim=1
            )
        kwargs["attention_mask"] = attention_mask
        return super().forward(**kwargs)  # Call the other parent's forward method

    def is_conditioned(self) -> bool:
        """Check whether all decoder layers are already conditioned."""
        return all(l.is_conditioned() for l in self._get_decoder_layers())

    def clear_conditioned_layers(self):
        for layer in self._get_decoder_layers():
            layer.condition_av_x(None)

class MSALM(nn.Module):
    def __init__(self, config):
        """
        Args:
            config (dict): configuration for all other functionalities
        """
        super(MSALM, self).__init__()
        # useful args
        self.max_token_len = config.max_token_len
        # BC-MLF's native regression head is one-dimensional.  The protocol
        # adapter sets task_out=5 and keeps every BC branch in logit space.
        self.task_out = int(config.get("task_out", 1))
        if self.task_out < 1:
            raise ValueError(f"task_out must be positive, got {self.task_out}")
        self.n_bn_fusion = config.get("n_bn_fusion", -1)
        self.modded_loss = config.get("modded_loss", False)
        self.use_ulgm = config.get("use_ulgm", False)
        self.lm_flavor = config['mmgpt'].get("type", 'llama')
        # Explicit prediction-layer aggregation mode.  Existing configs that
        # only set use_contribution_calibrated_head retain their old behavior.
        requested_branch_head_mode = config.get("branch_head_mode", None)
        if requested_branch_head_mode is None:
            requested_branch_head_mode = (
                "adaptive"
                if config.get("use_contribution_calibrated_head", False)
                else "none"
            )
        self.branch_head_mode = str(requested_branch_head_mode).lower()
        valid_branch_head_modes = {"none", "uniform", "global", "adaptive"}
        if self.branch_head_mode not in valid_branch_head_modes:
            raise ValueError(
                f"Unknown branch_head_mode={self.branch_head_mode!r}; "
                f"expected one of {sorted(valid_branch_head_modes)}"
            )
        self.use_contribution_calibrated_head = (
            self.branch_head_mode != "none"
        )
        self.contribution_head_temperature = 1.0
        self.contribution_min_weight = 0.0

        # ------ language model ----------
        # This vendored preparation package deliberately requires a local model
        # directory.  It must not silently resolve a model ID or download from
        # the network when a configured path is absent.
        lm_hf_path = os.path.abspath(os.path.expanduser(os.fspath(config.lm)))
        if not os.path.isdir(lm_hf_path):
            raise FileNotFoundError(
                "BC-MLF requires the locally prepared Chinese GPT-2 checkpoint "
                f"at {lm_hf_path!r}; prepare it explicitly before training."
            )
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            lm_hf_path,
            model_max_length=config.max_token_len,
            local_files_only=True,
        )
        self.tokenizer.padding_side = "right"
        if 'chinese' in config.lm:
            # due to BERT tokenizer has pad and no eos
            self.tokenizer.eos_token = self.tokenizer.pad_token
        else:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        def _load_causal_lm(load_kwargs):
            return AutoModelForCausalLM.from_pretrained(
                lm_hf_path,
                local_files_only=True,
                **load_kwargs,
            )
        
        if config.get("use_bf16", False):
            llm_config = AutoConfig.from_pretrained(lm_hf_path, local_files_only=True)
            llm_config.output_attentions = False  # Ensure this is set to False
            llm_config.output_hidden_states = True
            self.lang_encoder = _load_causal_lm(
                {
                    "torch_dtype": torch.bfloat16,
                    "return_dict_in_generate": True,
                    "output_hidden_states": True,
                    "use_cache": False,
                }
            )
        else:
            if 'gpt' in self.lm_flavor:
                self.lang_encoder = _load_causal_lm(
                    {
                        "output_hidden_states": True,
                        "use_cache": False,
                    }
                )
            else:                
                self.lang_encoder = _load_causal_lm(
                    {
                        "return_dict_in_generate": True,
                        "output_hidden_states": True,
                        "use_cache": False,
                    }
                )
        
        self._trim_static_causal_mask(config)

        # add mixing utilities
        extend_instance(self.lang_encoder, msaLMMixin)

        # add cross attentions
        self.lang_encoder.init_msalm(
            config["mmgpt"]["mm_layer"],
            config,
        )

        # if 'gpt' in self.lm_flavor:
        #     # manually deactivate layers that are not used
        # #     self.lang_encoder.transformer.wpe = None
        #     self.lang_encoder.transformer.drop = Identity()
        
        # task mapping
        # self.W_task = nn.Linear(
        #     config.mmgpt.d_mm + config.av_enc.d_enc_out,
        #     1
        # )
        ##############################################################################
        # original code here - before bn - uncomment afterwards
        ##########################################################################
        # self.W_task = nn.Linear(config.mmgpt.d_mm, 1)
        if self.use_ulgm:
            # task
            task_in_dim = 2*config.mmgpt.d_mm + config["av_enc"]["d_enc_out"]
            self.W_task_0 = nn.Sequential(
                nn.Linear(
                    task_in_dim,
                    config.mmgpt.d_out
                ),
                nn.BatchNorm1d(config.mmgpt.d_out),
                nn.ReLU(inplace=True)
            )
            self.W_task_1 = nn.Linear(config.mmgpt.d_out, self.task_out)
            # bn fusion
            self.W_bn = nn.Linear(config.mmgpt.d_mm, self.task_out)
            # text map
            self.W_text = nn.Linear(config.mmgpt.d_mm, self.task_out)
            # av map
            self.W_av = nn.Linear(config["av_enc"]["d_enc_out"], self.task_out)
        elif (self.n_bn_fusion > 0) and (self.modded_loss):
            self.d_task = config["mmgpt"].get("d_out", 64)
            task_in_dim = 2*config.mmgpt.d_mm + config["av_enc"]["d_enc_out"]
            self.W_task = nn.Sequential(
                nn.Linear(task_in_dim, self.d_task),
                nn.BatchNorm1d(self.d_task),
                nn.ReLU(inplace=True),
                nn.Linear(self.d_task, self.task_out)
            )
            self.W_bn = nn.Linear(config.mmgpt.d_mm, self.task_out)
            self.W_text = nn.Linear(config.mmgpt.d_mm, self.task_out)
            self.W_av = nn.Linear(config["av_enc"]["d_enc_out"], self.task_out)
            self.contribution_head_temperature = float(
                config.get("contribution_head_temperature", 1.0)
            )
            self.contribution_min_weight = float(
                config.get("contribution_min_weight", 0.0)
            )
            if self.branch_head_mode == "adaptive":
                self.contribution_gate = nn.Sequential(
                    nn.Linear(task_in_dim, self.d_task),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.d_task, 3),
                )
            elif self.branch_head_mode == "global":
                # Three sample-independent logits.  They use the same
                # temperature and minimum-weight transformation as BCHead.
                self.global_branch_logits = nn.Parameter(torch.zeros(3))
        else:
            self.W_task = nn.Linear(config.mmgpt.d_mm, self.task_out)
            self.use_contribution_calibrated_head = False

        # ------ audiovisual encoder ----------
        self.av_encoder = AV_Enc(config)
        if config["av_enc"].get("from_pretrained", False):
            print("----------------->>> Pretrained AudioVisual Encoder <<<<<----------------")
            self.av_encoder = self.av_encoder.from_pretrained(
                config["av_enc"]["path_to_pretrained"],
                config
            )
        else:
            print("From Scratch AudioVisual Encoder Initialization")

        # add layer normalization
        self.use_lnorm = config.get("use_lnorm", False)
        if self.use_lnorm:
            print("------------------ Adding LNorm ------------------------")
            d_av = config["av_enc"]["d_enc_out"]
            # self.RMS = nn.RMSNorm(d_av)
            self.LN = nn.LayerNorm(d_av)

        # Cross-Modal Consistency Mapping
        self.av_distil = config.get("av_distil", False)
        if self.av_distil:
            layers = []
            layers.append(nn.Linear(d_av, 1))
            self.av_dec = nn.Sequential(*layers)

        # max_AV_length == n_fusion_embeddings
        self.n_bn_fusion = config.get("n_bn_fusion", -1) # bottleneck fusion
        if self.n_bn_fusion > 0:
            kernel_size = config["max_token_len"] // self.n_bn_fusion
            stride = kernel_size
            padding = 0
            self.av_pooling = nn.AvgPool1d(
                kernel_size, stride, padding
            )
        
        # m3 - layer before classification
        self.p_m3 = config.get("p_m3", -1.0)

    def _split_bn_tokens(self, bn_tokens):
        avg_bn = torch.mean(bn_tokens, dim=1)
        return avg_bn

    def _task_input(self, avg_bn, last_hidden_text, av_x):
        return torch.cat((avg_bn, last_hidden_text, av_x), dim=1)

    def _contribution_calibrated_task(self, task_inputs, bn_logits, text_logits, av_logits):
        bn_pred = self._mean_bn_logits(bn_logits)
        text_pred = text_logits
        av_pred = av_logits

        # Branch outputs are [B, 1] for native regression or [B, C] for the
        # five-class protocol port.  Never broadcast incompatible class heads.
        predictions = {
            "bn": bn_pred,
            "text": text_pred,
            "av": av_pred,
        }
        for name, prediction in predictions.items():
            if prediction is None:
                raise RuntimeError(f"Missing {name} branch prediction")
            if prediction.dim() == 1:
                prediction = prediction.unsqueeze(-1)
                predictions[name] = prediction
            if prediction.dim() != 2 or prediction.size(1) != self.task_out:
                raise RuntimeError(
                    f"Expected {name} prediction shape [B, {self.task_out}], "
                    f"got {tuple(prediction.shape)}"
                )
        bn_pred = predictions["bn"]
        text_pred = predictions["text"]
        av_pred = predictions["av"]
        if not (bn_pred.shape == text_pred.shape == av_pred.shape):
            raise RuntimeError(
                "BC-MLF branch predictions must have identical [B, C] shapes; "
                f"got bn={tuple(bn_pred.shape)}, text={tuple(text_pred.shape)}, "
                f"av={tuple(av_pred.shape)}"
            )

        temperature = max(self.contribution_head_temperature, 1e-6)
        batch_size = bn_pred.size(0)
        if self.branch_head_mode == "uniform":
            weights = torch.full(
                (batch_size, 3),
                1.0 / 3.0,
                device=bn_pred.device,
                dtype=bn_pred.dtype,
            )
        elif self.branch_head_mode == "global":
            global_weights = torch.softmax(
                self.global_branch_logits / temperature,
                dim=-1,
            )
            weights = global_weights.unsqueeze(0).expand(batch_size, -1)
        elif self.branch_head_mode == "adaptive":
            weights = torch.softmax(
                self.contribution_gate(task_inputs) / temperature,
                dim=-1,
            )
        else:
            raise RuntimeError(
                "_contribution_calibrated_task called with "
                f"branch_head_mode={self.branch_head_mode!r}"
            )

        min_weight = min(max(self.contribution_min_weight, 0.0), 1.0 / 3.0 - 1e-6)
        if min_weight > 0:
            weights = weights * (1.0 - 3.0 * min_weight) + min_weight

        task_logits = (
            weights[:, 0:1] * bn_pred
            + weights[:, 1:2] * text_pred
            + weights[:, 2:3] * av_pred
        )
        return task_logits, weights

    def _branch_occlusion_task_logits(
        self,
        avg_bn,
        last_hidden_text,
        av_x,
        bn_tokens,
    ):
        """Recompute task predictions after zeroing one branch representation."""
        occlusion_logits = {}
        for branch in ("fusion", "text", "av"):
            masked_bn = torch.zeros_like(avg_bn) if branch == "fusion" else avg_bn
            masked_text = (
                torch.zeros_like(last_hidden_text) if branch == "text" else last_hidden_text
            )
            masked_av = torch.zeros_like(av_x) if branch == "av" else av_x
            masked_bn_tokens = (
                torch.zeros_like(bn_tokens) if branch == "fusion" else bn_tokens
            )

            masked_inputs = self._task_input(masked_bn, masked_text, masked_av)
            if self.use_contribution_calibrated_head:
                masked_bn_logits = self.W_bn(masked_bn_tokens)
                masked_text_logits = self.W_text(masked_text)
                masked_av_logits = self.W_av(masked_av)
                masked_task_logits, _ = self._contribution_calibrated_task(
                    masked_inputs,
                    masked_bn_logits,
                    masked_text_logits,
                    masked_av_logits,
                )
            else:
                masked_task_logits = self._task_map(masked_inputs)
            occlusion_logits[branch] = masked_task_logits
        return occlusion_logits

    def _trim_static_causal_mask(self, config):
        """Avoid repeating the full Llama causal mask for short MSA sequences."""
        llama_model = getattr(self.lang_encoder, "model", None)
        causal_mask = getattr(llama_model, "causal_mask", None)
        if causal_mask is None:
            return

        n_bn_fusion = max(config.get("n_bn_fusion", 0), 0)
        max_len = config.max_token_len + n_bn_fusion
        if causal_mask.size(-1) <= max_len:
            return

        if causal_mask.dim() == 2:
            llama_model.causal_mask = causal_mask[:max_len, :max_len].contiguous()
        elif causal_mask.dim() == 4:
            llama_model.causal_mask = causal_mask[:, :, :max_len, :max_len].contiguous()
        
    def forward(
        self,
        lang_x: torch.Tensor,
        audio_x: torch.Tensor,
        vision_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        Forward pass of Flamingo.

        Args:
            lang_x (torch.Tensor): Language input ids
                shape (B, max_len)
            av_x (torch.Tensor): audiovisual fused input
                shape (B, max_len, D)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
        """
        assert (
            self.lang_encoder.initialized_msalm
        ), "Flamingo layers are not initialized. Please call `init_flamingo` first."

        return_branch_occlusion = bool(kwargs.pop("return_branch_occlusion", False))

        av_x = self._encode_av(audio_x=audio_x, vision_x=vision_x)
        av_memory_for_fusion = av_x
        # print("calling _av_conditioning")
        self._av_conditioning(av_memory_for_fusion)

        outputs = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            **kwargs
        )
        lm_logits = outputs.logits
        last_hidden_states = outputs.hidden_states[-1]

        if self.use_ulgm:
            # last hidden state of the LM
            B, _, _ = last_hidden_states.size()
            with torch.no_grad():
                last_valid_indices = \
                                torch.sum(attention_mask, dim=1).long() - 1
            last_hidden_text = \
                last_hidden_states[
                    torch.arange(B, device=last_hidden_states.device),
                    last_valid_indices
                ]
            
            # modality masking
            if self.p_m3 > 0:
                # mask the language modality
                if self.p_m3 > random.random():
                    last_hidden_states = last_hidden_states * 0.0
            
            text_logits = self.W_text(last_hidden_text)
            # BN-fusion output
            bn_tokens = last_hidden_states[:, self.max_token_len:, :]
            bn_logits = self.W_bn(bn_tokens)
            # av output
            av_x = torch.mean(av_x, dim=1)
            av_logits = self.W_av(av_x)
            # global fusion output
            avg_bn = self._split_bn_tokens(bn_tokens)
            task_inputs = self._task_input(avg_bn, last_hidden_text, av_x)
            task_outputs = self._task_map(task_inputs)
            task_logits = task_outputs["task_logits"]
            result = {
                "task_logits": task_logits,
                "text_logits": text_logits,
                "av_logits": av_logits,
                "bn_logits": bn_logits,
                "lm_logits": None,
                "Feature_f": task_outputs["Feature_f"],
                "Feature_t": last_hidden_text,
                "Feature_av": av_x,
                "Feature_bn": avg_bn,
                "Feature_text_tokens": last_hidden_states[:, :self.max_token_len, :],
                "Feature_bn_tokens": bn_tokens,
                "Contribution_weights": None,
            }
            return result
        else:
            if (self.n_bn_fusion > 0) and (self.modded_loss):
                # get tha last valid text representation
                # Linearly increasing weights for each batch (B, L)
                # linear_weights_batch = \
                #     torch.linspace(0, 1, self.max_token_len, device=self.args.device).repeat(B, 1)
                # row_sums = linear_weights_batch.sum(dim=1, keepdim=True)
                # dense_mask = linear_weights_batch / row_sums
                
                # last hidden state of the LM
                B, _, _ = last_hidden_states.size()
                with torch.no_grad():
                    last_valid_indices = \
                                    torch.sum(attention_mask, dim=1).long() - 1
                last_hidden_text = \
                    last_hidden_states[
                        torch.arange(B, device=last_hidden_states.device),
                        last_valid_indices
                    ]
                # modality masking
                if self.p_m3 > 0:
                    # mask the language modality
                    if self.p_m3 > random.random():
                        last_hidden_states = last_hidden_states * 0.0

                text_logits = self.W_text(last_hidden_text)
                # BN-fusion output
                bn_tokens = last_hidden_states[:, self.max_token_len:, :]
                bn_logits = self.W_bn(bn_tokens)
                # av output
                av_x = torch.mean(av_x, dim=1)
                av_logits = self.W_av(av_x)
                # global fusion output
                avg_bn = self._split_bn_tokens(bn_tokens)
                task_inputs = self._task_input(avg_bn, last_hidden_text, av_x)
                task_logits = self._task_map(task_inputs)
                contribution_weights = None
                if self.use_contribution_calibrated_head:
                    task_logits, contribution_weights = self._contribution_calibrated_task(
                        task_inputs, bn_logits, text_logits, av_logits
                    )
                branch_occlusion_logits = None
                if return_branch_occlusion:
                    branch_occlusion_logits = self._branch_occlusion_task_logits(
                        avg_bn,
                        last_hidden_text,
                        av_x,
                        bn_tokens,
                    )
            else:        
                # _, l_t, _ = last_hidden_states.size()
                # task_logits = self._task_map(
                #     torch.cat((last_hidden_states, torch.mean(av_x, dim=1, keepdim=True).expand(-1, l_t, -1)), dim=2)
                # )
                bn_logits = None
                text_logits = None
                contribution_weights = None
                branch_occlusion_logits = None
                task_logits = self._task_map(last_hidden_states)
                if self.av_distil:
                    av_x = torch.mean(av_x, dim=1)
                    av_logits = self.av_dec(av_x)
            
            result = {
                "task_logits": task_logits,
                "text_logits": text_logits,
                "av_logits": av_logits,
                "bn_logits": bn_logits,
                "lm_logits": lm_logits,
                "Feature_t": last_hidden_text if (self.n_bn_fusion > 0) and (self.modded_loss) else None,
                "Feature_av": av_x if (self.n_bn_fusion > 0) and (self.modded_loss) else None,
                "Feature_bn": avg_bn if (self.n_bn_fusion > 0) and (self.modded_loss) else None,
                "Feature_text_tokens": last_hidden_states[:, :self.max_token_len, :] if (self.n_bn_fusion > 0) and (self.modded_loss) else None,
                "Feature_bn_tokens": bn_tokens if (self.n_bn_fusion > 0) and (self.modded_loss) else None,
                "Contribution_weights": contribution_weights,
                "Branch_occlusion_logits": branch_occlusion_logits,
            }
            return result

            # return lm_logits, task_logits, av_logits, bn_logits, text_logits
    
    def _task_map(self, h_last):
        # uses the final norm layer of the encoder (frozen)
        # h_last = self.lang_encoder.norm(h_last)
        if self.use_ulgm:
            h_fusion = self.W_task_0(h_last)
            fusion_logits = self.W_task_1(h_fusion)
            return {"Feature_f": h_fusion, "task_logits": fusion_logits}
        return self.W_task(h_last)

    def _mean_bn_logits(self, bn_logits):
        if bn_logits is None:
            return None
        if bn_logits.dim() > 2:
            return bn_logits.mean(dim=1)
        return bn_logits

    def _av_conditioning(self, av_x):
        # clear previous conditioning
        self.lang_encoder.clear_conditioned_layers()

        # condition
        for layer in self.lang_encoder._get_decoder_layers():
            if self.n_bn_fusion > 0: # bottleneck fusion
                # _av_x = av_x.permute(0, 2, 1) # (B,L,D) -> (B,D,L)
                # _av_x = self.av_pooling(_av_x)
                # _av_x = _av_x.permute(0, 2, 1) # (B,D,n_bn_fusion) -> (B,n_bn_fusion,D)
                # layer.condition_av_x(_av_x)
                
                # No pooling
                layer.condition_av_x(av_x)
            else:
                layer.condition_av_x(av_x)

    def _encode_av(self, audio_x: torch.Tensor, vision_x: torch.Tensor):
        """
        Compute audiovisual latent representations and condition language model.
        Args:
            audio_x (torch.Tensor): Audio input
            vision_x (torch.Tensor): Vision input
                shape (B, L_{m}, D_{m})
        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        # get new conditioning
        av_x = self.av_encoder(audio_x, vision_x)
        if self.use_lnorm:
            av_x = self.LN(av_x)
        return av_x
    

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
