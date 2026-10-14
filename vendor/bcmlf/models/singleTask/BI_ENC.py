"""
Bimodal transformer encoder
"""

import math
from typing import List
from random import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
# import torch.nn.MultiheadAttention as MultiheadAttention
from torch.nn import functional as F
from ...transformations.pool.mmaug import SoftPerm_Fast
from ..subNets.transformers_encoder.transformer import SinusoidalPositionalEmbedding


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

            # # Calculate the maximum length after masking
            # max_len1 = int((mask1.sum(dim=1)).max())
            # max_len2 = int((mask2.sum(dim=1)).max())

            # # Initialize the output tensors with reduced sizes
            # x1_masked = \
            #     torch.zeros(batch_size, max_len1, D1, dtype=x1.dtype, device=x1.device)
            # x2_masked = \
            #     torch.zeros(batch_size, max_len2, D2, dtype=x2.dtype, device=x2.device)

            # Apply masks and pad sequences to the calculated maximum length
            # for b in range(batch_size):
            #     x1_len = int(mask1[b].sum())
            #     x2_len = int(mask2[b].sum())

            #     x1_masked[b, :x1_len] = x1[b][mask1[b]]
            #     x2_masked[b, :x2_len] = x2[b][mask2[b]]
            # return x1_masked, x2_masked
            x1 = x1 * mask1.unsqueeze_(2)
            x2 = x2 * mask2.unsqueeze_(2)
        return x1, x2


class Masking(nn.Module):
    """Unimodal Masking with dropping dead timesteps
    and padding to max active len in the batch
    """
    def __init__(self, p):
        super(Masking, self).__init__()
        assert 0 <= p <= 1, "Probability p must be between 0 and 1"
        self.p = p

    def forward(self, x):
        if self.training:
            batch_size, L, D = x.shape
            # Create masks for each sample in the batch
            mask = torch.rand(batch_size, L, device=x.device) > self.p
            x = x * mask.unsqueeze_(2)
        return x


class BI_ENC(nn.Module):
    def __init__(self, args):
        super(BI_ENC, self).__init__()
        _, self.orig_d_a, self.orig_d_v = args.feature_dims
        _, self.l_a, self.l_v = args.seq_lens
        # if args["av_enc"].get("d_enc", False):
        self.d_a = self.d_v = args["av_enc"]["d_enc"]
        self.d_enc_out = args["av_enc"]["d_enc_out"]
        self.layers = args["av_enc"]["nlevels"]
        self.maxlen = args["av_enc"]["maxlen"]  # text's maximum length
        self.a_pdrop = args["av_enc"]["enc_dropout"]
        self.v_pdrop = args["av_enc"]["enc_dropout"]
        self.d_enc = args["av_enc"]["d_enc"]
        self.n_head = args["av_enc"]["n_head"]
        self.p_mask = args["av_enc"]["p_mask"]
        self.d_enc = args["av_enc"]["d_enc"]
        self.nlevels = args["av_enc"]["nlevels"]
        self.use_sperm = args["av_enc"]["use_softperm"]
        self.p_perm = args["av_enc"]["p_perm"]
        self.tf_fusion = args["av_enc"].get("tf_fusion", False)
        self.use_bn = args["av_enc"].get("use_bn", False)
        self.use_m3 = args["av_enc"].get("use_m3", False)
        # Batch Normalization 
        if isinstance(self.use_bn, dict):
            self.use_bn = True
            self.use_bn_a = args["av_enc"]["use_bn"].get("use_bn_a", False)
            self.use_bn_v = args["av_enc"]["use_bn"].get("use_bn_v", False)
        elif self.use_bn:
            self.use_bn = self.use_bn_a = self.use_bn_v = True 
        else:
            self.use_bn = False
        # Modality masking
        if isinstance(self.use_m3, dict):
            self.use_m3 = True
            self.p_a = args["av_enc"]["use_m3"].get("p_a", 0.0)
            self.p_v = args["av_enc"]["use_m3"].get("p_v", 0.0)
        else:
            self.use_m3 = False
        self.use_ln = args["av_enc"].get("use_ln", False)
        combined_dim = self.d_a + self.d_v
        # positional encodings
        self.embed_scale_a = math.sqrt(self.d_a)
        self.embed_positions_a = SinusoidalPositionalEmbedding(self.d_a)
        self.embed_scale_v = math.sqrt(self.d_v)
        self.embed_positions_v = SinusoidalPositionalEmbedding(self.d_v)

        if self.use_ln:
            self.LN_a = nn.LayerNorm(self.orig_d_a)
            self.LN_v = nn.LayerNorm(self.orig_d_v)
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

        if self.use_m3:
            print(f"-----Using AUDIO masking with probability {self.p_a}")
            self.mask_a = Masking(p=self.p_a)
            print(f"-----Using VISION masking with probability {self.p_v}")
            self.mask_v = Masking(p=self.p_v)

        # Masking as feture augmentation
        # self.m3_drop = M3(p=self.p_mask)
        if self.use_sperm:
            l_t, l_a, l_v = args["seq_lens"]
            self.a_sperm = SoftPerm_Fast(
                p_feat=self.p_perm,
                maxlen=l_a
            )
            self.v_sperm = SoftPerm_Fast(
                p_feat=self.p_perm,
                maxlen=l_v
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
        if not(self.tf_fusion):
            self.fusion = nn.Linear(combined_dim, self.d_enc_out)
        else:
            print(f"Using transformer encoder layer as fusion")
            self.fusion = nn.Sequential( 
                nn.Linear(combined_dim, self.d_enc_out),
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=self.d_enc_out,
                        nhead=self.n_head,
                        dropout=self.v_pdrop,
                        dim_feedforward=4*self.d_enc_out,
                        activation="gelu",
                        batch_first=True, norm_first=True
                        ),
                    num_layers=1
                ),
            )

        # # Clf - not actually used here yet
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

    def forward(self, _, x_a, x_v):
        """x_a, x_v: (B, L, D)
        """
        # layer norm directly at inputs (optional)
        if self.use_ln:
            x_a = self.LN_a(x_a)
            x_v = self.LN_v(x_v)
        
        # apply seqaug
        if self.training and self.use_sperm:
            # x_a, x_v = self.m3_drop(x_a, x_v)
            x_a = self.a_sperm(x_a)
            x_v = self.v_sperm(x_v)
        
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
        # x_a = self.LN_a(self.align(x_a))
        # x_v = self.LN_v(self.align(x_v))
        
        # project_in, rescaling adopted from fairseq repo
        x_a = self.proj_a(x_a) * self.embed_scale_a
        x_v = self.proj_v(x_v) * self.embed_scale_v
        
        # positional ambeddings
        x_a += self.embed_positions_a(x_a[:, :, 0])
        x_a = F.dropout(x_a, p=self.a_pdrop, training=self.training)
        x_v += self.embed_positions_v(x_v[:, :, 0])
        x_v = F.dropout(x_v, p=self.v_pdrop, training=self.training)

        # modality masking
        if self.use_m3 and self.training:
            if self.p_a > 0:
                x_a = self.mask_a(x_a)
            if self.p_v > 0:
                x_v = self.mask_v(x_v)
        
        # encode
        x_a = self.enc_a(x_a)
        x_v = self.enc_v(x_v)
        
        # fusion
        if not(self.tf_fusion):
            # fusion: [B,L,D] --> [B,L,2D]
            x_f = \
                torch.cat((x_a, x_v), dim=2)[:, -1, :] # keep last
            x_f = self.fusion(x_f)
        else:
            x_f = torch.cat((x_a, x_v), dim=2)
            x_f = self.fusion(x_f)[:, 0, :]
        
        # classification map
        x_f = self.clf(x_f)
        return x_f
