'''
This is the implementation used in the ICASSP submission of MMAug.
Some hyperparameters which are redundant are set to their default value
in this implementation. Moreover, the padding is at the end (suffix) so the
implementation takes into account this fomat. The overall pipeline is
identical to not affect any results.
'''
import random
from typing import List, Optional, Dict

import torch
import numpy as np
import torch.nn as nn
import torch.distributions.beta as beta
import torch.distributions.half_normal as half_normal
import torch.distributions as D


class PartialPermutation(nn.Module):
    def __init__(self, p_perm):
        super(PartialPermutation, self).__init__()
        self.p_perm = p_perm[0]
        self.std_perm = p_perm[1]

    def forward(self, x, real_len=None):
        # x: (B, L)
        # Create a mask to identify the real part of the sequence
        if self.training:
            mask = torch.arange(x.size(1)).expand_as(x).to(x.device) < real_len.view(-1, 1)
            bsz, maxlen = x.size(0), x.size(1)
            x_pp = x.clone()  # Use .clone() to create a copy of x
            for b in range(bsz):
                # isolate x
                x_b = x_pp[b]
                b_mask = mask[b]
                b_mask[0] = False # exclude <CLS>
                b_mask[real_len[b]-1] = False # exclude <EOS>
                b_mask_float = b_mask.float()
                # import pdb; pdb.set_trace()
                perm_idx_mask = torch.bernoulli(b_mask_float*self.p_perm).int()
                while torch.sum(perm_idx_mask) <= 1:
                    perm_idx_mask = torch.bernoulli(b_mask_float*self.p_perm).int()
                x_to_perm = x_b[perm_idx_mask.bool()]
                x_to_perm = x_to_perm[torch.randperm((x_to_perm.size(0)))]
                # Create a permutation mapping for the real part
                x_b[perm_idx_mask.bool()] = x_to_perm
                # Apply the permutation to the real part of the sequence
                x_pp[b] = x_b
            return x_pp
        else:
            return x

# # Example usage
# p_perm = 0.3  # Permutation probability
# max_seq_len = 12  # Maximum sequence length
# batch_size = 3  # Batch size
# real_lens = torch.tensor([7, 5, 8], dtype=torch.int64)  # Real lengths of sequences

# # Create synthetic input data with zero padding
# input_data = torch.tensor([
#     [1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0],
#     [11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0],
#     [21, 22, 23, 24, 25, 26, 27, 28, 0, 0, 0, 0]
# ], dtype=torch.float32)

# # Create the PartialPermutation module
# partial_permute_module = PartialPermutation(p_perm)

# # Set the module to training mode (this activates the permutation)
# partial_permute_module.train()

# # Apply the module to the input data
# permuted_data = partial_permute_module(input_data, real_lens)

# print("Original Data:")
# print(input_data)

# print("\nPermuted Data:")
# print(permuted_data)



class SoftPerm(nn.Module):
    def __init__(
        self,
        p_t_mod: (List[float]) = [1.0, 1.0, 1.0],
        alpha: (List[tuple]) = [(.1, 0.05), (0.1, 0.05), (.1, .05)],
        maxlen: Optional[int] = 50,
    ):
        """
        SoftPerm: SoftPermutation, randomly samples a beta distribution
        with hyperparam $B(a_1, a_2)$. By sampling, it gets the `copy_area`
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
            discard_zero_pad (bool): when true resamples only from the real length sequence
                and ignores the zero padded part at the beginning
            use_beta (bool): when true samples from a Betta distribution for each one
                of the involved modalities, otherwise from a uniform
        """
        super(SoftPerm, self).__init__()
        self.p_t_mod = p_t_mod
        self.n_modalities = len(self.p_t_mod)
        self.alpha = alpha
        self.maxlen = maxlen

        self.beta_mod = self.set_beta_mod()

        # OPTIONAL
        # distribution which defines which timesteps are going to be permuted (time-oriented mask)
        self.time_distribution = []
        for k in range(self.n_modalities):
            # p_t_mod: "0 probability"
            self.time_distribution.append(
                self.get_bernoulli_distribution(
                    torch.tensor(float(self.p_t_mod[k]), device="cuda")
                )
            )

    def set_beta_mod(self):
        beta_mod = []
        for (a_1, a_2) in self.alpha:
            # print(f"------------using BEta  with {a_1} and {a_2}------------")
            # beta_mod.append(
            #     D.beta.Beta(
            #         torch.tensor(a_1, device="cuda"),
            #         torch.tensor(a_2, device="cuda")
            #     )
            # )
            print(f"------------using Normal  with mean {a_1} and std {a_2}------------")
            if a_1 == 0.0:
                beta_mod.append([
                    torch.tensor(a_1, device="cuda"),
                    torch.tensor(a_2, device="cuda"),
                ])
            else:
                beta_mod.append([
                    torch.tensor(a_1, device="cuda"),
                    half_normal.HalfNormal(torch.tensor(a_2, device="cuda"))
                ])
        print(f"Beta mod is {beta_mod}")
        return beta_mod

    @staticmethod
    def get_bernoulli_distribution(zero_out_prob):
        '''defines a distribution from which we sample which feature-dimensions are going 
        to be blended for every feature tensor (area-oriented masking)
        Tip: probability of drawing 1 is probs
        '''
        return D.bernoulli.Bernoulli(probs=1.0 - torch.tensor(zero_out_prob))

    def forward(self, mods, real_len=None):
        """
        SoftPerm forward implementation
        Sample from $n_modalities$ independent Uniform distributions to mask "mask_area"
        for each one of the modalities.
        Args:
            mods (varargs torch.Tensor): [B, L, D_m] Modality representations
            m_ra (int): repeated augmentation index, default = 1
            real_len (torch.Tensor): [B] tensor of ints with the unpadded seq len
        Returns:
            (List[torch.Tensor]): The modality representations. Some of them are dropped
        """
        # List of [B, L, D] and T,A,V order is assumed
        if self.training:
            bsz, seqlen = mods[0].size(0), mods[0].size(1)
            copy_area = []
            for i, stats in enumerate(self.beta_mod):
                if stats[0] == 0 and stats[1] == 0:
                    copy_area.append(torch.zeros((bsz,), device="cuda"))
                else:
                    copy_area.append(torch.clip(stats[0] + stats[1].sample((bsz, )), max=1.0))

            for i_modal, p_modal in enumerate(self.p_t_mod):
                d_modal = mods[i_modal].size(2)

                # distribution which samples feature-dimensions to be permuted (area-masking)
                self.area_distribution = self.get_bernoulli_distribution(copy_area[i_modal])

                # TODO: define a zero tensor rather than sampling (for p_t_mod=1.0 is zeros)
                modal_timestep_mask = \
                    self.time_distribution[i_modal].sample((bsz, seqlen))
                modal_timestep_mask = modal_timestep_mask.unsqueeze(2)
                # keep constant masking across feature dimensions through timesteps
                area_mask = self.area_distribution.sample((1, d_modal))
                area_mask = area_mask.permute(2, 0, 1)

                # Reverse logic because we want to sample masks, i.e. zeros
                # when modal mask is 0 then copy the area mask part
                # when modal mask is 1 then do not mask-> 1's everywhere

                tmp_tensor = mods[i_modal].clone().detach().to(mods[0].device)
                # print("---------Sample from the real length only-------")
                multinomial_probs = torch.ones(bsz, seqlen).to(mods[0].device)
                # normalization
                if real_len is not None:
                    if i_modal > 0:
                        for k in range(bsz):
                            # zero the probability for the padded parts of the sequence
                            multinomial_probs[k, real_len[k]:] = 0
                        # probability normalization
                        multinomial_probs = multinomial_probs / real_len.view(-1, 1)
                    else:
                        multinomial_probs = multinomial_probs / self.maxlen
                else:
                    multinomial_probs = multinomial_probs / self.maxlen

                randperm_mask = \
                    torch.multinomial(
                        multinomial_probs,
                        seqlen,
                        replacement=False,
                    ) #.to(mods[0].device)
                randperm_mask = \
                    randperm_mask \
                    + torch.arange(start=0, end=bsz, device=mods[0].device).reshape(bsz, 1)*seqlen
                tmp_tensor = tmp_tensor.reshape(-1, d_modal)
                tmp_tensor = tmp_tensor[randperm_mask, :].reshape(bsz, seqlen, d_modal)

                # cutmix -like approach / softperm
                mods[i_modal] = \
                    (mods[i_modal]*area_mask + (1-area_mask)*tmp_tensor)*(1-modal_timestep_mask) \
                    + mods[i_modal] * modal_timestep_mask
            return mods

    def __repr__(self):
        shout = (
            self.__class__.__name__
            + "("
            + "p_mod="
            + str(self.p_t_mod)
            + f", mask_dim={self.mask_dim}"
            + f", alpha={self.alpha}"
            + f", permute={self.permute}"
            + f", reaplacement={self.replacement}"
            + ")"
        )
        return shout


class SoftPerm_Fast(nn.Module):
    def __init__(
        self,
        p_feat: float = 1.0,
        maxlen: Optional[int] = 50,
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
        super(SoftPerm_Fast, self).__init__()
        self.p_feat = p_feat  # resampled features
        self.maxlen = maxlen

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
        if self.training and self.p_feat > 0:
            bsz, seqlen, d = x.size()
            weights = torch.full(
                (seqlen,),
                1.0 / seqlen,
                device=x.device,
                dtype=x.dtype,
            )
            permutation = torch.multinomial(weights, seqlen, replacement=False)
            x_perm = x[:, permutation, :]
            keep_prob = 1.0 - self.p_feat
            area_mask = torch.bernoulli(
                torch.full(
                    (bsz, d),
                    keep_prob,
                    device=x.device,
                    dtype=x.dtype,
                )
            )
            # (B,D) --> (B, 1, D)
            area_mask = area_mask.unsqueeze(1)
            x = area_mask * x + (1 - area_mask) * x_perm

        return x

    # def __repr__(self):
    #     shout = (
    #         self.__class__.__name__
    #         + "("
    #         + "p_feat="
    #         + str(self.feat)
    #         + ")"
    #     )
    #     return shout
