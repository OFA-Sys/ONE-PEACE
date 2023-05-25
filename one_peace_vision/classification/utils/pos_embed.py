# Copyright 2023 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np
import torch
from scipy import interpolate


def interpolate_pos_embed(model, checkpoint_model):
    if 'image_adapter.pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['image_adapter.pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        bucket_size = model.image_adapter.bucket_size
        num_patches = bucket_size ** 2
        num_extra_tokens = model.image_adapter.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
            extra_tokens = pos_embed_checkpoint[:num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
                                            embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
              pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(0, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)
            checkpoint_model['image_adapter.pos_embed'] = new_pos_embed


def interpolate_rel_pos_embed(model, checkpoint_model, pos_embed_key):
    if pos_embed_key in checkpoint_model:
        rel_pos_bias = checkpoint_model[pos_embed_key]
        src_num_pos, num_attn_heads = rel_pos_bias.size()
        dst_num_pos, _ = model.state_dict()[pos_embed_key].size()
        num_extra_tokens = 3
        src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
        dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
        if src_size != dst_size:
            print(
                f"Relative position interpolate from {src_size}x{src_size} to {dst_size}x{dst_size}")
            extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
            rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

            def geometric_progression(a, r, n):
                return a * (1.0 - r ** n) / (1.0 - r)

            left, right = 1.01, 1.5
            while right - left > 1e-6:
                q = (left + right) / 2.0
                gp = geometric_progression(1, q, src_size // 2)
                if gp > dst_size // 2:
                    right = q
                else:
                    left = q

            dis = []
            cur = 1
            for i in range(src_size // 2):
                dis.append(cur)
                cur += q ** (i + 1)

            r_ids = [-_ for _ in reversed(dis)]

            x = r_ids + [0] + dis
            y = r_ids + [0] + dis

            t = dst_size // 2.0
            dx = np.arange(-t, t + 0.1, 1.0)
            dy = np.arange(-t, t + 0.1, 1.0)
            print(f"x = {x}")
            print(f"dx = {dx}")

            all_rel_pos_bias = []

            for i in range(num_attn_heads):
                z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                f = interpolate.interp2d(x, y, z, kind='cubic')
                all_rel_pos_bias.append(
                  torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

            rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
            new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
            checkpoint_model[pos_embed_key] = new_rel_pos_bias
