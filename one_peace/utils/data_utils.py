import torch
import torch.nn.functional as F
import torch.distributed as dist

import math
from typing import Tuple


def new_islice(iterable, start, stop, step, disable_tolist=False):
    allocate_list = [
        len([j for j in range(i, stop, step)])
        for i in range(0, step)
    ]
    start_idx = sum(allocate_list[:start])
    end_idx = start_idx + allocate_list[start]
    if disable_tolist:
        return iterable[start_idx:end_idx]
    else:
        return list(iterable[start_idx:end_idx])


def collate_tokens(
    values,
    pad_idx,
    pad_to_length=None,
    pad_to_multiple=1
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    if values[0].dim() == 1:
        res = values[0].new(len(values), size).fill_(pad_idx)
    elif values[0].dim() == 2:
        res = values[0].new(len(values), size, values[0].size(1)).fill_(pad_idx)
    else:
        raise NotImplementedError

    for i, v in enumerate(values):
        copy_tensor(v, res[i][: len(v)])
    return res


def all_gather(q, exclude_self=False):
    """
    Gathers tensor arrays of different lengths across multiple gpus

    Parameters
    ----------
        q : tensor array

    Returns
    -------
        all_q : gathered tensor arrays from all the gpus

    """
    ws = dist.get_world_size()
    device = q.device

    local_size = torch.tensor(q.size(0), device=device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(ws)]
    dist.all_gather(all_sizes, local_size)
    max_size = max([sz.item() for sz in all_sizes])

    size_diff = max_size - local_size.item()
    if size_diff:
        padding = torch.zeros(size_diff, *q.shape[1:], device=device, dtype=q.dtype)
        q = torch.cat((q, padding))

    all_qs_padded = [torch.zeros_like(q) for _ in range(ws)]
    dist.all_gather(all_qs_padded, q)
    all_qs = []
    rank_id = dist.get_rank()
    for i, (q, size) in enumerate(zip(all_qs_padded, all_sizes)):
        if exclude_self and i == rank_id:
            continue
        all_qs.append(q[:size])
    all_qs = torch.cat(all_qs, dim=0)
    return all_qs


def get_whole_word_mask(bpe, dictionary):
    if bpe is not None:

        def is_beginning_of_word(i):
            if i < dictionary.nspecial:
                # special elements are always considered beginnings
                return True
            tok = dictionary[i]
            if tok.startswith("madeupword"):
                return True
            try:
                return bpe.is_beginning_of_word(tok)
            except ValueError:
                return True

        mask_whole_words = torch.ByteTensor(
            list(map(is_beginning_of_word, range(len(dictionary))))
        )
        return mask_whole_words
    return None


def compute_block_mask_1d(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    mask_prob_adjust: float = 0,
    inverse_mask: bool = False,
    require_same_masks: bool = True,
    expand_adjcent: bool = False,
    mask_dropout: float = 0,
    non_overlapping: bool = False,
) -> torch.Tensor:

    B, L = shape

    if inverse_mask:
        mask_prob = 1 - mask_prob

    if non_overlapping:
        sz = math.ceil(L / mask_length)

        inp = torch.zeros((B, 1, sz))
        w = torch.ones((1, 1, mask_length))

        mask_inds = torch.multinomial(
            1 - inp.view(B, -1),
            int(sz * (mask_prob + mask_prob_adjust) * (1 + mask_dropout)),
            replacement=False,
        )
        inp.view(B, -1).scatter_(1, mask_inds, 1)

        mask = torch.nn.functional.conv_transpose1d(inp, w, stride=mask_length).squeeze(
            1
        )
        if mask.size(-1) > L:
            mask = mask[..., :L]

    else:
        mask = torch.zeros((B, L))
        mask_inds = torch.randint(
            0,
            L,
            size=(
                B,
                int(
                    L
                    * ((mask_prob + mask_prob_adjust) / mask_length)
                    * (1 + mask_dropout)
                ),
            ),
        )

        mask.view(B, -1).scatter_(1, mask_inds, 1)
        centers = mask.nonzero(as_tuple=True)

        inds = ([], [])

        offset = mask_length // 2
        for i in range(mask_length):
            k1 = i - offset
            inds[0].append(centers[0])
            inds[1].append(centers[1] + k1)

        i0 = torch.cat(inds[0])
        i1 = torch.cat(inds[1]).clamp_(min=0, max=L - 1)

        mask[(i0, i1)] = 1

    def get_nbs(b, m, w):
        all_nbs = torch.nn.functional.conv1d(m.unsqueeze(1), w, padding="same")
        all_nbs = all_nbs.clamp_max_(1).view(b, -1)
        return all_nbs

    if require_same_masks and expand_adjcent:
        w = torch.ones((1, 1, 3))
        w[..., 1] = 0
        all_nbs = get_nbs(B, mask, w)

    mask = mask.view(B, -1)

    if require_same_masks:
        n_masks = mask.sum(dim=-1)
        final_target_len = int(L * (mask_prob))
        target_len = int(final_target_len * (1 + mask_dropout))

        for i in range(len(mask)):
            n = n_masks[i]
            m = mask[i]
            r = 0
            while expand_adjcent and n < target_len:
                if r == 0:
                    nbs = all_nbs[i]
                else:
                    nbs = get_nbs(1, m.unsqueeze(0), w).squeeze(0)

                cands = (1 - m + nbs) > 1
                cand_sz = int(cands.sum().item())

                assert cand_sz > 0, f"{nbs} {cand_sz}"

                to_mask = torch.multinomial(
                    cands.float(), min(cand_sz, int(target_len - n)), replacement=False
                )
                m[to_mask] = 1
                assert to_mask.numel() > 0
                n += to_mask.numel()
                r += 1

            if n > final_target_len:
                to_unmask = torch.multinomial(
                    m, int(n - final_target_len), replacement=False
                )
                m[to_unmask] = 0
            elif n < final_target_len:
                to_mask = torch.multinomial(
                    (1 - m), int(final_target_len - n), replacement=False
                )
                m[to_mask] = 1

    if inverse_mask:
        mask = 1 - mask

    return mask