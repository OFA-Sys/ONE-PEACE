import torch
import numpy as np
from ..utils.data_utils import collate_tokens


def collate_fn(samples, pad_idx, pad_to_length=None):
    if len(samples) == 0:
        return {}

    def merge(key, pad=pad_idx, pad_to_length=None):
        if isinstance(samples[0][key], list):
            return collate_tokens([item for s in samples for item in s[key]], pad_idx=pad, pad_to_length=pad_to_length)
        else:
            return collate_tokens([s[key] for s in samples], pad_idx=pad, pad_to_length=pad_to_length)

    id = np.array([s["id"] for s in samples])

    src_tokens = None
    if samples[0].get("source_text", None) is not None:
        src_tokens = merge("source_text", pad_to_length=pad_to_length)

    src_images = None
    if samples[0].get("source_image", None) is not None:
        src_images = torch.stack([sample['source_image'] for sample in samples], dim=0)

    src_audios = None
    audio_padding_masks = None
    if samples[0].get("source_audio", None) is not None:
        assert samples[0].get("audio_padding_mask") is not None
        src_audios = merge("source_audio", pad=0)
        audio_padding_masks = merge("audio_padding_mask", pad=True)

    target = None
    if samples[0].get("target", None) is not None:
        target = torch.cat([s['target'] for s in samples])

    batch = {
        "id": id,
        "ntokens": len(samples),
        "nsentences": len(samples),
        "net_input": {
            "src_tokens": src_tokens,
            "src_images": src_images,
            "src_audios": src_audios,
            "audio_padding_masks": audio_padding_masks
        },
        "target": target
    }
    # for language pretraining
    if samples[0].get("text_mask_indices", None) is not None:
        batch["net_input"]["text_mask_indices"] = merge("text_mask_indices", pad=False)
    if samples[0].get("text_preserve_ids", None) is not None:
        batch["net_input"]["text_preserve_ids"] = merge("text_preserve_ids", pad=-1)
    # for vision pretraining
    if samples[0].get("image_mask_indices", None) is not None:
        batch["net_input"]["image_mask_indices"] = merge("image_mask_indices", pad=False)
    if samples[0].get("image_preserve_ids", None) is not None:
        batch["net_input"]["image_preserve_ids"] = merge("image_preserve_ids", pad=-1)
    # for audio pretraining
    if samples[0].get("audio_mask_indices", None) is not None:
        batch["net_input"]["audio_mask_indices"] = merge("audio_mask_indices", pad=False)
    if samples[0].get("audio_preserve_ids", None) is not None:
        batch["net_input"]["audio_preserve_ids"] = merge("audio_preserve_ids", pad=-1)
    # for vision-language pretraining
    if samples[0].get("vl_text_mask_indices", None) is not None:
        batch["net_input"]["vl_text_mask_indices"] = merge("vl_text_mask_indices", pad=False)
    if samples[0].get("vl_text_preserve_ids", None) is not None:
        batch["net_input"]["vl_text_preserve_ids"] = merge("vl_text_preserve_ids", pad=-1)
    if samples[0].get("vl_image_mask_indices", None) is not None:
        batch["net_input"]["vl_image_mask_indices"] = merge("vl_image_mask_indices", pad=False)
    if samples[0].get("vl_image_preserve_ids", None) is not None:
        batch["net_input"]["vl_image_preserve_ids"] = merge("vl_image_preserve_ids", pad=-1)
    # for audio-language pretraining
    if samples[0].get("al_text_mask_indices", None) is not None:
        batch["net_input"]["al_text_mask_indices"] = merge("al_text_mask_indices", pad=False)
    if samples[0].get("al_text_preserve_ids", None) is not None:
        batch["net_input"]["al_text_preserve_ids"] = merge("al_text_preserve_ids", pad=-1)
    if samples[0].get("al_audio_mask_indices", None) is not None:
        batch["net_input"]["al_audio_mask_indices"] = merge("al_audio_mask_indices", pad=False)
    if samples[0].get("al_audio_preserve_ids", None) is not None:
        batch["net_input"]["al_audio_preserve_ids"] = merge("al_audio_preserve_ids", pad=-1)
    # for nlvr2
    if samples[0].get("source_image2", None) is not None:
        batch["net_input"]["src_images_2"] = torch.stack([sample['source_image2'] for sample in samples], dim=0)
    # for refcoco
    if samples[0].get("w_resize_ratio", None) is not None:
        batch["w_resize_ratios"] = torch.tensor([s["w_resize_ratio"] for s in samples])
    if samples[0].get("h_resize_ratio", None) is not None:
        batch["h_resize_ratios"] = torch.tensor([s["h_resize_ratio"] for s in samples])
    if samples[0].get("region_coord", None) is not None:
        batch["region_coords"] = torch.stack([s['region_coord'] for s in samples], dim=0)

    return batch