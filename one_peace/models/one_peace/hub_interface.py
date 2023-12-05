
import os
import urllib
import math
import librosa
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from fairseq import checkpoint_utils, utils

from ...data.base_dataset import CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD
from ...utils.data_utils import collate_tokens
from ... import tasks
from ... import models


_MODELS = {
    "ONE-PEACE": "http://one-peace-shanghai.oss-accelerate.aliyuncs.com/one-peace.pt",
    "ONE-PEACE_Grounding": "https://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_checkpoints/finetune_refcocog.pt",
    "ONE-PEACE_VGGSound": "https://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_checkpoints/finetune_vggsound.pt"
}

def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True,
                  unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return download_target


def from_pretrained(
    model_name_or_path,
    model_type='one_peace_retrieval',
    device=("cuda" if torch.cuda.is_available() else "cpu"),
    dtype="float32",
    download_root=None
):

    if os.path.isfile(model_name_or_path):
        model_path = model_name_or_path
    else:
        model_path = _download(_MODELS[model_name_or_path], download_root or os.path.expanduser("~/.cache/one-peace"))

    overrides = {'model':{'_name': model_type}}
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        arg_overrides=overrides
    )
    model = models[0]

    return OnePeaceHubInterface(saved_cfg, task, model, device, dtype)


class OnePeaceHubInterface:
    """A simple PyTorch Hub interface to ONE-PEACE."""

    def __init__(self, cfg, task, model, device="cpu", dtype="float32"):
        super().__init__()
        self.model = model
        self.device = device
        self.dtype = dtype
        self.model_type = cfg.model._name

        # for text
        self.dict = task.dict
        self.bpe = task.bpe
        self.eos = self.dict.eos()
        self.pad = self.dict.pad()
        # for image
        mean = CLIP_DEFAULT_MEAN
        std = CLIP_DEFAULT_STD
        self.transform = transforms.Compose([
            transforms.Resize(
                (cfg.task.patch_image_size, cfg.task.patch_image_size),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        # for audio
        feature_encoder_spec = '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]'
        self.feature_encoder_spec = eval(feature_encoder_spec)
        self._features_size_map = {}

        self.model.to(device)
        self.model.eval()
        if self.dtype == "bf16":
            self.model.bfloat16()
        elif self.dtype == "fp16":
            self.model.half()
        else:
            self.model.float()

    def cast_data_dtype(self, t):
        if self.dtype == "bf16":
            return t.to(dtype=torch.bfloat16)
        elif self.dtype == "fp16":
            return t.to(dtype=torch.half)
        else:
            return t

    def _get_mask_indices_dims(self, size, feature_encoder_spec, padding=0, dilation=1):
        if size not in self._features_size_map:
            L_in = size
            for (_, kernel_size, stride) in feature_encoder_spec:
                L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
                L_out = 1 + L_out // stride
                L_in = L_out
            self._features_size_map[size] = L_out
        return self._features_size_map[size]

    def process_text(self, text_list):
        tokens_list = []
        for text in text_list:
            text = ' {}'.format(text.lower())
            s = self.dict.encode_line(
                line=self.bpe.encode(text),
                add_if_not_exist=False,
                append_eos=False
            ).long()
            s = s[:70]
            s = torch.cat([s, torch.LongTensor([self.eos])])
            tokens_list.append(s)
        src_tokens = collate_tokens(tokens_list, pad_idx=self.pad).to(self.device)
        src_tokens = self.cast_data_dtype(src_tokens)
        return src_tokens

    def process_image(self, image_list, return_image_sizes=False):
        patch_images_list = []
        image_width_list = []
        image_height_list = []
        for image_path in image_list:
            image = Image.open(image_path).convert("RGB")
            w, h = image.size
            patch_image = self.transform(image)
            patch_images_list.append(patch_image)
            image_width_list.append(w)
            image_height_list.append(h)
        src_images = torch.stack(patch_images_list, dim=0).to(self.device)
        src_images = self.cast_data_dtype(src_images)
        if return_image_sizes:
            image_widths = torch.tensor(image_width_list).to(self.device)
            image_heights = torch.tensor(image_height_list).to(self.device)
            return src_images, image_widths, image_heights
        else:
            return src_images

    def process_audio(self, audio_list):
        feats_list = []
        audio_padding_mask_list = []
        for audio in audio_list:
            wav, curr_sample_rate = librosa.load(audio, sr=16000)
            assert curr_sample_rate == 16000
            feats = torch.tensor(wav)
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
            if feats.size(-1) > curr_sample_rate * 15:
                start_idx = 0
                end_idx = start_idx + curr_sample_rate * 15
                feats = feats[start_idx:end_idx]
            if feats.size(-1) < curr_sample_rate * 15:
                feats = feats.repeat(math.ceil(curr_sample_rate * 15 / feats.size(-1)))
                feats = feats[:curr_sample_rate * 15]
            T = self._get_mask_indices_dims(feats.size(-1), self.feature_encoder_spec)
            audio_padding_mask = torch.zeros(T + 1).bool()
            feats_list.append(feats)
            audio_padding_mask_list.append(audio_padding_mask)
        src_audios = collate_tokens(feats_list, pad_idx=0).to(self.device)
        src_audios = self.cast_data_dtype(src_audios)
        audio_padding_masks = collate_tokens(audio_padding_mask_list, pad_idx=True).to(self.device)
        return src_audios, audio_padding_masks

    def process_image_text_pairs(self, image_text_list, return_image_sizes=False):
        image_list = [image_text_pair[0] for image_text_pair in image_text_list]
        text_list = [image_text_pair[1] for image_text_pair in image_text_list]
        src_tokens = self.process_text(text_list)
        if return_image_sizes:
            src_images, image_widths, image_heights = self.process_image(image_list, return_image_sizes=True)
            return (src_images, image_widths, image_heights), src_tokens
        else:
            src_images = self.process_image(image_list)
            return src_images, src_tokens

    def extract_text_features(self, src_tokens):
        if self.model_type == 'one_peace_classify':
            self.model(src_tokens=src_tokens)
        else:
            return self.model(src_tokens=src_tokens, encoder_type="text")

    def extract_image_features(self, src_images):
        if self.model_type == 'one_peace_classify':
            return self.model(src_images=src_images)
        else:
            return self.model(src_images=src_images, encoder_type="image")

    def extract_audio_features(self, src_audios, audio_padding_masks):
        if self.model_type == 'one_peace_classify':
            return self.model(src_audios=src_audios, audio_padding_masks=audio_padding_masks)
        else:
            return self.model(src_audios=src_audios, audio_padding_masks=audio_padding_masks, encoder_type="audio")

    def extract_vl_features(self, src_images, src_tokens):
        return self.model(src_tokens=src_tokens, src_images=src_images)
