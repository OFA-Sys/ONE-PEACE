# ONE-PEACE: Exploring One General Representation Model Toward Unlimited modalities

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-peace-exploring-one-general/semantic-segmentation-on-ade20k)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k?p=one-peace-exploring-one-general)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-peace-exploring-one-general/audio-to-text-retrieval-on-audiocaps)](https://paperswithcode.com/sota/audio-to-text-retrieval-on-audiocaps?p=one-peace-exploring-one-general)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-peace-exploring-one-general/text-to-audio-retrieval-on-audiocaps)](https://paperswithcode.com/sota/text-to-audio-retrieval-on-audiocaps?p=one-peace-exploring-one-general)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-peace-exploring-one-general/audio-to-text-retrieval-on-clotho)](https://paperswithcode.com/sota/audio-to-text-retrieval-on-clotho?p=one-peace-exploring-one-general)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-peace-exploring-one-general/text-to-audio-retrieval-on-clotho)](https://paperswithcode.com/sota/text-to-audio-retrieval-on-clotho?p=one-peace-exploring-one-general)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-peace-exploring-one-general/image-to-text-retrieval-on-coco)](https://paperswithcode.com/sota/image-to-text-retrieval-on-coco?p=one-peace-exploring-one-general)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-peace-exploring-one-general/image-to-text-retrieval-on-flickr30k)](https://paperswithcode.com/sota/image-to-text-retrieval-on-flickr30k?p=one-peace-exploring-one-general)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-peace-exploring-one-general/audio-classification-on-fsd50k)](https://paperswithcode.com/sota/audio-classification-on-fsd50k?p=one-peace-exploring-one-general)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-peace-exploring-one-general/self-supervised-image-classification-on-1)](https://paperswithcode.com/sota/self-supervised-image-classification-on-1?p=one-peace-exploring-one-general)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-peace-exploring-one-general/referring-expression-comprehension-on-refcoco-1)](https://paperswithcode.com/sota/referring-expression-comprehension-on-refcoco-1?p=one-peace-exploring-one-general)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-peace-exploring-one-general/referring-expression-comprehension-on-refcoco)](https://paperswithcode.com/sota/referring-expression-comprehension-on-refcoco?p=one-peace-exploring-one-general)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-peace-exploring-one-general/referring-expression-comprehension-on)](https://paperswithcode.com/sota/referring-expression-comprehension-on?p=one-peace-exploring-one-general)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-peace-exploring-one-general/referring-expression-comprehension-on-1)](https://paperswithcode.com/sota/referring-expression-comprehension-on-1?p=one-peace-exploring-one-general)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-peace-exploring-one-general/visual-question-answering-on-vqa-v2-test-std)](https://paperswithcode.com/sota/visual-question-answering-on-vqa-v2-test-std?p=one-peace-exploring-one-general)

[[`Paper`](https://arxiv.org/abs/2305.11172)] [Demo] [Checkpoints]

ONE-PEACE is a general representation model across vision, audio, and language modalities,
Without using any vision or language pretrained model for initialization, ONE-PEACE achieves leading results in vision, 
audio, audio-language, and vision-language tasks.
Furthermore, ONE-PEACE possesses a strong emergent zero-shot retrieval capability, enabling it to align modalities
that are not paired in the training data.

Below shows the architecture and pretraining tasks of ONE-PEACE.
With the scaling-friendly architecture and modality-agnostic tasks, ONE-PEACE has the potential to expand to unlimited modalities.

<p align="center">
<img src="assets/one_peace.png" width=100%>
</p>

## News
* 2023.5.19: Released the paper and code. Pretrained & finetuned checkpoints, training & inference scripts, as well as demos will be released as soon as possible.
<br></br>

## Emergent Zero-shot Retrieval

### Audio-to-Image
![a2i](assets/audio2img.png)

### Audio+Text-to-Image
![a+t2i](assets/audio+text2img.png)

### Audio+Image-to-Image
![a+i2i](assets/audio+img2img.png)
<br></br>

## Related Codebase
* [Fairseq](https://github.com/pytorch/fairseq)
* [xFormers](https://github.com/facebookresearch/xformers)
* [FlashAttention](https://github.com/HazyResearch/flash-attention)
* [Apex](https://github.com/NVIDIA/apex)
<br></br>

## Getting Involved
Feel free to submit Github issues or pull requests. Welcome to contribute to our project!

To contact us, never hestitate to send an email to `zheluo.wp@alibaba-inc.com` or `saimeng.wsj@alibaba-inc.com`!
<br></br>

### Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)

```BibTeX
@article{ONEPEACE,
  title={ONE-PEACE: Exploring one general Representation Model toward unlimited modalities},
  author={Wang, Peng and Wang, Shijie and Lin, Junyang and Bai, Shuai and Zhou, Xiaohuan and Zhou, Jingren and Wang, Xinggang and Zhou, Chang},
  journal={arXiv preprint arXiv:},
  year={2023}
}
```
