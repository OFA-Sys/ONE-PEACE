# Getting Started

Below we provide instructions for training and inference on audio and vision-language tasks.
Pretrained and finetuned checkpoints are provided in [checkpoints.md](../checkpoints.md).

We recommend that your workspace directory should be organized like this:
```
ONE-PEACE/
├── assets/
├── fairseq/
├── one_peace/
│   ├── checkpoints
│   │   ├── one-peace.pt
│   ├── criterions
│   ├── data
│   ├── dataset
│   │   ├── esc50/
│   │   ├── flickr30k/
│   ├── metrics
│   └── ...
├── .gitignore
├── LICENSE
├── README.md
├── checkpoints.md
├── datasets.md
├── requirements.txt
```
<br>

**Please note that if your device does not support bf16 precision, you can switch to fp16 precision for fine-tuning or inference.**
```yaml
common:
  # # use bf16
  # fp16: false
  # memory_efficient_fp16: false
  # bf16: true
  # memory_efficient_bf16: true

  # use fp16
  fp16: true
  memory_efficient_fp16: true
  bf16: false
  memory_efficient_bf16: false
```
<br>

## ESC-50
1. Download [ESC-50](http://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_datasets/esc50.zip)
2. Inference
```bash
cd one_peace/run_scripts/esc50
bash zero_shot_evaluate.sh
```

## Image-Text Retrieval
1. Download [COCO](http://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_datasets/mscoco.zip) and [Flickr](http://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_datasets/flickr30k.zip)
2. Finetuning
```bash
cd one_peace/run_scripts/image_text_retrieval
bash finetune_coco.sh
bash finetune_flickr.sh
```
3. Inference
```bash
cd one_peace/run_scripts/image_text_retrieval
bash zero_shot_evaluate_coco.sh  # zero-shot retrieval for COCO
bash zero_shot_evaluate_flickr.sh  # zero-shot retrieval for Flickr30K
bash evaluate_coco.sh  # evaluation for COCO
bash evaluate_flickr.sh  # evaluation for Flickr30K
```

## NLVR2
1. Download [NLVR2](http://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_datasets/nlvr2.zip)
2. Finetuning
```bash
cd one_peace/run_scripts/nlvr2
bash finetune.sh
```
3. Inference
```bash
cd one_peace/run_scripts/nlvr2
bash evaluate.sh
```

## Visual Grounding
1. Download [RefCOCO](http://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_datasets/refcoco.zip), [RefCOCO+](http://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_datasets/refcoco%2B.zip) and [RefCOCOg](http://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_datasets/refcocog.zip)
2. Finetuning
```bash
cd one_peace/run_scripts/visual_grounding
bash finetune_refcoco.sh
bash finetune_refcoco+.sh
bash finetune_refcocog.sh
```
3. Inference
```bash
cd one_peace/run_scripts/visual_grounding
bash evaluate_refcoco.sh  # evaluation for RefCOCO
bash evaluate_refcoco+.sh  # evaluation for RefCOCO+
bash evaluate_refcocog.sh  # evaluation for RefCOCOg
```

## VQA
1. Download [VQAv2](http://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_datasets/vqa.zip)
2. Finetuning
```bash
cd one_peace/run_scripts/vqa
bash finetune.sh
```
3. Inference
```bash
cd one_peace/run_scripts/vqa
bash evaluate.sh
```

## Audio-Text Retrieval
1. Download [AudioCaps](http://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_datasets/audiocaps.zip), [Clotho](https://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_datasets/clotho.zip) and [MACS](https://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_datasets/macs.zip)
2. Finetuning
```bash
cd one_peace/run_scripts/audio_text_retrieval
bash finetune.sh
```
3. Inference
```bash
cd one_peace/run_scripts/audio_text_retrieval
bash evaluate.sh
```

## Audio Question Answering (AQA)
1. Download [AVQA](https://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_datasets/avqa.zip)
2. Finetuning
```bash
cd one_peace/run_scripts/aqa
bash finetune.sh
```
3. Inference
```bash
cd one_peace/run_scripts/aqa
bash evaluate.sh
```

## FSD50K
1. Download [FSD50K](https://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_datasets/fsd50K.zip)
2. Finetuning
```bash
cd one_peace/run_scripts/fsd50k
bash finetune.sh
```
3. Inference
```bash
cd one_peace/run_scripts/fsd50k
bash evaluate.sh
```

## Vggsound
1. Download [Vggsound](https://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_datasets/vggsound.zip)
2. Finetuning
```bash
cd one_peace/run_scripts/vggsound
bash finetune.sh
```
3. Inference
```bash
cd one_peace/run_scripts/vggsound
bash evaluate.sh
```



