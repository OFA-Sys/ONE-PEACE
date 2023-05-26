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
bash zero_shot_evaluate_coco.sh
bash zero_shot_evaluate_flickr.sh
```

## NLVR2
1. Download [NLVR2](http://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_datasets/nlvr2.zip)
2. Finetuning
```bash
cd one_peace/run_scripts/nlvr2
bash finetune.sh
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

## VQA
1. Download [VQAv2](http://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_datasets/vqa.zip)
2. Finetuning
```bash
cd one_peace/run_scripts/vqa
bash finetune.sh
```





