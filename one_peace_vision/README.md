# ONE-PEACE for Vision

[one-peace-vision.pkl](https://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_checkpoints/one-peace-vision.pkl) can be extracted by [the conversion script](./convert_to_vision.py). We use it for downstream vision tasks.
```python
python convert_to_vision.py /path/to/one-peace.pt /save_path/one-peace-vision.pkl
```

<!-- TOC -->
* [Image Classification](./classification/README.md)
* [Semantic Segmentation](./seg/README.md)
* [Object Detection & Instance Segmentation](./det/README.md)
* [Video Action Recognition](./video/README.md)
<!-- TOC -->
