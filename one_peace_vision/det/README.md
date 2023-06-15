# ONE-PEACE for Object Detection

### Pretrained Models
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">batch size</th>
<th valign="bottom">iter</th>
<th valign="bottom">box AP</th>
<th valign="bottom">mask AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="configs/onepeace/cascade_mask_rcnn_vitdet_50ep.py">onepeace_det</a></td>
<td align="center">64</td>
<td align="center">90k</td>
<td align="center">60.4</td>
<td align="center">52.9</td>
<td align="center">-</a></td>
</tr>
</tbody></table>

### Installation
```
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2[all]
```

### Evaluation
Object detection:
```
python lazyconfig_train_net.py --config-file ./configs/onepeace/cascade_mask_rcnn_vitdet_50ep.py --num-gpus 4 --eval-only train.init_checkpoint=/path/to/model_checkpoint
```
Expected results:
```
Task: bbox
AP,AP50,AP75,APs,APm,APl
60.3704,79.2223,65.5209,44.8737,64.9417,75.5091
```

Instance segmentation:
```
python lazyconfig_train_net.py --config-file ./configs/onepeace/cascade_mask_rcnn_vitdet_50ep.py --num-gpus 4 --eval-only train.init_checkpoint=/path/to/model_checkpoint model.roi_heads.maskness_thresh=0.5
```
Expected results:
```
Task: segm
AP,AP50,AP75,APs,APm,APl
52.8513,77.0620,58.0247,34.1789,56.3485,71.3040
```

### Training
```
python lazyconfig_train_net.py --config-file ./configs/onepeace/cascade_mask_rcnn_vitdet_50ep.py --num-gpus 8 --num-machines 8 --machine-rank ${RANK} --dist-url "tcp://$MASTER_ADDR:60900" model.backbone.net.pretrained=${CHECKPOINT_PATH} train.output_dir=${OUTPUT_DIR}
```