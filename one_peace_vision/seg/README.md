# ONE-PEACE for Semantic Segmentation

### Pretrained Models
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">batch size</th>
<th valign="bottom">iter</th>
<th valign="bottom">mIoU (ss)</th>
<th valign="bottom">mIoU (ms)</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="configs/ade20k/mask2former_onepeace_adapter_g_896_80k_ade20k_ss.py">onepeace_seg_ade20k</a></td>
<td align="center">16</td>
<td align="center">40k</td>
<td align="center">62.0</td>
<td align="center">63.0</td>
<td align="center">-</a></td>
</tr>
</tbody></table>

### Installation
```
pip install mmcv-full==1.5.0 mmdet==2.22.0 mmsegmentation==0.30.0 timm==0.5.4
cd ops & sh make.sh
```

### Evaluation
Single-scale:
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=12355 --use_env test.py --launcher pytorch configs/ade20k/mask2former_onepeace_adapter_g_896_40k_ade20k_ss.py /path/to/onepeace_seg_cocostuff2ade20k.pth --eval mIoU
```
Expected results:
```
Summary:
+-------+------+-------+
|  aAcc | mIoU |  mAcc |
+-------+------+-------+
| 87.18 | 62.0 | 76.42 |
+-------+------+-------+
```

Multi-scale:
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=12355 --use_env test.py --launcher pytorch configs/ade20k/mask2former_onepeace_adapter_g_896_40k_ade20k_ms.py /path/to/onepeace_seg_cocostuff2ade20k.pth --eval mIoU
```
Expected results:
```
Summary:
+-------+-------+-------+
|  aAcc |  mIoU |  mAcc |
+-------+-------+-------+
| 87.59 | 62.98 | 76.82 |
+-------+-------+-------+
```

### Training
COCO-Stuff-164K:
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=12355 --use_env train.py --launcher pytorch configs/coco_stuff164k/mask2former_onepeace_adapter_g_896_80k_cocostuff164k_ss.py --work-dir ${OUTPUT_DIR} --cfg-options model.pretrained=${CHECKPOINT_PATH}
```

ADE20K:
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=12355 --use_env train.py --launcher pytorch configs/ade20k/mask2former_onepeace_adapter_g_896_40k_ade20k_ss.py --load-from=${LOAD_FROM_PATH} --work-dir ${OUTPUT_DIR}
```