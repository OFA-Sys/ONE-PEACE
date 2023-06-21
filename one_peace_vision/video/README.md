# ONE-PEACE for Video Action Recognition

### Pretrained Models
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">batch size</th>
<th valign="bottom">epochs</th>
<th valign="bottom">frames</th>
<th valign="bottom">top1 acc</th>
<th valign="bottom">top5 acc</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="configs/recognition/onepeace_k400.py">onepeace_k400</a></td>
<td align="center">64</td>
<td align="center">30</td>
<td align="center">16</td>
<td align="center">88.0</td>
<td align="center">97.8</td>
<td align="center">-</a></td>
</tr>
<tr><td align="left"><a href="configs/recognition/onepeace_k400_frame32.py">onepeace_k400</a></td>
<td align="center">64</td>
<td align="center">30</td>
<td align="center">32</td>
<td align="center">88.1</td>
<td align="center">97.8</td>
<td align="center">-</a></td>
</tr>
</tbody></table>

### Installation
```
pip install -r requirements.txt
```

### Datasets
Following [here](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/kinetics/README.md) to prepare Kinetics-400 dataset. Note that we use the [AcademicTorrents](https://academictorrents.com/details/184d11318372f70018cf9a72ef867e2fb9ce1d26) version.

### Evaluation
16 Frame:
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=12355 --use_env test.py --launcher pytorch configs/recognition/onepeace_k400.py /path/to/onepeace_video_k400.pth --eval top_k_accuracy
```
Expected results:
```
top1_acc: 0.8800
top5_acc: 0.9776
```

32 Frame:
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=12355 --use_env test.py --launcher pytorch configs/recognition/onepeace_k400_frame32.py /path/to/onepeace_video_k400.pth --eval top_k_accuracy
```
Expected results:
```
top1_acc: 0.8810
top5_acc: 0.9785
```

### Training
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=12355 --use_env train.py --launcher pytorch configs/recognition/onepeace_k400_frame32.py --test-last --validate --cfg-options model.backbone.pretrained=${CHECKPOINT_PATH} work_dir=${OUTPUT_DIR}
```
