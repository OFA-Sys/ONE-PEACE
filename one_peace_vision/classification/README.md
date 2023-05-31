# ONE-PEACE for Image Classification

### Installation
The code requires `python>=3.7`, as well as `pytorch>=1.10` and `torchvision>=0.8`.
```
python -m pip install -r requirements.txt
```

### Evaluation
As a sanity check, run evaluation using our ImageNet **fine-tuned** models:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ONE-PEACE-384</th>
<th valign="bottom">ONE-PEACE-512</th>
<!-- TABLE BODY -->
<tr><td align="left">fine-tuned checkpoint</td>
<td align="center">-</a></td>
<td align="center">-</a></td>
<tr><td align="left">reference ImageNet accuracy</td>
<td align="center">89.6</td>
<td align="center">89.8</td>
</tr>
</tbody></table>

Evaluate ONE-PEACE-384 in a single GPU (`${IMAGENET_DIR}` is a directory containing `{train, val}` sets of ImageNet):
```
python main_ft.py --eval --resume onepeace_ft_21kto1k_384.pth --model one_piece_g_384 --input_size 384 --batch_size 64 --data_path ${IMAGENET_DIR}
```
This should give:
```
* Acc@1 89.558 Acc@5 98.984 loss 0.653
```

Evaluate ONE-PEACE-512
```
python main_ft.py --eval --resume onepeace_ft_21kto1k_512.pth --model one_piece_g_512 --input_size 512 --batch_size 64 --data_path ${IMAGENET_DIR}
```
This should give:
```
* Acc@1 89.752 Acc@5 98.982 loss 0.656
```

### Fine-tuning

Intermediate fine-tune on ImageNet-21k
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=24 --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=6000 --use_env main_ft.py \
    --batch_size 16 \
    --input_size 256 \
    --disable_eval_during_finetuning \
    --nb_classes 19167 \
    --model one_piece_g_256 \
    --finetune ${PRETRAIN_CHKPT} \
    --output_dir ${SAVE_DIR} \
    --log_dir ${SAVE_DIR} \
    --epochs 40 \
    --warmup_epochs 5 \
    --num_workers 4 \
    --lr 1e-4 \
    --min_lr 0.0 \
    --layer_decay 0.85 \
    --opt_betas 0.9 0.98 \
    --opt_eps 1e-6 \
    --weight_decay 0.05 \
    --drop_path 0.4 \
    --color_jitter 0.4 \
    --smoothing 0.1 \
    --reprob 0.0 \
    --mixup 0.0 \
    --cutmix 0.0 \
    --enable_deepspeed \
    --zero_stage 1
```
Fine-tune ONE-PEACE-384 on ImageNet-1k
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=8 --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=6000 --use_env main_ft.py \
    --batch_size 16 \
    --input_size 384 \
    --model one_piece_g_384 \
    --finetune ${21K_CHKPT} \
    --output_dir ${SAVE_DIR} \
    --log_dir ${SAVE_DIR} \
    --epochs 15 \
    --warmup_epochs 3 \
    --num_workers 4 \
    --lr 3e-5 \
    --min_lr 0.0 \
    --layer_decay 0.9 \
    --weight_decay 0.05 \
    --drop_path 0.4 \
    --color_jitter 0.4 \
    --smoothing 0.3 \
    --reprob 0.0 \
    --mixup 0.0 \
    --cutmix 0.0 \
    --use_checkpoint \
    --enable_deepspeed \
    --zero_stage 1
```
Fine-tune ONE-PEACE-512 on ImageNet-1k
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=8 --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=6000 --use_env main_ft.py \
    --batch_size 16 \
    --input_size 512 \
    --model one_piece_g_512 \
    --finetune ${21K_CHKPT} \
    --output_dir ${SAVE_DIR} \
    --log_dir ${SAVE_DIR} \
    --epochs 15 \
    --warmup_epochs 3 \
    --num_workers 4 \
    --lr 5e-5 \
    --min_lr 0.0 \
    --layer_decay 0.9 \
    --weight_decay 0.05 \
    --drop_path 0.4 \
    --color_jitter 0.4 \
    --smoothing 0.3 \
    --reprob 0.0 \
    --mixup 0.0 \
    --cutmix 0.0 \
    --use_checkpoint \
    --enable_deepspeed \
    --zero_stage 1
```