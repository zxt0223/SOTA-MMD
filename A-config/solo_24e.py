import os
os.environ["NCCL_P2P_DISABLE"] = "1"

_base_ = ['../configs/solo/solo_r50_fpn_1x_coco.py']

metainfo = dict(classes=('stone', ), palette=[(220, 20, 60)])

model = dict(mask_head=dict(num_classes=1))

data_root = 'A-datas/'
train_dataloader = dict(dataset=dict(data_root=data_root, metainfo=metainfo, ann_file='annotations/instances_train2017.json', data_prefix=dict(img='train/')))
val_dataloader = dict(dataset=dict(data_root=data_root, metainfo=metainfo, ann_file='annotations/instances_val2017.json', data_prefix=dict(img='val/')))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = val_evaluator

# 降低学习率，并强制限制梯度最大范数，防止 SOLO 权重跑飞
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0006, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2)
)
optim_wrapper = dict(optimizer=dict(lr=0.01))