import os
os.environ["NCCL_P2P_DISABLE"] = "1"

_base_ = [
    '../configs/_base_/models/mask-rcnn_r50_fpn.py',
    '../configs/_base_/datasets/coco_instance.py',
    '../configs/_base_/schedules/schedule_2x.py', 
    '../configs/_base_/default_runtime.py'
]

metainfo = dict(classes=('stone', ), palette=[(220, 20, 60)])
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)
    )
)

data_root = 'A-datas/'

# [剥离装甲 1]：简化数据增强管道，关闭多尺度，只保留最基础的 Resize 和 Flip
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True), # 固定单一尺度
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline # 应用朴素版数据增强
    )
)

val_dataloader = dict(dataset=dict(data_root=data_root, metainfo=metainfo, ann_file='annotations/instances_val2017.json', data_prefix=dict(img='val/')))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = val_evaluator

# [剥离装甲 2]：残酷地干掉 Warmup (预热)
param_scheduler = [
    # 这里直接删除了 LinearLR Warmup 阶段，开局直接硬刚
    dict(type='MultiStepLR', begin=0, end=24, by_epoch=True, milestones=[16, 22], gamma=0.1)
]

# 单卡 BS=2，学习率严格按比例缩小
optim_wrapper = dict(optimizer=dict(lr=0.0025))