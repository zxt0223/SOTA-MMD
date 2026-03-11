import os
os.environ["NCCL_P2P_DISABLE"] = "1"

_base_ = [
    '../configs/_base_/models/mask-rcnn_r50_fpn.py',
    '../configs/_base_/datasets/coco_instance.py',
    '../configs/_base_/schedules/schedule_1x.py',
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

train_dataloader = dict(dataset=dict(data_root=data_root, metainfo=metainfo, ann_file='annotations/instances_train2017.json', data_prefix=dict(img='train/')))
val_dataloader = dict(dataset=dict(data_root=data_root, metainfo=metainfo, ann_file='annotations/instances_val2017.json', data_prefix=dict(img='val/')))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = val_evaluator

# [冒烟测试专属] 仅训练 1 个 epoch，验证 1 次
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)
optim_wrapper = dict(optimizer=dict(lr=0.005))