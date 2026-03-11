import os
os.environ["NCCL_P2P_DISABLE"] = "1"

_base_ = ['../configs/yolact/yolact_r50_1xb8-55e_coco.py']

metainfo = dict(classes=('stone', ), palette=[(220, 20, 60)])
model = dict(bbox_head=dict(num_classes=1))

data_root = 'A-datas/'
train_dataloader = dict(dataset=dict(data_root=data_root, metainfo=metainfo, ann_file='annotations/instances_train2017.json', data_prefix=dict(img='train/')))
val_dataloader = dict(dataset=dict(data_root=data_root, metainfo=metainfo, ann_file='annotations/instances_val2017.json', data_prefix=dict(img='val/')))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = val_evaluator

# 恢复 YOLACT 原生的 55 轮训练，匹配其特有的多阶梯衰减
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=55, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=55, by_epoch=True, milestones=[20, 42, 49, 52], gamma=0.1)
]
optim_wrapper = dict(optimizer=dict(lr=0.001)) # 保持极小学习率
