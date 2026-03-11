import os
os.environ["NCCL_P2P_DISABLE"] = "1"

# [修复点] 替换为 3.x 版本的真实文件名
_base_ = [
    '../configs/yolact/yolact_r50_1xb8-55e_coco.py'
]

metainfo = dict(classes=('stone', ), palette=[(220, 20, 60)])

# YOLACT 的类别数控制在 bbox_head 中
model = dict(
    bbox_head=dict(num_classes=1)
)

data_root = 'A-datas/'

train_dataloader = dict(dataset=dict(data_root=data_root, metainfo=metainfo, ann_file='annotations/instances_train2017.json', data_prefix=dict(img='train/')))
val_dataloader = dict(dataset=dict(data_root=data_root, metainfo=metainfo, ann_file='annotations/instances_val2017.json', data_prefix=dict(img='val/')))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = val_evaluator

# [冒烟测试专属] 仅训练 1 个 epoch
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)
optim_wrapper = dict(optimizer=dict(lr=0.001))