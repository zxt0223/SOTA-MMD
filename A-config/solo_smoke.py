import os
os.environ["NCCL_P2P_DISABLE"] = "1"

# 继承 SOLO 的官方配置
_base_ = [
    '../configs/solo/solo_r50_fpn_1x_coco.py'
]

metainfo = dict(classes=('stone', ), palette=[(220, 20, 60)])

# 单阶段 SOLO 只有 mask_head
model = dict(
    mask_head=dict(num_classes=1)
)

data_root = 'A-datas/'

train_dataloader = dict(dataset=dict(data_root=data_root, metainfo=metainfo, ann_file='annotations/instances_train2017.json', data_prefix=dict(img='train/')))
val_dataloader = dict(dataset=dict(data_root=data_root, metainfo=metainfo, ann_file='annotations/instances_val2017.json', data_prefix=dict(img='val/')))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = val_evaluator

# [冒烟测试专属] 仅训练 1 个 epoch
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)
# SOLO 默认 BS 较大，学习率调整为适合双卡的 0.005
optim_wrapper = dict(optimizer=dict(lr=0.005))