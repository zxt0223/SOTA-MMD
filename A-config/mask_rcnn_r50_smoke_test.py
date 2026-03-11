import os
# [关键] 禁用 P2P，防止在特定拓扑(如SYS连接)下NCCL死锁
os.environ["NCCL_P2P_DISABLE"] = "1"

# 1. 继承官方标准配置 (注意相对路径跳回 SOTA_MMD 根目录下的 configs)
_base_ = [
    '../configs/_base_/models/mask-rcnn_r50_fpn.py',
    '../configs/_base_/datasets/coco_instance.py',
    '../configs/_base_/schedules/schedule_1x.py',
    '../configs/_base_/default_runtime.py'
]

# 2. 定义数据集元信息
metainfo = dict(classes=('stone', ), palette=[(220, 20, 60)])

# 3. 覆盖模型输出类别数 (80 -> 1)
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)
    )
)

# 4. 覆盖数据集路径 (使用我们的软链接目录)
data_root = 'A-datas/'

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train/')
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val/')
    )
)
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = val_evaluator

# 5. [测试专用设置] 只跑 1 个 Epoch，学习率设为 2张卡的基准
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)
optim_wrapper = dict(optimizer=dict(lr=0.005))