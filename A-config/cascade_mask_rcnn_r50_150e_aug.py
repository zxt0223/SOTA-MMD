_base_ = [
    '../configs/_base_/models/cascade-mask-rcnn_r50_fpn.py',
    '../configs/_base_/datasets/coco_instance.py',
    '../configs/_base_/schedules/schedule_1x.py',
    '../configs/_base_/default_runtime.py'
]

# 1. 模型结构调整为 1 类
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(type='Shared2FCBBoxHead', num_classes=1),
            dict(type='Shared2FCBBoxHead', num_classes=1),
            dict(type='Shared2FCBBoxHead', num_classes=1)
        ],
        mask_head=dict(type='FCNMaskHead', num_classes=1)
    )
)

dataset_type = 'CocoDataset'
data_root = 'A-datas/'

metainfo = dict(
    classes=('stone', ),
    palette=[(0, 255, 0)]
)

# 2. 🚀 核心：强力数据增强 Pipeline
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # 多尺度训练，短边在 400 到 960 之间随机，长边最大 1333
    dict(
        type='RandomResize',
        scale=[(1333, 400), (1333, 960)],
        keep_ratio=True),
    # 随机水平翻转 (概率50%)
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # 随机垂直翻转 (概率50%，石头特供)
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    # 光度失真，抗阴影和条纹干扰
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train2017.json', 
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline # 挂载数据增强
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/')
    )
)

test_dataloader = val_dataloader
val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/instances_val2017.json', metric=['bbox', 'segm'])
test_evaluator = val_evaluator

# 3. 学习率与 150 轮训练策略
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
)

max_epochs = 150
# 将验证间隔设为 5，即每 5 个 epoch 评估一次，节省训练时间
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        # 在第 110 轮和 140 轮时，学习率下降 10 倍进行精细收敛
        milestones=[110, 140],
        gamma=0.1)
]

work_dir = './A-output/cascade_mask_rcnn_r50_150e_aug'
