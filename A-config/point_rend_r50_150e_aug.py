_base_ = [
    '../configs/point_rend/point-rend_r50-caffe_fpn_ms-1x_coco.py'
]

# 1. 核心模型调整 (类别改为 1，并在底层直接解除 100 个数量封印)
model = dict(
    roi_head=dict(
        # PointRend 特有的双网络结构：粗糙掩码头 + 高精度边缘点渲染头
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1),
        point_head=dict(num_classes=1)
    ),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.15,      # 过滤极低分杂音
            max_per_img=500      # 彻底解除数量封印，允许检出最多 500 块石头
        )
    )
)

dataset_type = 'CocoDataset'
data_root = 'A-datas/'

metainfo = dict(
    classes=('stone', ),
    palette=[(0, 255, 0)]
)

# 2. 复用极其成功的“石头特供”数据增强 Pipeline
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomResize',
        scale=[(1333, 400), (1333, 960)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),  # 垂直翻转
    dict(type='PhotoMetricDistortion'),                       # 抗光影/条纹失真
    dict(type='PackDetInputs')
]

# 挂载数据集
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train2017.json', 
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline
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

# 3. 学习率与 150 轮持久战策略
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
)

max_epochs = 150
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[110, 140],
        gamma=0.1)
]

work_dir = './A-output/point_rend_r50_150e_aug'
