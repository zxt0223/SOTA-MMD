# 继承我们刚刚完美跑通的解封+升维版本
_base_ = ['./mask_rcnn_legnet_24e.py']

# 1. 训练时间拉长到 150 轮 (给模型足够的时间去适应复杂的增强数据)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=150, val_interval=10)

# 2. 匹配 150 轮的学习率衰减策略 (在第 110 和 140 轮降速，让模型精细收敛)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=150,
        by_epoch=True,
        milestones=[110, 140],
        gamma=0.1)
]

# 3. 石头专属极限数据增强 Pipeline
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # 多尺度训练：随机缩放，应对大小不一的石头
    dict(
        type='RandomResize',
        scale=[(1333, 640), (1333, 800)],
        keep_ratio=True),
    # 随机水平翻转
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # 🚀 随机垂直翻转：石头特供增强，打破物理方向限制！
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    # 🚀 光度失真：随机改变亮度、对比度、色相，抗光照干扰！
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs')
]

# 将新的 Pipeline 覆盖到训练数据加载器中
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
