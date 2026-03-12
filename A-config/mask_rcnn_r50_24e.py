_base_ = ['../configs/mask_rcnn/mask-rcnn_r50_fpn_2x_coco.py']
model = dict(roi_head=dict(bbox_head=dict(num_classes=1), mask_head=dict(num_classes=1)))

data_root = 'A-datas/'
metainfo = dict(classes=('stone',), palette=[(20, 220, 20)])
train_dataloader = dict(batch_size=2, dataset=dict(data_root=data_root, metainfo=metainfo, ann_file='annotations/instances_train2017.json', data_prefix=dict(img='train2017/')))
val_dataloader = dict(batch_size=2, dataset=dict(data_root=data_root, metainfo=metainfo, ann_file='annotations/instances_val2017.json', data_prefix=dict(img='val2017/')))
test_dataloader = val_dataloader

# 开启 Tensorboard 并记录最详细的 AP50-95
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json', metric=['bbox', 'segm'])
test_evaluator = val_evaluator
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])

# 针对 600 张图片优化的长周期训练策略 (100轮)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=5)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500), # Warmup
    dict(type='MultiStepLR', begin=0, end=100, by_epoch=True, milestones=[70, 90], gamma=0.1)
]
optim_wrapper = dict(optimizer=dict(lr=0.01)) # 4卡*2图=8BS，基准LR减半
default_hooks = dict(checkpoint=dict(interval=10, max_keep_ckpts=3)) # 每10轮保存，最多留3个防爆盘
