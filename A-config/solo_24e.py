_base_ = ['../configs/solo/solo_r50_fpn_3x_coco.py']
model = dict(mask_head=dict(num_classes=1))

data_root = 'A-datas/'
metainfo = dict(classes=('stone',), palette=[(20, 220, 20)])
train_dataloader = dict(batch_size=2, dataset=dict(data_root=data_root, metainfo=metainfo, ann_file='annotations/instances_train2017.json', data_prefix=dict(img='train2017/')))
val_dataloader = dict(batch_size=2, dataset=dict(data_root=data_root, metainfo=metainfo, ann_file='annotations/instances_val2017.json', data_prefix=dict(img='val2017/')))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json', metric=['segm'])
test_evaluator = val_evaluator
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=120, val_interval=5)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=120, by_epoch=True, milestones=[80, 110], gamma=0.1)
]
optim_wrapper = dict(optimizer=dict(lr=0.005)) # SOLO基准学习率略低
default_hooks = dict(checkpoint=dict(interval=10, max_keep_ckpts=3))
