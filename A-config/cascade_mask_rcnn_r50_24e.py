_base_ = [
    '../configs/_base_/models/cascade-mask-rcnn_r50_fpn.py',
    '../configs/_base_/datasets/coco_instance.py',
    '../configs/_base_/schedules/schedule_1x.py',
    '../configs/_base_/default_runtime.py'
]

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

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        # 【修改点 1】指向 annotations 文件夹里的具体 json 文件名
        ann_file='annotations/instances_train2017.json', 
        # 【修改点 2】指向实际的图片文件夹
        data_prefix=dict(img='train2017/')
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        # 【修改点 3】验证集 json
        ann_file='annotations/instances_val2017.json',
        # 【修改点 4】验证集图片文件夹
        data_prefix=dict(img='val2017/')
    )
)

test_dataloader = val_dataloader
# 【修改点 5】评估器也要用正确的 val json
val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/instances_val2017.json', metric=['bbox', 'segm'])
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
)

max_epochs = 24
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

work_dir = './A-output/cascade_mask_rcnn_r50_24e'
