_base_ = ['../configs/yolact/yolact_r50_1xb8-55e_coco.py'] # 修正了3.x官方的文件名
model = dict(
    bbox_head=dict(num_classes=1),
    mask_head=dict(num_classes=1))

data_root = 'A-datas/'
metainfo = dict(classes=('stone',), palette=[(20, 220, 20)])

train_dataloader = dict(
    batch_size=4, 
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/')))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/')))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)
default_hooks = dict(checkpoint=dict(interval=1))
