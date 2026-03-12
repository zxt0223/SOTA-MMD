_base_ = ['../configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'] # 注意3.x这里有横杠

# 修改类别数
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# MMD 3.x 数据集与类别配置规范
data_root = 'A-datas/'
metainfo = dict(classes=('stone',), palette=[(20, 220, 20)])

train_dataloader = dict(
    batch_size=2,
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

# MMD 3.x 验证器配置
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = val_evaluator

# 只跑 1 Epoch 的核心配置
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)
default_hooks = dict(checkpoint=dict(interval=1))
