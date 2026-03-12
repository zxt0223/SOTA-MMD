_base_ = ['./mask_rcnn_r50_24e.py']
model = dict(
    backbone=dict(depth=18, init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512]) 
)
