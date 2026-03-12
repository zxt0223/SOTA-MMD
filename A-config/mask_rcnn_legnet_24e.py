_base_ = ['./mask_rcnn_r50_24e.py']

custom_imports = dict(imports=['A_models.legnet_3x'], allow_failed_imports=False)

model = dict(
    backbone=dict(
        _delete_=True,
        type='LWEGNet',
        in_chans=3,
        stem_dim=32,
        depths=(1, 4, 4, 2),
        drop_path_rate=0.1,
        fork_feat=True,
        # 依然加载预训练大还丹（新加的升维模块会自动从头训练，不受影响）
        init_cfg=dict(type='Pretrained', checkpoint='/group/chenjinming/LEG的权重文件/LWEGNet_tiny.pth')
    ),
    neck=dict(
        # 🚀 关键：对接升维后的新通道！
        in_channels=[64, 128, 256, 512] 
    )
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.05),
    clip_grad=dict(max_norm=35.0, norm_type=2) 
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        interval=10, 
        max_keep_ckpts=10, 
        save_best='auto'
    )
)

model_wrapper_cfg = dict(type='MMDistributedDataParallel', find_unused_parameters=True)
