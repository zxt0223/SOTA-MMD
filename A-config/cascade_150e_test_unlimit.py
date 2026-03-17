_base_ = ['./cascade_mask_rcnn_r50_150e_aug.py']

# 仅修改测试（推理）时的参数，不影响原模型结构
model = dict(
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.01,       # 【关键】稍微降低网络底层的置信度门槛 (原默认0.05)
            max_per_img=500       # 【关键】暴力解除 100 个的数量封印，最大允许输出 500 个框
        )
    )
)
