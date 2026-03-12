import os
import glob
from mmdet.apis import DetInferencer

def main():
    # 定义五个模型的配置和输出路径
    models = [
        ("Mask_RCNN_R50", "A-config/mask_rcnn_r50_24e.py", "A-output/mask_rcnn_r50_24e"),
        ("Mask_RCNN_R18", "A-config/mask_rcnn_r18_24e.py", "A-output/mask_rcnn_r18_24e"),
        ("SOLO", "A-config/solo_24e.py", "A-output/solo_24e"),
        ("YOLACT++", "A-config/yolact_24e.py", "A-output/yolact_24e"),
        ("LEGNet_Ours", "A-config/mask_rcnn_legnet_24e.py", "A-output/mask_rcnn_legnet_24e")
    ]

    # 获取验证集里的前 3 张图片（你可以改成你特定的测试图片名字）
    img_dir = "A-datas/val2017"
    if not os.path.exists(img_dir):
        print(f"找不到图片目录: {img_dir}")
        return
        
    all_imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
    test_imgs = all_imgs[:3] # 取前3张，如果想测更多可以把 3 改成 10

    # 统一的高清可视化输出目录
    vis_out_dir = "A-output/Vis_Results"
    os.makedirs(vis_out_dir, exist_ok=True)

    for model_name, config_path, work_dir in models:
        print(f"\n=============================================")
        print(f"🚀 正在处理模型: {model_name}")
        
        # 1. 自动寻找最佳权重 (优先找 best_mAP，找不到就找最新的 epoch)
        ckpts = glob.glob(os.path.join(work_dir, "best_*.pth"))
        if not ckpts:
            ckpts = glob.glob(os.path.join(work_dir, "epoch_*.pth"))
            
        if not ckpts:
            print(f"⚠️ 警告: 在 {work_dir} 中没有找到 .pth 权重文件，跳过...")
            continue
            
        # 按修改时间排序，取最后一个（最新的/最好的）
        ckpts.sort(key=os.path.getmtime)
        best_ckpt = ckpts[-1]
        print(f"📦 找到权重: {os.path.basename(best_ckpt)}")

        # 2. 初始化推理器
        try:
            inferencer = DetInferencer(model=config_path, weights=best_ckpt)
        except Exception as e:
            print(f"❌ 初始化 {model_name} 失败: {e}")
            continue

        # 3. 批量推理并保存画好 Mask 的图片
        model_out_dir = os.path.join(vis_out_dir, model_name)
        inferencer(test_imgs, out_dir=model_out_dir, no_save_pred=True)
        print(f"✅ {model_name} 推理完成！结果已保存至: {model_out_dir}/vis")

    print(f"\n🎉 所有模型的预测对比图已全部生成在: {vis_out_dir}")

if __name__ == "__main__":
    main()
