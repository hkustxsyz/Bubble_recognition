from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm
import os
import mmcv
import time

def run_inference_on_folder(
    model,
    image_folder,
    output_folder,
    save_vis=True,
    score_thr=0.1
):
    os.makedirs(output_folder, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')])

    with open(os.path.join(output_folder, 'image_names.txt'), 'w') as f:
        for img_name in image_files:
            f.write(img_name + '\n')

    all_results = {}
    start_time = time.time()

    for img_name in tqdm(image_files, desc=f"[{os.path.basename(image_folder)}] 推理中", ncols=80):
        img_path = os.path.join(image_folder, img_name)
        try:
            result = inference_detector(model, img_path)

            if save_vis:
                score_str = f"{score_thr:.1f}".replace('.', '_')
                visual_name = f"result_thr{score_str}_{img_name}"
                visual_path = os.path.join(output_folder, visual_name)
                model.show_result(img_path, result, out_file=visual_path, score_thr=score_thr)

            all_results[img_name] = result

        except Exception as e:
            print(f"❌ 处理失败：{img_name}，错误：{e}")
            continue

    result_pkl_path = os.path.join(output_folder, "all_results1.pkl")
    mmcv.dump(all_results, result_pkl_path)

    elapsed = time.time() - start_time
    print(f"\n✅ {image_folder} 推理完成，结果保存在：{result_pkl_path}")
    print(f"⏱️ 用时：{elapsed:.2f} 秒\n")

def run_batch_inference(
    config_file,
    checkpoint_file,
    root_folder,
    save_vis=True,
    device='cuda:0',
    score_thr=0.1
):
    print(f"📂 开始批量推理，总目录：{root_folder}")

    # 初始化模型
    model = init_detector(config_file, checkpoint_file, device=device)

    # 遍历子文件夹
    subdirs = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]

    for subdir in subdirs:
        input_path = os.path.join(root_folder, subdir)
        output_path = os.path.join(input_path, "out")

        run_inference_on_folder(
            model,
            image_folder=input_path,
            output_folder=output_path,
            save_vis=save_vis,
            score_thr=score_thr
        )

    print("✅ 所有子文件夹处理完成！")

# 示例调用
if __name__ == "__main__":
    config_path = r"my_deformable_detr_r50_16x2_50e_coco_top.py"
    checkpoint_path = r"E:\bubble_top\model\pic_15_wulixueyuan/latest.pth"
    root_input_folder = r"E:\bubble_top\top_bright_result\graphene with NiVRu1.0/"  #a ✅ 总文件夹路径，每个子目录为一个图片序列
    save_visualization =False
    score_threshold = 0.15
    run_batch_inference(
        config_file=config_path,
        checkpoint_file=checkpoint_path,
        root_folder=root_input_folder,
        save_vis=save_visualization,
        score_thr=score_threshold
    )
