# analyze_bubble_infos.py
import os
import math
import cv2
import pandas as pd
import numpy as np
from MIB_data_processing1 import build_bubble_infos
from MIB_reading import read_pkl_dict
from MIB_data_processing2 import track_bubbles_as_dataframe_fast
from tqdm import tqdm
import matplotlib.cm as cm

def draw_detected_bounding_boxes(tracked_df, image_folder, output_folder, draw_score=True):
    """
    在每帧图像上绘制识别的矩形框，并保存至输出文件夹。
    :param tracked_df: 含 ['filename', 'x1', 'y1', 'x2', 'y2', 'score', 'bubble_id'] 的DataFrame
    :param image_folder: 原始图像文件夹
    :param output_folder: 绘制结果保存文件夹
    :param draw_score: 是否绘制置信度分数
    """
    os.makedirs(output_folder, exist_ok=True)
    grouped = tracked_df.groupby('filename')

    for fname, group in tqdm(grouped, desc="📦 绘制识别框"):
        image_path = os.path.join(image_folder, fname)
        if not os.path.exists(image_path):
            print(f"⚠️ 跳过: 图像不存在 {image_path}")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ 跳过: 图像读取失败 {image_path}")
            continue

        for _, row in group.iterrows():
            x1, y1, x2, y2 = map(int, [row['x1'], row['y1'], row['x2'], row['y2']])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) #

            label = f"ID:{int(row['bubble_id'])}" if 'bubble_id' in row else ""
            if draw_score and 'score' in row:
                label += f" {row['score']:.2f}"

            # if label:
            #     cv2.putText(img, label.strip(), (x1, y1 - 5),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        save_path = os.path.join(output_folder, fname)
        cv2.imwrite(save_path, img)
def images_to_video(image_folder, output_video_path, fps=30, size=(1024, 1024)):
    """
    将文件夹中所有图像按顺序合成为一个视频（.mp4）。
    图像名应按帧顺序命名，例如 0001.jpg、0002.png 等。

    :param image_folder: 图像文件夹路径
    :param output_video_path: 输出视频路径（.mp4）
    :param fps: 帧率
    :param size: 视频尺寸，默认(1024,1024)
    """
    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png'))]
    )

    if not image_files:
        print(f"⚠️ 未找到图像文件: {image_folder}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, size)

    for fname in tqdm(image_files, desc=f"🎞️ 合成视频: {os.path.basename(output_video_path)}"):
        img_path = os.path.join(image_folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 读取失败：{img_path}")
            continue
        img_resized = cv2.resize(img, size)
        video_writer.write(img_resized)

    video_writer.release()
    print(f"✅ 已保存视频: {output_video_path}")

def analyze_bubble_infos(bubble_infos, fps=30):
    records = []
    frame_area_dict = {}

    for bubble in bubble_infos:
        # —— 新增：读取 growth_rate ——
        gr = bubble.growth_rate
        records.append({
            'bubble_id':      bubble.ID,
            'duration':       bubble.duration,
            'leave_flag':     bubble.leave_flag,
            'radius_mean':    bubble.radius_mean,
            'radius_max':     bubble.radius_max,
            'radius_min':     bubble.radius_min,
            'first_radius':   bubble.first_radius,
            'leave_radius':   bubble.last_radius if bubble.leave_flag else 0,
            'first_center_x': bubble.first_center[0],
            'first_center_y': bubble.first_center[1],

            # —— 新增字段 ——
            'growth_rate':    gr,

        })

        # （下面保持原来的帧面积累加逻辑…）
        if bubble.leave_flag:
            end_frame = bubble.leave_frame
        else:
            end_frame = bubble.last_frame
        for i in range(bubble.first_frame, end_frame + 1):
            idx = i - bubble.first_frame
            if idx < 0 or idx >= len(bubble.radius):
                print(f"⚠️ 跳过无效索引: bubble_id={bubble.ID}, frame={i}, idx={idx}, radius_len={len(bubble.radius)}")
                continue
            area = bubble.radius[i - bubble.first_frame] ** 2 * math.pi
            frame_area_dict[i] = frame_area_dict.get(i, 0) + area

    df = pd.DataFrame(records)
    return df, frame_area_dict

def draw_bubble_centers(bubble_infos, output_path="bubble_centers.png", shape='rect'):
    """
    绘制气泡初始中心点图，支持矩形 / 菱形，绿色表示。

    :param bubble_infos: List[BubbleInfo]
    :param output_path: 输出图像路径
    :param shape: 'rect' or 'diamond'
    """
    canvas = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
    size = 1  # 控制形状大小
    color = (0, 0, 255)  # 绿色 (B, G, R)

    for bubble in bubble_infos:
        if bubble.duration <= 10:
            continue
        cx, cy = map(int, bubble.first_center)

        if shape == 'rect':
            cv2.rectangle(canvas, (cx - size, cy - size), (cx + size, cy + size), color, -1)

        elif shape == 'diamond':
            points = np.array([
                [cx, cy - size],  # top
                [cx + size, cy],  # right
                [cx, cy + size],  # bottom
                [cx - size, cy]   # left
            ])
            cv2.fillPoly(canvas, [points], color)

    cv2.imwrite(output_path, canvas,[cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"✅ 已保存气泡初始中心图像至: {output_path}（形状: {shape}）")



def visualize_per_frame_coverage(tracked_df, bubble_infos, image_folder, output_folder, save_images=True):
    """
    对每一帧图像绘制未脱离气泡：
    - 用绿色圆形填充表示气泡（中心+半径）
    - 红色标注面积
    - 计算总覆盖率并显示在左上角（单位：圆形面积/1024^2）
    - 返回覆盖率 DataFrame
    """
    os.makedirs(output_folder, exist_ok=True)
    tracked_df = tracked_df.sort_values('frame_idx')

    img_groups = tracked_df.groupby('filename')

    bubble_dict = {b.ID: b for b in bubble_infos}

    frame_coverages = []
    for fname, df_group in tqdm(img_groups, desc="🖼️ 绘制帧覆盖图"):
        frame_idx = df_group['frame_idx'].iloc[0]
        image_path = os.path.join(image_folder, fname)
        # 检查分组后的数据是否为空
        if len(img_groups) == 0:
            print("⚠️ 警告：按文件名分组后没有数据，无法生成热力图")
            return
        if save_images:
            if not os.path.exists(image_path):
                continue
            img = cv2.imread(image_path)
            if img is None:
                continue
        else:
            img = None

        total_area = 0

        for _, row in df_group.iterrows():
            bubble_id = row['bubble_id']
            bubble = bubble_dict[bubble_id]

            # 跳过脱离后的气泡
            if bubble.leave_flag and frame_idx > bubble.leave_frame:
                print(f"[Skip-leave] frame={frame_idx}, bubble_id={bubble_id}, leave_frame={bubble.leave_frame}")
                continue

            idx = frame_idx - bubble.first_frame
            if idx < 0 or idx >= len(bubble.center):
                print(
                    f"[Skip-range] frame={frame_idx}, bubble_id={bubble_id}, idx={idx}, center_len={len(bubble.center)}")
                continue

            cx, cy = map(int, bubble.center[idx])
            r = int(bubble.radius[idx])
            area = math.pi * (r ** 2)
            total_area += area

            if save_images:
                cv2.circle(img, (cx, cy), r, (0, 255, 0), -1)
                cv2.putText(img, f"{area:.0f}", (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 可以选择是否打印无气泡的帧信息
            print(f"ℹ️ 信息：帧 {frame_idx} ({fname}) 中没有符合条件的气泡被绘制")
        # 这里再打印一下最终结果
        coverage_ratio = total_area / (1024 * 1024)
        frame_coverages.append({'frame_idx': frame_idx, 'coverage': coverage_ratio})
        # ============ Debug 2: 检查过小覆盖率 ============
        if coverage_ratio < 1e-3:
            print(f"[Debug] Frame={frame_idx}, coverage={coverage_ratio:.8f}, total_area={total_area:.2f}, "
                  "可能出现前几帧覆盖率极小的情况.")
        if save_images:
            cv2.putText(img, f"Coverage: {coverage_ratio:.4f}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            out_path = os.path.join(output_folder, fname)
            cv2.imwrite(out_path, img)

    return pd.DataFrame(frame_coverages)

def visualize_bubble_lifespan_heatmap(tracked_df, bubble_infos, image_folder, output_folder,
                                      min_duration=1, max_duration=200,init_radius=10):
    """
    可视化：依据气泡出现帧数生成热力图颜色填充。
    每帧图像绘制未脱离气泡，颜色深浅表示持续帧数。

    特别排除：在“全局第一帧”中出现，且初始半径 > 10 的气泡，以及总持续时间小于 5 的气泡。

    :param tracked_df: 带 bubble_id 的识别结果 DataFrame
    :param bubble_infos: List[BubbleInfo]
    :param image_folder: 原始图像文件夹
    :param output_folder: 可视化图像输出文件夹
    :param min_duration: 持续帧数最小值（用于颜色映射）
    :param max_duration: 持续帧数最大值（用于颜色映射）
    """
    os.makedirs(output_folder, exist_ok=True)
    colormap = cm.get_cmap('Reds', 100)
    # 检查输入数据是否为空
    if tracked_df.empty:
        print("⚠️ 警告：输入的tracked_df为空，没有可处理的数据")
        return
    print(len(tracked_df))
    if not bubble_infos:
        print("⚠️ 警告：bubble_infos为空，没有气泡信息可处理")
        return
    def map_duration_to_color(current_duration):
        norm_val = (current_duration - min_duration) / (max_duration - min_duration)
        norm_val = np.clip(norm_val, 0, 1)
        rgba = colormap(norm_val)
        bgr = tuple(int(255 * c) for c in rgba[:3][::-1])  # 转为 BGR
        return bgr

    frame_to_bubbles = {b.ID: b for b in bubble_infos}
    tracked_df = tracked_df.sort_values('frame_idx')
    img_groups = tracked_df.groupby('filename')

    # Step 1: 查找全局第一帧的 frame_idx
    min_frame_idx = tracked_df['frame_idx'].min()
    first_frame_df = tracked_df[tracked_df['frame_idx'] == min_frame_idx]

    # Step 2: 跳过那些在第一帧中首次出现且半径 > 阈值 的气泡
    skip_bubble_ids = set()
    for _, row in first_frame_df.iterrows():
        bubble_id = row['bubble_id']
        bubble = frame_to_bubbles[bubble_id]
        if bubble.first_frame == min_frame_idx and len(bubble.radius) > 0 and bubble.radius[0] > init_radius:
            skip_bubble_ids.add(bubble_id)

    # Step 3: 记录每个气泡的总持续帧数
    short_lived_bubble_ids = set()
    for bubble in bubble_infos:
        duration = len(bubble.radius)
        if duration < 10:  # 总持续时间小于 5 帧的气泡
            short_lived_bubble_ids.add(bubble.ID)

    for fname, df_group in tqdm(img_groups, desc="🌡️ 气泡寿命热力图"):
        frame_idx = df_group['frame_idx'].iloc[0]
        image_path = os.path.join(image_folder, fname)
        if not os.path.exists(image_path):
            continue

        img = cv2.imread(image_path)
        if img is None:
            continue

        for _, row in df_group.iterrows():
            bubble_id = row['bubble_id']
            if bubble_id in skip_bubble_ids or bubble_id in short_lived_bubble_ids:
                continue

            bubble = frame_to_bubbles[bubble_id]

            # 跳过已脱离的气泡
            if bubble.leave_flag and frame_idx > bubble.leave_frame:
                continue

            rel_idx = frame_idx - bubble.first_frame
            if rel_idx < 0 or rel_idx >= len(bubble.radius):
                continue
            box = [row['x1'], row['y1'], row['x2'], row['y2']]
            width = box[2] - box[0]
            height = box[3] - box[1]
            long_side = max(width, height)  # 关键修改：取长边作为直径
            radius = int(long_side/2) # 转为半径
            # radius = int(bubble.radius[rel_idx])
            center_x, center_y = map(int, bubble.center[rel_idx])

            # 逻辑1：检查首帧半径大于 10 的气泡，后续不填充颜色
            if bubble.first_frame == frame_idx and radius > init_radius:
                skip_bubble_ids.add(bubble.ID)
                continue

            color = map_duration_to_color(rel_idx + 1)

            # 绘制气泡
            cv2.circle(img, (center_x, center_y), radius, color, -1)

        out_path = os.path.join(output_folder, fname)
        cv2.imwrite(out_path, img)


def export_bubble_frames(bubble_infos):
    """
    导出每个气泡在每帧的半径、脱离半径和平均增长率。
    - 时间从0开始（相对时间，而非绝对帧号）
    - 新增相对时间列 `time_relative`
    - 保留绝对帧号 `frame_absolute`（可选）
    - 新增脱离标志列 `is_detached` (0:未脱离, 1:已脱离)
    - 新增气泡中心位置列 `cx` 和 `cy`
    """
    records = []
    for bubble in tqdm(bubble_infos, desc="📊 导出每帧数据"):
        bubble_id = bubble.ID
        growth_rate = bubble.growth_rate
        leave_radius = bubble.leave_radius if bubble.leave_flag else np.nan

        # 遍历每一帧的半径（相对时间从0开始）
        for idx, radius in enumerate(bubble.radius):
            time_relative = idx  # 相对时间（从0开始）
            frame_absolute = bubble.first_frame + idx  # 绝对帧号（可选）
            # 获取气泡中心位置，使用已有的 self.center 属性
            cx, cy = bubble.center[idx] if idx < len(bubble.center) else (np.nan, np.nan)

            # 判断当前帧是否已经脱离（仅对已脱离的气泡有效）
            is_detached = 0
            if bubble.leave_flag and frame_absolute > bubble.leave_frame:
                is_detached = 1

            records.append({
                'bubble_id': int(bubble_id),
                'time_relative': int(time_relative),  # 关键修改：相对时间
                'frame_absolute': int(frame_absolute),  # 保留绝对帧号（可选）
                'radius': float(radius),
                'leave_radius': float(leave_radius) if bubble.leave_flag else np.nan,
                'growth_rate': float(growth_rate),
                'is_detached': int(is_detached),  # 新增脱离标志
                'cx': float(cx),  # 新增气泡中心 x 坐标
                'cy': float(cy)   # 新增气泡中心 y 坐标
            })
    return pd.DataFrame(records)


if __name__ == "__main__":
    image_folder_path = r"E:\zl\output1\1\scaled_images\5000_1"
    out_path = image_folder_path + "_out"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f"✅ 创建出文件夹: {out_path}")
    # Step 1: 读取识别结果并追踪气泡
    result_df = read_pkl_dict(r"E:\zl\output1\1\scaled_images\5000_1\out\all_results1.pkl")
    # iou阈值设置（0：0.3，10 ），（10：0.2，20），（30：0.2，20），（45：0.3，30）催化剂与0一致
    tracked_df= track_bubbles_as_dataframe_fast(result_df,iou_thresh=0.3,max_backtrack=10,radius_thresh=20)
    # Step 2: 构建 BubbleInfo 实例集合
    bubble_infos = build_bubble_infos(tracked_df)

    # Step 3: 分析统计 + 面积信息
    analysis_df, frame_area_dict = analyze_bubble_infos(bubble_infos)
    merge_flags = tracked_df[['bubble_id', 'merged_flag']].drop_duplicates(subset='bubble_id')

    # 将 merge 标记合并到 analysis_df
    analysis_df = analysis_df.merge(merge_flags, on='bubble_id', how='left')

    # 替换空值为 0（有些轨迹可能没有被标记）
    analysis_df['merged_flag'] = analysis_df['merged_flag'].fillna(0).astype(int)

    print("融合气泡数量:", analysis_df[analysis_df['merged_flag'] == 1].shape[0])
    # 1. 先获取所有气泡的总数（未过滤前的完整数据）
    total_all_bubbles = len(analysis_df)

    # 2. 统计leave_flag == 0的气泡数量和占比
    leave_0_count = len(analysis_df[analysis_df['leave_flag'] == 0])
    leave_0_ratio = (leave_0_count / total_all_bubbles) * 100 if total_all_bubbles > 0 else 0

    # 3. 统计leave_flag == 1的气泡数量和占比（用于对比）
    leave_1_count = len(analysis_df[analysis_df['leave_flag'] == 1])
    leave_1_ratio = (leave_1_count / total_all_bubbles) * 100 if total_all_bubbles > 0 else 0

    # 4. 构建统计表格
    stats_data = {
        '气泡状态': ['未离开(leave_flag=0)', '已离开(leave_flag=1)', '总计'],
        '气泡数量': [leave_0_count, leave_1_count, total_all_bubbles],
        '占比(%)': [round(leave_0_ratio, 2), round(leave_1_ratio, 2), 100.0]
    }
    stats_df = pd.DataFrame(stats_data)
    # ======================== 新增核心：融合标记气泡全维度占比统计 ========================
    print("【新增统计-融合标记气泡占比 完整版】")
    # ✅ 统计1：所有气泡中 有融合标记的气泡数量+占比
    merged_1_total = len(analysis_df[analysis_df['merged_flag'] == 1])
    merged_1_ratio_total = (merged_1_total / total_all_bubbles) * 100 if total_all_bubbles > 0 else 0

    # ✅ 统计2：未离开气泡(leave_flag=0)中 有融合标记的数量+占比
    leave0_merged1 = analysis_df[(analysis_df['leave_flag'] == 0) & (analysis_df['merged_flag'] == 1)]
    leave0_merged1_count = len(leave0_merged1)
    leave0_merged1_ratio = (leave0_merged1_count / leave_0_count) * 100 if leave_0_count > 0 else 0

    # ✅ 统计3：已离开气泡(leave_flag=1)中 有融合标记的数量+占比
    leave1_merged1 = analysis_df[(analysis_df['leave_flag'] == 1) & (analysis_df['merged_flag'] == 1)]
    leave1_merged1_count = len(leave1_merged1)
    leave1_merged1_ratio = (leave1_merged1_count / leave_1_count) * 100 if leave_1_count > 0 else 0

    # ✅ 统计4：无融合标记的气泡数量+占比（补充对比）
    merged_0_total = total_all_bubbles - merged_1_total
    merged_0_ratio_total = 100 - merged_1_ratio_total

    # 打印独立的融合统计信息
    print(f"📊 所有气泡总数: {total_all_bubbles}")
    print(f"🔵 有融合标记气泡(merged_flag=1): {merged_1_total} 个 | 占比: {merged_1_ratio_total:.2f} %")
    print(f"⚪ 无融合标记气泡(merged_flag=0): {merged_0_total} 个 | 占比: {merged_0_ratio_total:.2f} %")
    print(f"🔸 未离开气泡中融合占比: {leave0_merged1_count}/{leave_0_count} | {leave0_merged1_ratio:.2f} %")
    print(f"🔹 已离开气泡中融合占比: {leave1_merged1_count}/{leave_1_count} | {leave1_merged1_ratio:.2f} %")

    # ✅ 构建【融合标记专项统计表格】，更规范直观
    merge_stats_data = {
        '气泡分类': ['所有气泡-有融合标记', '所有气泡-无融合标记',
                     '未离开气泡-有融合标记', '已离开气泡-有融合标记'],
        '气泡数量': [merged_1_total, merged_0_total, leave0_merged1_count, leave1_merged1_count],
        '对应群体占比(%)': [round(merged_1_ratio_total, 2), round(merged_0_ratio_total, 2),
                            round(leave0_merged1_ratio, 2), round(leave1_merged1_ratio, 2)]
    }
    merge_stats_df = pd.DataFrame(merge_stats_data)
    print("\n【融合标记气泡专项统计表格】")
    print(merge_stats_df)
    print("=" * 60)
    merge_stats_df.to_csv(out_path + "/merge_stats_df.csv", index=False)
    # 5. 打印表格并保存到文件
    print("\n===== 气泡离开状态统计 =====")
    print(stats_df)
    stats_df.to_csv(os.path.join(out_path, "leave_flag_stats.csv"), index=False)
    print(f"\n✅ 统计表格已保存到: {os.path.join(out_path, 'leave_flag_stats.csv')}")
    # ✅ 只保留符合条件的气泡
    # analysis_df = analysis_df[
    #     (analysis_df['leave_flag'] == 1) &
    #     (analysis_df['duration'] >= 1)
    #     ]
    analysis_df = analysis_df[(analysis_df['duration']>=0)]
    #125FPS为2，250FPS为4，500FPS为8
    analysis_df.to_csv(out_path + "/bubble_analysis.csv", index=False)
    print(analysis_df.head())

    # 生成每帧数据
    frame_df = export_bubble_frames(bubble_infos)
    frame_df.to_csv(out_path + "/bubble_frames_details.csv", index=False)
    # # 统计气泡数量
    total_bubbles = len(analysis_df)
    print(total_bubbles)
    # Step 4: 绘制初始中心点
    # draw_bubble_centers(bubble_infos, output_path=out_path + "/bubble_centers.png", shape='diamond')

    # Step 5: 绘制每帧覆盖图
    # coverage_df = visualize_per_frame_coverage(tracked_df, bubble_infos,
    #                                            image_folder=image_folder_path,
    #                                            output_folder=os.path.join(out_path, "coverage"),
    #                                            save_images=False)

    # ✅ 输出帧覆盖率表格
    # coverage_df.to_csv(os.path.join(out_path, "duration_analysis.csv"), index=False)
    # print("✅ 已保存每帧覆盖率分析表 duration_analysis.csv")
    # # Step 6: 绘制气泡寿命热力图
    # visualize_bubble_lifespan_heatmap(tracked_df, bubble_infos,
    #                                   image_folder=image_folder_path,
    #                                     output_folder= out_path+ "/heatmap",
    #                                     min_duration=1, max_duration=200)
