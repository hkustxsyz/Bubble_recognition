import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from io_utils import read_pkl_dict  # 假设你已封装了读取函数：返回 DataFrame
import re
import os
import cv2
def get_box_center(box: dict):
    """
    根据 box = {'x1':..., 'y1':..., 'x2':..., 'y2':...} 返回中心点 (cx, cy)
    """
    cx = 0.5 * (box['x1'] + box['x2'])
    cy = 0.5 * (box['y1'] + box['y2'])
    return cx, cy
def get_box_radius(box: dict) -> float:
    """
    根据 (x1,y1,x2,y2) 返回内接圆的半径，即 min(width, height)/2
    """
    w = box['x2'] - box['x1']
    h = box['y2'] - box['y1']
    r = min(w, h) / 2.0
    # 避免出现负值
    return max(r, 0.0)
def get_box_center_width_height(box):
    """
    给定 {x1,y1,x2,y2}，返回 (cx, cy, w, h)
    """
    x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
    cx = 0.5*(x1 + x2)
    cy = 0.5*(y1 + y2)
    w = x2 - x1
    h = y2 - y1
    return cx, cy, w, h

def build_box_from_center(cx, cy, w, h):
    """
    给定中心(cx,cy) 和宽高(w,h)，返回 dict(x1=..., y1=..., x2=..., y2=...)
    """
    x1 = cx - w/2
    x2 = cx + w/2
    y1 = cy - h/2
    y2 = cy + h/2
    return dict(x1=x1, y1=y1, x2=x2, y2=y2)
def compute_iou_batch(box, boxes):
    """
    向量化计算一个 box 与多个 boxes 之间的 IOU
    :param box: list[x1, y1, x2, y2]
    :param boxes: np.ndarray, shape (N, 4)
    :return: np.ndarray, shape (N,) 的 IOU 数组
    """
    # 🔥 新增：处理空数组情况
    if boxes.size == 0:
        return np.array([])
    x1, y1, x2, y2 = box
    box_area = (x2 - x1) * (y2 - y1)

    # 计算交集
    xx1 = np.maximum(x1, boxes[:, 0])
    yy1 = np.maximum(y1, boxes[:, 1])
    xx2 = np.minimum(x2, boxes[:, 2])
    yy2 = np.minimum(y2, boxes[:, 3])

    inter_w = np.maximum(0, xx2 - xx1)
    inter_h = np.maximum(0, yy2 - yy1)
    inter_area = inter_w * inter_h

    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area

    iou = inter_area / (union_area + 1e-6)
    return iou


def is_same_box(box1, box2):
    return (
        box1['x1'] == box2['x1'] and
        box1['y1'] == box2['y1'] and
        box1['x2'] == box2['x2'] and
        box1['y2'] == box2['y2']
    )
def merge_disconnected_tracks(result_df, max_frame_gap=5, iou_thresh_merge=0.4, center_dist_thresh=40):
    # 按轨迹ID分组并过滤短轨迹
    id_to_group = result_df.groupby('bubble_id')
    id_start_end = {
        bid: {
            'start_frame': group['frame_idx'].iloc[0],
            'end_frame': group['frame_idx'].iloc[-1],
            'start_box': group.iloc[0],
            'end_box': group.iloc[-1],
            'length': len(group)  # 记录轨迹长度
        }
        for bid, group in id_to_group if len(group) >= 1  # 过滤短轨迹
    }
    merge_map = {}  # id_b → id_a
    protected_ids = {}
    # 新增：记录每个目标轨迹id_a对应的最优源轨迹信息（用于冲突判断）
    best_match_for_target = defaultdict(lambda: {'max_iou': 0, 'source_id': None})
    for id_a, info_a in id_start_end.items():
        for id_b, info_b in id_start_end.items():
            if id_a == id_b or id_b in protected_ids or id_b in merge_map:
                continue
            frame_gap = info_b['start_frame'] - info_a['end_frame']
            if 1 <= frame_gap <= max_frame_gap:
                box_a = info_a['end_box']
                box_b = info_b['start_box']
                cx1, cy1 = get_box_center(box_a)
                cx2, cy2 = get_box_center(box_b)
                dist = np.hypot(cx1 - cx2, cy1 - cy2)

                iou_val = compute_iou_batch(
                    [box_a['x1'], box_a['y1'], box_a['x2'], box_a['y2']],
                    np.array([[box_b['x1'], box_b['y1'], box_b['x2'], box_b['y2']]])
                )[0]
                # 多对一冲突处理核心逻辑
                if dist < center_dist_thresh and iou_val > iou_thresh_merge:
                    # 1. 检查当前id_a是否已有源轨迹匹配
                    current_best = best_match_for_target[id_a]
                    if current_best['source_id'] is None:
                        # 无匹配：直接记录当前id_b为id_a的最优源轨迹
                        current_best['max_iou'] = iou_val
                        current_best['source_id'] = id_b
                        merge_map[id_b] = id_a
                        # print(f"[轨迹合并] {id_b} → {id_a} | Δframe={frame_gap}, dist={dist:.1f}, IoU={iou_val:.2f}")
                    else:
                        # 有匹配：比较当前IoU与已有最优IoU，保留更大的
                        if iou_val > current_best['max_iou']:
                            # 移除原有次优源轨迹的映射
                            old_source_id = current_best['source_id']
                            if old_source_id in merge_map:
                                del merge_map[old_source_id]
                            # 更新为当前更优的匹配
                            current_best['max_iou'] = iou_val
                            current_best['source_id'] = id_b
                            merge_map[id_b] = id_a
                            # print(f"[合并冲突修复] {old_source_id}→{id_a}（IoU={current_best['max_iou']:.2f}）替换为 {id_b}→{id_a}（IoU={iou_val:.2f}）")

                # if dist < center_dist_thresh and iou_val > iou_thresh_merge:
                #     # print(f"[轨迹合并] {id_b} → {id_a} | Δframe={frame_gap}, dist={dist:.1f}, IoU={iou_val:.2f}")
                #     merge_map[id_b] = id_a

    # 避免循环合并
    def get_merged_id(bid):
        while bid in merge_map:
            bid = merge_map[bid]
        return bid

    result_df['bubble_id'] = result_df['bubble_id'].apply(get_merged_id)

    merged_ids = set(merge_map.keys())
    return result_df, merged_ids
def ios_ratio(box1: dict, box2: dict) -> float:
    """计算两个框的交集面积与较小框面积之比"""
    inter_x1 = max(box1['x1'], box2['x1'])
    inter_y1 = max(box1['y1'], box2['y1'])
    inter_x2 = min(box1['x2'], box2['x2'])
    inter_y2 = min(box1['y2'], box2['y2'])

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    small_area = min(area1, area2)

    if small_area <= 0:
        return 0.0
    return inter_area / small_area


def optimize_dataframe_memory(df):
    """优化DataFrame内存使用（安全版本）"""
    df = df.copy()  # 确保不修改原始数据

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    for col in df.columns:
        if df[col].dtype in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(df[col].dtype)[:3] == 'int':
                # 检查范围是否在int16内
                if (c_min >= np.iinfo(np.int16).min and
                        c_max <= np.iinfo(np.int16).max):
                    df[col] = df[col].astype(np.int16)
                # 检查范围是否在int32内
                elif (c_min >= np.iinfo(np.int32).min and
                      c_max <= np.iinfo(np.int32).max):
                    df[col] = df[col].astype(np.int32)

            else:  # float类型
                # 检查是否需要保持高精度
                if (abs(c_min) <= 1e6 and abs(c_max) <= 1e6 and
                        np.issubdtype(df[col].dtype, np.floating)):
                    df[col] = df[col].astype(np.float32)

    return df

def track_bubbles_as_dataframe_fast(
    result_df: pd.DataFrame,
    iou_thresh=0.2,
    iou_thresh_prev2=0.2,
    max_backtrack=1,
    radius_thresh = 60,
) -> pd.DataFrame:
    """
    以逐框处理的简化逻辑，支持：单轨迹匹配、前一帧多轨迹匹配(融合: 选最近中心)、拆分(spilt) 以及多帧回溯(只单轨迹)。
    :param result_df: 输入 DataFrame，包含 ['frame_idx', 'x1', 'y1', 'x2', 'y2', 'filename']
    :param iou_thresh: 单轨迹 / 多轨迹匹配的 IoU 阈值
    :param max_backtrack: 最多向前回溯几帧（只做单轨迹匹配）
    :return: 带 'bubble_id' 的 DataFrame
    """
    # 初始化多轨迹融合计数器
    merged_track_ids = set()
    result_df = optimize_dataframe_memory(result_df)
    # 新增：融合事件记录器
    fusion_events = []

    def extract_frame_number(filename):
        """从 '1000mA_C001H001S0001021838' 中提取最后的数字作为帧号"""
        import re
        # 获取下划线后的部分
        if '_' in filename:
            suffix = filename.split('_')[-1]
        else:
            suffix = filename

        # 提取suffix中的所有数字
        numbers = re.findall(r'\d+', suffix)
        if numbers:
            # 取最后一个数字段作为帧号（应该是 1021838）
            return int(numbers[-1])
        return 0


    # --------- 新增：按文件名数字排序 ---------
    # # 提取文件名中的数字
    # result_df['frame_number'] = result_df['filename'].apply(
    #     lambda x: int(re.search(r'(\d+)', x).group(1))
    # )
    # # 按数字排序
    # result_df = result_df.sort_values(by='frame_number').reset_index(drop=True)
    # # 重新生成 frame_idx 从 1 开始
    # result_df['frame_idx'] = result_df.groupby('frame_number').ngroup() + 1
    # result_df.drop(columns=['frame_number'], inplace=True)
    # # 1) 按帧排序
    # result_df = result_df.sort_values(by='frame_idx').reset_index(drop=True)
    # 提取文件名中的数字（从最后一个下划线之后的部分）
    # result_df['frame_number'] = result_df['filename'].apply(
    #     lambda x: int(x.split('_')[-1].replace('.jpg', ''))
    # )
    # 提取帧号
    result_df['frame_number'] = result_df['filename'].apply(extract_frame_number)
    # 按数字排序
    result_df = result_df.sort_values(by='frame_number').reset_index(drop=True)

    # 重新生成 frame_idx 从 1 开始
    #result_df['frame_idx'] = result_df.groupby('frame_number').ngroup() + 1
    result_df['frame_idx'] = result_df['frame_number']
    result_df.drop(columns=['frame_number'], inplace=True)
    # 2) 预备数据结构
    bubble_ids = [-1] * len(result_df)         # 存放最终的 bubble_id
    bubble_tracks = []                         # 每个 track_id 对应的一组 box
    track_ends_by_frame = defaultdict(list)    # {frame_idx: [(track_id, last_box)]}
    # 方便边遍历边赋值
    index_map = {idx: row for idx, row in result_df.iterrows()}
    track_latest_id = -1                       # 轨迹 ID 递增计数
    # 3) 遍历每个检测框(按帧顺序)
    for idx, cur_box in tqdm(index_map.items(), desc="🧬 Tracking", total=len(result_df)):

        frame_idx = cur_box['frame_idx']
        matched_track_ids = []
        current_max_backtrack = max_backtrack
        # ---------- (A) 首先只看“前一帧” ----------
        # 优先匹配上一帧
        # ① 获取候选轨迹并记录是来自哪一帧
        candidates = []  # 空列表初始化
        frame_used = None

        # 添加上一帧的候选轨迹（来源帧为 frame_idx-1）
        prev1_candidates = track_ends_by_frame.get(frame_idx - 1, [])
        for tid, box in prev1_candidates:
            candidates.append((tid, box, frame_idx - 1))  # 记录来源帧

        # # 如果上一帧无候选，添加上上帧的候选轨迹（来源帧为 frame_idx-2）
        # if not candidates:
        #     prev2_candidates = track_ends_by_frame.get(frame_idx - 2, [])
        #     for tid, box in prev2_candidates:
        #         candidates.append((tid, box, frame_idx - 2))  # 记录来源帧

        # ② 提取轨迹ID、框、来源帧（适配三元组结构）
        if candidates:
            # 从三元组中分别提取 track_id、last_box、source_frame
            track_ids = [t[0] for t in candidates]
            last_boxes = [t[1] for t in candidates]
            source_frames = [t[2] for t in candidates]  # 此时 t[2] 有效，不会越界
        else:
            track_ids = []
            last_boxes = []
            source_frames = []
        # # ② 初始化
        if prev1_candidates:
            frame_used = frame_idx - 1  # 上一帧有候选，更新 frame_used
        # elif prev2_candidates:
        #     frame_used = frame_idx - 2  # 上上帧有候选，更新 frame_used
        else:
            frame_used = None  # 无候选
        best_ios = 0
        best_tid = None

        # source_frames = [t[2] for t in candidates]  # 来源帧：frame_idx-1或frame_idx-2
        if candidates:
            cur_box_array = [cur_box['x1'], cur_box['y1'], cur_box['x2'], cur_box['y2']]
            last_boxes_array = np.array([[b['x1'], b['y1'], b['x2'], b['y2']] for b in last_boxes])
            ious = compute_iou_batch(cur_box_array, last_boxes_array)

            # 按来源帧区分阈值
            for i, iou_val in enumerate(ious):
                source_frame = source_frames[i]
                if ((source_frame == frame_idx - 1 and iou_val > iou_thresh)):
                    matched_track_ids.append(i)

            # IoS 匹配逻辑（同样可区分阈值，可选）
            if not matched_track_ids:
                for i, last_box in enumerate(last_boxes):
                    ios = ios_ratio(cur_box, last_box)
                    source_frame = source_frames[i]
                    # 这里可以根据需要为上一帧/上上帧设置不同的 IoS 阈值
                    if 0.5 < ios < 1:  # 示例：暂用同一阈值，可改为区分逻辑
                        matched_track_ids.append(i)


        # 先根据上一帧匹配情况决定怎么做：
        if len(matched_track_ids) == 1:
            # ========== 新增：IoU二次验证（过滤仅满足IoS的情况） ==========
            i_match = matched_track_ids[0]
            base_id = track_ids[i_match]
            last_box = last_boxes[i_match]  # 匹配轨迹的历史框（上一帧/上上帧）
            source_frame = source_frames[i_match]  # 历史框来源帧

            # 计算当前框与历史框的IoU
            cur_box_array = [cur_box['x1'], cur_box['y1'], cur_box['x2'], cur_box['y2']]
            last_box_array = np.array([[last_box['x1'], last_box['y1'], last_box['x2'], last_box['y2']]])
            current_iou = compute_iou_batch(cur_box_array, last_box_array)[0]  # 取单个IoU值

            # 根据来源帧判断IoU是否达标（和前序IoU阈值保持一致）
            iou_pass = False
            if (source_frame == frame_idx - 1 and current_iou > iou_thresh):
                iou_pass = True
            if iou_pass:
                    # ========== (1) 单轨迹匹配 ==========
                i_match = matched_track_ids[0]
                base_id = track_ids[i_match]
                # 检查“拆分(split)”：
                #   如果本帧有多个 box 都匹配到同一个 base_id，就说明拆分。
                #   简化处理：若本帧同一个track_id已经分配过，就给它新开一条。
                already_matched = any(
                    (tid == base_id) for (tid, _) in track_ends_by_frame.get(frame_idx, [])
                )
                if not already_matched:
                    # 直接追加到 base_id
                    bubble_tracks[base_id].append(cur_box)
                    bubble_ids[idx] = base_id
                    track_ends_by_frame[frame_idx].append((base_id, cur_box))
                    # 获取中心坐标并打印
                    cx, cy = get_box_center(cur_box)
                    # print(
                    #     f"[Frame {frame_idx}] Bubble idx {idx} ✅ 单轨迹匹配成功，继承轨迹: {base_id}，中心坐标: ({cx:.1f}, {cy:.1f})")
                else:
                    i_match = matched_track_ids[0]
                    base_id = track_ids[i_match]
                    last_box = last_boxes[i_match]

                    cur_box_array = [cur_box['x1'], cur_box['y1'], cur_box['x2'], cur_box['y2']]
                    last_box_array = np.array([[last_box['x1'], last_box['y1'], last_box['x2'], last_box['y2']]])
                    iou = compute_iou_batch(cur_box_array, last_box_array)
                    iou = float(iou)

                    existing_matches = [box for tid, box in track_ends_by_frame.get(frame_idx, []) if tid == base_id]

                    if existing_matches:
                        existing_box = existing_matches[0]
                        existing_box_array = [existing_box['x1'], existing_box['y1'], existing_box['x2'],
                                              existing_box['y2']]
                        existing_iou = compute_iou_batch(existing_box_array, last_box_array)
                        existing_iou = float(existing_iou)

                        if iou > existing_iou:
                            for i, (tid, box) in enumerate(track_ends_by_frame[frame_idx]):
                                if tid == base_id:
                                    track_latest_id += 1
                                    new_id = track_latest_id
                                    bubble_tracks.append([box])
                                    prev_idx = next(
                                        (i for i, bid in enumerate(bubble_ids)
                                         if bid == base_id and result_df.loc[i, 'frame_idx'] == frame_idx and is_same_box(
                                            result_df.loc[i], box)),
                                        None
                                    )
                                    if prev_idx is not None:
                                        bubble_ids[prev_idx] = new_id
                                    track_ends_by_frame[frame_idx][i] = (new_id, box)
                                    break

                            bubble_tracks[base_id].append(cur_box)
                            bubble_ids[idx] = base_id
                            track_ends_by_frame[frame_idx].append((base_id, cur_box))
                            print(f"[Frame {frame_idx}] Bubble idx {idx} ⚠️ 抢占轨迹 {base_id}（原框回退）")
                        else:
                            track_latest_id += 1
                            new_id = track_latest_id
                            bubble_tracks.append([cur_box])
                            bubble_ids[idx] = new_id
                            track_ends_by_frame[frame_idx].append((new_id, cur_box))
                            print(f"[Frame {frame_idx}] Bubble idx {idx} 🚨新建轨迹2 {new_id}（无继承）")
            else:
                # ========== IoU不达标（仅满足IoS）：进入多帧回溯 ==========
                print(f"[Frame {frame_idx}] 单轨迹 {base_id} 仅满足IoS（IoU={current_iou:.2f}），尝试多帧回溯...")
                single_result = try_backtrack_match(cur_box, frame_idx, max_backtrack)
                if single_result is not None:
                    single_tid, max_iou = single_result
                    bubble_tracks[single_tid].append(cur_box)
                    bubble_ids[idx] = single_tid
                    track_ends_by_frame[frame_idx].append((single_tid, cur_box))
                else:
                    track_latest_id += 1
                    new_id = track_latest_id
                    bubble_tracks.append([cur_box])
                    bubble_ids[idx] = new_id
                    track_ends_by_frame[frame_idx].append((new_id, cur_box))
                continue  # 跳过后续处理，避免重复匹配
                # 这里不处理，直接让流程继续到后面的"无匹配"分支
                # 通过将 matched_track_ids 设置为空，使程序进入 else 分支
                pass  # 让程序继续执行到下面的 if-elif-else 结构
        elif len(matched_track_ids) > 1:
            # ===== Step 1: 当前帧中所有其他框（排除自己） =====
            cur_cx, cur_cy = get_box_center(cur_box)
            # print(
            #     f"\n[Frame {frame_idx}] Bubble idx {idx} 进入 Step 1：多轨迹融合逻辑，中心坐标 = ({cur_cx:.1f}, {cur_cy:.1f})")

            cur_frame_all = result_df[result_df['frame_idx'] == frame_idx]
            other_boxes = [
                {
                    'x1': row.x1, 'y1': row.y1, 'x2': row.x2, 'y2': row.y2,
                    'index': row.Index,
                    'center': get_box_center(row._asdict())
                }
                for row in cur_frame_all.itertuples()
                if row.Index != idx
            ]
            nearest_arr = np.array([[b['x1'], b['y1'], b['x2'], b['y2']] for b in other_boxes])
            # print(f"Step 1 完成：当前帧共有 {len(other_boxes)} 个其他框作为邻近参考")
            nearest_indices = [b['index'] for b in other_boxes]
            # ===== Step 2: 判断是否是“未融合”疑似情况 =====
            print("进入 Step 2：判断是否疑似未融合")
            iou_thresh_target = 0.7
            suspected_unmerged = False
            for i in matched_track_ids:
                tid = track_ids[i]
                last_box = last_boxes[i]
                last_arr = [last_box['x1'], last_box['y1'], last_box['x2'], last_box['y2']]
                ious = compute_iou_batch(last_arr, nearest_arr)
                for j, iou_val in enumerate(ious):
                    if iou_val > iou_thresh_target:
                        suspected_unmerged = True
                        print(f"  ⛔ 疑似未融合：轨迹 {tid} 与框 {nearest_indices[j]} 的 IoU = {iou_val:.2f} > 阈值")
                        break
                if suspected_unmerged:
                    break
            print(f"Step 2 完成：suspected_unmerged = {suspected_unmerged}")

            # ===== Step 3: 如果是疑似未融合 → 尝试独立继承最近轨迹 =====
            if suspected_unmerged:
                # 新增：打印当前帧索引，明确日志归属
                max_iou = -1.0  # 初始化最大IoU为-1（IoU范围0-1）
                best_tid = None
                # 🔥 修改：确保不重复计算
                processed_tids = set()
                for i in matched_track_ids:
                    tid = track_ids[i]
                    # 检查是否重复处理
                    if tid in processed_tids:
                        print(f"[Frame {frame_idx}]   ⚠️ 警告：轨迹 {tid} 被重复匹配，跳过重复计算")
                        continue
                    processed_tids.add(tid)

                    # 获取轨迹最新框并计算与当前框的IoU
                    last_box = last_boxes[i]
                    current_box = [cur_box['x1'], cur_box['y1'], cur_box['x2'], cur_box['y2']]
                    track_box = [last_box['x1'], last_box['y1'], last_box['x2'], last_box['y2']]
                    iou = compute_iou_batch(
                        current_box,
                        np.array([track_box])
                    )[0]

                    # 新增：打印当前帧IoU计算结果
                    print(f"[Frame {frame_idx}]   轨迹 {tid} 与当前框IoU = {iou:.2f}")

                    # 筛选IoU>0.5且保留最大IoU的轨迹
                    if iou > 0.5 and iou > max_iou:
                        max_iou = iou
                        best_tid = tid

                # 新增：打印当前帧最优轨迹选择结果
                if best_tid is not None:
                    print(f"[Frame {frame_idx}] Step 3 完成：选择IoU最大轨迹 {best_tid}，IoU = {max_iou:.2f}")
                else:
                    #
                    print(f"[Frame {frame_idx}] Step 3 完成：无满足IoU>0.5的轨迹")
                    # 新建轨迹的正常逻辑
                    track_latest_id += 1
                    new_id = track_latest_id
                    bubble_tracks.append([cur_box])  # 初始化轨迹
                    bubble_ids[idx] = new_id
                    track_ends_by_frame[frame_idx].append((new_id, cur_box))  # 记录末端框
                    continue
                # ===== Step 4: 若轨迹已被继承，判断是否抢占 =====
                # 新增：打印当前帧，明确冲突检查的帧
                print(f"[Frame {frame_idx}] 进入 Step 4：检查是否抢占当前帧已继承轨迹")
                existing = [
                    i for i, bid in enumerate(bubble_ids)
                    if bid == best_tid and result_df.loc[i, 'frame_idx'] == frame_idx and i != idx
                ]
                if existing:
                    prev_idx = existing[0]
                    prev_box = result_df.loc[prev_idx]
                    # 补充：获取轨迹历史框时，可打印历史框所属帧（frame_used），明确轨迹来源
                    print(f"[Frame {frame_idx}]   📌 冲突轨迹 {best_tid} 来源帧：{frame_used}")
                    try:
                        last_box = next(b for (tid, b) in track_ends_by_frame[frame_used] if tid == best_tid)
                    except StopIteration:
                        print(f"⚠️ 错误：在帧 {frame_used} 中未找到轨迹 {best_tid}")
                        # 根据情况处理，例如：
                        # 1. 跳过当前框
                        # 2. 或者重新计算 best_tid
                        # 3. 或者创建新轨迹
                        track_latest_id += 1
                        new_id = track_latest_id
                        bubble_ids[idx] = new_id
                        bubble_tracks.append([cur_box])
                        track_ends_by_frame[frame_idx].append((new_id, cur_box))
                        continue  # 跳过当前循环
                    iou_prev = compute_iou_batch(
                        [prev_box['x1'], prev_box['y1'], prev_box['x2'], prev_box['y2']],
                        np.array([[last_box['x1'], last_box['y1'], last_box['x2'], last_box['y2']]])
                    )[0]
                    iou_cur = compute_iou_batch(
                        [cur_box['x1'], cur_box['y1'], cur_box['x2'], cur_box['y2']],
                        np.array([[last_box['x1'], last_box['y1'], last_box['x2'], last_box['y2']]])
                    )[0]
                    # 新增：打印当前帧，明确冲突检测的帧和IoU值
                    print(f"[Frame {frame_idx}]   ⚔️ 冲突检测：cur_iou = {iou_cur:.2f}, prev_iou = {iou_prev:.2f}")
                    if iou_cur > iou_prev:
                        # 新增：打印当前帧，明确抢占操作的帧
                        print(f"[Frame {frame_idx}]   → 当前框抢占轨迹 {best_tid}")
                        bubble_ids[prev_idx] = -1
                        bubble_tracks[best_tid].pop()
                        track_ends_by_frame[frame_idx] = [
                            pair for pair in track_ends_by_frame[frame_idx]
                            if not is_same_box(pair[1], prev_box)
                        ]
                        track_latest_id += 1
                        new_id = track_latest_id
                        bubble_tracks.append([prev_box])
                        bubble_ids[prev_idx] = new_id
                        track_ends_by_frame[frame_idx].append((new_id, prev_box))
                        # 新增：打印当前帧，明确原框重新分配的帧
                        print(f"[Frame {frame_idx}]     → 撤销框 {prev_idx} 的继承，重新分配为轨迹 {new_id}")
                        bubble_ids[idx] = best_tid
                        bubble_tracks[best_tid].append(cur_box)
                        track_ends_by_frame[frame_idx].append((best_tid, cur_box))
                        merged_track_ids.add(best_tid)
                        # 新增：打印当前帧，明确当前框分配的帧
                        print(f"[Frame {frame_idx}]     → 当前框分配轨迹 {best_tid}")
                        # bubble_tracks[best_tid].pop()
                    else:
                        # 新增：打印当前帧，明确不抢占的帧
                        print(f"[Frame {frame_idx}]   → 当前框 IoU 较低（{iou_cur:.2f}），不抢占轨迹")
                        track_latest_id += 1
                        new_id = track_latest_id
                        bubble_ids[idx] = new_id
                        bubble_tracks.append([cur_box])
                        track_ends_by_frame[frame_idx].append((new_id, cur_box))
                        # 新增：打印当前帧，明确新建轨迹的帧
                        print(f"[Frame {frame_idx}]     → 分配新轨迹 {new_id}")
                else:
                    # 新增：打印当前帧，明确无冲突的帧
                    print(f"[Frame {frame_idx}]   → 当前轨迹 {best_tid} 未被继承，直接分配")
                    bubble_ids[idx] = best_tid
                    bubble_tracks[best_tid].append(cur_box)
                    track_ends_by_frame[frame_idx].append((best_tid, cur_box))
                    merged_track_ids.add(best_tid)
                    # 新增：打印当前帧，明确直接继承的帧
                    print(f"[Frame {frame_idx}]     → 当前框继承轨迹 {best_tid}")
                continue
            else:
                # ------- Step 4: 确认真正融合，进入继承逻辑 -------
                # === Step 1: 获取有历史轨迹的 matched 轨迹 ===
                historical_infos = []
                for i in matched_track_ids:
                    tid = track_ids[i]
                    track_hist = bubble_tracks[tid]
                    if len(track_hist) > 0:
                        radii_history = [get_box_radius(box) for box in track_hist]
                        has_small = any(r < radius_thresh for r in radii_history)
                        max_radius = max(radii_history)
                        hist_len = len(track_hist)  # 新增：历史轨迹帧数长度
                        # 元组扩展：tid、是否含小半径、历史最大半径、历史轨迹长度
                        historical_infos.append((tid, has_small, max_radius, hist_len))
                # 在融合逻辑开始处添加
                # print(f"调试信息 - 匹配轨迹详情:")
                # print(f"  matched_track_ids: {matched_track_ids}")
                # # print(f"  track_ids: {track_ids}")
                # print(f"  实际匹配的轨迹: {[track_ids[i] for i in matched_track_ids]}")

                fusion_source_tids = []
                for i in matched_track_ids:
                    tid = track_ids[i]
                    fusion_source_tids.append(tid)
                    print(f"  添加轨迹 {tid} 到融合列表")

                print(f"最终融合轨迹列表: {fusion_source_tids}")
                if len(historical_infos) == 1:
                    # 仅一个有历史信息 → 继承它
                    chosen_tid = historical_infos[0][0]
                    print(f"[Frame {frame_idx}] Bubble idx {idx} 仅一个有历史轨迹 {chosen_tid} → 直接继承")
                    bubble_ids[idx] = chosen_tid
                    bubble_tracks[chosen_tid].append(cur_box)
                    track_ends_by_frame[frame_idx].append((chosen_tid, cur_box))
                    # ✅ 修复：添加所有参与融合的轨迹ID，而非仅选中的主轨迹
                    merged_track_ids.update(fusion_source_tids)
                    # 记录融合事件
                    fusion_events.append({
                        'frame_idx': frame_idx,
                        'current_box_idx': idx,
                        'chosen_tid': chosen_tid,
                        'source_tids': fusion_source_tids.copy(),
                        'reason': 'single_historical',
                        'current_center': (cur_cx, cur_cy)
                    })
                elif len(historical_infos) > 1:
                    # 多个有历史信息：直接选历史轨迹最长
                    selection_reason = "max_history_len"
                    chosen_tid = max(historical_infos, key=lambda x: x[3])[0]

                    print(f"[Frame {frame_idx}] Bubble idx {idx} 多个历史轨迹中继承历史最长的轨迹 {chosen_tid}")
                    bubble_ids[idx] = chosen_tid
                    bubble_tracks[chosen_tid].append(cur_box)
                    track_ends_by_frame[frame_idx].append((chosen_tid, cur_box))
                    # ✅ 修复：添加所有参与融合的轨迹ID，而非仅选中的主轨迹
                    merged_track_ids.update(fusion_source_tids)
                    # 🔥 新增：详细融合信息打印
                    print(f"\n🎯 === 气泡融合事件 ===")
                    print(f"   帧: {frame_idx}, 当前框索引: {idx}")
                    print(f"   融合结果: 轨迹 {chosen_tid} 被选中继承")
                    print(f"   参与融合的轨迹: {sorted(fusion_source_tids)}")
                    print(f"   选择原因: {selection_reason}")
                    for tid, has_small, max_r, hist_len in historical_infos:
                        print(f"     - 轨迹 {tid}: 有小半径历史={has_small}, 最大半径={max_r:.1f}, 历史帧数={hist_len}")

                    # 记录融合事件
                    fusion_events.append({
                        'frame_idx': frame_idx,
                        'current_box_idx': idx,
                        'chosen_tid': chosen_tid,
                        'source_tids': fusion_source_tids.copy(),
                        'reason': selection_reason,
                        'current_center': (cur_cx, cur_cy),
                        'historical_infos': historical_infos.copy()
                    })
                else:
                    # 所有匹配轨迹均无历史信息 → 不新建，继承空间距离最近轨迹
                    cur_center = get_box_center(cur_box)
                    min_dist = float('inf')
                    nearest_tid = None

                    # 遍历所有匹配轨迹，计算中心距离
                    for i in matched_track_ids:
                        tid = track_ids[i]
                        last_box = last_boxes[i]
                        track_center = get_box_center(last_box)
                        # 欧式距离
                        dist = ((cur_center[0] - track_center[0]) ** 2 + (cur_center[1] - track_center[1]) ** 2) ** 0.5
                        if dist < min_dist:
                            min_dist = dist
                            nearest_tid = tid

                    chosen_tid = nearest_tid
                    print(f"[Frame {frame_idx}] 所有轨迹无历史记录 → 继承空间最近轨迹 {chosen_tid}，距离={min_dist:.2f}")
                    bubble_ids[idx] = chosen_tid
                    bubble_tracks[chosen_tid].append(cur_box)
                    track_ends_by_frame[frame_idx].append((chosen_tid, cur_box))
                    merged_track_ids.add(chosen_tid)

                    # 记录融合失败事件
                    fusion_events.append({
                        'frame_idx': frame_idx,
                        'current_box_idx': idx,
                        'chosen_tid': new_id,
                        'source_tids': fusion_source_tids.copy(),
                        'reason': 'no_history_new_track',
                        'current_center': (cur_cx, cur_cy)
                    })
                    # print(f"[Frame {frame_idx}] Bubble idx {idx} 所有轨迹都无历史记录 → 分配新轨迹 {new_id}")
        else:
            # ========== (3) 没有和上一帧匹配到 ==========
            def try_backtrack_match(cur_box, frame_idx, max_bt):
                """
                改进版：只对每个轨迹的最后一帧进行回溯，更准确
                    # 只取每个轨迹最后一帧
                    # 加距离约束
                    # 加半径约束
                    # 加小气泡特殊约束
                """
                # 1. 收集每个轨迹的最后一帧作为候选
                all_cands = []
                # 确定要回溯的帧范围
                backtrack_frames = range(max(0, frame_idx - max_bt), frame_idx - 1)
                # 为每个轨迹找到其最后一帧（最近的历史帧）
                track_last_frames = {}
                # 从近到远搜索，找到每个轨迹的最后一帧
                for t in sorted(backtrack_frames, reverse=True):  # 从最近的帧开始找
                    frame_boxes = track_ends_by_frame.get(t, [])
                    for tid, box in frame_boxes:
                        if tid not in track_last_frames:  # 每个轨迹只记录一次（最近的）
                            # 计算时间间隔
                            time_gap = frame_idx - t
                            track_last_frames[tid] = (box, time_gap, t)
                # 转换格式
                for tid, (last_box, time_gap, t) in track_last_frames.items():
                    all_cands.append((tid, last_box, time_gap))

                if not all_cands:
                    return None

                # 2. 批量计算IOU
                cur_arr = [cur_box['x1'], cur_box['y1'], cur_box['x2'], cur_box['y2']]
                boxes_arr = np.array([[b['x1'], b['y1'], b['x2'], b['y2']] for _, b, _ in all_cands])
                ious = compute_iou_batch(cur_arr, boxes_arr)

                # 3. 找到最大IOU
                max_idx = int(np.argmax(ious))
                max_iou = ious[max_idx]
                # 如果最大IoU不达标，直接返回
                if max_iou < 0.35:
                    return None
                chosen_tid, last_box, time_gap = all_cands[max_idx]
                # ========== 物理约束检查（保持不变）==========
                cur_cx, cur_cy = get_box_center(cur_box)
                last_cx, last_cy = get_box_center(last_box)
                distance = np.sqrt((cur_cx - last_cx) ** 2 + (cur_cy - last_cy) ** 2)

                cur_r = get_box_radius(cur_box)
                last_r = get_box_radius(last_box)

                # 检查1：距离约束
                max_allowed_distance = 50 * time_gap
                if distance > max_allowed_distance:
                    return None
                # 检查2：半径变化约束
                if cur_r < last_r * 0.5 or cur_r > last_r * 2.0:
                    return None
                return (chosen_tid, max_iou)

            # 遍历主循环里 “没有上一帧匹配” 的分支
            # ========== (3) 没有和上一帧匹配到 ==========
            #    尝试多帧回溯（支持拆分）
            single_result = try_backtrack_match(cur_box, frame_idx, max_backtrack)
            if single_result is not None:
                single_tid, max_iou = single_result
                bubble_tracks[single_tid].append(cur_box)
                bubble_ids[idx] = single_tid
                track_ends_by_frame[frame_idx].append((single_tid, cur_box))
                # print(f"[Frame {frame_idx}] Bubble idx {idx} 🕰️ 多帧回溯成功，继承轨迹: {single_tid} | IoU={max_iou:.4f}")
            else:
                track_latest_id += 1
                new_id = track_latest_id
                bubble_tracks.append([cur_box])
                bubble_ids[idx] = new_id
                track_ends_by_frame[frame_idx].append((new_id, cur_box))
                cx = (cur_box['x1'] + cur_box['x2']) / 2
                cy = (cur_box['y1'] + cur_box['y2']) / 2
                # ✅ 打印新建框信息
                # print(f"[Frame {frame_idx}] Bubble idx {idx} ❌中心坐标: ({cx:.1f}, {cy:.1f}) 未匹配成功，创建新轨迹 {new_id}")
    # 2. 将更新后的 bubble_ids 写回 result_df
    result_df['bubble_id'] = bubble_ids
    # ---------- (B) 补帧处理：分段插值中心和尺寸 ----------
    filled_rows = []
    # # ✅ 过滤掉未分配轨迹的框
    # result_df = result_df[result_df['bubble_id'] != -1].reset_index(drop=True)
    # result_df, merged_ids = merge_disconnected_tracks(
    #     result_df,
    #     max_frame_gap=10,  # 你可以改成你希望的间隔帧数
    #     iou_thresh_merge=0.6,  # IoU 合并阈值
    #     center_dist_thresh=150  # 中心点距离阈值
    # )
    frame_to_filename = (
         result_df.sort_values(['frame_idx'])[['frame_idx', 'filename']]
        .drop_duplicates('frame_idx')
        .set_index('frame_idx')['filename']
        .to_dict()
    )
    grouped = result_df.groupby('bubble_id')
    # 用于记录已处理的帧，避免重复
    processed_frames_per_track = {}
    for bubble_id, group in grouped:
        group = group.sort_values('frame_idx')
        frames = group['frame_idx'].tolist()
        if len(frames) <= 1:
            # 只有一个点的轨迹，不补帧
            tmp = group.copy()
            tmp['interpolated'] = False
            filled_rows.extend(tmp.to_dict(orient='records'))
            continue

        frame_to_row = {row['frame_idx']: row for _, row in group.iterrows()}

        # 初始化当前轨迹已处理的帧集合
        if bubble_id not in processed_frames_per_track:
            processed_frames_per_track[bubble_id] = set()

        for i in range(len(frames) - 1):
            f_start = frames[i]
            f_end = frames[i + 1]
            row_start = frame_to_row[f_start]
            row_end = frame_to_row[f_end]
            # 提取起点终点的中心和宽高
            cx1, cy1, w1, h1 = get_box_center_width_height(row_start)
            cx2, cy2, w2, h2 = get_box_center_width_height(row_end)
            # 只在起点帧未被处理过时添加
            if f_start not in processed_frames_per_track[bubble_id]:
                real_row = row_start.to_dict()
                real_row['interpolated'] = False  # 标记真实框
                filled_rows.append(real_row)
                processed_frames_per_track[bubble_id].add(f_start)
            # 插值中间帧
            for f in range(f_start + 1, f_end):
                # 跳过已处理的帧
                if f in processed_frames_per_track[bubble_id]:
                    continue
                alpha = (f - f_start) / (f_end - f_start)
                # 线性插值中心和宽高
                cx_f = cx1 + alpha * (cx2 - cx1)
                cy_f = cy1 + alpha * (cy2 - cy1)
                w_f = w1 + alpha * (w2 - w1)
                h_f = h1 + alpha * (h2 - h1)
                # 生成新 box
                new_box = build_box_from_center(cx_f, cy_f, w_f, h_f)
                # 基于起点拷贝一行，并替换数据
                fake_row = row_start.copy()
                fake_row['frame_idx'] = f
                fake_row['x1'] = new_box['x1']
                fake_row['y1'] = new_box['y1']
                fake_row['x2'] = new_box['x2']
                fake_row['y2'] = new_box['y2']
                fake_row['filename'] = frame_to_filename.get(f, None)
                # 添加插值标记
                fake_row['interpolated'] = True  # 标记插值框
                fake_row['bubble_id'] = bubble_id
                filled_rows.append(fake_row)
                processed_frames_per_track[bubble_id].add(f)
        # 添加终点帧（如果还没添加过）
        if f_end not in processed_frames_per_track[bubble_id]:
            real_row = row_end.to_dict()
            real_row['interpolated'] = False  # 标记真实框
            filled_rows.append(real_row)
            processed_frames_per_track[bubble_id].add(f_end)

    filled_df = pd.DataFrame(filled_rows)
    # 在 track_bubbles_as_dataframe_fast 内的多轨迹匹配（融合）逻辑下添加：
    # chosen_tid 就是融合后选中的轨迹 ID
    filled_df['merged_flag'] = filled_df['bubble_id'].apply(lambda x: 1 if x in merged_track_ids else 0)
    # filled_rows.append(row_end.to_dict())
    # 在函数结尾，添加标记列
    filled_df = filled_df.sort_values(['bubble_id', 'frame_idx']).reset_index(drop=True)
    print("filled_df shape:", filled_df.shape)
    print("filled_df columns:", filled_df.columns.tolist())
    return filled_df

def nms_filter(df, iou_thresh=0.6):
    """
    对单帧的检测结果做 IoU 过滤，避免高度重合的框
    优先保留真实框，再保留插值框
    """
    if df.empty:
        return df

    # 排序：真实框优先（interpolated=False），再按面积大到小
    df = df.copy()
    df["area"] = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"])
    df = df.sort_values(["interpolated", "area"], ascending=[True, False])

    kept_rows = []
    kept_boxes = []

    for _, row in df.iterrows():
        cur_box = np.array([row["x1"], row["y1"], row["x2"], row["y2"]])
        if kept_boxes:
            ious = compute_iou_batch(cur_box, np.array(kept_boxes))
            if np.max(ious) >= iou_thresh:
                continue
        kept_boxes.append(cur_box)
        kept_rows.append(row)

    return pd.DataFrame(kept_rows)
def visualize_bubble_tracking(bubble_df, image_folder=None, output_folder=None):
    """
    可视化每一帧的气泡识别与追踪结果，绘制检测框 + 气泡 ID
    :param bubble_df: 带 bubble_id 的 DataFrame
    :param image_folder: 原始图片文件夹
    :param output_folder: 可视化输出路径
    """
    os.makedirs(output_folder, exist_ok=True)
    grouped = bubble_df.groupby('filename')
    print("传入的 DataFrame 列名：", bubble_df.columns.tolist())
    print("前2行数据：", bubble_df.head(2))
    if image_folder is None:
        return
    if not os.path.exists(image_folder):
        print(f"⚠️ 图像文件夹不存在: {image_folder}")
        return
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"✅ 创建输出文件夹: {output_folder}")
    for filename, group in tqdm(grouped, desc='🎨 可视化气泡追踪'):
        image_path = os.path.join(image_folder, filename)
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ 未找到图像文件: {image_path}")
            continue
        # 🔥 在这里做 NMS 过滤，避免重复框
        # group = nms_filter(group, iou_thresh=0.5)
        for _, row in group.iterrows():
            # 添加ID过滤条件
            x1, y1, x2, y2 = map(int, [row['x1'], row['y1'], row['x2'], row['y2']])
            bubble_id = int(row['bubble_id'])
            # 获取是否插值标记（默认False）
            is_interpolated = row.get('interpolated', False)
            # 用红色表示插值框，绿色表示真实框
            #color = (0, 0, 255) if is_interpolated else (0, 255, 0)
            color = (0, 255, 0)
            # 绘制框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # # #标注ID
            cv2.putText(img, f"ID:{bubble_id}", (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), )
        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, img)


# ✅ 测试主入口
if __name__ == "__main__":
    result_df = read_pkl_dict(r"E:\bubble_pic\bright\250FPS\45_400mA_2\out/all_results.pkl")
    #iou阈值设置（0：0.2 10），（10：0.2,20），（30：0.2,20），（45：0.3,30）net(0.2,10)
    bubble_df= track_bubbles_as_dataframe_fast(result_df, iou_thresh=0.3,max_backtrack=30)
    # print(bubble_df.head())fenbid
    # print(f"✅ 共检测到气泡轨迹数：{bubble_df['bubble_id'].nunique()}")
    # bubble_df.to_csv("bubble_tracking_results.csv", index=False)

# 查看第一个气泡轨迹
#     bubble_id = bubble_df['bubble_id'].unique()[0]zhognw3  b
#     bubble_track = bubble_df[bubble_df['bubble_id'] == bubble_id]
#     print(bubble_track)
    # 可视化保存到 out/
    visualize_bubble_tracking(bubble_df, image_folder=r'E:\bubble_pic\bright\250FPS\45_400mA_2', output_folder=r'E:\bubble_pic\bright\250FPS\45_400mA_2/output32')