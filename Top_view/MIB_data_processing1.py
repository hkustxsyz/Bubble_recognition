# bubble_info.py
import math
import pandas as pd
import numpy as np
from bbox_utils import box_std

class BubbleInfo:
    def __init__(self, ID, bubble_data,fps=250):
        """
        构造单个气泡轨迹信息

        :param ID: int，气泡轨迹编号
        :param bubble_data: List[dict]，每帧包含字段：
               ['filename', 'frame_idx', 'box_id', 'x1', 'y1', 'x2', 'y2', 'score']
        """
        self.ID = ID
        self.bubble_data = bubble_data
        self.len = len(bubble_data)
        self.fps = fps
        self.frame_idxs = [d['frame_idx'] for d in bubble_data]
        self.filenames = [d['filename'] for d in bubble_data]

        self.first_frame = self.frame_idxs[0] if self.len > 0 else None
        self.last_frame = self.frame_idxs[-1] if self.len > 0 else None

        self.first_location = [bubble_data[0][c] for c in ['x1', 'y1', 'x2', 'y2']] if self.len > 0 else None
        self.last_location = [bubble_data[-1][c] for c in ['x1', 'y1', 'x2', 'y2']] if self.len > 0 else None

        self.radius = []              # 每一帧的气泡半径
        self.center = []              # 每一帧的气泡中心
        self.first_center = None      # 初始中心点
        self.first_radius = None
        self.last_radius = None
        self.radius_mean = None
        self.radius_max = None
        self.radius_min = None

        self.leave_frame = None       # 脱离帧号
        self.leave_flag = 0           # 脱离状态标志（0: 未脱离；1: 脱离）
        self.leave_radius = None      # 脱离帧的气泡半径
        self.duration = 0             # 气泡持续帧数
        self.duration_seconds = 0

        self._analyze()

    def _analyze(self):
        """
        内部分析函数：计算中心、半径、脱离状态等属性。
        """
        # 计算每帧中心点和半径
        for d in self.bubble_data:
            box = box_std([d['x1'], d['y1'], d['x2'], d['y2']])
            w, h = box[2] - box[0], box[3] - box[1]
            r = math.sqrt(w * h) / 2
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2

            self.radius.append(r)
            self.center.append((cx, cy))

        # 转为 DataFrame
        self.radius_pd = pd.DataFrame(self.radius, columns=["radius"])
        self.center_pd = pd.DataFrame(self.center, columns=["center_x", "center_y"])

        # 半径统计
        self.first_radius = self.radius[0] if self.radius else None
        self.last_radius = self.radius[-1] if self.radius else None
        self.radius_mean = np.mean(self.radius) if self.radius else None
        self.radius_max = np.max(self.radius) if self.radius else None
        self.radius_min = np.min(self.radius) if self.radius else None

        # 脱离判断逻辑
        if self.center:

            merge_indices = [i for i, d in enumerate(self.bubble_data) if d.get('merged_flag', 0) == 1]

            if merge_indices:
                merge_start_idx = merge_indices[0]
                self.first_center = self.center[merge_start_idx]
                print(f"使用融合点作为中心: 索引{merge_start_idx}, 坐标{self.first_center}")
            else:
                self.first_center = self.center[0]
            last_r = self.last_radius if self.radius else 0

            # 检查是否所有帧都在初始中心范围内
            all_within = True
            for c in self.center:
                dx = c[0] - self.first_center[0]
                dy = c[1] - self.first_center[1]
                dist = math.sqrt(dx ** 2 + dy ** 2)
                if dist >= last_r:
                    all_within = False
                    break

            if not all_within:
                for i in range(self.len - 1, -1, -1):
                    dx = self.center[i][0] - self.first_center[0]
                    dy = self.center[i][1] - self.first_center[1]
                    dist = math.sqrt(dx ** 2 + dy ** 2)
                    if dist < last_r:
                        self.leave_frame = self.frame_idxs[i]
                        self.leave_radius = self.radius[i]
                        self.leave_flag = 1
                        break
            else:
                self.leave_flag = 0

        # 气泡持续时间（帧数）
        if self.leave_frame is not None:
            self.duration = self.leave_frame - self.first_frame + 1
        else:
            self.duration = self.last_frame - self.first_frame + 1
        self.duration_seconds = self.duration / self.fps if self.fps > 0 else 0
        # # —— 新增：计算增长率 ——
        # # 取从第一帧到脱离帧（或最后一帧）的半径列表长度 n
        n = self.duration if self.leave_frame is not None else len(self.radius)
        diffs = []
        max_diffs = 100
        n = min(self.duration, len(self.radius))  # ✅ 防止 n 超出 radius 长度
        for i in range(1, n):
            if len(diffs) >= max_diffs:
                break
            # 先尝试与上一帧差值
            diff = self.radius[i] - self.radius[i - 1]
            if diff < 0:
                # 如果为负，就往前多帧累计平均
                for k in range(2, i + 1):
                    diff_k = (self.radius[i] - self.radius[i - k]) / k
                    if diff_k >= 0:
                        diff = diff_k
                        break
                else:
                    # 推到第一帧仍为负，就舍弃这个差值
                    continue
            diffs.append(diff)

        # 平均所有保留下来的非负差值
        if diffs:
            self.growth_rate = float(sum(diffs) / len(diffs))
        else:
            self.growth_rate = 0.0


def build_bubble_infos(tracked_df: pd.DataFrame):
    """
    将追踪后的 DataFrame（包含 bubble_id）构造成 BubbleInfo 实例集合
    :param tracked_df: DataFrame，包含识别框 + 气泡轨迹编号 bubble_id
    :return: List[BubbleInfo]
    """
    bubble_infos = []
    grouped = tracked_df.groupby('bubble_id')

    for bubble_id, group in grouped:
        bubble_data = group.to_dict('records')
        bubble_infos.append(BubbleInfo(ID=bubble_id, bubble_data=bubble_data))

    return bubble_infos


# ✅ 测试示例
if __name__ == "__main__":
    from io_utils import read_pkl_dict
    from bubble_utils1 import track_bubbles_as_dataframe_fast

    df = read_pkl_dict('all_results.pkl')
    tracked_df = track_bubbles_as_dataframe_fast(df)
    bubble_infos = build_bubble_infos(tracked_df)

    example = bubble_infos[0]
    print(f"✅ 共构建 BubbleInfo 实例数：{len(bubble_infos)}")
    print("🎯 气泡 ID:", example.ID)
    print("🎯 中心轨迹:", example.center)
    print("🎯 初始中心:", example.first_center)
    print("🎯 脱离帧:", example.leave_frame)
    print("🎯 脱离半径:", example.leave_radius)
    print("🎯 脱离标志:", example.leave_flag)
    print("📏 气泡持续帧数:", example.duration)
    print('初始半径:', example.first_radius)