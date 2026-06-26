import numpy as np
from bbox_utils import box_std, iou_calculate, box_proportion, box_contain

import numpy as np
#
# def nms(boxes, thresh=0.6, score_thr=0.2):
#     """
#     非极大值抑制（NMS），保留面积最大且得分高于阈值的检测框。
#     :param boxes: np.ndarray, shape=(N, 5)，格式为[x1, y1, x2, y2, score]
#     :param thresh: float, IOU 抑制阈值
#     :param score_thr: float, 最小分数阈值
#     :return: list[list]，保留的检测框，格式为[x1, y1, x2, y2, score]
#     """
#     if boxes is None or len(boxes) == 0:
#         return []
#
#     boxes = np.array(boxes)
#     if boxes.ndim != 2 or boxes.shape[1] < 5:
#         raise ValueError(f"NMS输入格式错误，应为 (N,5)，但收到 {boxes.shape}")
#
#     # 筛选得分高于阈值的框
#     boxes = boxes[boxes[:, -1] > score_thr]
#     if boxes.shape[0] == 0:
#         return []
#
#     # 计算面积（宽度*高度）
#     areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
#     # 【改动1】按面积从大到小排序（保留最大框优先）
#     order = np.argsort(areas)[::-1]   # 降序 = 最大面积优先
#     keep = []
#
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#
#         # 当前框与其他框的交集计算
#         xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
#         yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
#         xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
#         yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
#
#         w = np.maximum(0.0, xx2 - xx1)
#         h = np.maximum(0.0, yy2 - yy1)
#         inter = w * h
#
#         area_i = areas[i]   # 当前框面积
#         area_others = areas[order[1:]]
#
#         union = area_i + area_others - inter
#         ious = np.where(union > 0, inter / union, 0)
#
#         # 保留 IOU 小于阈值的框
#         order = order[np.where(ious <= thresh)[0] + 1]
#
#     filtered_boxes = boxes[keep]
#     return post_process(filtered_boxes)
#
#
# def post_process(boxes):
#     """
#     对 NMS 结果进一步清理冗余框：
#     - 去除长条形框
#     - 去除重复框（高 IOU 或包含关系），优先保留面积大的框
#     :param boxes: ndarray, shape=(N, 5)
#     :return: list of valid boxes
#     """
#     if len(boxes) == 0:
#         return []
#
#     boxes = boxes.tolist()
#     valid = []
#     skip = set()
#
#     # 【改动2】后处理也按面积从大到小遍历，保证优先保留最大框
#     boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
#
#     for i in range(len(boxes)):
#         if i in skip:
#             continue
#         if box_proportion(boxes[i]):
#             continue  # 丢弃长条框
#         keep = True
#         for j in range(i + 1, len(boxes)):
#             if j in skip:
#                 continue
#             if iou_calculate(boxes[i], boxes[j]) > 0.7 or box_contain(boxes[i], boxes[j]) or box_contain(boxes[j], boxes[i]):
#                 skip.add(j)
#         valid.append(boxes[i])
#
#     return valid
def nms(boxes, thresh=0.6, score_thr=0.4):
    """
    非极大值抑制（NMS），保留得分较高且冗余较低的检测框。
    :param boxes: np.ndarray, shape=(N, 5)，格式为[x1, y1, x2, y2, score]
    :param thresh: float, IOU 抑制阈值
    :param score_thr: float, 最小分数阈值
    :return: list[list]，保留的检测框，格式为[x1, y1, x2, y2, score]
    初始为阈值为0.15
    """
    if boxes is None or len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    if boxes.ndim != 2 or boxes.shape[1] < 5:
        raise ValueError(f"NMS输入格式错误，应为 (N,5)，但收到 {boxes.shape}")

    # 筛选得分高于阈值的框
    boxes = boxes[boxes[:, -1] > score_thr]
    if boxes.shape[0] == 0:
        return []

    # 得分排序，最大在前
    order = np.argsort(boxes[:, -1])[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        # 当前框与其他框的交集计算
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_others = (boxes[order[1:], 2] - boxes[order[1:], 0]) * \
                      (boxes[order[1:], 3] - boxes[order[1:], 1])

        union = area_i + area_others - inter
        ious = np.where(union > 0, inter / union, 0)

        # 保留 IOU 小于阈值的框
        order = order[np.where(ious <= thresh)[0] + 1]

    filtered_boxes = boxes[keep]
    return post_process(filtered_boxes)


def post_process(boxes):
    """
    对 NMS 结果进一步清理冗余框：
    - 去除长条形框
    - 去除重复框（高 IOU 或包含关系）

    :param boxes: ndarray, shape=(N, 5)
    :return: list of valid boxes
    初始为0.2
    """
    if len(boxes) == 0:
        return []

    boxes = boxes.tolist()
    valid = []
    skip = set()

    for i in range(len(boxes)):
        if i in skip:
            continue
        if box_proportion(boxes[i]):
            continue  # 丢弃长条框
        keep = True
        for j in range(i + 1, len(boxes)):
            if j in skip:
                continue
            if iou_calculate(boxes[i], boxes[j]) > 0.6 or box_contain(boxes[i], boxes[j]) :
                skip.add(j)
        valid.append(boxes[i])

    return valid
if __name__ == '__main__':
    print("🔍 测试 NMS")

    test_boxes = np.array([
        [50, 50, 100, 100, 0.9],
        [52, 52, 98, 98, 0.8],
        [200, 200, 300, 300, 0.7],
        [400, 400, 410, 405, 0.95],  # 长条框
        [60, 60, 105, 105, 0.85],
        [200, 200, 300, 300, 0.65],  # 重复框
    ])

    kept = nms(test_boxes, thresh=0.2, score_thr=0.1)

    print("保留的框：")
    for b in kept:
        print(f"  {b}")
