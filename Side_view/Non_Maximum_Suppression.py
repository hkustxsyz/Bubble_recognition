import numpy as np
from MIB_data_processing2 import box_std, iou_calculate, box_proportion, box_contain


def nms(boxes, thresh=0.6, score_thr=0.5):
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
    boxes = boxes[boxes[:, -1] >= score_thr]
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
                #45的为0.6
            if iou_calculate(boxes[i], boxes[j]) > 0.6 :
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
