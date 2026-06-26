# bbox_utils.py
import math
import numpy as np


def box_std(box):
    """
    将各种格式的box转换为[x1, y1, x2, y2]，并确保x1<x2, y1<y2
    支持输入格式：
    - list或np.array: [x1, y1, x2, y2, (score)]
    - list嵌套: [[x1, y1], [x2, y2]]
    """
    box = np.array(box).flatten()
    if box.shape[0] < 4:
        raise ValueError(f"无效box维度: {box}")
    x1, y1, x2, y2 = box[:4]
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def iou_calculate(box1, box2):
    """计算两个矩形框的交并比(IOU)"""
    box1 = box_std(box1)
    box2 = box_std(box2)

    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def box_proportion(box):
    """判断矩形框是否为长条形或无效框"""
    box = box_std(box)
    width = box[2] - box[0]
    height = box[3] - box[1]
    if width == 0 or height == 0:
        return True
    ratio = max(width / height, height / width)
    return ratio >= 1.6


def box_contain(box1, box2):
    """判断 box1 是否包含 box2"""
    box1 = box_std(box1)
    box2 = box_std(box2)
    return (box1[0] <= box2[0] and box1[1] <= box2[1] and
            box1[2] >= box2[2] and box1[3] >= box2[3])


def long_error(box1, box2):
    """计算两个框左上/右下点的误差，归一化到对角线长度"""
    box1 = box_std(box1)
    box2 = box_std(box2)

    err1 = (box1[0] - box2[0]) ** 2 + (box1[1] - box2[1]) ** 2
    err2 = (box1[2] - box2[2]) ** 2 + (box1[3] - box2[3]) ** 2
    diag = (box2[0] - box2[2]) ** 2 + (box2[1] - box2[3]) ** 2
    diag = diag if diag != 0 else 1

    return abs((err1 ** 0.5 + err2 ** 0.5) / (diag ** 0.5))


def area(box1, box2):
    """计算两个框的面积误差（以box1为基准）"""
    box1 = box_std(box1)
    box2 = box_std(box2)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return abs(area1 - area2) / area1 if area1 > 0 else 0.0


if __name__ == '__main__':
    print("🔍 测试 box_std")
    test_boxes = [
        [50, 40, 100, 120],
        [[100, 120], [50, 40]],
        [100, 120, 50, 40],
        [[100, 120, 50, 40, 0.95]],
        np.array([100, 120, 50, 40, 0.9]),
    ]
    for i, b in enumerate(test_boxes):
        try:
            std_box = box_std(b)
            print(f"  box{i}: {b} → 标准化: {std_box}")
        except Exception as e:
            print(f"  box{i} 异常: {e}")

    print("\n📏 测试 iou")
    box_a = [30, 30, 70, 70]
    box_b = [50, 50, 100, 100]
    print("  IOU:", iou_calculate(box_a, box_b))

    print("\n📐 测试 box_proportion")
    print("  长条框判断:", box_proportion([10, 10, 210, 30]))  # True
    print("  正常框判断:", box_proportion([10, 10, 60, 60]))  # False

    print("\n📦 测试 box_contain")
    print("  包含关系:", box_contain([0, 0, 100, 100], [10, 10, 50, 50]))  # True
    print("  非包含关系:", box_contain([0, 0, 40, 40], [10, 10, 50, 50]))  # False

    print("\n📏 测试 long_error")
    print("  长边误差:", long_error([0, 0, 100, 100], [5, 5, 105, 105]))

    print("\n📐 测试 area")
    print("  面积误差:", area([0, 0, 100, 100], [0, 0, 90, 90]))
