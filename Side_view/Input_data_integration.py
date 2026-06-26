import os
import json
import numpy as np
import glob
import shutil
import cv2
from PIL import Image
import io
import base64

np.random.seed(41)
classname_to_id = {

    "bubble": 1,
}


class Lableme2CoCo:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示

    # 由json文件构建COCO
    def to_coco(self, json_path_list):
        self._init_categories()
        for json_path in json_path_list:
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']
            for shape in shapes:
                annotation = self._annotation(shape)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, obj, path):
        image = {}
        # 使用PIL处理TIF图像
        try:
            if 'imageData' in obj and obj['imageData'] is not None:
                image_data = base64.b64decode(obj['imageData'])
                img = Image.open(io.BytesIO(image_data))
            else:
                # 如果json中没有imageData，尝试从文件路径读取
                img_path = path.replace('.json', '.tif')
                if not os.path.exists(img_path):
                    img_path = path.replace('.json', '.jpg')
                img = Image.open(img_path)

            # 转换为RGB如果图像是RGBA或其他模式
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img_array = np.array(img)
            h, w = img_array.shape[:2]
        except Exception as e:
            print(f"Error reading image: {e}")
            # 如果无法读取图像，使用默认尺寸
            h, w = 1024, 1024

        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        # 修改文件扩展名为jpg，因为COCO格式通常使用jpg
        image['file_name'] = os.path.basename(path).replace(".json", ".jpg")
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape):
        # print('shape', shape)
        label = shape['label']
        points = shape['points']
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    # 读取json文件，返回一个json对象
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]


def json2coco(labelme_path, save_coco_path):
    print('reading...')
    # 创建文件
    if not os.path.exists("%s/annotations/" % save_coco_path):
        os.makedirs("%s/annotations/" % save_coco_path)
    if not os.path.exists("%s/images/train2017/" % save_coco_path):
        os.makedirs("%s/images/train2017" % save_coco_path)
    if not os.path.exists("%s/images/val2017/" % save_coco_path):
        os.makedirs("%s/images/val2017" % save_coco_path)
    # 获取images目录下所有的json文件列表
    print(labelme_path + "/*.json")
    json_list_path = glob.glob(labelme_path + "/*.json")
    print('json_list_path: ', len(json_list_path))
    train_path = json_list_path
    val_path = json_list_path
    print("train_n:", len(train_path))
    # 把训练集转化为COCO的json格式
    l2c_train = Lableme2CoCo()
    train_instance = l2c_train.to_coco(train_path)
    l2c_train.save_coco_json(train_instance, '%s/annotations/instances_train2017.json' % save_coco_path)

    # 处理图像文件复制和转换
    for file in train_path:
        # 尝试不同的图像扩展名
        for ext in ['.tif', '.jpg', '.png']:
            img_name = file.replace('.json', ext)
            if os.path.exists(img_name):
                try:
                    # 读取图像
                    if ext == '.tif':
                        img = Image.open(img_name)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img_array = np.array(img)
                    else:
                        img_array = cv2.imread(img_name)

                    # 保存为jpg格式
                    output_name = img_name.split('\\')[-1].split('/')[-1].replace(ext, '.jpg')
                    cv2.imwrite(
                        "%s/images/train2017/%s" % (save_coco_path, output_name),
                        img_array if ext == '.tif' else img_array
                    )
                    print(f'{img_name} --> {output_name}')
                    break
                except Exception as e:
                    print(e)
                    print('Wrong Image:', img_name)
                    continue
        else:
            print(f'No image file found for {file}')

    for file in val_path:
        # 尝试不同的图像扩展名
        for ext in ['.tif', '.jpg', '.png']:
            img_name = file.replace('.json', ext)
            if os.path.exists(img_name):
                try:
                    # 读取图像
                    if ext == '.tif':
                        img = Image.open(img_name)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img_array = np.array(img)
                    else:
                        img_array = cv2.imread(img_name)

                    # 保存为jpg格式
                    output_name = img_name.split('\\')[-1].split('/')[-1].replace(ext, '.jpg')
                    cv2.imwrite(
                        "%s/images/val2017/%s" % (save_coco_path, output_name),
                        img_array if ext == '.tif' else img_array
                    )
                    print(f'{img_name} --> {output_name}')
                    break
                except Exception as e:
                    print(e)
                    print('Wrong Image:', img_name)
                    continue
        else:
            print(f'No image file found for {file}')

    # 把验证集转化为COCO的json格式
    l2c_val = Lableme2CoCo()
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, '%s/annotations/instances_val2017.json' % save_coco_path)


# 训练过程中，如果遇到Index put requires the source and destination dtypes match, got Long for the destination and Int for the source
# 参考：https://github.com/open-mmlab/mmdetection/issues/6706
if __name__ == '__main__':
    labelme_path = r"E:\zl"
    saved_coco_path = r"E:\zl/output"
    json2coco(labelme_path, saved_coco_path)