
from tools.cocotools import get_classes
from tools.visualize import get_colors, draw
from paddle_serving_client import Client
import cv2
import sys
import numpy as np




classes_path = 'data/coco_classes.txt'
all_classes = get_classes(classes_path)
num_classes = len(all_classes)
colors = get_colors(num_classes)




prototxt_path = sys.argv[1]
client = Client()
client.load_client_config(prototxt_path)
client.connect(['127.0.0.1:9494'])


img_path = sys.argv[3]
print(img_path)   # 这是图片的路径
input_shape = (608, 608)
image = cv2.imread(img_path)
h, w = image.shape[:2]
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
scale_x = float(input_shape[1]) / w
scale_y = float(input_shape[0]) / h
img = cv2.resize(img, None, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)
pimage = img.astype(np.float32) / 255.
pimage = pimage.transpose(2, 0, 1)


fetch_map = client.predict(
    feed={
        "image": pimage,
        "origin_shape": np.array([h, w]),
    },
    fetch=["multiclass_nms_0.tmp_0"])
print('===============================================')
pred = fetch_map['multiclass_nms_0.tmp_0']
print(pred.shape)
print(pred)

if pred[0][0] < 0.0:
    boxes = np.array([])
    classes = np.array([])
    scores = np.array([])
else:
    boxes = pred[:, 2:]
    scores = pred[:, 1]
    classes = pred[:, 0].astype(np.int32)


draw_image = True
draw_thresh = 0.0
if len(scores) > 0 and draw_image:
    pos = np.where(scores >= draw_thresh)
    boxes2 = boxes[pos]  # [M, 4]
    classes2 = classes[pos]  # [M, ]
    scores2 = scores[pos]  # [M, ]
    draw(image, boxes2, scores2, classes2, all_classes, colors)

cv2.imwrite('a.jpg', image)



