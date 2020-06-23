#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-10 10:20:27
#   Description : paddlepaddle_yolov4
#
# ================================================================
from collections import deque
import datetime
import cv2
import os
import copy
import json
import time
import numpy as np
from tools.cocotools import eval
import paddle.fluid as fluid
import paddle.fluid.layers as P
from tools.cocotools import get_classes, clsid2catid
from model.yolov4 import YOLOv4
from model.decode_np import Decode

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)



# 6G的卡，训练时如果要预测，则设置use_gpu = False，否则显存不足。
use_gpu = False
use_gpu = True


if __name__ == '__main__':
    # classes_path = 'data/voc_classes.txt'
    classes_path = 'data/coco_classes.txt'
    # model_path可以是'yolov4'、'./weights/step00001000'这些。
    # model_path = 'yolov4'
    model_path = './weights/step00252000'

    # input_shape越大，精度会上升，但速度会下降。
    # input_shape = (320, 320)
    # input_shape = (416, 416)
    input_shape = (608, 608)
    # 验证时的分数阈值和nms_iou阈值
    conf_thresh = 0.001
    nms_thresh = 0.45
    # 是否画出验证集图片
    draw_image = False
    # 验证时的批大小
    eval_batch_size = 4

    # 验证集图片的相对路径
    # eval_pre_path = '../VOCdevkit/VOC2012/JPEGImages/'
    # anno_file = 'annotation_json/voc2012_val.json'
    eval_pre_path = '../data/data7122/val2017/'
    anno_file = '../data/data7122/annotations/instances_val2017.json'
    with open(anno_file, 'r', encoding='utf-8') as f2:
        for line in f2:
            line = line.strip()
            dataset = json.loads(line)
            images = dataset['images']

    num_anchors = 3
    all_classes = get_classes(classes_path)
    num_classes = len(all_classes)


    startup_prog = fluid.Program()
    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            # 多尺度训练
            inputs = P.data(name='input_1', shape=[-1, 3, -1, -1], append_batch_size=False, dtype='float32')
            output_l, output_m, output_s = YOLOv4(inputs, num_classes, num_anchors, is_test=False, trainable=True)
            eval_fetch_list = [output_l, output_m, output_s]
    eval_prog = eval_prog.clone(for_test=True)
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    fluid.load(eval_prog, model_path, executor=exe)
    _decode = Decode(conf_thresh, nms_thresh, input_shape, exe, eval_prog, all_classes)


    _clsid2catid = copy.deepcopy(clsid2catid)
    if num_classes != 80:   # 如果不是COCO数据集，而是自定义数据集
        _clsid2catid = {}
        for k in range(num_classes):
            _clsid2catid[k] = k
    box_ap = eval(_decode, eval_fetch_list, images, eval_pre_path, anno_file, eval_batch_size, _clsid2catid, draw_image)

