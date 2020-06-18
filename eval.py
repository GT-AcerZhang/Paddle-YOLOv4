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


    train_program = fluid.Program()
    startup_program = fluid.Program()
    test_program = None
    with fluid.program_guard(train_program, startup_program):
        inputs = fluid.layers.data(name='input_1', shape=[-1, 3, -1, -1], append_batch_size=False, dtype='float32')
        pred_outs = YOLOv4(inputs, num_classes, num_anchors, is_test=True, trainable=False)
        fetch_list = pred_outs

        # 在使用Optimizer之前，将train_program复制成一个test_program。之后使用测试数据运行test_program，就可以做到运行测试程序，而不影响训练结果。
        test_program = train_program.clone(for_test=True)
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_program)

    fluid.io.load_persistables(exe, model_path, main_program=startup_program)
    _decode = Decode(conf_thresh, nms_thresh, input_shape, exe, test_program, all_classes)


    _clsid2catid = copy.deepcopy(clsid2catid)
    if num_classes != 80:   # 如果不是COCO数据集，而是自定义数据集
        _clsid2catid = {}
        for k in range(num_classes):
            _clsid2catid[k] = k
    box_ap = eval(_decode, fetch_list, images, eval_pre_path, anno_file, eval_batch_size, _clsid2catid, draw_image)

