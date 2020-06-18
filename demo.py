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
import time
import numpy as np
import paddle.fluid as fluid
from tools.cocotools import get_classes
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
    model_path = './weights/step00061000'

    # input_shape越大，精度会上升，但速度会下降。
    # input_shape = (320, 320)
    input_shape = (416, 416)
    # input_shape = (608, 608)

    # 验证时的分数阈值和nms_iou阈值
    conf_thresh = 0.05
    nms_thresh = 0.45

    # 是否给图片画框。不画可以提速。读图片、后处理还可以继续优化。
    draw_image = True
    # draw_image = False


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


    path_dir = os.listdir('images/test')
    # warm up
    if use_gpu:
        for k, filename in enumerate(path_dir):
            image = cv2.imread('images/test/' + filename)
            image, boxes, scores, classes = _decode.detect_image(image, fetch_list, draw_image=False)
            if k == 10:
                break


    time_stat = deque(maxlen=20)
    start_time = time.time()
    end_time = time.time()
    num_imgs = len(path_dir)
    start = time.time()
    for k, filename in enumerate(path_dir):
        image = cv2.imread('images/test/' + filename)
        image, boxes, scores, classes = _decode.detect_image(image, fetch_list, draw_image)

        # 估计剩余时间
        start_time = end_time
        end_time = time.time()
        time_stat.append(end_time - start_time)
        time_cost = np.mean(time_stat)
        eta_sec = (num_imgs - k) * time_cost
        eta = str(datetime.timedelta(seconds=int(eta_sec)))

        logger.info('Infer iter {}, num_imgs={}, eta={}.'.format(k, num_imgs, eta))
        if draw_image:
            cv2.imwrite('images/res/' + filename, image)
            logger.info("Detection bbox results save in images/res/{}".format(filename))
    cost = time.time() - start
    logger.info('total time: {0:.6f}s'.format(cost))
    logger.info('Speed: %.6fs per image,  %.1f FPS.'%((cost / num_imgs), (num_imgs / cost)))


