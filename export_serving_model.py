#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-10 10:20:27
#   Description : paddlepaddle_yolov4
#
# ================================================================
import os
import tempfile
import copy
import shutil
from collections import OrderedDict
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as P

from export_model import load_params, dump_infer_config, prune_feed_vars
from model.head import YOLOv3Head
from model.resnet import Resnet50Vd
from model.yolov3 import YOLOv3
from tools.cocotools import get_classes
from model.yolov4 import YOLOv4
from config import *

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)



def save_serving_model(save_dir, exe, feed_vars, test_fetches, infer_prog):
    feed_var_names = [var.name for var in feed_vars.values()]
    fetch_list = sorted(test_fetches.items(), key=lambda i: i[0])
    target_vars = [var[1] for var in fetch_list]
    feed_var_names = prune_feed_vars(feed_var_names, target_vars, infer_prog)
    output_dir = save_dir
    serving_client = os.path.join(output_dir, 'serving_client')
    serving_server = os.path.join(output_dir, 'serving_server')
    logger.info(
        "Export serving model to {}, client side: {}, server side: {}. input: {}, output: "
        "{}...".format(output_dir, serving_client, serving_server,
                       feed_var_names, [str(var.name) for var in target_vars]))
    feed_dict = {x: infer_prog.global_block().var(x) for x in feed_var_names}
    fetch_dict = {x.name: x for x in target_vars}
    import paddle_serving_client.io as serving_io
    serving_client = os.path.join(save_dir, 'serving_client')
    serving_server = os.path.join(save_dir, 'serving_server')
    serving_io.save_model(
        client_config_folder=serving_client,
        server_model_folder=serving_server,
        feed_var_dict=feed_dict,
        fetch_var_dict=fetch_dict,
        main_program=infer_prog)
    shutil.copy(serving_client+'/serving_client_conf.prototxt', save_dir+'/serving_server_conf.prototxt')
    shutil.rmtree('-p')



if __name__ == '__main__':
    # 推理模型保存目录
    save_dir = 'serving_model'

    # 导出时用fastnms还是不后处理
    # postprocess = 'fastnms'
    postprocess = 'multiclass_nms'
    # postprocess = 'numpy_nms'

    # need 3 for YOLO arch
    min_subgraph_size = 3

    # 是否使用Padddle Executor进行推理。
    use_python_inference = False

    # 使用GPU时，默认为fluid, 可选（fluid/trt_fp32/trt_fp16）
    mode = 'fluid'

    # 对模型输出的预测框再进行一次分数过滤的阈值。设置为0.0表示不再进行分数过滤。
    # 与conf_thresh不同，需要修改这个值的话直接编辑导出的inference_model/infer_cfg.yml配置文件，不需要重新导出模型。
    # 总之，inference_model/infer_cfg.yml里的配置可以手动修改，不需要重新导出模型。
    draw_threshold = 0.0

    # 选择配置
    cfg = YOLOv4_Config_1()
    # cfg = YOLOv3_Config_1()


    # =============== 以下不用设置 ===============
    algorithm = cfg.algorithm
    classes_path = cfg.classes_path

    # 读取的模型
    model_path = cfg.infer_model_path

    # input_shape越大，精度会上升，但速度会下降。
    input_shape = cfg.infer_input_shape

    # 推理时的分数阈值和nms_iou阈值。注意，这些值会写死进模型，如需修改请重新导出模型。
    conf_thresh = cfg.infer_conf_thresh
    nms_thresh = cfg.infer_nms_thresh
    keep_top_k = cfg.infer_keep_top_k
    nms_top_k = cfg.infer_nms_top_k


    # 初始卷积核个数
    initial_filters = 32
    # 先验框
    _anchors = copy.deepcopy(cfg.anchors)
    num_anchors = len(cfg.anchor_masks[0])  # 每个输出层有几个先验框
    _anchors = np.array(_anchors)
    _anchors = np.reshape(_anchors, (-1, num_anchors, 2))
    _anchors = _anchors.astype(np.float32)
    num_anchors = len(_anchors[0])  # 每个输出层有几个先验框

    all_classes = get_classes(classes_path)
    num_classes = len(all_classes)


    startup_prog = fluid.Program()
    infer_prog = fluid.Program()
    with fluid.program_guard(infer_prog, startup_prog):
        with fluid.unique_name.guard():
            inputs = P.data(name='image', shape=[-1, 3, -1, -1], append_batch_size=False, dtype='float32')

            if postprocess == 'fastnms' or postprocess == 'multiclass_nms':
                resize_shape = P.data(name='resize_shape', shape=[-1, 2], append_batch_size=False, dtype='int32')
                origin_shape = P.data(name='origin_shape', shape=[-1, 2], append_batch_size=False, dtype='int32')
                param = {}
                param['resize_shape'] = resize_shape
                param['origin_shape'] = origin_shape
                param['anchors'] = _anchors
                param['conf_thresh'] = conf_thresh
                param['nms_thresh'] = nms_thresh
                param['keep_top_k'] = keep_top_k
                param['nms_top_k'] = nms_top_k
                param['num_classes'] = num_classes
                param['num_anchors'] = num_anchors
                # 输入字典
                feed_vars = [('image', inputs), ('resize_shape', resize_shape), ('origin_shape', origin_shape)]
                feed_vars = OrderedDict(feed_vars)
            if postprocess == 'numpy_nms':
                param = None
                # 输入字典
                feed_vars = [('image', inputs), ]
                feed_vars = OrderedDict(feed_vars)

            if algorithm == 'YOLOv4':
                if postprocess == 'fastnms':
                    boxes, scores, classes = YOLOv4(inputs, num_classes, num_anchors, is_test=False, trainable=True, export=True, postprocess=postprocess, param=param)
                    test_fetches = {'boxes': boxes, 'scores': scores, 'classes': classes, }
                if postprocess == 'multiclass_nms':
                    pred = YOLOv4(inputs, num_classes, num_anchors, is_test=False, trainable=True, export=True, postprocess=postprocess, param=param)
                    test_fetches = {'pred': pred, }
            elif algorithm == 'YOLOv3':
                backbone = Resnet50Vd()
                head = YOLOv3Head(keep_prob=1.0)   # 一定要设置keep_prob=1.0, 为了得到一致的推理结果
                yolov3 = YOLOv3(backbone, head)
                if postprocess == 'fastnms':
                    boxes, scores, classes = yolov3(inputs, export=True, postprocess=postprocess, param=param)
                    test_fetches = {'boxes': boxes, 'scores': scores, 'classes': classes, }
                if postprocess == 'multiclass_nms':
                    pred = yolov3(inputs, export=True, postprocess=postprocess, param=param)
                    test_fetches = {'pred': pred, }
    infer_prog = infer_prog.clone(for_test=True)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)


    logger.info("postprocess: %s" % postprocess)
    load_params(exe, infer_prog, model_path)

    save_serving_model(save_dir, exe, feed_vars, test_fetches, infer_prog)
    logger.info("Done.")


