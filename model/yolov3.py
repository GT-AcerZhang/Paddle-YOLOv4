#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-10 10:20:27
#   Description : paddlepaddle_yolov4
#
# ================================================================
import paddle.fluid as fluid
import paddle.fluid.layers as P
import numpy as np

from model.fastnms import fastnms
from model.yolov4 import decode



class YOLOv3(object):
    def __init__(self, backbone, head):
        super(YOLOv3, self).__init__()
        self.backbone = backbone
        self.head = head

    def __call__(self, x, export=False, postprocess=None, param=None):
        body_feats = self.backbone(x)
        output_l, output_m, output_s = self.head(body_feats)
        if export:
            # 用张量操作实现后处理
            if postprocess == 'fastnms' or postprocess == 'multiclass_nms':
                resize_shape = param['resize_shape']
                origin_shape = param['origin_shape']
                anchors = param['anchors']
                conf_thresh = param['conf_thresh']
                nms_thresh = param['nms_thresh']
                keep_top_k = param['keep_top_k']
                nms_top_k = param['nms_top_k']
                num_classes = param['num_classes']
                num_anchors = param['num_anchors']

                use_yolo_box = False

                # 先对坐标解码
                # 第一种方式。慢一点，但支持修改。
                if not use_yolo_box:
                    # 相当于numpy的transpose()，交换下标
                    output_l = fluid.layers.transpose(output_l, perm=[0, 2, 3, 1], name='output_l')
                    output_m = fluid.layers.transpose(output_m, perm=[0, 2, 3, 1], name='output_m')
                    output_s = fluid.layers.transpose(output_s, perm=[0, 2, 3, 1], name='output_s')
                    pred_xywh_s, pred_conf_s, pred_prob_s = decode(output_s, anchors[0], 8, num_classes, conf_thresh)
                    pred_xywh_m, pred_conf_m, pred_prob_m = decode(output_m, anchors[1], 16, num_classes, conf_thresh)
                    pred_xywh_l, pred_conf_l, pred_prob_l = decode(output_l, anchors[2], 32, num_classes, conf_thresh)
                    # 获取分数。可以不用将pred_conf_s第2维重复80次，paddle支持直接相乘。
                    pred_score_s = pred_conf_s * pred_prob_s
                    pred_score_m = pred_conf_m * pred_prob_m
                    pred_score_l = pred_conf_l * pred_prob_l
                    # 所有输出层的预测框集合后再执行nms
                    all_pred_boxes = P.concat([pred_xywh_s, pred_xywh_m, pred_xywh_l], axis=1)       # [batch_size, -1, 4]
                    all_pred_scores = P.concat([pred_score_s, pred_score_m, pred_score_l], axis=1)   # [batch_size, -1, 80]

                    if postprocess == 'fastnms':
                        # cx_cy_w_h格式不用转换成x0y0x1y1格式
                        pass
                    if postprocess == 'multiclass_nms':
                        # 把cx_cy_w_h格式转换成x0y0x1y1格式
                        all_pred_boxes = P.concat([all_pred_boxes[:, :, :2] - all_pred_boxes[:, :, 2:] * 0.5,
                                                   all_pred_boxes[:, :, :2] + all_pred_boxes[:, :, 2:] * 0.5], axis=-1)
                        all_pred_boxes /= 608
                        all_pred_scores = fluid.layers.transpose(all_pred_scores, perm=[0, 2, 1])


                # 第二种方式。用官方yolo_box()函数快一点
                if use_yolo_box:
                    anchors = anchors.astype(np.int32)
                    anchors = np.reshape(anchors, (-1, num_anchors*2))
                    anchors = anchors.tolist()
                    # [bz, ?1, 4]  [bz, ?1, 80]   注意，是过滤置信度位小于conf_thresh的，而不是过滤最终分数！
                    bbox_l, prob_l = fluid.layers.yolo_box(
                        x=output_l,
                        img_size=fluid.layers.ones(shape=[1, 2], dtype="int32"),   # 返回归一化的坐标，而且是x0y0x1y1格式
                        anchors=anchors[2],
                        class_num=num_classes,
                        conf_thresh=conf_thresh,
                        downsample_ratio=32,
                        clip_bbox=False)
                    bbox_m, prob_m = fluid.layers.yolo_box(
                        x=output_m,
                        img_size=fluid.layers.ones(shape=[1, 2], dtype="int32"),   # 返回归一化的坐标，而且是x0y0x1y1格式
                        anchors=anchors[1],
                        class_num=num_classes,
                        conf_thresh=conf_thresh,
                        downsample_ratio=16,
                        clip_bbox=False)
                    bbox_s, prob_s = fluid.layers.yolo_box(
                        x=output_s,
                        img_size=fluid.layers.ones(shape=[1, 2], dtype="int32"),   # 返回归一化的坐标，而且是x0y0x1y1格式
                        anchors=anchors[0],
                        class_num=num_classes,
                        conf_thresh=conf_thresh,
                        downsample_ratio=8,
                        clip_bbox=False)
                    boxes = []
                    scores = []
                    boxes.append(bbox_l)
                    boxes.append(bbox_m)
                    boxes.append(bbox_s)
                    scores.append(prob_l)
                    scores.append(prob_m)
                    scores.append(prob_s)
                    all_pred_boxes = fluid.layers.concat(boxes, axis=1)  # [batch_size, -1, 4]
                    all_pred_scores = fluid.layers.concat(scores, axis=1)  # [batch_size, -1, 80]
                    if postprocess == 'fastnms':
                        # 把x0y0x1y1格式转换成cx_cy_w_h格式
                        all_pred_boxes = P.concat([(all_pred_boxes[:, :, :2] + all_pred_boxes[:, :, 2:]) * 0.5,
                                                   all_pred_boxes[:, :, 2:] - all_pred_boxes[:, :, :2]], axis=-1)
                    if postprocess == 'multiclass_nms':
                        # x0y0x1y1格式不用转换成cx_cy_w_h格式
                        all_pred_scores = fluid.layers.transpose(all_pred_scores, perm=[0, 2, 1])
                # 官方的multiclass_nms()也更快一点。但是为了之后的深度定制。
                # 用fastnms
                if postprocess == 'fastnms':
                    boxes, scores, classes = fastnms(all_pred_boxes, all_pred_scores, resize_shape, origin_shape, conf_thresh,
                                                     nms_thresh, keep_top_k, nms_top_k, use_yolo_box)
                    return boxes, scores, classes
                if postprocess == 'multiclass_nms':
                    pred = fluid.layers.multiclass_nms(all_pred_boxes, all_pred_scores,
                                                       score_threshold=conf_thresh,
                                                       nms_top_k=nms_top_k,
                                                       keep_top_k=keep_top_k,
                                                       nms_threshold=nms_thresh,
                                                       background_label=-1)   # 对于YOLO算法，一定要设置background_label=-1，否则检测不出人。
                    return pred

        # 相当于numpy的transpose()，交换下标
        output_l = fluid.layers.transpose(output_l, perm=[0, 2, 3, 1], name='output_l')
        output_m = fluid.layers.transpose(output_m, perm=[0, 2, 3, 1], name='output_m')
        output_s = fluid.layers.transpose(output_s, perm=[0, 2, 3, 1], name='output_s')
        return output_l, output_m, output_s




