#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-10 10:20:27
#   Description : paddlepaddle_yolov4
#
# ================================================================
import datetime
import json
from collections import deque
import paddle.fluid as fluid
import paddle.fluid.layers as P
import sys
import time
import shutil
import math
import copy
import random
import threading
import numpy as np
import os

from config import *
from model.head import YOLACTHead
from model.resnet import Resnet50Vd
from model.yolact import YOLACT
from tools.cocotools import get_classes, catid2clsid, clsid2catid
from model.decode_np import Decode
from tools.cocotools import eval
from tools.data_process import data_clean, get_samples
from tools.transform import *
from pycocotools.coco import COCO

import logging

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def bbox_ciou(boxes1, boxes2):
    '''
    计算ciou = iou - p2/c2 - av
    :param boxes1: (8, 13, 13, 3, 4)   pred_xywh
    :param boxes2: (8, 13, 13, 3, 4)   label_xywh
    :return:
    '''

    # 变成左上角坐标、右下角坐标
    boxes1_x0y0x1y1 = P.concat([boxes1[:, :, :, :, :2] - boxes1[:, :, :, :, 2:] * 0.5,
                                boxes1[:, :, :, :, :2] + boxes1[:, :, :, :, 2:] * 0.5], axis=-1)
    boxes2_x0y0x1y1 = P.concat([boxes2[:, :, :, :, :2] - boxes2[:, :, :, :, 2:] * 0.5,
                                boxes2[:, :, :, :, :2] + boxes2[:, :, :, :, 2:] * 0.5], axis=-1)
    '''
    逐个位置比较boxes1_x0y0x1y1[..., :2]和boxes1_x0y0x1y1[..., 2:]，即逐个位置比较[x0, y0]和[x1, y1]，小的留下。
    比如留下了[x0, y0]
    这一步是为了避免一开始w h 是负数，导致x0y0成了右下角坐标，x1y1成了左上角坐标。
    '''
    boxes1_x0y0x1y1 = P.concat([P.elementwise_min(boxes1_x0y0x1y1[:, :, :, :, :2], boxes1_x0y0x1y1[:, :, :, :, 2:]),
                                P.elementwise_max(boxes1_x0y0x1y1[:, :, :, :, :2], boxes1_x0y0x1y1[:, :, :, :, 2:])],
                               axis=-1)
    boxes2_x0y0x1y1 = P.concat([P.elementwise_min(boxes2_x0y0x1y1[:, :, :, :, :2], boxes2_x0y0x1y1[:, :, :, :, 2:]),
                                P.elementwise_max(boxes2_x0y0x1y1[:, :, :, :, :2], boxes2_x0y0x1y1[:, :, :, :, 2:])],
                               axis=-1)

    # 两个矩形的面积
    boxes1_area = (boxes1_x0y0x1y1[:, :, :, :, 2] - boxes1_x0y0x1y1[:, :, :, :, 0]) * (
                boxes1_x0y0x1y1[:, :, :, :, 3] - boxes1_x0y0x1y1[:, :, :, :, 1])
    boxes2_area = (boxes2_x0y0x1y1[:, :, :, :, 2] - boxes2_x0y0x1y1[:, :, :, :, 0]) * (
                boxes2_x0y0x1y1[:, :, :, :, 3] - boxes2_x0y0x1y1[:, :, :, :, 1])

    # 相交矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    left_up = P.elementwise_max(boxes1_x0y0x1y1[:, :, :, :, :2], boxes2_x0y0x1y1[:, :, :, :, :2])
    right_down = P.elementwise_min(boxes1_x0y0x1y1[:, :, :, :, 2:], boxes2_x0y0x1y1[:, :, :, :, 2:])

    # 相交矩形的面积inter_area。iou
    inter_section = P.relu(right_down - left_up)
    inter_area = inter_section[:, :, :, :, 0] * inter_section[:, :, :, :, 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + 1e-9)

    # 包围矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    enclose_left_up = P.elementwise_min(boxes1_x0y0x1y1[:, :, :, :, :2], boxes2_x0y0x1y1[:, :, :, :, :2])
    enclose_right_down = P.elementwise_max(boxes1_x0y0x1y1[:, :, :, :, 2:], boxes2_x0y0x1y1[:, :, :, :, 2:])

    # 包围矩形的对角线的平方
    enclose_wh = enclose_right_down - enclose_left_up
    enclose_c2 = P.pow(enclose_wh[:, :, :, :, 0], 2) + P.pow(enclose_wh[:, :, :, :, 1], 2)

    # 两矩形中心点距离的平方
    p2 = P.pow(boxes1[:, :, :, :, 0] - boxes2[:, :, :, :, 0], 2) + P.pow(boxes1[:, :, :, :, 1] - boxes2[:, :, :, :, 1],
                                                                         2)

    # 增加av。
    atan1 = P.atan(boxes1[:, :, :, :, 2] / (boxes1[:, :, :, :, 3] + 1e-9))
    atan2 = P.atan(boxes2[:, :, :, :, 2] / (boxes2[:, :, :, :, 3] + 1e-9))
    v = 4.0 * P.pow(atan1 - atan2, 2) / (math.pi ** 2)
    a = v / (1 - iou + v)

    ciou = iou - 1.0 * p2 / enclose_c2 - 1.0 * a * v
    return ciou


def bbox_iou(boxes1, boxes2):
    '''
    预测框          boxes1 (?, grid_h, grid_w, 3,   1, 4)，神经网络的输出(tx, ty, tw, th)经过了后处理求得的(bx, by, bw, bh)
    图片中所有的gt  boxes2 (?,      1,      1, 1, 70, 4)
    '''
    boxes1_area = boxes1[:, :, :, :, :, 2] * boxes1[:, :, :, :, :, 3]  # 所有格子的3个预测框的面积
    boxes2_area = boxes2[:, :, :, :, :, 2] * boxes2[:, :, :, :, :, 3]  # 所有ground truth的面积

    # (x, y, w, h)变成(x0, y0, x1, y1)
    boxes1 = P.concat([boxes1[:, :, :, :, :, :2] - boxes1[:, :, :, :, :, 2:] * 0.5,
                       boxes1[:, :, :, :, :, :2] + boxes1[:, :, :, :, :, 2:] * 0.5], axis=-1)
    boxes2 = P.concat([boxes2[:, :, :, :, :, :2] - boxes2[:, :, :, :, :, 2:] * 0.5,
                       boxes2[:, :, :, :, :, :2] + boxes2[:, :, :, :, :, 2:] * 0.5], axis=-1)

    # 所有格子的3个预测框 分别 和  70个ground truth  计算iou。 所以left_up和right_down的shape = (?, grid_h, grid_w, 3, 70, 2)
    expand_boxes1 = P.expand(boxes1, [1, 1, 1, 1, P.shape(boxes2)[4], 1])  # 不同于pytorch和tf，boxes1和boxes2都要扩展为相同shape
    expand_boxes2 = P.expand(boxes2, [1, P.shape(boxes1)[1], P.shape(boxes1)[2], P.shape(boxes1)[3], 1,
                                      1])  # 不同于pytorch和tf，boxes1和boxes2都要扩展为相同shape
    left_up = P.elementwise_max(expand_boxes1[:, :, :, :, :, :2], expand_boxes2[:, :, :, :, :, :2])  # 相交矩形的左上角坐标
    right_down = P.elementwise_min(expand_boxes1[:, :, :, :, :, 2:], expand_boxes2[:, :, :, :, :, 2:])  # 相交矩形的右下角坐标

    inter_section = P.relu(right_down - left_up)  # 相交矩形的w和h，是负数时取0  (?, grid_h, grid_w, 3, 70, 2)
    inter_area = inter_section[:, :, :, :, :, 0] * inter_section[:, :, :, :, :, 1]  # 相交矩形的面积   (?, grid_h, grid_w, 3, 70)
    expand_boxes1_area = P.expand(boxes1_area, [1, 1, 1, 1, P.shape(boxes2)[4]])
    expand_boxes2_area = P.expand(boxes2_area, [1, P.shape(expand_boxes1_area)[1], P.shape(expand_boxes1_area)[2],
                                                P.shape(expand_boxes1_area)[3], 1])
    union_area = expand_boxes1_area + expand_boxes2_area - inter_area  # union_area                (?, grid_h, grid_w, 3, 70)
    iou = 1.0 * inter_area / union_area  # iou                       (?, grid_h, grid_w, 3, 70)

    return iou


def sanitize_coordinates(_x1, _x2, img_size, padding: int = 0, cast: bool = True):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.

    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    x1 = fluid.layers.elementwise_min(_x1, _x2)
    x2 = fluid.layers.elementwise_max(_x1, _x2)
    x1 = fluid.layers.relu(x1 - padding)  # 下限是0
    img_size2 = fluid.layers.expand(img_size, (fluid.layers.shape(x2)[0],))
    img_size2 = fluid.layers.cast(img_size2, 'float32')
    x2 = img_size2 - fluid.layers.relu(img_size2 - (x2 + padding))  # 上限是img_size
    if cast:
        x1 = fluid.layers.cast(x1, 'int32')
        x2 = fluid.layers.cast(x2, 'int32')
    return x1, x2

def crop(masks, boxes, padding: int = 1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks    。n是正样本数量
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = fluid.layers.shape(masks)[0], fluid.layers.shape(masks)[1], fluid.layers.shape(masks)[2]
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=False)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=False)

    rows = fluid.layers.range(0, w, 1, 'int32')
    cols = fluid.layers.range(0, h, 1, 'int32')
    rows = fluid.layers.expand(fluid.layers.reshape(rows, (1, -1, 1)), [h, 1, n])
    cols = fluid.layers.expand(fluid.layers.reshape(cols, (-1, 1, 1)), [1, w, n])
    rows.stop_gradient = True
    cols.stop_gradient = True

    x1 = fluid.layers.reshape(x1, (1, 1, -1))
    x2 = fluid.layers.reshape(x2, (1, 1, -1))
    y1 = fluid.layers.reshape(y1, (1, 1, -1))
    y2 = fluid.layers.reshape(y2, (1, 1, -1))
    x1.stop_gradient = True
    x2.stop_gradient = True
    y1.stop_gradient = True
    y2.stop_gradient = True
    masks_left = fluid.layers.cast(rows >= fluid.layers.expand(x1, [h, w, 1]), 'float32')
    masks_right = fluid.layers.cast(rows < fluid.layers.expand(x2, [h, w, 1]), 'float32')
    masks_up = fluid.layers.cast(cols >= fluid.layers.expand(y1, [h, w, 1]), 'float32')
    masks_down = fluid.layers.cast(cols < fluid.layers.expand(y2, [h, w, 1]), 'float32')

    crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * crop_mask

def loss_layer(conv, pred, label, bboxes, gt_label, stride, mcf, proto_out, gt_mask, batch_label_idx, num_class, iou_loss_thresh, batch_size_int):
    conv_shape = P.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = stride * output_size

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]
    pred_prob = pred[:, :, :, :, 5:]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    ciou = P.reshape(bbox_ciou(pred_xywh, label_xywh),
                     (batch_size, output_size, output_size, 3, 1))  # (8, 13, 13, 3, 1)
    input_size = P.cast(input_size, dtype='float32')

    # 每个预测框xxxiou_loss的权重 = 2 - (ground truth的面积/图片面积)
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    ciou_loss = respond_bbox * bbox_loss_scale * (1 - ciou)  # 1. respond_bbox作为mask，有物体才计算xxxiou_loss

    # 2. respond_bbox作为mask，有物体才计算类别loss
    prob_pos_loss = label_prob * (0 - P.log(pred_prob + 1e-9))  # 二值交叉熵，tf中也是加了极小的常数防止nan
    prob_neg_loss = (1 - label_prob) * (0 - P.log(1 - pred_prob + 1e-9))  # 二值交叉熵，tf中也是加了极小的常数防止nan
    prob_mask = P.expand(respond_bbox, [1, 1, 1, 1, num_class])
    prob_loss = prob_mask * (prob_pos_loss + prob_neg_loss)

    # 3. xxxiou_loss和类别loss比较简单。重要的是conf_loss，是一个二值交叉熵损失
    # 分两步：第一步是确定 grid_h * grid_w * 3 个预测框 哪些作为反例；第二步是计算二值交叉熵损失。
    expand_pred_xywh = P.reshape(pred_xywh, (batch_size, output_size, output_size, 3, 1, 4))  # 扩展为(?, grid_h, grid_w, 3,   1, 4)
    expand_bboxes = P.reshape(bboxes, (batch_size, 1, 1, 1, P.shape(bboxes)[1], 4))  # 扩展为(?,      1,      1, 1, 70, 4)
    iou = bbox_iou(expand_pred_xywh, expand_bboxes)  # 所有格子的3个预测框 分别 和  70个ground truth  计算iou。   (?, grid_h, grid_w, 3, 70)
    max_iou, max_iou_indices = P.topk(iou, k=1)  # 与70个ground truth的iou中，保留最大那个iou。  (?, grid_h, grid_w, 3, 1)

    # respond_bgd代表  这个分支输出的 grid_h * grid_w * 3 个预测框是否是 反例（背景）
    # label有物体，respond_bgd是0。 没物体的话：如果和某个gt(共70个)的iou超过iou_loss_thresh，respond_bgd是0；如果和所有gt(最多70个)的iou都小于iou_loss_thresh，respond_bgd是1。
    # respond_bgd是0代表有物体，不是反例（或者是忽略框）；  权重respond_bgd是1代表没有物体，是反例。
    # 有趣的是，模型训练时由于不断更新，对于同一张图片，两次预测的 grid_h * grid_w * 3 个预测框（对于这个分支输出）  是不同的。用的是这些预测框来与gt计算iou来确定哪些预测框是反例。
    # 而不是用固定大小（不固定位置）的先验框。
    respond_bgd = (1.0 - respond_bbox) * P.cast(max_iou < iou_loss_thresh, 'float32')

    # 二值交叉熵损失
    pos_loss = respond_bbox * (0 - P.log(pred_conf + 1e-9))
    neg_loss = respond_bgd  * (0 - P.log(1 - pred_conf + 1e-9))

    conf_loss = pos_loss + neg_loss
    # 回顾respond_bgd，某个预测框和某个gt的iou超过iou_loss_thresh，不被当作是反例。在参与“预测的置信位 和 真实置信位 的 二值交叉熵”时，这个框也可能不是正例(label里没标这个框是1的话)。这个框有可能不参与置信度loss的计算。
    # 这种框一般是gt框附近的框，或者是gt框所在格子的另外两个框。它既不是正例也不是反例不参与置信度loss的计算。（论文里称之为ignore）


    # ===================== 4.Mask Loss =====================
    mask_alpha = 6.125
    # mask_alpha = 1.0
    mask_h = fluid.layers.shape(proto_out)[1]
    mask_w = fluid.layers.shape(proto_out)[2]

    proto_outs = fluid.layers.split(proto_out, batch_size_int, dim=0)  # 里面每个元素形状[1, -1, -1, 32]
    mcfs = fluid.layers.split(mcf, batch_size_int, dim=0)    # 里面每个元素形状[1, -1, -1, 3, 32]
    tobjs = fluid.layers.split(respond_bbox, batch_size_int, dim=0)  # 里面每个元素形状[1, -1, -1, 3, 1]
    gts = fluid.layers.split(bboxes, batch_size_int, dim=0)  # 每个样本（图片）切分成一个张量组成list。里面每个元素形状[1, 70, 4]
    lbs = fluid.layers.split(gt_label, batch_size_int, dim=0)  # 每个样本（图片）切分成一个张量组成list。里面每个元素形状[1, 70]
    gtms = fluid.layers.split(gt_mask, batch_size_int, dim=0)  # 里面每个元素形状[1, 70, s4, s4]
    target_pidx_midx_s = fluid.layers.split(batch_label_idx, batch_size_int, dim=0)  # 里面每个元素形状[1, 70, 4]
    losses_mask = []
    for_maskiou_loss = []
    for pred_proto, pred_mcf, gt_obj, gt, lb, gtm, pmidx in zip(proto_outs, mcfs, tobjs, gts, lbs, gtms,
                                                                target_pidx_midx_s):  # 遍历这一批的每张图片
        pred_proto = fluid.layers.squeeze(pred_proto, axes=[0])  # [-1, -1, 32]，即去掉第0维。
        pred_mcf = fluid.layers.squeeze(pred_mcf, axes=[0])  # [-1, -1, 3, 32]，即去掉第0维。
        gt_obj = fluid.layers.squeeze(gt_obj, axes=[0])       # [-1, -1, 3, 1]，即去掉第0维。
        gt_obj = fluid.layers.squeeze(gt_obj, axes=[-1])      # [-1, -1, 3]，   即去掉最后一维。
        gt = fluid.layers.squeeze(gt, axes=[0])    # [70, 4]，即去掉第0维。
        lb = fluid.layers.squeeze(lb, axes=[0])    # [70, ]，即去掉第0维。
        gtm = fluid.layers.squeeze(gtm, axes=[0])    # [70, -1, -1]，即去掉第0维。
        pmidx = fluid.layers.squeeze(pmidx, axes=[0])   # [70, 4]，即去掉第0维。

        # 去掉-1的坐标值
        # 准备数据时有一个小细节。如果这一输出层没有gt，一定要分配一个坐标使得layers.gather()函数成功。
        # 没有坐标的话（这里的keep是[None]时），gather()函数会出现难以解决的错误。
        idx_sum = fluid.layers.reduce_sum(pmidx, dim=1)
        keep = fluid.layers.where(idx_sum > -1)
        pmidx = fluid.layers.gather(pmidx, keep)
        p_idx = pmidx[:, :3]
        m_idx = pmidx[:, 3:]
        p_idx.stop_gradient = True
        m_idx.stop_gradient = True

        _pos_coef = fluid.layers.gather_nd(pred_mcf, p_idx)    # [-1, 32]
        _gt_obj = fluid.layers.gather_nd(gt_obj, p_idx)        # [-1, ]
        mask_t = fluid.layers.gather(gtm, m_idx)         # [?, -1, -1]
        gt_t = fluid.layers.gather(gt, m_idx)            # [?, 4]
        lb_t = fluid.layers.gather(lb, m_idx)            # [?, ]

        # shape: [s4, s4, ?]  =  原型*系数转置
        _pos_coef = fluid.layers.tanh(_pos_coef)
        pred_masks = fluid.layers.matmul(pred_proto, _pos_coef, transpose_y=True)
        pred_masks = fluid.layers.sigmoid(pred_masks)   # sigmoid激活

        gt_x1y1x2y2 = fluid.layers.concat([gt_t[:, :2] - gt_t[:, 2:] * 0.5,
                                           gt_t[:, :2] + gt_t[:, 2:] * 0.5], axis=-1)
        pred_masks = crop(pred_masks, gt_x1y1x2y2)   # 不计较超出gt外的损失
        pred_masks = fluid.layers.transpose(pred_masks, perm=[2, 0, 1])  # [?, s4, s4]


        # _gt_obj     (-1, )         是否是真正的正例，要么是个单独的0，要么全是1
        # pred_masks  (-1, s4, s4)   ?个正例预测的掩码
        # mask_t      (-1, s4, s4)   ?个正例的真实mask
        # lb_t        (-1, )         ?个正例的真实cid
        for_maskiou_loss.append([_gt_obj, pred_masks, mask_t, lb_t])

        masks_pos_loss = mask_t * (0 - fluid.layers.log(pred_masks + 1e-9))            # 二值交叉熵，加了极小的常数防止nan
        masks_neg_loss = (1 - mask_t) * (0 - fluid.layers.log(1 - pred_masks + 1e-9))  # 二值交叉熵，加了极小的常数防止nan
        pre_loss = (masks_pos_loss + masks_neg_loss)
        pre_loss = fluid.layers.reduce_sum(pre_loss, dim=[1, 2])

        # gt面积越小，对应mask损失权重越大
        gt_box_width = gt_t[:, 2]
        gt_box_height = gt_t[:, 3]
        pre_loss = pre_loss / (gt_box_width * gt_box_height + 1e-9)   # 有的图片一个gt都没有，加了极小的常数防止nan

        # 这一输出层没有gt时，过滤掉巧妙添加的那个坐标（_gt_obj是0）。这一输出层有gt时，_gt_obj全是1。
        # XXX this is hackish, but seems to be the least contrived way...
        pre_loss = _gt_obj * pre_loss
        pre_loss = fluid.layers.reduce_sum(pre_loss)   # 每个样本（图片）所有物体的loss_mask求和
        pre_loss = mask_alpha * pre_loss / mask_h / mask_w
        losses_mask.append(pre_loss)
    loss_mask = fluid.layers.stack(losses_mask, axis=0)  # [batch_size, 1]

    ciou_loss = P.reduce_sum(ciou_loss) / batch_size
    conf_loss = P.reduce_sum(conf_loss) / batch_size
    prob_loss = P.reduce_sum(prob_loss) / batch_size
    mask_loss = P.reduce_sum(loss_mask) / batch_size

    return ciou_loss, conf_loss, prob_loss, mask_loss


def decode(conv_output, anchors, stride, num_class, mcf):
    conv_shape = P.shape(conv_output)
    batch_size = conv_shape[0]
    n_grid = conv_shape[1]
    anchor_per_scale = len(anchors)

    conv_output = P.reshape(conv_output, (batch_size, n_grid, n_grid, anchor_per_scale, 5 + num_class))
    mcf = P.reshape(mcf, (batch_size, n_grid, n_grid, anchor_per_scale, -1))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5:]

    rows = P.range(0, n_grid, 1, 'float32')
    cols = P.range(0, n_grid, 1, 'float32')
    rows = P.expand(P.reshape(rows, (1, -1, 1)), [n_grid, 1, 1])
    cols = P.expand(P.reshape(cols, (-1, 1, 1)), [1, n_grid, 1])
    offset = P.concat([rows, cols], axis=-1)
    offset = P.reshape(offset, (1, n_grid, n_grid, 1, 2))
    offset = P.expand(offset, [batch_size, 1, 1, anchor_per_scale, 1])

    pred_xy = (P.sigmoid(conv_raw_dxdy) + offset) * stride
    anchor_t = fluid.layers.assign(np.copy(anchors).astype(np.float32))
    pred_wh = (P.exp(conv_raw_dwdh) * anchor_t)
    pred_xywh = P.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = P.sigmoid(conv_raw_conf)
    pred_prob = P.sigmoid(conv_raw_prob)

    return P.concat([pred_xywh, pred_conf, pred_prob], axis=-1), mcf


def yolact_loss(args, num_classes, iou_loss_thresh, anchors, batch_size_int):
    conv_lbbox = args[0]  # (?, ?, ?, 3*(num_classes+5))
    conv_mbbox = args[1]  # (?, ?, ?, 3*(num_classes+5))
    conv_sbbox = args[2]  # (?, ?, ?, 3*(num_classes+5))
    label_sbbox = args[3]  # (?, ?, ?, 3, num_classes+5)
    label_mbbox = args[4]  # (?, ?, ?, 3, num_classes+5)
    label_lbbox = args[5]  # (?, ?, ?, 3, num_classes+5)
    gt_bboxes = args[6]    # (?, 70, 4)
    gt_classes = args[7]   # (?, 70, )
    gt_segment = args[8]   # (?, 80, s8, s8)
    gt_masks = args[9]     # (?, 70, s4, s4)
    batch_label_idx_lbbox = args[10]  # (?, 70, 4)
    batch_label_idx_mbbox = args[11]  # (?, 70, 4)
    batch_label_idx_sbbox = args[12]  # (?, 70, 4)
    mcf_l = args[13]         # (?, ?, ?, 3*32)
    mcf_m = args[14]         # (?, ?, ?, 3*32)
    mcf_s = args[15]         # (?, ?, ?, 3*32)
    proto_out = args[16]     # (?, s4, s4, 32)
    segm = args[17]          # (?, 80, s8, s8)
    pred_sbbox, mcf_s = decode(conv_sbbox, anchors[0], 8, num_classes, mcf_s)
    pred_mbbox, mcf_m = decode(conv_mbbox, anchors[1], 16, num_classes, mcf_m)
    pred_lbbox, mcf_l = decode(conv_lbbox, anchors[2], 32, num_classes, mcf_l)
    sbbox_ciou_loss, sbbox_conf_loss, sbbox_prob_loss, sbbox_mask_loss = loss_layer(conv_sbbox, pred_sbbox, label_sbbox, gt_bboxes, gt_classes, 8,
                                                                   mcf_s, proto_out, gt_masks, batch_label_idx_sbbox, num_classes, iou_loss_thresh, batch_size_int)
    mbbox_ciou_loss, mbbox_conf_loss, mbbox_prob_loss, mbbox_mask_loss = loss_layer(conv_mbbox, pred_mbbox, label_mbbox, gt_bboxes, gt_classes, 16,
                                                                   mcf_m, proto_out, gt_masks, batch_label_idx_mbbox, num_classes, iou_loss_thresh, batch_size_int)
    lbbox_ciou_loss, lbbox_conf_loss, lbbox_prob_loss, lbbox_mask_loss = loss_layer(conv_lbbox, pred_lbbox, label_lbbox, gt_bboxes, gt_classes, 32,
                                                                   mcf_l, proto_out, gt_masks, batch_label_idx_lbbox, num_classes, iou_loss_thresh, batch_size_int)

    ciou_loss = sbbox_ciou_loss + mbbox_ciou_loss + lbbox_ciou_loss
    conf_loss = sbbox_conf_loss + mbbox_conf_loss + lbbox_conf_loss
    prob_loss = sbbox_prob_loss + mbbox_prob_loss + lbbox_prob_loss
    mask_loss = sbbox_mask_loss + mbbox_mask_loss + lbbox_mask_loss
    return [ciou_loss, conf_loss, prob_loss, mask_loss]


def multi_thread_op(i, samples, decodeImage, context, train_dataset,
                    photometricDistort, randomCrop, randomFlipImage, normalizeBox, padBox, bboxXYXY2XYWH):
    samples[i] = decodeImage(samples[i], context, train_dataset)
    samples[i] = photometricDistort(samples[i], context)
    samples[i] = randomCrop(samples[i], context)
    samples[i] = randomFlipImage(samples[i], context)
    samples[i] = normalizeBox(samples[i], context)
    samples[i] = padBox(samples[i], context)
    samples[i] = bboxXYXY2XYWH(samples[i], context)

def clear_model(save_dir):
    path_dir = os.listdir(save_dir)
    it_ids = []
    for name in path_dir:
        sss = name.split('.')
        if sss[0] == '':
            continue
        if sss[0] == 'best_model':   # 不会删除最优模型
            it_id = 9999999999
        else:
            it_id = int(sss[0])
        it_ids.append(it_id)
    if len(it_ids) >= 11 * 3:
        it_id = min(it_ids)
        pdopt_path = '%s/%d.pdopt' % (save_dir, it_id)
        pdmodel_path = '%s/%d.pdmodel' % (save_dir, it_id)
        pdparams_path = '%s/%d.pdparams' % (save_dir, it_id)
        if os.path.exists(pdopt_path):
            os.remove(pdopt_path)
        if os.path.exists(pdmodel_path):
            os.remove(pdmodel_path)
        if os.path.exists(pdparams_path):
            os.remove(pdparams_path)

if __name__ == '__main__':
    use_gpu = False
    use_gpu = True

    # 选择配置
    cfg = YOLACT_Config_1()


    algorithm = cfg.algorithm

    class_names = get_classes(cfg.classes_path)
    num_classes = len(class_names)
    _anchors = copy.deepcopy(cfg.anchors)
    num_anchors = len(cfg.anchor_masks[0])  # 每个输出层有几个先验框
    _anchors = np.array(_anchors)
    _anchors = np.reshape(_anchors, (-1, num_anchors, 2))
    _anchors = _anchors.astype(np.float32)

    # 步id，无需设置，会自动读。
    iter_id = 0
    batch_size = cfg.batch_size

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            # 多尺度训练
            inputs = P.data(name='input_1', shape=[-1, 3, -1, -1], append_batch_size=False, dtype='float32')
            backbone = Resnet50Vd()
            head = YOLACTHead()
            yolact = YOLACT(backbone, head)
            output_l, output_m, output_s, mcf_l, mcf_m, mcf_s, proto_out, segm = yolact(inputs)

            # 建立损失函数
            # batch_image, batch_label, batch_gt_bbox, batch_gt_segment, batch_gt_mask, batch_label_idx
            label_sbbox = P.data(name='input_2', shape=[-1, -1, -1, 3, (num_classes + 5)], append_batch_size=False, dtype='float32')
            label_mbbox = P.data(name='input_3', shape=[-1, -1, -1, 3, (num_classes + 5)], append_batch_size=False, dtype='float32')
            label_lbbox = P.data(name='input_4', shape=[-1, -1, -1, 3, (num_classes + 5)], append_batch_size=False, dtype='float32')
            gt_bboxes = P.data(name='input_5', shape=[cfg.num_max_boxes, 4], dtype='float32')
            gt_classes = P.data(name='gt_classes', shape=[cfg.num_max_boxes, ], dtype='int32')
            gt_segment = P.data(name='gt_segment', shape=[-1, num_classes, -1, -1], append_batch_size=False, dtype='float32')
            gt_masks = P.data(name='gt_masks', shape=[-1, -1, -1, -1], append_batch_size=False, dtype='float32')
            batch_label_idx_lbbox = P.data(name='idx_lbbox', shape=[-1, -1, 4], append_batch_size=False, dtype='int32')
            batch_label_idx_mbbox = P.data(name='idx_mbbox', shape=[-1, -1, 4], append_batch_size=False, dtype='int32')
            batch_label_idx_sbbox = P.data(name='idx_sbbox', shape=[-1, -1, 4], append_batch_size=False, dtype='int32')
            args = [output_l, output_m, output_s, label_sbbox, label_mbbox, label_lbbox, gt_bboxes, gt_classes, gt_segment, gt_masks,
                    batch_label_idx_lbbox, batch_label_idx_mbbox, batch_label_idx_sbbox, mcf_l, mcf_m, mcf_s, proto_out, segm]
            ciou_loss, conf_loss, prob_loss, mask_loss = yolact_loss(args, num_classes, cfg.iou_loss_thresh, _anchors, batch_size)
            loss = ciou_loss + conf_loss + prob_loss + mask_loss

            optimizer = fluid.optimizer.Adam(learning_rate=cfg.lr)
            optimizer.minimize(loss)


    # eval_prog = fluid.Program()
    # with fluid.program_guard(eval_prog, startup_prog):
    #     with fluid.unique_name.guard():
    #         # 多尺度训练
    #         inputs = P.data(name='input_1', shape=[-1, 3, -1, -1], append_batch_size=False, dtype='float32')
    #         backbone = Resnet50Vd()
    #         head = YOLACTHead(keep_prob=1.0)   # 一定要设置keep_prob=1.0, 为了得到一致的推理结果
    #         yolact = YOLACT(backbone, head)
    #         outputs, mcf_outputs, proto_out, segm = yolact(inputs)
    #         eval_fetch_list = [outputs[0], outputs[1], outputs[2]]
    # eval_prog = eval_prog.clone(for_test=True)

    # 参数随机初始化
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    # compiled_eval_prog = fluid.compiler.CompiledProgram(eval_prog)
    # _decode = Decode(algorithm, cfg.anchors, cfg.conf_thresh, cfg.nms_thresh, cfg.input_shape, exe, compiled_eval_prog, class_names)

    if cfg.pattern == 1:
        fluid.load(train_prog, cfg.model_path, executor=exe)
        strs = cfg.model_path.split('weights/')
        if len(strs) == 2:
            iter_id = int(strs[1])

    # 种类id
    _catid2clsid = copy.deepcopy(catid2clsid)
    _clsid2catid = copy.deepcopy(clsid2catid)
    if num_classes != 80:  # 如果不是COCO数据集，而是自定义数据集
        _catid2clsid = {}
        _clsid2catid = {}
        for k in range(num_classes):
            _catid2clsid[k] = k
            _clsid2catid[k] = k
    # 训练集
    train_dataset = COCO(cfg.train_path)
    train_img_ids = train_dataset.getImgIds()
    train_records = data_clean(train_dataset, train_img_ids, _catid2clsid, cfg.train_pre_path)
    num_train = len(train_records)
    train_indexes = [i for i in range(num_train)]
    # 验证集
    with open(cfg.val_path, 'r', encoding='utf-8') as f2:
        for line in f2:
            line = line.strip()
            dataset = json.loads(line)
            val_images = dataset['images']
    with_mixup = cfg.with_mixup
    context = cfg.context
    # 预处理
    # sample_transforms
    decodeImage = DecodeImage(with_mixup=with_mixup, process_mask=True)   # 对图片解码。最开始的一步。
    photometricDistort = PhotometricDistort()   # 颜色扭曲
    randomCrop = RandomCrop(process_mask=True)                   # 随机裁剪
    randomFlipImage = RandomFlipImage(process_mask=True)         # 随机翻转
    normalizeBox = NormalizeBox()               # 将物体的左上角坐标、右下角坐标中的横坐标/图片宽、纵坐标/图片高 以归一化坐标。
    padBox = PadBox(cfg.num_max_boxes, process_mask=True)          # 如果gt_bboxes的数量少于num_max_boxes，那么填充坐标是0的bboxes以凑够num_max_boxes。
    bboxXYXY2XYWH = BboxXYXY2XYWH()             # sample['gt_bbox']被改写为cx_cy_w_h格式。
    # batch_transforms
    randomShape = RandomShape(process_mask=True)                 # 多尺度训练。随机选一个尺度。也随机选一种插值方式。
    normalizeImage = NormalizeImage(algorithm, is_scale=True, is_channel_first=False)   # 图片归一化。直接除以255。
    gt2YolactTarget = Gt2YolactTarget(cfg.anchors,
                                  cfg.anchor_masks,
                                  cfg.downsample_ratios,
                                  num_classes)  # 填写target0、target1、target2张量。

    # 保存模型的目录
    if not os.path.exists('./weights'): os.mkdir('./weights')

    time_stat = deque(maxlen=20)
    start_time = time.time()
    end_time = time.time()

    # 一轮的步数。丢弃最后几个样本。
    train_steps = num_train // batch_size
    best_ap_list = [0.0, 0]  # [map, iter]
    while True:  # 无限个epoch
        # 每个epoch之前洗乱
        np.random.shuffle(train_indexes)
        for step in range(train_steps):
            iter_id += 1

            # 估计剩余时间
            start_time = end_time
            end_time = time.time()
            time_stat.append(end_time - start_time)
            time_cost = np.mean(time_stat)
            eta_sec = (cfg.max_iters - iter_id) * time_cost
            eta = str(datetime.timedelta(seconds=int(eta_sec)))

            # ==================== train ====================
            samples = get_samples(train_records, train_indexes, step, batch_size, with_mixup)
            # sample_transforms用多线程
            threads = []
            for i in range(batch_size):
                t = threading.Thread(target=multi_thread_op,
                                     args=(i, samples, decodeImage, context, train_dataset,
                                           photometricDistort, randomCrop, randomFlipImage, normalizeBox, padBox,
                                           bboxXYXY2XYWH))
                threads.append(t)
                t.start()
            # 等待所有线程任务结束。
            for t in threads:
                t.join()

            # 看掩码图片
            # for sample in samples:
            #     im = sample['image']
            #     gt_bbox = sample['gt_bbox']
            #     gt_mask = sample['gt_mask']  # HWM，M是最大正样本数量，是50
            #     gt_class = sample['gt_class']
            #     gt_score = sample['gt_score']
            #     im_name = np.random.randint(0, 1000000)
            #     n = gt_mask.shape[2]
            #     cv2.imwrite('%d.jpg' % im_name, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            #     for i in range(n):
            #         if gt_score[i] < 0.001:
            #             continue
            #         mm = gt_mask[:, :, i]
            #         un = np.unique(mm)
            #         cv2.imwrite('%d_%.2d.jpg' % (im_name, i), mm * 255)

            # batch_transforms
            samples = randomShape(samples, context)
            samples = normalizeImage(samples, context)
            batch_image, batch_label, batch_gt_bbox, batch_gt_class, batch_gt_segment, batch_gt_mask, batch_label_idx = gt2YolactTarget(samples, context)

            # 一些变换
            batch_image = batch_image.transpose(0, 3, 1, 2)
            batch_image = batch_image.astype(np.float32)

            batch_label[2] = batch_label[2].astype(np.float32)
            batch_label[1] = batch_label[1].astype(np.float32)
            batch_label[0] = batch_label[0].astype(np.float32)

            batch_gt_bbox = batch_gt_bbox.astype(np.float32)
            batch_gt_segment = batch_gt_segment.astype(np.float32)
            batch_gt_mask = batch_gt_mask.astype(np.float32)
            # batch_label_idx = batch_label_idx.astype(np.int32)

            losses = exe.run(train_prog, feed={"input_1": batch_image, "input_2": batch_label[2],
                                               "input_3": batch_label[1], "input_4": batch_label[0],
                                               "input_5": batch_gt_bbox, "gt_classes": batch_gt_class, "gt_segment": batch_gt_segment,
                                               "gt_masks": batch_gt_mask, "idx_lbbox": batch_label_idx[0],
                                               "idx_mbbox": batch_label_idx[1], "idx_sbbox": batch_label_idx[2], },
                             fetch_list=[loss, ciou_loss, conf_loss, prob_loss, mask_loss])

            # ==================== log ====================
            if iter_id % 20 == 0:
                strs = 'Train iter: {}, all_loss: {:.6f}, ciou_loss: {:.6f}, conf_loss: {:.6f}, prob_loss: {:.6f}, mask_loss: {:.6f}, eta: {}'.format(
                    iter_id, losses[0][0], losses[1][0], losses[2][0], losses[3][0], losses[4][0], eta)
                logger.info(strs)

            # ==================== save ====================
            if iter_id % cfg.save_iter == 0:
                save_path = './weights/%d' % iter_id
                fluid.save(train_prog, save_path)
                logger.info('Save model to {}'.format(save_path))
                clear_model('weights')

            # ==================== eval ====================
            # if iter_id % cfg.eval_iter == 0:
            #     box_ap = eval(_decode, eval_fetch_list, val_images, cfg.val_pre_path, cfg.val_path, cfg.eval_batch_size,
            #                   _clsid2catid, cfg.draw_image)
            #     logger.info("box ap: %.3f" % (box_ap[0],))
            #
            #     # 以box_ap作为标准
            #     ap = box_ap
            #     if ap[0] > best_ap_list[0]:
            #         best_ap_list[0] = ap[0]
            #         best_ap_list[1] = iter_id
            #         save_path = './weights/best_model'
            #         fluid.save(train_prog, save_path)
            #         logger.info('Save model to {}'.format(save_path))
            #         clear_model('weights')
            #     logger.info("Best test ap: {}, in iter: {}".format(
            #         best_ap_list[0], best_ap_list[1]))

            # ==================== exit ====================
            if iter_id == cfg.max_iters:
                logger.info('Done.')
                exit(0)

