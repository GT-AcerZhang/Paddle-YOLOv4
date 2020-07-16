#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-10 10:20:27
#   Description : paddlepaddle_yolov4
#
# ================================================================
class YOLOv3(object):
    def __init__(self, backbone, head):
        super(YOLOv3, self).__init__()
        self.backbone = backbone
        self.head = head

    def __call__(self, x):
        body_feats = self.backbone(x)
        output_l, output_m, output_s = self.head(body_feats)
        return output_l, output_m, output_s




