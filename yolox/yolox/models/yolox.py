#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.


import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P


from .yolo_fpn import YOLOFPN
from .yolo_pafpn import YOLOPAFPN
from .yolo_head import YOLOXHead


class YOLOX(nn.Cell):
    """ connect yolox backbone and head """

    def __init__(self, config, backbone="yolopafpn", exp=None):
        super(YOLOX, self).__init__()
        self.num_classes = config.num_classes
        self.attr_num = self.num_classes + 5
        self.depthwise = config.depth_wise
        self.strides = Tensor([8, 16, 32], mindspore.float32)
        self.input_size = config.input_size
        self.depth = exp.depth
        self.width = exp.width
        # network
        if backbone == "yolopafpn":
            self.backbone = YOLOPAFPN(depth=self.depth, width=self.width, input_w=self.input_size[1], input_h=self.input_size[0])
            self.head_inchannels = [1024, 512, 256]
            self.activation = "silu"
            self.width = exp.width
        else:
            self.backbone = YOLOFPN(input_w=self.input_size[1], input_h=self.input_size[0])
            self.head_inchannels = [512, 256, 128]
            self.activation = "lrelu"
            self.width = exp.width

        self.head_l = YOLOXHead(in_channels=self.head_inchannels, num_classes=self.num_classes, scale='l',
                                act=self.activation, width=self.width)
        self.head_m = YOLOXHead(in_channels=self.head_inchannels, num_classes=self.num_classes, scale='m',
                                act=self.activation, width=self.width)
        self.head_s = YOLOXHead(in_channels=self.head_inchannels, num_classes=self.num_classes, scale='s',
                                act=self.activation, width=self.width)

    def construct(self, x):
        """ forward """
        outputs = []
        x_l, x_m, x_s = self.backbone(x)
        cls_output_l, reg_output_l, obj_output_l = self.head_l(x_l)  # (bs, 80, 80, 80)(bs, 4, 80, 80)(bs, 1, 80, 80)
        cls_output_m, reg_output_m, obj_output_m = self.head_m(x_m)  # (bs, 80, 40, 40)(bs, 4, 40, 40)(bs, 1, 40, 40)
        cls_output_s, reg_output_s, obj_output_s = self.head_s(x_s)  # (bs, 80, 20, 20)(bs, 4, 20, 20)(bs, 1, 20, 20)
        if self.training:
            output_l = P.Concat(axis=1)((reg_output_l, obj_output_l, cls_output_l))  # (bs, 85, 80, 80)
            output_m = P.Concat(axis=1)((reg_output_m, obj_output_m, cls_output_m))  # (bs, 85, 40, 40)
            output_s = P.Concat(axis=1)((reg_output_s, obj_output_s, cls_output_s))  # (bs, 85, 20, 20)

            output_l = self.mapping_to_img(output_l, stride=self.strides[0])  # (bs, 6400, 85)x_c, y_c, w, h
            output_m = self.mapping_to_img(output_m, stride=self.strides[1])  # (bs, 1600, 85)x_c, y_c, w, h
            output_s = self.mapping_to_img(output_s, stride=self.strides[2])  # (bs,  400, 85)x_c, y_c, w, h

        else:

            output_l = P.Concat(axis=1)(
                (reg_output_l, P.Sigmoid()(obj_output_l), P.Sigmoid()(cls_output_l)))  # bs, 85, 80, 80

            output_m = P.Concat(axis=1)(
                (reg_output_m, P.Sigmoid()(obj_output_m), P.Sigmoid()(cls_output_m)))  # bs, 85, 40, 40

            output_s = P.Concat(axis=1)(
                (reg_output_s, P.Sigmoid()(obj_output_s), P.Sigmoid()(cls_output_s)))  # bs, 85, 20, 20
            output_l = self.mapping_to_img(output_l, stride=self.strides[0])  # (bs, 6400, 85)x_c, y_c, w, h
            output_m = self.mapping_to_img(output_m, stride=self.strides[1])  # (bs, 1600, 85)x_c, y_c, w, h
            output_s = self.mapping_to_img(output_s, stride=self.strides[2])  # (bs,  400, 85)x_c, y_c, w, h
        outputs.append(output_l)
        outputs.append(output_m)
        outputs.append(output_s)
        return P.Concat(axis=1)(outputs)  # batch_size, 8400, 85

    def mapping_to_img(self, output, stride):
        """ map to origin image scale for each fpn """
        batch_size = P.Shape()(output)[0]
        n_ch = self.attr_num
        grid_size = P.Shape()(output)[2:4]
        range_x = range(grid_size[1])
        range_y = range(grid_size[0])
        stride = P.Cast()(stride, output.dtype)
        grid_x = P.Cast()(F.tuple_to_array(range_x), output.dtype)
        grid_y = P.Cast()(F.tuple_to_array(range_y), output.dtype)
        grid_y = P.ExpandDims()(grid_y, 1)
        grid_x = P.ExpandDims()(grid_x, 0)
        yv = P.Tile()(grid_y, (1, grid_size[1]))
        xv = P.Tile()(grid_x, (grid_size[0], 1))
        grid = P.Stack(axis=2)([xv, yv])  # (80, 80, 2)
        grid = P.Reshape()(grid, (1, 1, grid_size[0], grid_size[1], 2))  # (1,1,80,80,2)
        output = P.Reshape()(output,
                             (batch_size, n_ch, grid_size[0], grid_size[1]))  # bs, 6400, 85-->(bs,85,80,80)
        output = P.Transpose()(output, (0, 2, 1, 3))  # (bs,85,80,80)-->(bs,80,85,80)
        output = P.Transpose()(output, (0, 1, 3, 2))  # (bs,80,85,80)--->(bs, 80, 80, 85)
        output = P.Reshape()(output, (batch_size, 1 * grid_size[0] * grid_size[1], -1))  # bs, 6400, 85
        grid = P.Reshape()(grid, (1, -1, 2))  # grid(1, 6400, 2)

        # reconstruct
        output_xy = output[..., :2]
        output_xy = (output_xy + grid) * stride
        output_wh = output[..., 2:4]
        output_wh = P.Exp()(output_wh) * stride
        output_other = output[..., 4:]
        output_t = P.Concat(axis=-1)([output_xy, output_wh, output_other])
        return output_t  # bs, 6400, 85           grid(1, 6400, 2)


