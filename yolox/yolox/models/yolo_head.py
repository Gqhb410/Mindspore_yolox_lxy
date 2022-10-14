
import mindspore.nn as nn
from .network_blocks import BaseConv



class YOLOXHead(nn.Cell):
    """ head  """

    def __init__(self, num_classes, scale, in_channels=None, act="silu", width=1.0):
        super(YOLOXHead, self).__init__()
        if in_channels is None:
            in_channels = [1024, 512, 256]
        self.scale = scale
        self.num_classes = num_classes
        Conv = BaseConv
        if scale == 's':
            self.stem = BaseConv(in_channels=int(in_channels[0] * width), out_channels=int(256 * width), ksize=1,
                                 stride=1, act=act)
        elif scale == 'm':
            self.stem = BaseConv(in_channels=int(in_channels[1] * width), out_channels=int(256 * width), ksize=1,
                                 stride=1, act=act)
        elif scale == 'l':
            self.stem = BaseConv(in_channels=int(in_channels[2] * width), out_channels=int(256 * width), ksize=1,
                                 stride=1, act=act)
        else:
            raise KeyError("Invalid scale value for DetectionBlock")

        self.cls_convs = nn.SequentialCell(
            [
                Conv(
                    in_channels=int(256 * width),
                    out_channels=int(256 * width),
                    ksize=3,
                    stride=1,
                    act=act,
                ),
                Conv(
                    in_channels=int(256 * width),
                    out_channels=int(256 * width),
                    ksize=3,
                    stride=1,
                    act=act,
                ),
            ]
        )
        self.reg_convs = nn.SequentialCell(
            [
                Conv(
                    in_channels=int(256 * width),
                    out_channels=int(256 * width),
                    ksize=3,
                    stride=1,
                    act=act,
                ),
                Conv(
                    in_channels=int(256 * width),
                    out_channels=int(256 * width),
                    ksize=3,
                    stride=1,
                    act=act,
                ),
            ]
        )
        self.cls_preds = nn.Conv2d(in_channels=int(256 * width), out_channels=self.num_classes, kernel_size=1, stride=1,
                                   pad_mode="pad", has_bias=True)

        self.reg_preds = nn.Conv2d(in_channels=int(256 * width), out_channels=4, kernel_size=1, stride=1,
                                   pad_mode="pad",
                                   has_bias=True)

        self.obj_preds = nn.Conv2d(in_channels=int(256 * width), out_channels=1, kernel_size=1, stride=1,
                                   pad_mode="pad",
                                   has_bias=True)

    def construct(self, x):
        """ forward """
        x = self.stem(x)
        cls_x = x
        reg_x = x
        cls_feat = self.cls_convs(cls_x)

        cls_output = self.cls_preds(cls_feat)

        reg_feat = self.reg_convs(reg_x)
        reg_output = self.reg_preds(reg_feat)
        obj_output = self.obj_preds(reg_feat)

        return cls_output, reg_output, obj_output