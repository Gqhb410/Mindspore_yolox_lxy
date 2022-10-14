# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =======================================================================================
"""
for evaluate
"""
import os
import datetime
import argparse
from tqdm import tqdm

from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context

from yolox.models.yolox import YOLOX
from yolox.exp.build import get_exp
from yolox.exp.yolo_base import DetectionEngine
from yolox.logger import get_logger
from yolox.config import config
from datasets.yolox_dataset import create_yolox_dataset


def get_parser():
    parser = argparse.ArgumentParser(description='Yolox eval.')
    parser.add_argument('--name', type=str, default="yolox-m", help='model name, yolox-s, yolox-m, yolox-l, yolox-x')
    parser.add_argument('--val_data_dir', type=str, default="/home/psy/workplace/datasets/coco2017",
                        help='Location of data.')
    parser.add_argument('--log_dir', type=str, default='./logs/val', help='Location of logs.')
    parser.add_argument('--singel_eval', type=bool, default=False, help='eval one model or a dir')
    parser.add_argument('--val_ckpt', type=str, default='./save/yolox/yolox-m-1_1250.ckpt', help='Location of ckpt.')
    parser.add_argument('--vai_ckpt_dir', type=str, default='./save/yolox3', help='Location of ckpt dir.')
    parser.add_argument('--backbone', type=str, default="yolofpn", help='yolofpn or yolopafpn')
    parser.add_argument('--device_target', type=str, default="GPU", help='Ascend or GPU')
    parser.add_argument('--rank', type=int, default=0, help='logger related, rank id')
    parser.add_argument('--per_batch_size', type=int, default=4, help='dataset related, batch_size')
    parser.add_argument('--group_size', type=int, default=1, help='dataset related, device_num')
    return parser


def run_test():
    args = get_parser().parse_args()
    exp = get_exp(args.name)

    # 设置logger
    config.outputs_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    logger = get_logger(config.outputs_dir, args.rank)

    # 设置datasets
    data_root = os.path.join(args.val_data_dir, 'val2017')
    annFile = os.path.join(args.val_data_dir, 'annotations/instances_val2017.json')
    ds = create_yolox_dataset(image_dir=data_root, anno_path=annFile, is_training=False,
                              batch_size=args.per_batch_size, device_num=args.group_size, rank=args.rank)
    data_size = ds.get_dataset_size()

    # 设置device
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=False)
    # 定义网络
    network = YOLOX(config, backbone=args.backbone, exp=exp)
    network.set_train(False)    # 设置为eval模式

    if args.singel_eval:
        val_ckpt = args.val_ckpt
        param_dict = load_checkpoint(val_ckpt)
        load_param_into_net(network, param_dict)
        get_result(network, ds, data_size, logger)
    else:
        val_ckpt_dir = args.vai_ckpt_dir
        val_ckpt_list = os.listdir(val_ckpt_dir)
        for ckpt in val_ckpt_list:
            logger.info(f'model name:{ckpt}')
            val_ckpt = os.path.join(val_ckpt_dir, ckpt)
            param_dict = load_checkpoint(val_ckpt)
            load_param_into_net(network, param_dict)
            get_result(network, ds, data_size, logger)

def get_result(network, ds, data_size, logger):
    # init detection engine
    detection = DetectionEngine(config)
    for _, data in enumerate(tqdm(ds.create_dict_iterator(num_epochs=1), total=data_size, colour="GREEN")):
        image = data['image']
        img_info = data['image_shape']
        img_id = data['img_id']
        prediction = network(image)
        prediction = prediction.asnumpy()
        img_shape = img_info.asnumpy()
        img_id = img_id.asnumpy()
        detection.detection(prediction, img_shape, img_id)

    logger.info('Calculating mAP...')
    result_file_path = detection.evaluate_prediction()
    logger.info('result file path: %s', result_file_path)
    eval_result, _ = detection.get_eval_result()
    eval_print_str = '\n=============coco eval result=========\n' + eval_result
    logger.info(eval_print_str)


if __name__ == '__main__':
    run_test()
