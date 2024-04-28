import cv2
import mmengine
import mmcv
import torch
import argparse
import tempfile

from collections import deque
from threading import Thread

from mmengine import DictAction, Config
from mmengine.utils import track_iter_progress

from mmaction.apis import (detection_inference, inference_skeleton,
                           init_recognizer, pose_inference)
from mmaction.registry import VISUALIZERS
from mmaction.utils import frame_extract, get_str_type

from mmengine.dataset import Compose, pseudo_collate

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1
EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]


config_dict = {
    'config': "configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_custom.py",
    'checkpoint': "work_dirs/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint/best_acc_top1_epoch_9.pth",
    'det-config': "demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py",
    'det-checkpoint': "http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth",
    'det-score-thr': 0.9,
    'det-cat-id': 0,
    'pose-config': "demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py",
    'pose-checkpoint': "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth",
    'label-map': "demo/boxing_demo/labels_boxing.txt",
    'predict-stepsize': 30,
    'output-fps': 30,
    'camera-id': 0,
    'average-size': 1,
    'device': 'cpu'

}


def main():
    global average_size, det_score_thr, drawing_fps, inference_fps, \
        device, model, det_config, det_ckpt, pose_config, pose_ckpt,\
        camera, data, det_cat_id
    
    average_size = config_dict['average-size']
    drawing_fps = config_dict['output-fps']
    inference_fps = config_dict['predict-stepsize']

    device = torch.device(config_dict['device'])

    model_config = Config.fromfile(config_dict['config'])

    model = init_recognizer(model_config, config_dict['checkpoint'], config_dict['device'])
    camera = cv2.VideoCapture(config_dict['camera-id'])
    data = dict(img_shape=None, modality='RGB', label=-1)

    with open(config_dict['label-map'], 'r') as f:
        label = [line.strip() for line in f]
    
    det_config = config_dict['det-config']
    det_ckpt = config_dict['det-checkpoint']
    det_score_thr = config_dict['det-score-thr']
    det_cat_id = config_dict['det-cat-id']

    pose_config = config_dict['pose-config']
    pose_ckpt = config_dict['pose-checkpoint']

    
