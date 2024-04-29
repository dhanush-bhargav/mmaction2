import cv2
import mmengine
import mmcv
import torch
import argparse
import tempfile
import numpy as np

from collections import deque
from threading import Thread

from mmengine import DictAction, Config
from mmengine.utils import track_iter_progress
from mmengine.structures import InstanceData

from mmdet.apis import init_detector, inference_detector

from mmpose.apis import init_model, inference_topdown
from mmpose.structures import merge_data_samples

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
    'config': "projects/punch_recognition/config/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_custom.py",
    'checkpoint': "work_dirs/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint/best_acc_top1_epoch_9.pth",
    'det-config': "demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py",
    'det-checkpoint': "http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth",
    'det-score-thr': 0.9,
    'det-cat-id': 0,
    'pose-config': "demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py",
    'pose-checkpoint': "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth",
    'label-map': "projects/punch_recognition/demo/labels_boxing.txt",
    'predict-stepsize': 30,
    'output-fps': 30,
    'camera-id': 0,
    'average-size': 1,
    'device': 'cpu'

}


def main():
    global average_size, det_score_thr, drawing_fps, inference_fps, \
        device, recognizer_model, det_model, pose_model,\
        camera, data, det_cat_id
    
    average_size = config_dict['average-size']
    drawing_fps = config_dict['output-fps']
    inference_fps = config_dict['predict-stepsize']

    det_cat_id = config_dict['det-cat-id']
    det_score_thr = config_dict['det-score-thr']

    device = torch.device(config_dict['device'])

    det_model = init_detector(config=config_dict['det-config'], checkpoint=config_dict['det-checkpoint'],
                              device=device)
    
    pose_model = init_model(config=config_dict['pose-config'], checkpoint=config_dict['pose-checkpoint'],
                            device=device)

    recognizer_model = init_recognizer(config_dict['config'], config_dict['checkpoint'], config_dict['device'])
    
    camera = cv2.VideoCapture(config_dict['camera-id'])
    data = dict(img_shape=None, modality='RGB', label=-1)

    with open(config_dict['label-map'], 'r') as f:
        label = [line.strip() for line in f]

    while(True):
        ret, frame = camera.read()
        cv2.imshow('frame', frame)

        det_data_sample = inference_detector(det_model, frame)
        pred_instance = det_data_sample.pred_instances.cpu().numpy()
        bboxes = pred_instance.bboxes
        scores = pred_instance.scores

        valid_idx = np.logical_and(pred_instance.labels == det_cat_id,
                                   pred_instance.scores > det_score_thr)
        bboxes = bboxes[valid_idx]
        scores = scores[valid_idx]
        torch.cuda.empty_cache()

        # pose_data_samples = inference_topdown(pose_model, frame, bboxes, bbox_format='xyxy')
        # pose_data_sample = merge_data_samples(pose_data_samples)
        # pose_data_sample.dataset_meta = pose_model.dataset_meta

        # if not hasattr(pose_data_sample, 'pred_instances'):
        #     num_keypoints = pose_model.dataset_meta['num_keypoints']
        #     pred_instances_data = dict(
        #         keypoints=np.empty(shape=(0, num_keypoints, 2)),
        #         keypoints_scores=np.empty(shape=(0, 17), dtype=np.float32),
        #         bboxes=np.empty(shape=(0, 4), dtype=np.float32),
        #         bbox_scores=np.empty(shape=(0), dtype=np.float32))
        #     pose_data_sample.pred_instances = InstanceData(
        #         **pred_instances_data)
            
        # poses = pose_data_sample.pred_instances.to_dict()

        print(bboxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    camera.release()

    cv2.destroyAllWindows()

if __name__=="__main__":
    main()


