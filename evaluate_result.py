# -*- coding: utf-8 -*-
# read frames of ori and result and calc score

import cv2
import numpy as np
from core.metrics import calc_psnr_and_ssim, calculate_i3d_activations, calculate_vfid, init_i3d_model


def evaluate_mp4(ori_video_path, result_video_path):
    videocap_ori = cv2.VideoCapture(ori_video_path)
    videocap_res = cv2.VideoCapture(result_video_path)
    success_ori, img_ori = videocap_ori.read()
    success_res, img_res = videocap_res.read()
    need_resize = img_ori.shape != img_res.shape
    while success_res and success_ori:
        # todo resize if shape is different
        if need_resize:
            img_ori_r = img_ori.resize((240, 432, 3))
        psnr, ssim = calc_psnr_and_ssim(img_ori, img_res)
        print('psnr={}, ssim={}'.format(psnr, ssim))
        success_ori, img_ori = videocap_ori.read()
        success_res, img_res = videocap_res.read()


ori_dir = '/home/wegatron/win-data/data/video_inpating/input/delogo_examples'
output_dir = '/home/wegatron/win-data/opensource_code/E2FGVI/results'
evaluate_mp4(ori_dir+'/test_01.mp4', output_dir+'/test_01_results.mp4');
