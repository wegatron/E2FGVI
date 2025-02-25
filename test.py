#--model e2fgvi_hq --video examples/tennis --mask examples/tennis_mask --ckpt release_model/E2FGVI-HQ-CVPR22.pth
#--model e2fgvi --video /home/wegatron/win-data/data/video_inpating/input/delogo_examples/test_01.mp4 --mask /home/wegatron/win-data/data/video_inpating/input/delogo_examples/mask/test_01_mask.png --ckpt release_model/E2FGVI-CVPR22.pth
#--model e2fgvi_hq --video /home/wegatron/win-data/data/video_inpating/input/delogo_examples/test_01.mp4 --mask /home/wegatron/win-data/data/video_inpating/input/delogo_examples/mask/test_01_mask.png --ckpt release_model/E2FGVI-HQ-CVPR22.pth
# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import numpy as np
import importlib
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
import torch

from core.utils import to_tensors
from text_det import MaskDetect
import argparse

import time

class Timer:
    def __init__(self):
        self.timers = {}

    def start(self, label):
        if label not in self.timers:
            self.timers[label] = 0
        self.timers[label] -= time.time()

    def stop(self, label):
        if label not in self.timers:
            raise ValueError(f"Timer '{label}' not started.")
        self.timers[label] += time.time()
        return self.timers[label]
    
    def print(self):
        return print(self.timers)


parser = argparse.ArgumentParser(description="E2FGVI")
parser.add_argument("-v", "--video", type=str, required=True)
parser.add_argument("-c", "--ckpt", type=str, required=True)
parser.add_argument("--model", type=str, choices=['e2fgvi', 'e2fgvi_hq'])
parser.add_argument("--step", type=int, default=10)
parser.add_argument("--num_ref", type=int, default=7)
parser.add_argument("--neighbor_stride", type=int, default=5)
parser.add_argument("--inpaint_board", type=int, default = 0)
parser.add_argument("--max_frame", type=int, default=200)

args = parser.parse_args()

# params from parser
ref_length = args.step  # ref_step
num_ref = args.num_ref
neighbor_stride = args.neighbor_stride
video_path = args.video
use_mp4 = True if video_path.endswith('.mp4') else False
ckpt = args.ckpt
model_name = args.model

# sample reference frames from the whole video
def get_ref_index(f, neighbor_ids, length):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref // 2))
        end_idx = min(length - 1, f + ref_length * (num_ref // 2))
        for i in range(start_idx, end_idx + 1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) > num_ref:
                    break
                ref_index.append(i)
    return ref_index


# read frame-wise masks
def read_mask(mpath, size):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for mp in mnames:
        m = Image.open(os.path.join(mpath, mp))
        #m = m.resize(size, Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m,
                       cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                       iterations=4)
        masks.append(Image.fromarray(m * 255))
    return masks


# read frame-wise masks
def read_mask_lst(mpath, lst):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for i in lst:        
        #for mp in mnames:
        mp = mnames[i]
        m = Image.open(os.path.join(mpath, mp))
        #m = m.resize(size, Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m,
                       cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                       iterations=4)
        masks.append(Image.fromarray(m * 255))
    return masks


def read_mask_static(mpath, n):
    masks = []
    m = Image.open(mpath)
    #m = m.resize(size, Image.NEAREST)
    m = np.array(m.convert('L'))
    m = np.array(m > 0).astype(np.uint8)
    m = cv2.dilate(m,
                   cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                   iterations=4)
    mm = Image.fromarray(m * 255)
    for i in range(0, n):
        masks.append(mm)
    return masks


def get_video_info():
    vidcap = cv2.VideoCapture(video_path)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    vidcap.release()
    return fps, min(length, args.max_frame), width, height


def read_frame_from_videos_by_index_list(index_lst):
    vname = video_path
    frames = []
    vidcap = cv2.VideoCapture(vname)
    for i in index_lst:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if not success:
            print(f'read frame {i} error!')
            exit(1)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frames.append(image)
    vidcap.release()
    return frames


# #  read frames from video
# def read_frame_from_videos():
#     vname = video_path
#     frames = []
#     vidcap = cv2.VideoCapture(vname)
#     success, image = vidcap.read()
#     count = 0
#     while success:
#         image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         frames.append(image)
#         success, image = vidcap.read()
#         count += 1
#     vidcap.release()
#     return frames

def read_video(frame_cnt):
    vidcap = cv2.VideoCapture(video_path)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    success, image = vidcap.read()
    count = 0
    frames = []
    while success:
        if count == frame_cnt:
            break        
        frames.append(image)
        success, image = vidcap.read()
        count += 1
    vidcap.release()
    return frames, size

# resize frames
def resize_frames(frames, size=None):
    if size is not None:
        frames = [f.resize(size) for f in frames]
    else:
        size = frames[0].size
    return frames, size

def crop_frames(frames, bbox):
    cropped_frames = []
    for frame in frames:
        # Extracting bbox coordinates
        x1, y1, x2, y2 = bbox
        
        # Crop the frame using the bounding box
        cropped_frame = frame.crop((x1, y1, x2, y2))
        
        # Append the cropped frame to the list of cropped frames
        cropped_frames.append(cropped_frame)
    
    return cropped_frames 


def merge_frame(frame, inpainted, bbox):
    x1, y1, x2, y2 = bbox
    frame[y1:y2, x1:x2] = inpainted
    

def main_worker():
    # set up models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = importlib.import_module('model.' + model_name)
    model = net.InpaintGenerator().to(device)
    data = torch.load(ckpt, map_location=device)
    model.load_state_dict(data)
    print(f'Loading model from: {ckpt}')
    model.eval()

    # prepare datset
    print(
        f'Loading videos and masks from: {video_path} | INPUT MP4 format: {use_mp4}'
    )

    fps, video_length, v_width, v_height = get_video_info()
    print(f'input video {v_width} {v_height}')
    bbox = np.array([0, v_height*0.6, v_width, v_height*4/5], dtype='uint32')
    #bbox = np.array([0, v_height*2/3, v_width, v_height*4/5], dtype='uint32')
    print('video_length={}'.format(video_length))

    #h, w = size[1], size[0]
    h, w = bbox[3]-bbox[1], bbox[2]-bbox[0]
    comp_frames = [None] * video_length

    v_frames, v_size = read_video(video_length)    
    mask_det = MaskDetect(args.inpaint_board != 0)
    # completing holes by e2fgvi
    print(f'Start test...')
    timer = Timer()
    for f in tqdm(range(0, video_length, neighbor_stride)):
        timer.start('preprocess')
        neighbor_ids = [
            i for i in range(max(0, f - neighbor_stride),
                             min(video_length, f + neighbor_stride + 1))
        ]
        ref_ids = get_ref_index(f, neighbor_ids, video_length)

        # read temp imgs and masks
        index_lst = neighbor_ids+ref_ids
        #selected_frames = read_frame_from_videos_by_index_list(index_lst)
        selected_frames = [Image.fromarray(cv2.cvtColor(v_frames[i], cv2.COLOR_BGR2RGB)) for i in index_lst]
        selected_frames = crop_frames(selected_frames, bbox)
        selected_imgs = to_tensors()(selected_frames).unsqueeze(0) * 2 - 1
        selected_imgs = selected_imgs.to(device)

        #selected_frames, size = resize_frames(selected_frames, size)        
        selected_frames = [np.array(f).astype(np.uint8) for f in selected_frames]

        timer.start('gen_mask')
        binary_masks = mask_det.mask(selected_frames)
        timer.stop('gen_mask')

        selected_masks = torch.from_numpy(binary_masks.astype(np.float32)).unsqueeze(0).permute(0, 1, 4, 2, 3).to(device)        
        #selected_masks = to_tensors()(selected_masks_data).unsqueeze(0).to(device)

        timer.stop('preprocess')
        timer.start('inpainting')
        with torch.no_grad():
            masked_imgs = selected_imgs * (1 - selected_masks)
            mod_size_h = 60
            mod_size_w = 108
            h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
            w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
            masked_imgs = torch.cat(
                [masked_imgs, torch.flip(masked_imgs, [3])],
                3)[:, :, :, :h + h_pad, :]
            masked_imgs = torch.cat(
                [masked_imgs, torch.flip(masked_imgs, [4])],
                4)[:, :, :, :, :w + w_pad]
            pred_imgs, _ = model(masked_imgs, len(neighbor_ids))
            pred_imgs = pred_imgs[:, :, :h, :w]
            pred_imgs = (pred_imgs + 1) / 2
            pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_imgs[i]).astype(
                    np.uint8) * binary_masks[i] + selected_frames[i] * (
                        1 - binary_masks[i])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(
                        np.float32) * 0.5 + img.astype(np.float32) * 0.5
        timer.stop('inpainting')
    # saving videos
    timer.start('save_video')
    print('Saving videos...')
    save_dir_name = 'results'
    ext_name = '_results.mp4'
    save_base_name = video_path.split('/')[-1]
    save_name = save_base_name.replace(
        '.mp4', ext_name) if use_mp4 else save_base_name + ext_name
    if not os.path.exists(save_dir_name):
        os.makedirs(save_dir_name)
    save_path = os.path.join(save_dir_name, save_name)

    # read ori video frames
    #frames, size = read_video(video_length)
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, v_size)
    for f in range(video_length):
        comp = cv2.cvtColor(comp_frames[f].astype(np.uint8), cv2.COLOR_BGR2RGB)
        merge_frame(v_frames[f], comp, bbox)
        writer.write(v_frames[f])
    writer.release()
    print(f'Finish test! The result video is saved in: {save_path}.')
    timer.stop('save_video')
    timer.print()


if __name__ == '__main__':
    main_worker()
