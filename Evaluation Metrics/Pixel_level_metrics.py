from skimage.metrics import structural_similarity as SSIM
import torch
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
import numpy as np
import os
import imageio.v3 as iio
import math
import cv2
import argparse

def channel_last(img):
    if img.shape[-1] == 3:
        return img
    if len(img.shape) == 3:
        img = rearrange(img, 'c h w -> h w c')
    elif len(img.shape) == 4:
        img = rearrange(img, 'f c h w -> f h w c')
    else:
        raise ValueError(f'img shape should be 3 or 4, but got {len(img.shape)}')
    return img

def ssim_score_only(
                pred_videos: np.array,
                gt_videos: np.array,
                **kwargs
                ):
    pred_videos = channel_last(pred_videos)
    gt_videos = channel_last(gt_videos)
    scores = []
    for pred, gt in zip(pred_videos, gt_videos):
        scores.append(ssim_metric(pred, gt))
    return np.mean(scores), np.std(scores)

def mse_metric(img1, img2):
    return F.mse_loss(torch.FloatTensor(img1/255.0), torch.FloatTensor(img2/255.0), reduction='mean').item()

def ssim_metric(img1, img2):
    return SSIM(img1, img2, data_range=255, channel_axis=-1)

def psnr(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1e-10:
        return 100
    psnr1 = 20 * math.log10(255 / math.sqrt(mse))
    return psnr1

def calculate_hue_similarity(img1, img2):
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    img1_hue = img1_hsv[:, :, 0]
    img2_hue = img2_hsv[:, :, 0]

    img1_hue = img1_hue / 360.0
    img2_hue = img2_hue / 360.0

    similarity = np.sum(img1_hue * img2_hue) / (np.sqrt(np.sum(img1_hue ** 2)) * np.sqrt(np.sum(img2_hue ** 2)))
    return similarity

def caculation(pred_list, gt_list, frame_num, type):
    SS = []
    if type == 'SSIM':
        for i in tqdm(range(1200)):
            ssim = []
            for j in range(frame_num):
                recons = pred_list[i, j, :, :, :]
                gt = gt_list[i, j, :, :, :]
                c = SSIM(recons, gt, data_range=255, channel_axis=-1)
                ssim.append(c)
            s = np.mean(ssim)
            SS.append(s)
    if type == 'PSNR':
        for i in tqdm(range(1200)):
            ssim = []
            for j in range(frame_num):
                recons = pred_list[i, j, :, :, :]
                gt = gt_list[i, j, :, :, :]
                c = psnr(recons, gt)
                ssim.append(c)
            s = np.mean(ssim)
            SS.append(s)
    if type == 'PCC':
        for i in tqdm(range(1200)):
            corr = []
            for j in range(frame_num):
                recons = pred_list[i, j, :, :, :]
                gt = gt_list[i, j, :, :, :]
                c = calculate_hue_similarity(recons, gt)
                corr.append(c)
            s = np.mean(corr)
            SS.append(s)
    return SS


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--recons_results_root', type=str, default = None)
  args = parser.parse_args()

  gt_list = []
  pred_list = []
  for i in tqdm(range(1200)):
      gif = iio.imread(os.path.join(args.recons_results_root,f'test_and_gt{i + 1}.gif'), index=None)
      gt, pred = np.split(gif, 2, axis=2)

      gt_list.append(gt)
      pred_list.append(pred)

  gt_list = np.stack(gt_list)
  pred_list = np.stack(pred_list)
  print(f'pred shape: {pred_list.shape}, gt shape: {gt_list.shape}')

  SS = caculation(pred_list=pred_list, gt_list=gt_list, frame_num=8, type='SSIM')
  print("SSIM_mean:{},std:{}".format(np.mean(SS), np.std(SS)))

  PSNR = caculation(pred_list=pred_list, gt_list=gt_list, frame_num=8, type='PSNR')
  print("PSNR_mean:{},std:{}".format(np.mean(PSNR), np.std(PSNR)))

  C = caculation(pred_list=pred_list, gt_list=gt_list, frame_num=8, type='PCC')
  print("Hue_pcc_mean:{},std:{}".format(np.mean(C), np.std(C)))

