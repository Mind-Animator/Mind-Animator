from skimage.metrics import structural_similarity as SSIM
import torch
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
import numpy as np
from PIL import Image
import os
import imageio.v3 as iio
import math
import cv2
import clip
import argparse
device = torch.device('cuda:0')
model, preprocess = clip.load("ViT-B/32", device=device)

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
        for i in tqdm(range(pred_list.shape[0])):
            ssim = []
            for j in range(frame_num):
                recons = pred_list[i, j, :, :, :]
                gt = gt_list[i, j, :, :, :]
                c = SSIM(recons, gt, data_range=255, channel_axis=-1)
                ssim.append(c)
            s = np.mean(ssim)
            SS.append(s)
    if type == 'PSNR':
        for i in tqdm(range(pred_list.shape[0])):
            ssim = []
            for j in range(frame_num):
                recons = pred_list[i, j, :, :, :]
                gt = gt_list[i, j, :, :, :]
                c = psnr(recons, gt)
                ssim.append(c)
            s = np.mean(ssim)
            SS.append(s)
    if type == 'PCC':
        for i in tqdm(range(pred_list.shape[0])):
            corr = []
            for j in range(frame_num):
                recons = pred_list[i, j, :, :, :]
                gt = gt_list[i, j, :, :, :]
                c = calculate_hue_similarity(recons, gt)
                corr.append(c)
            s = np.mean(corr)
            SS.append(s)
    return SS

def cal_CLIP_pcc( predlist, VIFI_CLIP_list, bar):
    pcc_list = []
    for i in range(predlist.shape[0]):
        if VIFI_CLIP_list[i] <= bar:
            pcc_list.append(0)
        else:
            cos_simlist = []
            pred_frames = predlist[i,:,:,:,:]
            for j in range(pred_frames.shape[0]-1):
                img_recons1 = Image.fromarray(np.uint8(pred_frames[j, :, :, :]))
                recons1 = preprocess(img_recons1).unsqueeze(0).to(device)
                recons1_features = model.encode_image(recons1)

                img_recons2 = Image.fromarray(np.uint8(pred_frames[j + 1, :, :, :]))
                recons2 = preprocess(img_recons2).unsqueeze(0).to(device)
                recons2_features = model.encode_image(recons2)

                cos_sim = torch.cosine_similarity(recons1_features, recons2_features).cpu().detach().numpy()
                cos_simlist.append(cos_sim)
            pcc_list.append(np.mean(cos_simlist))
    return pcc_list


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--Successful_idxs', type=list, default=None)
  parser.add_argument('--recons_results_root', type=str, default = None)
  parser.add_argument('--VIFICLIP_score_save_root', type=str, default=None)
  args = parser.parse_args()

  gt_list = []
  pred_list = []
  for i in tqdm(args.Successful_idxs):
      gif = iio.imread(os.path.join(args.recons_results_root,f'test_and_gt{i}.gif'), index=None)
      gt, pred = np.split(gif, 2, axis=2)
      gt_list.append(gt)
      pred_list.append(pred)

  gt_list = np.stack(gt_list)
  pred_list = np.stack(pred_list)
  print(f'pred shape: {pred_list.shape}, gt shape: {gt_list.shape}')


  SS_y = caculation(pred_list=pred_list, gt_list=gt_list, frame_num=8, type='SSIM')
  PSNR_y = caculation(pred_list=pred_list, gt_list=gt_list, frame_num=8, type='PSNR')
  C_y = caculation(pred_list=pred_list, gt_list=gt_list, frame_num=8, type='PCC')
  VIFICLIP = np.load(args.VIFICLIP_score_save_root + '/VIFICLIP.npy').flatten()[(np.array(args.Successful_idxs) - 1).tolist()]
  consist_y = cal_CLIP_pcc(pred_list, VIFICLIP, 0.6)

  SSIM_K_list = []
  PSNR_K_list = []
  PCC_K_list = []
  CCCC_list = []

  for rep in range(5):
      Ssim_mean = []
      Psnr_mean = []
      Pcc_mean = []
      consist = []

      SSIM_K = 0
      PSNR_K = 0
      PCC_K = 0
      CCCC = 0

      for num in tqdm(range(100)):
          shuffled_indices = np.random.permutation(8)
          Pred_list = pred_list[:, shuffled_indices, :, :, :]

          SS = caculation(pred_list=Pred_list, gt_list=gt_list, frame_num=8, type='SSIM')
          Ssim_mean.append(np.mean(SS))

          PSNR = caculation(pred_list=Pred_list, gt_list=gt_list, frame_num=8, type='PSNR')
          Psnr_mean.append(np.mean(PSNR))

          C = caculation(pred_list=Pred_list, gt_list=gt_list, frame_num=8, type='PCC')
          Pcc_mean.append(np.mean(C))

          c = cal_CLIP_pcc(Pred_list, VIFICLIP, 0.6)
          pcc_shuffled = np.mean(c)
          consist.append(pcc_shuffled)

      for j in range(100):
          if Ssim_mean[j] > np.mean(SS_y):
              SSIM_K = SSIM_K + 1
          if Psnr_mean[j] > np.mean(PSNR_y):
              PSNR_K = PSNR_K + 1
          if Pcc_mean[j] > np.mean(C_y):
              PCC_K = PCC_K + 1
          if consist[j] > np.mean(consist):
              CCCC = CCCC + 1

      SSIM_K_list.append(SSIM_K)
      PSNR_K_list.append(PSNR_K)
      PCC_K_list.append(PCC_K)
      CCCC_list.append(CCCC)


  print(f'SSIM_mean: {np.mean(SSIM_K_list)}, SSIM_std: {np.std(SSIM_K_list)}')
  print(f'PSNR_mean: {np.mean(PSNR_K_list)}, SSIM_std: {np.std(PSNR_K_list)}')
  print(f'PCC_mean: {np.mean(PCC_K_list)}, SSIM_std: {np.std(PCC_K_list)}')
  print(f'CCCC_mean: {np.mean(CCCC_list)}, SSIM_std: {np.std(CCCC_list)}')




