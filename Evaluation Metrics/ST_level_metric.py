import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch
import clip
from PIL import Image
import cv2
import imageio.v3 as iio
import numpy as np
import argparse
device = torch.device('cuda:0')
model, preprocess = clip.load("ViT-B/32", device=device)

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
  parser.add_argument('--recons_results_root', type=str, default = None)
  parser.add_argument('--VIFICLIP_score_save_root', type=str, default=None)
  args = parser.parse_args()

  pred_list = []
  for i in range(1200):
      gif = iio.imread(os.path.join(args.recons_results_root,f'test_and_gt{i + 1}.gif'), index=None)
      _, pred = np.split(gif, 2, axis=2)
      pred_frames = np.array([cv2.resize(pred_f, (512, 512)) for pred_f in pred])
      pred_list.append(pred_frames)

  Pred_list = np.stack(pred_list)
  VIFICLIP = np.load(args.VIFICLIP_score_save_root+ '/VIFICLIP.npy').flatten()

  predpcc_list = cal_CLIP_pcc(Pred_list, VIFICLIP, 0.6)
  print('ST-consistency:')
  print(np.mean(predpcc_list))
  print(np.std(predpcc_list))
