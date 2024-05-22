import torch
import clip
from PIL import Image
import numpy as np
import h5py
from tqdm import tqdm
import argparse
device = torch.device('cuda:5')

weight_dtype = torch.float32
model, preprocess = clip.load("ViT-B/32", device=device)
model = model.to(torch.float32)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--Train_captions_save_path', type=str, default=None)
  parser.add_argument('--Train_text_CLIPfeature_save_path', type=str, default=None)
  parser.add_argument('--Train_frames_save_path', type=str, default=None)
  parser.add_argument('--Train_img_CLIPfeature_save_path', type=str, default=None)

  parser.add_argument('--Test_captions_save_path', type=str, default=None)
  parser.add_argument('--Test_text_CLIPfeature_save_path', type=str, default=None)
  parser.add_argument('--Test_frames_save_path', type=str, default=None)
  parser.add_argument('--Test_img_CLIPfeature_save_path', type=str, default=None)
  args = parser.parse_args()

  # Text-------------------------------------------------------------------------------------------------------------------------------------
  Train_prompts = h5py.File(args.Train_captions_save_path + '/Train_captions_new.h5', 'r')
  Train_captions = Train_prompts['Train_captions']
  Train_embeddings = []
  for j in tqdm(range(len(Train_captions))):
      caption = str(Train_captions[j], 'utf-8')
      text = clip.tokenize([caption]).to(device)
      with torch.no_grad():
          text_features = model.encode_text(text).cpu().detach().numpy()
      Train_embeddings.append(text_features)
  Train_ = np.concatenate(Train_embeddings, axis=0)
  np.save(args.Train_text_CLIPfeature_save_path + '/Train_text_embeddings_CLIP_512.npy', Train_)

  Test_prompts = h5py.File(args.Test_captions_save_path + '/Test_captions_new.h5', 'r')
  Test_captions = Test_prompts['Test_captions']
  Test_embeddings = []
  for j in tqdm(range(len(Test_captions))):
      caption = str(Test_captions[j], 'utf-8')
      text = clip.tokenize([caption]).to(device)
      with torch.no_grad():
          text_features = model.encode_text(text).cpu().detach().numpy()
      Test_embeddings.append(text_features)
  Test_ = np.concatenate(Test_embeddings, axis=0)
  np.save(args.Test_text_CLIPfeature_save_path + '/Test_text_embeddings_CLIP_512.npy', Test_)

  # Img----------------------------------------------------------------------------------------------------------------------------------------------------
  Train_imgs = []
  for i in tqdm(range(18)):
      for j in tqdm(range(240)):
          frames_root = args.Train_frames_save_path + 'seg{}_{}/'.format(i + 1, j + 1)
          image = preprocess(Image.open(frames_root + '0000024.jpg')).unsqueeze(0).to(device)
          with torch.no_grad():
              image_features = model.encode_image(image).cpu().detach().numpy()
          Train_imgs.append(image_features)
  Train_ = np.concatenate(Train_imgs, axis=0)
  np.save(args.Train_img_CLIPfeature_save_path + '/Train_img_embeddings_CLIP_512.npy', Train_)


  Test_imgs = []
  for i in tqdm(range(5)):
      for j in tqdm(range(240)):
          frames_root = args.Test_frames_save_path + 'test{}_{}/'.format(i + 1, j + 1)
          image = preprocess(Image.open(frames_root + '0000024.jpg')).unsqueeze(0).to(device)
          with torch.no_grad():
              image_features = model.encode_image(image).cpu().detach().numpy()
          Test_imgs.append(image_features)
  Test_ = np.concatenate(Test_imgs, axis=0)
  np.save(args.Test_img_CLIPfeature_save_path + '/Test_img_embeddings_CLIP_512.npy', Test_)
