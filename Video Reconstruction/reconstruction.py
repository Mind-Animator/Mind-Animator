import os
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import copy
import argparse
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import decord
decord.bridge.set_bridge('torch')
from Inflated_T2I_Model.util import save_videos_grid, ddim_inversion
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL
from diffusers.schedulers import (
    DDPMScheduler,
    DDIMScheduler,
)
from einops import rearrange
from Inflated_T2I_Model.models.unet import UNet3DConditionModel
from Inflated_T2I_Model.pipelines.Inflated_T2I_pipeline import Inflated_T2IPipeline
from Mind_Animator_Models.modules import MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding, EncoderDecoder, BrainEncoder, Decoder, DecoderLayer,Embeddings
from Mind_Animator_Models.utils import greedy_decode
from Mind_Animator_Models.modules import Semantic_Decoder, Structure_Decoder

device = torch.device('cuda:2')
seed = 42
weight_dtype = torch.float32

parser = argparse.ArgumentParser(description='reconstruction')
parser.add_argument('--subj_ID', help='subj_ID', default=1, type=int)
parser.add_argument('--fMRI_data_path', help='fMRI data saved path', default=None, type=str)
parser.add_argument('--Semantic_model_dir', help='model saved path', default=None, type=str)
parser.add_argument('--Structure_model_dir', help='model saved path', default=None, type=str)
parser.add_argument('--CMG_model_dir', help='model saved path', default=None, type=str)
parser.add_argument('--results_save_root', help='results_save_root', default=None, type=str)
args = parser.parse_args()

def make_model( N=2, d_model=4096*4, d_ff=768, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    CMG_model = EncoderDecoder(
        BrainEncoder(out_dim=d_model, in_dim=4500, h=768, n_blocks=1, norm_type='ln', act_first=False),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model), c(position)))

    for p in CMG_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return CMG_model

noise_scheduler = DDPMScheduler.from_pretrained("/data0/home/Datastation/checkpoints/scheduler")
inference_scheduler = DDIMScheduler.from_pretrained("/data0/home/Datastation/checkpoints/scheduler")
tokenizer = CLIPTokenizer.from_pretrained("/data0/home/Datastation/checkpoints/tokenizer")
text_encoder = CLIPTextModel.from_pretrained("/data0/home/Datastation/checkpoints/text_encoder").to(device, dtype=weight_dtype)
vae = AutoencoderKL.from_pretrained("/data0/home/Datastation/checkpoints/vae").to(device, dtype=weight_dtype)
unet = UNet3DConditionModel.from_pretrained_2d("/data0/home/Datastation/checkpoints/unet").to(device, dtype=weight_dtype)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)
unet.enable_xformers_memory_efficient_attention()

inference_pipeline = Inflated_T2IPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,scheduler=inference_scheduler)
inference_pipeline.enable_vae_slicing()
ddim_inv_scheduler = inference_scheduler
ddim_inv_scheduler.set_timesteps(50)

generator = torch.Generator(device=device)
generator.manual_seed(seed)

Semantic_model = Semantic_Decoder(in_dim=4500, out_dim=19 * 768, h=512, n_blocks=3, norm_type='ln', act_first=False)
Semantic_model.load_state_dict(torch.load(os.path.join(args.Semantic_model_dir,'/Sub_{}/'.format(args.subj_ID) , 'Semantic_Decoder_{}.pth'.format(30))))
Semantic_model.to(device)
Semantic_model.eval()

Structure_model = Structure_Decoder(in_dim=4500, out_dim=4*64*64, h=1024, n_blocks=2, norm_type='ln', act_first=False)
Structure_model.load_state_dict(torch.load(os.path.join(args.Structure_model_dir,'/Sub_{}/'.format(args.subj_ID) , 'Structure_Decoder_{}.pth'.format(75))))
Structure_model.to(device)
Structure_model.eval()

CMG_model = make_model(2, d_model=4096*4, d_ff=768, h=4, dropout=0.2)
CMG_model.load_state_dict(torch.load(os.path.join(args.CMG_model_dir,'/Sub_{}/'.format(args.subj_ID) , 'CMG_model_{}.pth'.format(70))))
CMG_model.to(device)
CMG_model.eval()

fMRI_data = np.load(args.fMRI_data_path+ '/Sub_{}/'.format(args.subj_ID) +  '/masked4500_test_data.npy')
test_gts = torch.tensor(np.load('/nfs/diskstation/DataStation/reconstruction_results/test_gt_npy/test_groundtruth.npy'), dtype=torch.float32)

def reconstructing_videos(subj_ID, test_src, test_gts, Semantic_model, Structure_model, CMG_model, video_save_root):
    fMRI_ = torch.tensor(test_src, dtype=torch.float32).to(device)
    for video_ID in tqdm(range(1200)):
        fMRI = fMRI_[video_ID:video_ID + 1, :]

        first_frame = Structure_model(fMRI).to(torch.float32)
        latents = rearrange(
            greedy_decode(first_frame=first_frame, model=CMG_model, src=fMRI, src_mask=None, max_len=8,
                          mask_ratio=0), "b f (c h w) -> b c f h w ", f=8, c=4, h=64, w=64)  # .view(-1,4,8,32,32)
        ddim_inv_latent = \
        ddim_inversion(inference_pipeline, ddim_inv_scheduler, video_latent=latents, num_inv_steps=50, prompt="")[
            -1].to(weight_dtype)

        noise = torch.randn_like(latents)
        timesteps = torch.tensor([200]).long()
        noisy_latents = noise_scheduler.add_noise(ddim_inv_latent, noise, timesteps)

        # semantic decoder----------------------------------------------------------------------------------------------------------------------------------------
        cls = np.expand_dims(np.load('/nfs/diskstation/DataStation/reconstruction_results/cls.npy'), 0)
        _, sem = Semantic_model(fMRI)
        sem = sem.cpu().detach().numpy()
        prompt = torch.tensor(np.concatenate([cls, sem], axis=1), dtype=torch.float32).to(device)

        # --------------------------------------------------------------------------------------------------------------------------------------
        output_dir1 = video_save_root + '/sub{}/recons_only_n200'.format(subj_ID)
        output_dir2 = video_save_root + '/sub{}/recons_and_gt_n200'.format(subj_ID)

        sample = inference_pipeline(prompt, generator=generator, latents=noisy_latents).videos
        gt = test_gts[video_ID:video_ID + 1, :, :, :, :]

        save_videos_grid(sample, output_dir1 + '/test{}.gif'.format(video_ID + 1))
        save_videos_grid(torch.concat([gt, sample], dim=0), output_dir2 + '/test_and_gt{}.gif'.format(video_ID + 1))
    return


if __name__ == "__main__":
    reconstructing_videos(subj_ID = args.subj_ID, test_src = fMRI_data, test_gts = test_gts, Semantic_model = Semantic_model, Structure_model = Structure_model, CMG_model = CMG_model, video_save_root = args.results_save_root)




