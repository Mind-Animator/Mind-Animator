from .utils.config import get_config
from .trainers import vificlip
import torch
from tqdm import tqdm
from .datasets.pipeline import *
import argparse

### Set values here ###
config_ = '/data1/home/Datastation/Python_project/ViFi-CLIP-main/configs/zero_shot/train/k400/16_16_vifi_clip.yaml'
output_folder_name = "exp"
pretrained_model_path = "/data0/home/Datastation/Pretrained_Models/k400_seed1_vifi_clip_base2novel.pth"
class_names = ['dancing', 'drum beating', 'swimming', "climbing stairs"]

# Step 1:
# Configuration class
class parse_option():
    def __init__(self):
        self.config = config_
        self.output =  output_folder_name   # Name of output folder to store logs and save weights
        self.resume = pretrained_model_path
        # No need to change below args.
        self.only_test = True
        self.opts = None
        self.batch_size = None
        self.pretrained = None
        self.accumulation_steps = None
        self.local_rank = 0
args = parse_option()
config = get_config(args)

# Step 2:
# Create the ViFi-CLIP models and load pretrained weights
model = vificlip.returnCLIP(config,
                            #logger=logger,
                            class_names=class_names,)
model = model.float().cuda()  # changing to cuda here
checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
load_state_dict = checkpoint['model']
# now remove the unwanted keys:
if "module.prompt_learner.token_prefix" in load_state_dict:
    del load_state_dict["module.prompt_learner.token_prefix"]

if "module.prompt_learner.token_suffix" in load_state_dict:
    del load_state_dict["module.prompt_learner.token_suffix"]

if "module.prompt_learner.complete_text_embeddings" in load_state_dict:
    del load_state_dict["module.prompt_learner.complete_text_embeddings"]
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in load_state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

# load params
msg = model.load_state_dict(new_state_dict, strict=False)

# Step 3:
# Preprocessing for video
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
scale_resize = int(512 / 224 * config.DATA.INPUT_SIZE)

val_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=8, frame_interval=1, num_clips=config.DATA.NUM_FRAMES, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(int(256), int(256))),
    dict(type='CenterCrop', crop_size=config.DATA.INPUT_SIZE),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

pipeline = Compose(val_pipeline)

def cal_VIFI_CLIP(gt_path, recons_path):
    CLIP_feature = []
    for i in tqdm(range(1200)):
        recons_video = recons_path + 'video_{}.avi'.format(i + 1)
        gt_video = gt_path + 'video_{}.avi'.format(i + 1)
        dict_file_gt = {'filename': gt_video, 'tar': False, 'modality': 'RGB', 'start_index': 0}
        dict_file_recons = {'filename': recons_video, 'tar': False, 'modality': 'RGB', 'start_index': 0}

        video_gt = pipeline(dict_file_gt)
        video_recons = pipeline(dict_file_recons)

        video_tensor_gt = video_gt['imgs'].unsqueeze(0).cuda().float()
        video_tensor_recons = video_recons['imgs'].unsqueeze(0).cuda().float()

        # Inference through ViFi-CLIP
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                img_feature_gt = model(video_tensor_gt)
                img_feature_recons = model(video_tensor_recons)
                cos_sim = torch.cosine_similarity(img_feature_gt, img_feature_recons).cpu().detach().numpy()
                CLIP_feature.append(cos_sim)
    return CLIP_feature

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--avi_gt_results_root', type=str, default=None)
  parser.add_argument('--avi_recons_results_root', type=str, default=None)
  parser.add_argument('--VIFICLIP_score_save_root', type=str, default=None)
  args = parser.parse_args()

  CLIP_feature = cal_VIFI_CLIP(args.avi_gt_results_root, args.avi_recons_results_root)
  np.save(args.VIFICLIP_score_save_root+ '/VIFICLIP.npy', np.array(CLIP_feature))

  print(np.mean(CLIP_feature))
  print(np.std(CLIP_feature))



