import subprocess
import os
from tqdm import tqdm
import numpy as np
import cv2
import imageio.v3 as iio
import argparse

def save_numpy_video_to_avi(numpy_video, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in tqdm(range(numpy_video.shape[0])):
        video_frames = numpy_video[i]
        video_name = f"video_{i + 1}.avi"
        output_path = os.path.join(output_folder, video_name)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 4
        height, width, _ = video_frames[0].shape
        video_frames_uint8 = video_frames.astype(np.uint8)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in video_frames_uint8:
            out.write(frame)
        out.release()


def convert_gif_to_avi(input_path, output_path):
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', input_path,
        output_path
    ]

    subprocess.run(ffmpeg_cmd)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--recons_results_root', type=str, default = None)
  parser.add_argument('--avi_gt_results_root', type=str, default=None)
  parser.add_argument('--avi_recons_results_root', type=str, default=None)
  args = parser.parse_args()

  gt_list = []
  pred_list = []
  for i in tqdm(range(1200)):
      gif = iio.imread(os.path.join(args.recons_results_root, f'test_and_gt{i + 1}.gif'), index=None)
      gt, pred = np.split(gif, 2, axis=2)
      gt_list.append(gt)
      pred_list.append(pred)

  gt_list = np.stack(gt_list)
  pred_list = np.stack(pred_list)
  print(f'pred shape: {pred_list.shape}, gt shape: {gt_list.shape}')

  save_numpy_video_to_avi(gt_list, args.avi_gt_results_root)
  save_numpy_video_to_avi(pred_list, args.avi_recons_results_root)




