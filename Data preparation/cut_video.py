from moviepy.video.io.VideoFileClip import VideoFileClip
import argparse

def clip_video(source_file, target_file, start_time, stop_time):
    source_video = VideoFileClip(source_file)
    video = source_video.subclip(int(start_time), int(stop_time))
    video.write_videofile(target_file)
    return

def cut_videos(source_file_root, target_file_root_Train, target_file_root_Test):
    """
    :param source_file_root: load from CC2017
    :param target_file_root_Train:  The path you have set for storing the training set video clips.
    :param target_file_root_Test:   The path you have set for storing the testing set video clips.
    :return:
    """
    # Train_video
    for i in range(18):
        source_file = source_file_root + 'seg{}.mp4'.format(i + 1)
        for j in range(240):
            target_file = target_file_root_Train + 'seg{}_{}.mp4'.format(i + 1, j + 1)
            start_time = 2 * j
            stop_time = 2 * (j + 1)
            a = clip_video(source_file, target_file, start_time, stop_time)
            print('')

    print("Training set Done")
    # Test_video
    for i in range(5):
        source_file = source_file_root + 'test{}.mp4'.format(i + 1)
        for j in range(240):
            target_file = target_file_root_Test + 'test{}_{}.mp4'.format(i + 1, j + 1)
            start_time = 2 * j
            stop_time = 2 * (j + 1)
            a = clip_video(source_file, target_file, start_time, stop_time)

    print("Testing set Done")
    return



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--source_file_root', type=str, default = None)
  parser.add_argument('--target_file_root_Train', type=str, default = None)
  parser.add_argument('--target_file_root_Test', type=str, default=None)

  args = parser.parse_args()
  cut_videos(args.source_file_root, args.target_file_root_Train, args.target_file_root_Test)
