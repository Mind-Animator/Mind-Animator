# <p align="center"> Mind-Animator </p>
## <p align="center"> TODOList </p>
- [x] Reorganize all the code.
- [x] Test the data preprocessing code for CC2017 dataset.
- [ ] Upload the data preprocessing code for CC2017 dataset.
- [x] Upload the code of evaluation metrics.
- [ ] Test the project code for Mind-Animator.
- [ ] Upload the project code for Mind-Animator.
- [ ] Release the reconstruction results of Mind-Animator on all test sets.
- [ ] Release the checkpoints of Mind-Animator.

## <p align="center">  Preliminaries  </p> 
This code was developed and tested with: 

*  Python version 3.9.16 
*  PyTorch version 1.12.1 
*  A100 80G 
*  The conda environment defined in environment.yml 

## <p align="center">  Dataset downloading and preparation </p> 

### <p align="center">  Dataset downloading </p> 

**CC2017  dataset** https://purr.purdue.edu/publications/2809/1  <br>
**HCP dataset** https://www.humanconnectome.org/  <br>
**Algonauts2021 dataset** http://algonauts.csail.mit.edu/2021/index.html  <br>

### <p align="center">  Data preparation </p> 
#### <p align="center">  CC2017  dataset </p> 
* After downloading the CC2017 dataset, you will obtain a folder organized as shown in the figure. In this project, we utilize the fMRI data located within the cifti folder.

![](https://github.com/Zuskd/Mind-Animator/blob/main/imgs/folder.png)

* To label each video segment with text captions, you should first install and run the BLIP2 project. https://github.com/salesforce/LAVIS/tree/main/projects/blip2 <br>

* Segment the video into 2-second clips and downsample the frame rate of each segment to 4 Hz. You are required to initially create the following four directories for the respective storage purposes: one for the training set video segments (Train_video_path), another for the training set video frames (Train_frames_path), a third for the testing set video segments (Test_video_path), and a fourth for the testing set video frames (Test_frames_path).<br>

    python cut_video.py --source_file_root "path/to/your/root/stimuli" --target_file_root_Train "path/to/Train_video_path" --target_file_root_Test "path/to/Test_video_path"
    python Feature extractor/detail_extracttion.py

* 





















