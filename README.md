# <p align="center"> Mind-Animator </p>

## <p align="center"> The human brain's comprehension of dynamic visual scenes </p>

<div align=center>
<img src="https://github.com/Mind-Animator/Mind-Animator/blob/main/imgs/decouple.png">
</div>

When receiving dynamic visual information, human brain gradually comprehends low-level **structural** details such as position, shape and color in the primary visual cortex, discerns **motion** information, and ultimately constructs high-level **semantic** information in the higher visual cortex, such as an overall description of the scene.

## <p align="center"> The overall architecture of Mind-Animator </p>

<div align=center>
<img src="https://github.com/Mind-Animator/Mind-Animator/blob/main/imgs/overview.png">
</div>

The overall architecture of Mind-Animator, a two-stage video reconstruction model based on fMRI. Three decoders are trained during the **fMRI-to-feature** stage to disentangle semantic, structural, and motion feature from fMRI, respectively.  In the **feature-to-video** stage, the decoded information is input into an inflated Text-to-Image (T2I) model for video reconstruction.

## <p align="center"> Experiments </p>

<div align=center>
<img src="https://github.com/Mind-Animator/Mind-Animator/blob/main/imgs/results.png">
</div>

<div align=center>
<img src="https://github.com/Mind-Animator/Mind-Animator/blob/main/imgs/results2.png">
</div>



## <p align="center"> TODOList </p>
- [x] Reorganize all the code.
- [x] Test the data preprocessing code for CC2017 dataset.
- [ ] Upload the data preprocessing code for CC2017 dataset.
- [x] Upload the code of evaluation metrics.
- [ ] Test the project code for Mind-Animator.
- [x] Upload the project code for Mind-Animator.
- [ ] Release the checkpoints of Mind-Animator.

## <p align="center">  Preliminaries  </p> 
This code was developed and tested with: 

*  Python version 3.9.16 
*  PyTorch version 1.12.1 
*  A100 80G 
*  The conda environment defined in environment.yml

First, set up the conda enviroment as follows:<br>
```
conda env create -f environment.yml  # create conda env
conda activate Mind-Animator          # activate conda env
cd path/to/project/Mind-Animator
```


## <p align="center">  Dataset downloading and preparation </p> 

### <p align="center">  Dataset downloading </p> 

**CC2017  dataset** https://purr.purdue.edu/publications/2809/1  <br>
**HCP dataset** https://www.humanconnectome.org/  <br>
**Algonauts2021 dataset** http://algonauts.csail.mit.edu/2021/index.html  <br>

### <p align="center">  Data preparation </p> 
#### <p align="center">  CC2017  dataset </p> 
* After downloading the CC2017 dataset, you will obtain a folder organized as shown in the figure. In this project, we utilize the fMRI data located within the cifti folder.

<div align=center>
<img src="https://github.com/Zuskd/Mind-Animator/blob/main/imgs/folder.png">
</div>

* Segment the video into 2-second clips and downsample the frame rate of each segment to 4 Hz. You are required to initially create the following four directories for the respective storage purposes: one for the training set video segments (Train_video_path), another for the training set video frames (Train_frames_path), a third for the testing set video segments (Test_video_path), and a fourth for the testing set video frames (Test_frames_path).<br>
```
    cd path/to/project/Mind-Animator/Data preparation
    python cut_video.py --source_file_root "path/to/your/root/stimuli" --target_file_root_Train "path/to/Train_video_path" --target_file_root_Test "path/to/Test_video_path"
```

```
    python get_video_frames.py --Train_videopath_root "path/to/Train_video_path" --Train_target_root "path/to/Train_frames_path" --Test_videopath_root "path/to/Test_video_path" --Test_target_root "path/to/Test_frames_path"
```

 
* To label each video segment with text captions, you should first install and run the BLIP2 project. https://github.com/salesforce/LAVIS/tree/main/projects/blip2 <br>
```
conda activate BLIP2
python video_captioning.py --Train_video_path_root "path/to/Train_frames_path" --Train_captions_save_path "path/to/Train_frames_path" --Test_video_path_root "path/to/Test_frames_path" --Test_captions_save_path "path/to/Test_frames_path"
conda deactivate BLIP2
```
* Extract the voxels within the fMRI data that are indicative of activation in the visual cortex. You are required to create directories in advance for each subject to store the following pre-processed data: fMRI data for individual trails of the training set (Train_fMRI_singletrail), averaged fMRI data across multiple trails of the training set (Train_fMRI_multitrail_average), averaged fMRI data across multiple trails of the test set (Test_fMRI_multitrail_average), and mask files (mask_save_root).

```
python fMRI_preparation_FSLR.py --fMRI_volumes_root "path/to/CC2017_Purdue" --raw_train_data_root "path/to/Train_fMRI_singletrail" --averaged_train_data_root "path/to/Train_fMRI_multitrail_average" --averaged_test_data_root "path/to/Test_fMRI_multitrail_average" --mask_save_root "path/to/mask_save_root"
```


## <p align="center">  Feature Extraction </p> 

### <p align="center">  Semantic Feature Extraction </p> 

* Before extracting the text conditions, it is necessary to establish several directories in advance to store the following data: the text condition for the training set (Train_condition) and the text condition for the test set (Test_condition).

```
cd path/to/project/Mind-Animator/Feature extraction
python Text_condition.py --Train_captions_save_path "path/to/Train_frames_path" --Train_text_condition_save_path "path/to/Train_condition" --Test_captions_save_path "path/to/Test_frames_path" --Test_text_condition_save_path "path/to/Test_condition"
```

* Before extracting the CLIP features, it is necessary to first create the following directories to store the extracted features: image features of the training set (Train_img_CLIP_512), text features of the training set (Train_text_CLIP_512), image features of the test set (Test_img_CLIP_512), and text features of the test set (Test_text_CLIP_512).

```
python CLIP_feature.py --Train_captions_save_path "path/to/Train_frames_path" --Train_text_CLIPfeature_save_path "path/to/Train_text_CLIP_512" --Train_frames_save_path "path/to/Train_frames_path" --Train_img_CLIPfeature_save_path "path/to/Train_img_CLIP_512" --Test_captions_save_path "path/to/Test_frames_path" --Test_text_CLIPfeature_save_path "path/to/Test_text_CLIP_512" --Test_frames_save_path "path/to/Test_frames_path" --Test_img_CLIPfeature_save_path "path/to/Test_img_CLIP_512" 
```

### <p align="center">  Structure and Motion Feature Extraction </p> 

Run the following code to extract the structural information and consistent motion information from the video using a pre-trained VQVAE.

```
python Structure_feature.py --target_file_root "path/to/{}_video_path"
```

## <p align="center">  Feature Decoding </p> 



## <p align="center">  Video Reconstruction </p> 

Run the following code snippet to generate reconstruction results for each subject's test dataset. Please adjust the subj_ID variable accordingly (valid options include: 1, 2, 3).  To ensure that the reconstruction results for each subject are stored in the correct location, please modify the output_dir1 and output_dir2 variables within the reconstruction.py file accordingly.

```
cd path/to/project/Mind-Animator/Video Reconstruction
python reconstruction.py --subj_ID 1 --fMRI_data_path "path/to/Test_fMRI_multitrail_average" --Semantic_model_dir "path/to/Semantic_decoder" --Structure_model_dir "path/to/Structure_decoder" --CMG_model_dir "path/to/CMG_decoder" --results_save_root "path/to/results_save_root"
```


## <p align="center">  Calculate the Evaluation Metrics </p> 

To calculate the following evaluation metrics, you will first need to download the following models: ViT-Base Patch16 224 (https://huggingface.co/google/vit-base-patch16-224), VideoMAE (https://huggingface.co/MCG-NJU/videomae-base), and ViFi-CLIP (https://github.com/muzairkhattak/ViFi-CLIP).


### <p align="center">  Semantic-level Metrics </p> 

Run the following code to calculate the 2-way-I and 2-way-V metrics.

```
cd path/to/project/Mind-Animator/Evaluation Metrics
python 2-way-I.py --recons_results_root "path/to/results_save_root"
python 2-way-V.py --recons_results_root "path/to/results_save_root"
```

Before calculating VIFI_score, it is necessary to first convert the reconstruction results from the .gif format to the .avi format. Please ensure to create folders in advance to store the .avi format ground truth (gt_avi) , reconstruction results (recons_avi) , and VIFI_score results (VIFI_score_path).

```
cd path/to/project/Mind-Animator/Evaluation Metrics/ViFi_CLIP
python gif_to_avi.py --recons_results_root "path/to/results_save_root" --avi_gt_results_root "path/to/gt_avi" --avi_recons_results_root "path/to/recons_avi"
python VIFI_score.py --avi_gt_results_root "path/to/gt_avi" --avi_recons_results_root "path/to/recons_avi" --VIFICLIP_score_save_root "path/to/VIFI_score_path"
```
### <p align="center">  Pixel-level Metrics </p> 

Run the following code to calculate Pixel-level Metrics.

```
cd path/to/project/Mind-Animator/Evaluation Metrics
python Pixel_level_metrics.py --recons_results_root "path/to/results_save_root"
```

### <p align="center">  Spatiotemporal-level Metric </p> 

Run the following code to calculate the Spatiotemporal-level Metric.

```
cd path/to/project/Mind-Animator/Evaluation Metrics
python ST_level_metric.py --recons_results_root "path/to/results_save_root" --VIFICLIP_score_save_root "path/to/VIFI_score_path"
```




