# CP-VTON+ (CVPRW 2020)
Official implementation for "CP-VTON+: Clothing Shape and Texture Preserving Image-Based Virtual Try-On" from CVPRW 2020.
<br/>Project page: https://minar09.github.io/cpvtonplus/. 
<br/>Saved/Pre-trained models: [Checkpoints](https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP)
<br/>Dataset: [VITON_PLUS](https://1drv.ms/u/s!Ai8t8GAHdzVUiQQYX0azYhqIDPP6?e=4cpFTI)
<br/>The code and pre-trained models are tested with pytorch 0.4.1, torchvision 0.2.1, opencv-python 4.1 and pillow 5.4 (Python 3 env).
<br/><br/>
[Project page](https://minar09.github.io/cpvtonplus/) | [Paper](https://minar09.github.io/cpvtonplus/cvprw20_cpvtonplus.pdf) | [Dataset](https://1drv.ms/u/s!Ai8t8GAHdzVUiQRFmTPrtrAy0ZP5?e=rS1aK8) | [Model](https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=6PO4gq) | [Video](https://www.youtube.com/watch?v=MPB_PYLOfd8)
<br/><br/>
	
## Usage
This pipeline is a combination of consecutive training and testing of GMM + TOM. GMM generates the warped clothes according to the target human. Then, TOM blends the warped clothes outputs from GMM into the target human properties, to generate the final try-on output.

1) Install the requirements
2) Download/Prepare the dataset
3) Train GMM network
4) Get warped clothes for training set with trained GMM network, and copy warped clothes & masks inside `data/train` directory
5) Train TOM network
6) Test GMM for testing set
7) Get warped clothes for testing set, copy warped clothes & masks inside `data/test` directory
8) Test TOM testing set

## Installation
This implementation is built and tested in PyTorch 0.4.1.
Pytorch and torchvision are recommended to install with conda: `conda install pytorch=0.4.1 torchvision=0.2.1 -c pytorch`
<br/>For all packages, run `pip install -r requirements.txt`

## Data preparation
For training/testing VITON dataset, our full and processed dataset is available here: https://1drv.ms/u/s!Ai8t8GAHdzVUiQRFmTPrtrAy0ZP5?e=rS1aK8. After downloading, unzip to your data directory.

## Training
Run `python train.py` with your specific usage options for GMM and TOM stage.
<br/>For example, GMM: ```python train.py --name GMM --stage GMM --workers 4 --save_count 5000 --shuffle```
<br/> Then run test.py for GMM network with the training dataset, which will generate the warped clothes and masks in "warp-cloth" and "warp-mask" folders inside the "result/GMM/train/" directory. Copy the "warp-cloth" and "warp-mask" folders into your data directory, for example inside "data/train" folder.
<br/>Run TOM stage, ```python train.py --name TOM --stage TOM --workers 4 --save_count 5000 --shuffle```

## Testing
Run 'python test.py' with your specific usage options.
<br/>For example, GMM: ```python test.py --name GMM --stage GMM --workers 4 --datamode test --data_list test_pairs.txt --checkpoint checkpoints/GMM/gmm_final.pth```
<br/> Then run test.py for GMM network with the testing dataset, which will generate the warped clothes and masks in "warp-cloth" and "warp-mask" folders inside the "result/GMM/test/" directory. Copy the "warp-cloth" and "warp-mask" folders into your data directory, for example inside "data/test" folder.
<br/>Run TOM stage: ```python test.py --name TOM --stage TOM --workers 4 --datamode test --data_list test_pairs.txt --checkpoint checkpoints/TOM/tom_final.pth```

## Inference/Demo
Download the pre-trained models from here: https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP.
Then run the same step as Testing to test/inference our model.
The code and pre-trained models are tested with pytorch 0.4.1, torchvision 0.2.1, opencv 4.1 and pillow 5.4.

### Testing with custom images
to run the model with custom internet images, make sure you have the following:

1) image (image of a person, crop/resize to 192 x 256 (width x height) pixels)
2) image-parse (you can generate with CIHP_PGN or Graphonomy pretrained networks from the person image. See this [comment](https://github.com/minar09/cp-vton-plus/issues/15#issuecomment-683403388))
3) cloth (in-shop cloth image, crop/resize to 192 x 256 (width x height) pixels)
4) cloth-mask (binary mask of cloth image, you can generate it with simple pillow/opencv function)
5) pose (pose keypoints of the person, generate with openpose COCO-18 model (OpenPose from the official repository is preferred))
6) Also, make a test_pairs.txt file for your custom images. Follow the VITON dataset format to keep same arrangements, otherwise you can modify the code.

### What to do in case of unexpected results
There are many factors that can make distorted/unexpected results. Can you please do the following?

1) First try the original viton dataset and test pair combinations, check the intermediate results and the final output. Check if they are as expected.
2) If the original viton results are not as expected, please check the issues raised in this github repo, people have already found several issues and see how they solved it.
3) If the original viton test results are as expected, then run your custom test sets and check the intermediate results and debug where its going wrong.
4) If you are testing with custom images then check the github repository readme and related issues on how to run with custom images.

Its difficult to understand your issue from only single image/output. As I mentioned, there are various factors. Please debug yourself step by step and see where its going wrong. Check all the available intermediate/final inputs/outputs visually, and check multiple cases to see if the issue is happening for all cases. Good luck to you!


## Citation
Please cite our paper in your publications if it helps your research:
```
@InProceedings{Minar_CPP_2020_CVPR_Workshops,
	title={CP-VTON+: Clothing Shape and Texture Preserving Image-Based Virtual Try-On},
	author={Minar, Matiur Rahman and Thai Thanh Tuan and Ahn, Heejune and Rosin, Paul and Lai, Yu-Kun},
	booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
	month = {June},
	year = {2020}
}
```

### Acknowledgements
This implementation is largely based on the PyTorch implementation of [CP-VTON](https://github.com/sergeywong/cp-vton). We are extremely grateful for their public implementation.
