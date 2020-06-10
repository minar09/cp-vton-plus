# CP-VTON+
Official implementation for "CP-VTON+: Clothing Shape and Texture Preserving Image-Based Virtual Try-On" from CVPRW 2020.
<br/>Project page: https://minar09.github.io/cpvtonplus/.

## Usage
1) Install the requirements
2) Prepare the dataset
3) Train GMM network
4) Get warped clothes for training set with trained GMM network, and copy inside `data/viton_resize/train` directory
5) Train TOM network
6) Test/evaluate with test set

## Installation
This implementation is built and tested in PyTorch 0.4.1.
<br/>Run `pip install -r requirements.txt`

## Data preparation
1) Run `python data_download.py`
2) Run `python dataset_neck_skin_correction.py`
3) Run `python body_binary_masking.py`

## Training
Run `python train.py` with your specific usage options for GMM and TOM stage.
<br/>For example, GMM: ```python train.py --name GMM --stage GMM --workers 4 --save_count 5000 --shuffle```
<br/>and for TOM stage, ```python train.py --name TOM --stage TOM --workers 4 --save_count 5000 --shuffle```

## Testing
Run 'python test.py' with your specific usage options.
<br/>For example, GMM: ```python test.py --name GMM --stage GMM --workers 4 --datamode test --data_list test_pairs.txt --checkpoint checkpoints/GMM/gmm_final.pth```
<br/>and for TOM stage: ```python test.py --name TOM --stage TOM --workers 4 --datamode test --data_list test_pairs.txt --checkpoint checkpoints/TOM/tom_final.pth```

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
