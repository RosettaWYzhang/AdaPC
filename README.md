# On automatic data augmentation for 3D point cloud classification (BMVC 2021)

This repository contains the code for our BMVC 2021 paper: On automatic data augmentation for 3D point cloud classification. You can find the PDF of our paper <a href="https://www.bmvc2021-virtualconference.com/assets/papers/0911.pdf" target="_blank">here</a>.

### Dependencies
Following is the suggested way to install these dependencies:
```bash
# Create a new conda environment
conda create -n AdaPC python=3.7
conda activate AdaPC
conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch -y
pip install -r requirements.txt
```

### Data
Download the ModelNet40 dataset from <a href="https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip" target="_blank">here</a> and put in into /data folder.


### Training
You can train our model using the command below:

    python train.py --data_dir data/modelnet40_ply_hdf5_2048 --gpu 0 --log_dir modelnet40_AdaPC_STRy
The default options have scaling, translation and rotation augmentations.

### Acknowledgement
Part of our code is borrowed from <a href="https://github.com/liruihui/PointAugment" target="_blank">PointAugment(Li et al)</a> and <a href="https://github.com/quark0/darts" target="_blank">DARTS (Liu et al)</a>.

### Citation
If you find our work useful in your research, please consider citing:
```
@article{zhang2021on,
  title={On Automatic Data Augmentation for 3D Point Cloud Classification.},
  author={Zhang, Wanyue and Xu, Xun and Liu, Fayao and Zhang, Le and Foo, Chuan-Sheng},
  booktitle={BMVC},
  year={2021}
}

```
