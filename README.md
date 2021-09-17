# RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching
This repository contains the source code for our paper:

[RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching](https://arxiv.org/pdf/2109.07547.pdf)<br/>
Lahav Lipson, Zachary Teed and Jia Deng<br/>

<img src="RAFTStereo.png">

## Requirements
The code has been tested with PyTorch 1.7 and Cuda 10.2.
```Shell
conda env create -f environment.yaml
conda activate raftstereo
```




## Required Data
To evaluate/train RAFT, you will need to download the required datasets. 
* [Sceneflow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html#:~:text=on%20Academic%20Torrents-,FlyingThings3D,-Driving) (Includes FlyingThings3D, Driving & Monkaa
* [Middlebury](https://vision.middlebury.edu/stereo/data/)
* [ETH3D](https://www.eth3d.net/datasets#low-res-two-view-test-data)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

To download the ETH3D and Middlebury test datasets for the [demos](#demos), run 
```Shell
chmod ug+x download_datasets.sh && ./download_datasets.sh
```

By default `stereo_datasets.py` will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder

```Shell
├── datasets
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── disparity
    ├── Monkaa
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── disparity
    ├── Driving
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── disparity
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── Middlebury
        ├── MiddEval3
    ├── ETH3D
        ├── lakeside_1l
        ├── ...
        ├── tunnel_3s
```

## Demos
Pretrained models can be downloaded by running
```Shell
chmod ug+x download_models.sh && ./download_models.sh
```
or downloaded from [google drive](https://drive.google.com/drive/folders/1booUFYEXmsdombVuglatP0nZXb5qI89J)

You can demo a trained model on pairs of images. To predict stereo for Middlebury, run
```Shell
python demo.py --restore_ckpt models/raftstereo-sceneflow.pth
```
Or for ETH3D:
```Shell
python demo.py --restore_ckpt models/raftstereo-eth3d.pth -l=datasets/ETH3D/*/im0.png -r=datasets/ETH3D/*/im1.png
```
Using our fastest model:
```Shell
python demo.py --restore_ckpt models/raftstereo-realtime.pth  --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru 
```

To save the disparity values as `.npy` files, run any of the demos with the `--save_numpy` flag. 

## Converting Disparity to Depth 

If the camera focal length and camera baseline are known, disparity predictions can be converted to depth values using

<img src="depth_eq.png" width="320">

Note that the units of the focal length are _pixels_ not millimeters.

## Evaluation

To evaluate a trained model on a validation set (e.g. Middlebury), run
```Shell
python evaluate_stereo.py --restore_ckpt models/raftstereo-middlebury.pth --dataset middlebury_H
```

## Training

Our model is trained on two RTX-6000 GPUs using the following command. Training logs will be written to `runs/` which can be visualized using tensorboard.

```Shell
python train_stereo.py --batch_size 8 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 200000 --mixed_precision
```
To train using significantly less memory, change `--n_downsample 2` to `--n_downsample 3`. This will slightly reduce accuracy.

## (Optional) Faster Implementation

We provide a faster CUDA implementation of the correlation volume which works with mixed precision feature maps.
```Shell
cd sampler && python setup.py install && cd ..
```
Running demo.py, train_stereo.py or evaluate.py with `--corr_implementation reg_cuda` together with `--mixed_precision` will speed up the model without impacting performance.

To significantly decrease memory consumption on high resolution images, use `--corr_implementation alt`. This implementation is slower than the default, however.
