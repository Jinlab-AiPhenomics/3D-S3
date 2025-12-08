# 3D-S3

*****3D-S3***: A Comprehensive semi-supervised Segmentation Framework for Crop Point Cloud under Limited Labeling Conditions**

-  Innovation: We introduced a two-stage semi-supervised segmentation framework (***3D-S3***) to alleviate the challenge of leaf ear segmentation in wheat and rice point clouds. The framework is mainly composed of four modules: adaptive point cloud transformation, global similarity, intra-class similarity, and inter-class separation.

<p align="center">
<img src="docs/projects/misc/3D-S3.png" width=85% height=85% class="center">
</p>

## 

## Module Introduction
[***3D-S3***](https://github.com/Tang-855/3D-Seg/) consists of four major components：. 

1. **Adaptive point cloud branching transformation**: randomly selects a combination of rotation, scaling and translation strategies in a local neighborhood to generate branching point clouds with similar global semantic features;
   
2. **Global similarity module**: calculates the similarity of two branching point clouds on a global scale;

3. **Intra-class similarity module**: evaluates the similarity between each neighboring point and the sample centroid in the same superpoint by constructing a loss function in the KNN grouped neighborhood;

4. **Inter-class similarity module**: semi-supervised clustering grouping through clustering algorithms (e.g., K-means++), initial clustering superpoints based on the realization of the three-dimensional coordinate features of the two branch point clouds, based on the fusion of the three-dimensional coordinate features and high-dimensional features, to complete further iterative clustering process
---


## ***3D-S3*** showed a significant improvement in segmentation performance for rice and wheat point clouds.  

**Wheat**
-  Under the 10 sample sizes of [10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%], the segmentation performance of *DGCNN, Point Cloud Transformer, PointNet++, PointNeXt, PointMLP* and *PointConv* based on fully supervised learning and ***3D-S3*** has been improved to varying degrees.
<p align="center">
<img src="docs/projects/misc/Wheat_training size.png" width=85% height=85% class="center">
</p>

-  Under the 12 annotation ratio of [1%, 5%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%], the segmentation performance of *PointConv* based on fully supervised learning and ***3D-S3*** , Significantly reduce PointConv's reliance on the proportion of labeled data.
<p align="center">
<img src="docs/projects/misc/wheat_annotation levels.png" width=85% height=85% class="center">
</p>


**Rice**
-  Based on the *PointMLP* model, the mIoU trend of semi-supervised and fully supervised learning in different rice samples
<p align="center">
<img src="docs/projects/misc/rice_samle.png" width=85% height=85% class="center">
</p>

-  Based on the *PointMLP* model, the mIoU trend of semi-supervised and fully supervised learning in different annotation ratio of rice sample
<p align="center">
<img src="docs/projects/misc/rice_annotation.png" width=85% height=85% class="center">
</p>

## Model Performance
Performance of ***3D-S3*** improvements are shown in the following areas:

1. **Robustness**: ***3D-S3*** show segmentation performance on *DGCNN, Point Cloud Transformer, PointNet++, PointNeXt, PointMLP* and *PointConv* models.

2. **Smaller sample size**：***3D-S3*** provide more improvement in segmentation accuracy when the amount of training data is smaller.

3.  **Lower annotation ratio**：***3D-S3*** It can reduce the model's dependence on the proportion of labeled data in training.

4. **Extensibility**：***3D-S3*** Significantly Improves Accuracy of Small-Sample Training on Wheat and Rice Datasets


## Installation
We provide a simple bash file to install the environment:

```
git clone --recurse-submodules git@github.com:Tang-855/3D-USF.git
cd 3D-USF 
source update.sh
source install.sh
```
If the above command fails to install successfully, please open the `install.sh` and follow the steps to install manually. See [Install](https://github.com/Tang-855/3D-USF/blob/main/install.sh) for detail. 



## Run 

Check our [***3D-s3***](https://github.com/Tang-855/3D-USF/) for detailed instructions. 

The installation commands for *DGCNN, Point Cloud Transformer, PointNet++, PointNeXt, PointMLP* and *PointConv* models are basically the same, using DGCNN as an example:
For example: 
-  Running the PointConv model:

```
cd your root/output/
chmod +x nohup.sh
./nohup.sh &
```

If you want to run other models, please modify the run command in the nohup.sh file. For example, if you want to run the DGCNN model, change the command to:
```
nohup python dgcnn_main.py > /your root/Pointnext/output/dgcnn.log 2>&1 &
```

If you want to run other models, please modify the run command in the nohup.sh file. For example, if you want to run the Point Transformer model, change the command to:
```
nohup python pct_main.py > /your root/Pointnext/output/pct.log 2>&1 &
```

Other models are similar; you only need to change the python file name of ""，that you want to run in the run command of the n file.

### Acknowledgment
This library is inspired by [PointNeXt](https://guochengqian.github.io/PointNeXt/modelzoo/). 

### Citation
If you find ***3D-S3*** useful, please cite.
