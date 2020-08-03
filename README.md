# RGBD semantic segmentation

A paper list of RGBD semantic segmentation.

*Last updated: 2020/08/03

#### Update log

*2020/May* - update all of recent papers and make some diagram about history of RGBD semantic segmentation.  
*2020/July* - update some recent papers (CVPR2020) of RGBD semantic segmentation.  
*2020/August* - update some recent papers (ECCV2020) of RGBD semantic segmentation.  

##

## Table of Contents

- [Datasets](https://github.com/Yangzhangcst/RGBD-semantic-segmentation/blob/master/README.md#Datasets)
- [Metrics](https://github.com/Yangzhangcst/RGBD-semantic-segmentation/blob/master/README.md#Metrics)
- [Performance tables](https://github.com/Yangzhangcst/RGBD-semantic-segmentation/blob/master/README.md#Performance-tables)
- [Paper list](https://github.com/Yangzhangcst/RGBD-semantic-segmentation/blob/master/README.md#paper-list)

##

## Datasets

The papers related to datasets used mainly in natural/color image segmentation are as follows.

- **[`[NYUDv2]`](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)** The NYU-Depth V2 dataset consists of 1449 RGB-D images showing interior scenes, which all labels are  usually mapped to 40 classes. The standard training and test set contain 795 and 654 images, respectively.
- **[`[SUN RGB-D]`](http://rgbd.cs.princeton.edu)** The SUN RGB-D dataset contains 10,335 RGBD images with semantic labels organized in 37 categories. The 5,285 images are used for training, and 5050 images are used for testing.
- **[`[2D-3D-S]`](http://buildingparser.stanford.edu/dataset.html)** Stanford-2D-3D-Semantic dataset contains 70496 RGB and depth images as well as 2D annotation with 13 object categories. Areas 1, 2, 3, 4, and 6 are utilized as the training and Area 5 is used as the testing set.

##

## Metrics

The papers related to metrics used mainly in RGBD semantic segmentation are as follows.

- **[PixAcc]**  Pixel  accuracy
- **[mAcc]**  Mean accuracy
- **[mIoU]** Mean intersection over union
- **[f.w.IOU]** Frequency weighted IOU


##

## Performance tables

Speed  is related to the hardware spec (e.g. CPU, GPU, RAM, etc), so it is hard to make an equal comparison. We select four indexes namely PixAcc, mAcc, mIoU, and f.w.IOU to make comparison. The closer the segmentation result is to the ground truth, the higher the above four indexes are.

### NYUDv2

|           Method           |   PixAcc    |    mAcc     |    mIoU    |     f.w.IOU      | Input |    Ref. from    |  Published  | Year |
| :------------------------: | :---------: | :---------: | :--------: | :--------------: | :---: | :-------------: | :---------: | ---- |
|          **POR**           |    59.1     |    28.4     |    29.1    |                  | RGBD  |                 |    CVPR     | 2013 |
|       **RGBD R-CNN**       |    60.3     |    35.1     |    31.3    |  47(in LSD-GF)   | RGBD  |                 |    ECCV     | 2014 |
|       **DeconvNet**        |    69.9     |    56.4     |    42.7    |        56        |  RGB  |   **LSD-GF**    |    ICCV     | 2015 |
|        **DeepLab**         |    68.7     |    46.9     |    36.8    |       52.5       | RGBD  |    **STD2P**    |    ICLR     | 2015 |
|        **CRF-RNN**         |    66.3     |    48.9     |    35.4    |        51        | RGBD  |    **STD2P**    |    ICCV     | 2015 |
|    **Multi-Scale  CNN**    |    65.6     |    45.1     |    34.1    |       51.4       |  RGB  | **LCSF-Deconv** |    ICCV     | 2015 |
|          **FCN**           |    65.4     |    46.1     |     34     |       49.5       | RGBD  | **LCSF-Deconv** |    CVPR     | 2015 |
|   **Mutex  Constraints**   |    63.8     |    31.5     |            | 48.5 (in LSD-GF) | RGBD  |                 |    ICCV     | 2015 |
|          **E2S2**          |    58.1     |    52.9     |     31     |       44.2       | RGBD  |    **STD2P**    |    ECCV     | 2016 |
|        **BI-3000**         |    58.9     |    39.3     |    27.7    |        43        | RGBD  |    **STD2P**    |    ECCV     | 2016 |
|        **BI-1000**         |    57.7     |    37.8     |    27.1    |       41.9       | RGBD  |    **STD2P**    |    ECCV     | 2016 |
|      **LCSF-Deconv**       |             |    47.3     |            |                  | RGBD  |                 |    ECCV     | 2016 |
|        **LSTM-CF**         |             |    49.4     |            |                  | RGBD  |                 |    ECCV     | 2016 |
|       **RDFNet-152**       |     76      |    62.8     |    50.1    |                  | RGBD  |                 |    ICCV     | 2017 |
|     **SCN-ResNet152**      |             |             |    49.6    |                  | RGBD  |                 |    ICCV     | 2017 |
|       **RDFNet-50**        |    74.8     |    60.4     |    47.7    |                  | RGBD  |                 |    ICCV     | 2017 |
|     **CFN(RefineNet)**     |             |             |    47.7    |                  | RGBD  |                 |    ICCV     | 2017 |
|     **RefineNet-152**      |    73.6     |    58.9     |    46.5    |                  |  RGB  |                 |    CVPR     | 2017 |
|         **LSD-GF**         |    71.9     |    60.7     |    45.9    |       59.3       | RGBD  |                 |    CVPR     | 2017 |
|         **3D-GNN**         |             |    55.7     |    43.1    |                  | RGBD  |                 |    ICCV     | 2017 |
|       **DML-Res50**        |             |             |    40.2    |                  |  RGB  |                 |    IJCAI    | 2017 |
|         **STD2P**          |    70.1     |    53.8     |    40.1    |       55.7       | RGBD  |                 |    CVPR     | 2017 |
|        **PBR-CNN**         |             |             |    33.2    |                  |  RGB  |                 |    ICCBS    | 2017 |
|        **B-SegNet**        |     68      |    45.8     |    32.4    |                  |  RGB  |                 |    BMVC     | 2017 |
|         **FC-CRF**         |    63.1     |     39      |    29.5    |       48.4       | RGBD  |                 |     TIP     | 2017 |
|          **LCR**           |    55.6     |    31.7     |    21.8    |       39.9       | RGBD  |                 |    ICIP     | 2017 |
|         **SegNet**         |    54.1     |    30.5     |     21     |       38.5       | RGBD  |     **LCR**     |    TPAMI    | 2017 |
|      **D-Refine-152**      |    74.1     |    59.5     |     47     |                  |  RGB  |                 |    ICPR     | 2018 |
|      **TRL-ResNet50**      |    76.2     |    56.3     |    46.4    |                  |  RGB  |                 |    ECCV     | 2018 |
|         **D-CNN**          |             |    56.3     |    43.9    |                  | RGBD  |                 |    ECCV     | 2018 |
|        **RGBD-Geo**        |    70.3     |    51.7     |    41.2    |       54.2       | RGBD  |                 |     MTA     | 2018 |
|        **Context**         |     70      |    53.6     |    40.6    |                  |  RGB  |                 |    TPAMI    | 2018 |
|      **DeepLab-LFOV**      |    70.3     |    49.6     |    39.4    |       54.7       | RGBD  |    **STD2P**    |    TPAMI    | 2018 |
|      **D-depth-reg**       |    66.7     |    46.3     |    34.8    |       50.6       | RGBD  |                 |     PRL     | 2018 |
|         **PU-Loop**        |    72.1     |             |    44.5    |                  |  RGB  |                 |    CVPR     | 2018 |
|          **GAD**           | ***84.8***  | ***68.7***  | ***59.6*** |                  |  RGB  |                 |    CVPR     | 2019 |
|         **CTS-IM**         |    76.3     |             |    50.6    |                  | RGBD  |                 |    ICIP     | 2019 |
|          **PAP**           |    76.2     |    62.5     |    50.4    |                  |  RGB  |                 |    CVPR     | 2019 |
|     **KIL-ResNet101**      |    75.1     |    58.4     |    50.2    |                  |  RGB  |                 |    ACPR     | 2019 |
|       **2.5D-Conv**        |    75.9     |             |    49.1    |                  | RGBD  |                 |    ICIP     | 2019 |
|         **ACNet**          |             |             |    48.3    |                  | RGBD  |                 |    ICIP     | 2019 |
|        **3M2RNet**         |     76      |     63      |     48     |                  | RGBD  |                 |     SIC     | 2019 |
|       **FDNet-16s**        |    73.9     |    60.3     |    47.4    |                  |  RGB  |                 |    AAAI     | 2019 |
|         **DMFNet**         |    74.4     |    59.3     |    46.8    |                  | RGBD  |                 | IEEE Access | 2019 |
|      **MMAF-Net-152**      |    72.2     |    59.2     |    44.8    |                  | RGBD  |                 |    arXiv    | 2019 |
|         **RTJ-AA**         |             |             |     42     |                  |  RGB  |                 |    ICRA     | 2019 |
|      **JTRL-ResNet50**     |    81.3     |    60.0     |    50.3    |                  |  RGB  |                 |    TPAMI    | 2019 |
|         **SGNet**          | <u>76.8</u> | <u>63.1</u> | <u>51</u>  |                  | RGBD  |                 |    arXiv    | 2020 |
|     **SCN-ResNet101**      |             |             |    48.3    |                  | RGBD  |                 |    TCYB     | 2020 |
| **RefineNet-Res152-Pool4** |    74.4     |    59.6     |    47.6    |                  |  RGB  |                 |    TPAMI    | 2020 |
|         **TSNet**          |    73.5     |    59.6     |    46.1    |                  | RGBD  |                 |   IEEE IS   | 2020 |
|     **PSD-ResNet50**       |    77.0     |    58.6     |    51.0    |                  |  RGB  |                 |    CVPR     | 2020 |
|     **Malleable 2.5D**     |    76.9     |             |    50.9    |                  | RGBD  |                 |    ECCV     | 2020 |
|      **BCMFP+SA-Gate**     |    77.9     |             |    52.4    |                  | RGBD  |                 |    ECCV     | 2020 |

### SUN RGB-D

|           Method           |   PixAcc    |    mAcc     |    mIoU    | f.w.IOU | Input |     Ref. from    | Published | Year |
| :------------------------: | :---------: | :---------: | :--------: | :-----: | :---: | :--------------: | :-------: | ---- |
|          **FCN**           |    68.2     |    38.4     |    27.4    |         |  RGB  |    **SegNet**    |   CVPR    | 2015 |
|       **DeconvNet**        |    66.1     |    32.3     |    22.6    |         |  RGB  |    **SegNet**    |   ICCV    | 2015 |
|          **IFCN**          |    77.7     |    55.5     |    42.7    |         |  RGB  |                  |   arXiv   | 2016 |
|     **CFN(RefineNet)**     |             |             |    48.1    |         | RGBD  |                  |   ICCV    | 2017 |
|       **RDFNet-152**       |    81.5     |    60.1     |    47.7    |         | RGBD  |                  |   ICCV    | 2017 |
|    **RefineNet-Res152**    |    80.6     |    58.5     |    45.9    |         |  RGB  |                  |   CVPR    | 2017 |
|         **3D-GNN**         |             |     57      |    45.9    |         | RGBD  |                  |   ICCV    | 2017 |
|       **DML-Res50**        |             |             |    42.3    |         |  RGB  |                  |   IJCAI   | 2017 |
|         **HP-SPS**         |    75.7     |    50.1     |     38     |         |  RGB  |                  |   BMVC    | 2017 |
|        **FuseNet**         |    76.3     |    48.3     |    37.3    |         | RGBD  |                  |   ACCV    | 2017 |
|          **LRN**           |    72.5     |    46.8     |    33.1    |         |  RGB  |                  |   arXiv   | 2017 |
|         **SegNet**         |    72.6     |    44.8     |    31.8    |         |  RGB  | **MMAF-Net-152** |   TPAMI   | 2017 |
|        **B-SegNet**        |    71.2     |    45.9     |    30.7    |         |  RGB  |                  |   BMVC    | 2017 |
|         **LSD-GF**         |             |     58      |            |         | RGBD  |                  |   CVPR    | 2017 |
|     **TRL-ResNet101**      |    84.3     |    58.9     |    50.3    |         |  RGB  |                  |   ECCV    | 2018 |
|        **CCF-GMA**         |    81.4     |    60.3     |    47.1    |         |  RGB  |                  |   CVPR    | 2018 |
|      **D-Refine-152**      |    80.8     |    58.9     |    46.3    |         |  RGB  |                  |   ICPR    | 2018 |
|        **Context**         |    78.4     |    53.4     |    42.3    |         |  RGB  |                  |   TPAMI   | 2018 |
|         **D-CNN**          |             |    53.5     |     42     |         | RGBD  |                  |   ECCV    | 2018 |
|     **G-FRNet-Res101**     |    75.3     |    47.5     |    36.9    |         |  RGB  |                  |   arXiv   | 2018 |
|      **DeepLab-LFOV**      |    71.9     |    42.2     |    32.1    |         |  RGB  |                  |   TPAMI   | 2018 |
|         **PU-Loop**        |    80.3     |             |    45.1    |         |  RGB  |                  |    CVPR   | 2018 |
|          **GAD**           | ***85.5***  | ***74.9***  | ***54.5*** |         |  RGB  |                  |   CVPR    | 2019 |
|     **KIL-ResNet101**      | <u>84.8</u> |     58      | <u>52</u>  |         |  RGB  |                  |   ACPR    | 2019 |
|          **PAP**           |    83.8     |    58.4     |    50.5    |         |  RGB  |                  |   CVPR    | 2019 |
|        **3M2RNet**         |    83.1     | <u>63.5</u> |    49.8    |         | RGBD  |                  |    SIC    | 2019 |
|          **CTS**           |    82.4     |             |    48.5    |         | RGBD  |                  |   ICIP    | 2019 |
|       **2.5D-Conv**        |    82.4     |             |    48.2    |         | RGBD  |                  |   ICIP    | 2019 |
|         **ACNet**          |             |             |    48.1    |         | RGBD  |                  |   ICIP    | 2019 |
|      **MMAF-Net-152**      |     81      |    58.2     |     47     |         | RGBD  |                  |   arXiv   | 2019 |
|        **LCR-RGBD**        |             |             |    42.4    |         | RGBD  |                  |   CVPRW   | 2019 |
|        **EFCN-8s**         |    76.9     |    53.5     |    40.7    |         |  RGB  |                  |    TIP    | 2019 |
|         **DSNet**          |    75.6     |             |    32.1    |         |  RGB  |                  |  ICASSP   | 2019 |
|      **JTRL-ResNet101**    |    84.8     |    59.1     |    50.8    |         |  RGB  |                  |   TPAMI   | 2019 |
|     **SCN-ResNet152**      |             |             |    50.7    |         | RGBD  |                  |   TCYB    | 2020 |
|         **SGNet**          |    81.8     |    60.9     |    48.5    |         | RGBD  |                  |   arXiv   | 2020 |
|         **CGBNet**         |    82.3     |    61.3     |    48.2    |         |  RGB  |                  |    TIP    | 2020 |
|    **CANet-ResNet101**     |    81.9     |             |    47.7    |         |  RGB  |                  |   arXiv   | 2020 |
| **RefineNet-Res152-Pool4** |    81.1     |    57.7     |     47     |         |  RGB  |                  |   TPAMI   | 2020 |
|     **PSD-ResNet50**       |    84.0     |    57.3     |    50.6    |         |  RGB  |                  |    CVPR   | 2020 |
|     **BCMFP+SA-Gate**      |    82.5     |             |    49.4    |         | RGBD  |                  |    ECCV   | 2020 |

### 2D-3D-S

|      Method      | PixAcc | mAcc | mIoU | f.w.IOU | Input |    Ref. from     | Published | Year |
| :--------------: | :----: | :--: | :--: | :-----: | :---: | :--------------: | :-------: | ---- |
|   **Deeplab**    |  64.3  | 46.7 | 35.5 |  48.5   | RGBD  | **MMAF-Net-152** |   ICLR    | 2015 |
| **DeepLab-LFOV** |  88.0  | 42.2 | 69.8 |         |  RGB  |   **PU-Loop**    |   TPAMI   | 2018 |
|    **D-CNN**     |  65.4  | 55.5 | 39.5 |  49.9   | RGBD  |                  |   ECCV    | 2018 |
|   **PU-Loop**    |  91.0  |      | 76.5 |         |  RGB  |                  |    CVPR   | 2018 |
| **MMAF-Net-152** |  76.5  | 62.3 | 52.9 |         | RGBD  |                  |   arXiv   | 2019 |
|   **3M2RNet**    |  79.8  | 75.2 |  63  |         | RGBD  |                  |    SIC    | 2019 |

##

## Paper list

- **[POR]** Gupta, S., et al. (2013). Perceptual Organization and Recognition of Indoor Scenes from RGB-D Images. IEEE Conference on Computer Vision and Pattern Recognition: 564-571.
- **[RGBD  R-CNN]** Gupta, S., et al. (2014). Learning Rich Features from RGB-D Images for Object Detection and Segmentation. European Conference on Computer Vision: 345-360. 
- **[FCN]** Long, J., et al. (2015). Fully convolutional networks for semantic segmentation. IEEE Conference on Computer Vision and Pattern Recognition: 3431-3440.
- **[CRF-RNN]** Zheng, S., et al. (2015). Conditional Random Fields as Recurrent Neural Networks. IEEE International Conference on Computer Vision: 1529-1537.
- **[Mutex Constraints]** Deng, Z., et al. (2015). Semantic Segmentation of RGBD Images with Mutex Constraints. IEEE International Conference on Computer Vision: 1733-1741.
- **[DeepLab]** Chen, L., et al. (2015). Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs. International Conference on Learning Representations.
- **[Multi-Scale  CNN]** Eigen, D. and R. Fergus (2015). Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-scale Convolutional Architecture. IEEE International Conference on Computer Vision: 2650-2658.
- **[DeconvNet]** Noh, H., et al. (2015). Learning Deconvolution Network for Semantic Segmentation. International Conference on Computer Vision: 1520-1528.
- **[LSTM-CF]** Li, Z., et al. (2016). LSTM-CF: Unifying Context Modeling and Fusion with LSTMs for RGB-D Scene Labeling. European Conference on Computer Vision: 541-557.
- **[LCSF-Deconv]** Wang, J., et al. (2016). Learning Common and Specific Features for RGB-D Semantic Segmentation with Deconvolutional Networks. European Conference on Computer Vision: 664-679.
- **[BI]** Gadde, R., et al. (2016). Superpixel Convolutional Networks using Bilateral Inceptions. European Conference on Computer Vision: 597-613.
- **[E2S2]** Caesar, H., et al. (2016). Region-Based Semantic Segmentation with End-to-End Training. European Conference on Computer Vision: 381-397.
- **[IFCN]** Shuai, B., et al. (2016). Improving Fully Convolution Network for Semantic Segmentation. arXiv:1611.08986.
- **[SegNet]** Badrinarayanan, V., et al. (2017). SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence 39(12): 2481-2495.
- **[LSD-GF]** Cheng, Y., et al. (2017). Locality-Sensitive Deconvolution Networks with Gated Fusion for RGB-D Indoor Semantic Segmentation. IEEE Conference on Computer Vision and Pattern Recognition: 1475-1483.
- **[LCR]** Chu, J., et al. (2017). Learnable contextual regularization for semantic segmentation of indoor scene images. IEEE International Conference on Image Processing: 1267-1271.
- **[RefineNet]** Lin, G., et al. (2017). RefineNet: Multi-path Refinement Networks for High-Resolution Semantic Segmentation. IEEE Conference on Computer Vision and Pattern Recognition : 5168-5177,
- **[FuseNet]** Hazirbas, C., et al. (2017). FuseNet: Incorporating Depth into Semantic Segmentation via Fusion-Based CNN Architecture. Asian Conference on Computer Vision: 213-228.
- **[STD2P]** He, Y., et al. (2017). STD2P: RGBD Semantic Segmentation Using Spatio-Temporal Data-Driven Pooling. IEEE Conference on Computer Vision and Pattern Recognition: 7158-7167.
- **[RDFNet]** Lee, S., et al. (2017). RDFNet: RGB-D Multi-level Residual Feature Fusion for Indoor Semantic Segmentation. IEEE International Conference on Computer Vision: 4990-4999.
- **[CFN(RefineNet)]** Lin, D., et al. (2017). Cascaded Feature Network for Semantic Segmentation of RGB-D Images. IEEE International Conference on Computer Vision: 1320-1328.
- **[3D-GNN]** Qi, X., et al. (2017). 3D Graph Neural Networks for RGBD Semantic Segmentation. IEEE International Conference on Computer Vision: 5209-5218.
- **[DML-Res50]** Shen, T., et al. (2017). Learning Multi-level Region Consistency with Dense Multi-label Networks for Semantic Segmentation. Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence: 2708-2714.
- **[PBR-CNN]** Zhang, Y., et al. (2017). Physically-Based Rendering for Indoor Scene Understanding Using Convolutional Neural Networks. IEEE Conference on Computer Vision and Pattern Recognition: 5057-5065.
- **[FC-CRF]** Liu, F., et al. (2017). Discriminative Training of Deep Fully Connected Continuous CRFs With Task-Specific Loss. IEEE Transactions on Image Processing 26(5), 2127-2136.
- **[HP-SPS]** Park, H., et al. (2017). Superpixel-based semantic segmentation trained by statistical process control. British Machine Vision Conference.
- **[LRN]** Islam, M. A., et al. (2017). Label Refinement Network for Coarse-to-Fine Semantic Segmentation. arXiv1703.00551.
- **[G-FRNet-Res101]** Islam, M. A., et al. (2018). Gated Feedback Refinement Network for Coarse-to-Fine Dense Semantic Image Labeling. arXiv:1806.11266
- **[CCF-GMA]** Ding, H., et al. (2018). Context Contrasted Feature and Gated Multi-scale Aggregation for Scene Segmentation. IEEE Conference on Computer Vision and Pattern Recognition: 2393-2402.
- **[Context]** Lin, G., et al. (2018). Exploring Context with Deep Structured Models for Semantic Segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(6), 1352-1366.
- **[D-Refine-152]** Chang, M., et al. (2018). Depth-assisted RefineNet for Indoor Semantic Segmentation. International Conference on Pattern Recognition: 1845-1850.
- **[D-depth-reg]** Guo, Y. and T. Chen (2018). Semantic segmentation of RGBD images based on deep depth regression. Pattern Recognition Letters 109: 55-64.
- **[RGBD-Geo]** Liu, H., et al. (2018). RGB-D joint modeling with scene geometric information for indoor semantic segmentation. Multimedia Tools and Applications 77(17): 22475-22488.
- **[D-CNN]** Wang, W. and U. Neumann (2018). Depth-aware CNN for RGB-D Segmentation. European Conference on Computer Vision: 144-161.
- **[TRL-ResNet50/101]** Zhang, Z., et al. (2018). Joint Task-Recursive Learning for Semantic Segmentation and Depth Estimation. European Conference on Computer Vision.
- **[DeepLab-LFOV]** Chen, L., et al. (2018). DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(4), 834-848.
- **[PU-Loop]** Kong, S. and C. Fowlkes (2018). Recurrent Scene Parsing with Perspective Understanding in the Loop. IEEE Conference on Computer Vision and Pattern Recognition: 956-965.
- **[EFCN-8s]** Shuai, B., et al. (2019). Toward Achieving Robust Low-Level and High-Level Scene Parsing. IEEE Transactions on Image Processing, 28(3), 1378-1390.
- **[3M2RNet]** Fooladgar, F., and Kasaei, S. (2019). 3M2RNet: Multi-Modal Multi-Resolution Refinement Network for Semantic Segmentation. Science and Information Conference: 544-557.
- **[RFBNet]** Deng, L., et al. (2019). RFBNet: Deep Multimodal Networks with Residual Fusion Blocks for RGB-D Semantic Segmentation. arXiv:1907.00135 
- **[MMAF-Net-152]** Fooladgar, F. and S. Kasaei (2019). "Multi-Modal Attention-based Fusion Model for Semantic Segmentation of RGB-Depth Images." arXiv:1912.11691.
- **[LCR-RGBD]** Giannone, G. and B. Chidlovskii (2019). Learning Common Representation from RGB and Depth Images. IEEE Conference on Computer Vision and Pattern Recognition Workshops.
- **[ACNet]** Hu, X., et al. (2019). ACNET: Attention Based Network to Exploit Complementary Features for RGBD Semantic Segmentation. IEEE International Conference on Image Processing: 1440-1444.
- **[DSNet]** Jiang, F., et al. (2019). DSNET: Accelerate Indoor Scene Semantic Segmentation. IEEE International Conference on Acoustics, Speech and Signal Processing: 3317-3321.
- **[GAD]** Jiao, J., et al. (2019). Geometry-Aware Distillation for Indoor Semantic Segmentation. IEEE Conference on Computer Vision and Pattern Recognition: 2864-2873.
- **[RTJ-AA]** Nekrasov, V., et al. (2019). Real-Time Joint Semantic Segmentation and Depth Estimation Using Asymmetric Annotations. International Conference on Robotics and Automation: 7101-7107.
- **[CTS-IM]** Xing, Y., et al. (2019). Coupling Two-Stream RGB-D Semantic Segmentation Network by Idempotent Mappings. IEEE International Conference on Image Processing: 1850-1854.
- **[2.5D-Conv]** Xing, Y. J., et al. (2019). 2.5d Convolution for RGB-D Semantic Segmentation. IEEE International Conference on Image Processing: 1410-1414.
- **[DMFNet]** Yuan, J., et al. (2019). DMFNet: Deep Multi-Modal Fusion Network for RGB-D Indoor Scene Segmentation. IEEE Access 7: 169350-169358.
- **[PAP]** Zhang, Z., et al. (2019). Pattern-Affinitive Propagation Across Depth, Surface Normal and Semantic Segmentation. IEEE Conference on Computer Vision and Pattern Recognition: 4101-4110.
- **[KIL-ResNet101]** Zhou, L., et al. (2019). KIL: Knowledge Interactiveness Learning for Joint Depth Estimation and Semantic Segmentation. Asian Conference on Pattern Recognition: 835-848.
- **[FDNet-16s]** Zhen, M., et al. (2019). Learning Fully Dense Neural Networks for Image Semantic Segmentation. The Thirty-Third AAAI Conference on Artificial Intelligence: 9283-9290.
- **[JTRL-ResNet50/101]** Zhang, Z., et al. (2019). "Joint Task-Recursive Learning for RGB-D Scene Understanding." IEEE Transactions on Pattern Analysis and Machine Intelligence.
- **[SGNet]** Chen, L.-Z., et al. (2020). Spatial Information Guided Convolution for Real-Time RGBD Semantic Segmentation. arXiv:2004.04534.
- **[SCN-ResNet101]** Lin, D., et al. (2020). SCN: Switchable Context Network for Semantic Segmentation of RGB-D Images. IEEE Transactions on Cybernetics 50(3): 1120-1131.
- **[RefineNet-Res152-Pool4]** Lin, G., et al. (2020). RefineNet: Multi-Path Refinement Networks for Dense Prediction. IEEE Transactions on Pattern Analysis and Machine Intelligence 42(5): 1228-1242.
- **[CANet-ResNet101]** Tang, Q., et al. (2020). Attention-guided Chained Context Aggregation for Semantic Segmentation. arXiv:2002.12041 
- **[CGBNet]** Ding, H., et al. (2020). Semantic Segmentation with Context Encoding and Multi-Path Decoding. IEEE Transactions on Image Processing 29: 3520-3533.
- **[TSNet]** Zhou, W., et al. (2020). TSNet: Three-stream Self-attention Network for RGB-D Indoor Semantic Segmentation. IEEE Intelligent Systems.
- **[PSD-ResNet50]** Zhou, L., et al. (2020). Pattern-Structure Diffusion for Multi-Task Learning. IEEE Conference on Computer Vision and Pattern Recognition.
- **[Malleable 2.5D]** Xing, Y., et al. (2020). Malleable 2.5D Convolution: Learning Receptive Fields along the Depth-axis for RGB-D Scene Parsing. European Conference on Computer Vision.
- **[BCMFP+SA-Gate]** Chen X., et al. (2020). Bi-directional Cross-Modality Feature Propagation with Separation-and-Aggregation Gate for RGB-D Semantic Segmentation. European Conference on Computer Vision.

##

## Contact & Feedback

If you have any suggestions about this project, feel free to contact me.

- [e-mail: yzhangcst[at]smail.nju.edu.cn]
