### Disclaimer

An implementation of our method. The code is constructed based on [Faster R-CNN](https://github.com/ShaoqingRen/faster_rcnn) and [ORN](https://github.com/ZhouYanzhao/ORN).

### Software

1. Ubuntu 16.04 + Anaconda + Python 2.7 + CUDA 8.0 + CUDNN 6.0

2. Other dependencies required to install [caffe](http://caffe.berkeleyvision.org/).

### Hardware

Any NVIDIA GPU that supports CUDA acceleration and has at least 3G memory.

### Installation

1. Clone the repository
  ```Shell
  git clone https://github.com/lilinhao/ShipDetection.git
  ```
2. Build the Cython modules
    ```Shell
    cd $SDET_ROOT/lib
    make
    ```
3. Build Caffe and pycaffe
    ```Shell
    cd $SDET_ROOT/caffe-fast-rcnn
    make -j8 && make pycaffe
    ```

### Usage

Our dataset can be download from [here](https://pan.baidu.com/s/1gDypy10iHHMH_9eOL1VzWw) with password `4t5e`. The dataset is organized according to the VOC2007 dataset. If you want to train on your own data, you can convert it to VOC2007 format.

Examples of the annotations in our xml files are as follows:

```python
<rotbox>
<rbox_cx>174.268870</rbox_cx>
<rbox_cy>512.186141</rbox_cy>
<rbox_w>143.288621</rbox_w>
<rbox_h>17.665301</rbox_h>
<rbox_ang>-0.219796</rbox_ang>
```

The annotated information is the center coordinate, width, height and orientation of the bounding box from top to bottom. The orientation ranges from -π/2 to π/2 and the counterclockwise direction is positive

Download the VGG16 model to `$FRCN_ROOT/data/imagenet_models` and use `experiments/scripts/ship_detection_end2end.sh`to train and test a ship detector. Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT
./experiments/scripts/ship_detection_end2end.sh [GPU_ID] VGG16 pascal_voc
# GPU_ID is the GPU you want to train on
```
