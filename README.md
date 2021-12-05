# CoreML for YOLOv3 and YOLOv4

## Quick Start

I will demonstrate, how to create and use realtime object detection engine
using [YOLO](http://pjreddie.com/darknet/yolo/) and iOS. For network creation i use Ubuntu 19.04 with NVidia GPU. For
iOS conversion and compilation i use Monterey and Xcode 13.1.

1. Get and compile darknet, i recommend [AlexeyAB fork](https://github.com/AlexeyAB/darknet.git). Enable CUDA and OpenCV
   support.

2. Prepare image dataset. My network is for detection of SCRATCHES on 224x224 input. Refer to darknet docs if you need
   your own objects. Split images into scratch/positives and scratch/negatives. Positives must contain images with
   objects and txt files with boxes. Negatives must contain images without objects and empty txt files. You can
   use [Yolo_mark](https://github.com/AlexeyAB/Yolo_mark).

3. Create yolo darknet model.

## YOLOv3, YOLOv4, YOLOv4-TINY

Use this method for devices with iOS >= 13. Currently script generates iOS15 MLProgram mlpackage, but can be easily
modified for iOS13 and mlmodel. YOLOv4-TINY work well. Suddenly, large YOLOv4 mlpackage takes minutes to loading on
every iOS example app launch. At least on iPhone12 with iOS15.0.1

1. `coremltools` is very sensitive to packages versions. This is why you need dedicated python `anaconda` environment.
Install Anaconda from: https://repo.anaconda.com/archive/Anaconda3-5.3.1-MacOSX-x86_64.pkg.

2. In Terminal enter conda environment (assuming anaconda installed to /anaconda3):

```shell
. /anaconda3/etc/profile.d/conda.sh
conda create -n coremltools-env python=3.7
conda activate coremltools-env
pip install yolov4=3.2.0
pip install opencv-python=4.5.4.60
pip install h5py=1.5.2
pip install coremltools=5.1.0
pip install keras==2.2.4
pip install tensorflow==2.5.0
```

3. Prepare `yolov4-tiny.cfg` file (clear unsupported learning tags like `subdivisions` if any). Keep original `yolov4-tiny.cfg` for further trainings. Example:

```shell
sh ./prepare_cfg.sh yolov4-tiny.cfg yolov4-tiny_temp.cfg 
```

4. Use prepared `yolov4-tiny_temp.cfg`. Convert:

```shell
python ./convert_tiny.py -n coco.names -c yolov4-tiny_temp.cfg -w yolov4-tiny.weights -m yolov4.mlpackage
```

## YOLOv3, YOLOv3-TINY, YOLOv4-Mish for iOS12

Use this method for unsupported devices with iOS < 13. Also see appropriate iOS App example.

1. Install Anaconda from: https://repo.anaconda.com/archive/Anaconda3-5.3.1-MacOSX-x86_64.pkg

2. yolo -> coreml:

```
conda create -n yolo2coreml python=3.6 anaconda
conda activate yolo2coreml
conda install tensorflow=1.14.0
conda install keras=2.3.1
conda install coremltools=4.1
python convert_v4_old.py yolov4.cfg yolov4.weights yolov4.mlmodel
```

3. You can use ios project as reference. Copy yolov4.mlmodel to project folder. Check anchors in yolov4.cfg and swift
   code. Change classes names and count, anchors, network size if you use your owns.

## Performance

- YOLOv3-Tiny 224x224 (SCRATCH) network takes about 25 ms per detection on iPhone X.
- YOLOv4 old method 416x416 (COCO) network takes about 5 second per detection on iPhone 6.
- YOLOv4 608x608 (COCO) network takes about 10 seconds per detection on iPhone 12.
- YOLOv4-TINY 416x416 (COCO) network takes about 19 ms per detection on iPhone 12.

## References

* [YOLO](http://pjreddie.com/darknet/yolo)
* [AlexeyAB](https://github.com/AlexeyAB/darknet.git)
* [coremltools](https://coremltools.readme.io/docs)
* [python-yolov4](https://wiki.loliot.net/docs/lang/python/libraries/yolov4/python-yolov4-about/)

---
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)
