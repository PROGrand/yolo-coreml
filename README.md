# yolo-coreml for YOLOv3 and YOLOv4

## Quick Start
I will demonstrate, how to create and use realtime object detection engine using [YOLO](http://pjreddie.com/darknet/yolo/) and iOS.
For network creation i use Ubuntu 19.04 with NVidia GPU.
For iOS compilation i use Catalina and Xcode 11.
Also we need two virtualenvs in ubuntu - for python2 and python3.

1. Get and compile darknet, i recommend [AlexeyAB fork](https://github.com/AlexeyAB/darknet.git). Enable CUDA and OpenCV support.

2. Prepare image dataset. My network is for detection of SCRATCHES on 224x224 input. Refer to darknet docs if you need your own objects. Split images into scratch/positives and scratch/negatives. Positives must contain images with objects and txt files with boxes. Negatives must contain images without objects and empty txt files. You can use [Yolo_mark](https://github.com/AlexeyAB/Yolo_mark).

3. Create yolo darknet model. You can use my config and scripts from [scratch](scratch) folder for reference. 

4. Install Anaconda
```
wget https://repo.continuum.io/archive/Anaconda3-5.3.1-Linux-x86_64.sh
bash ./Anaconda3-5.3.1-Linux-x86_64.sh
```

5. yolo -> coreml:
```
conda create -n yolo2coreml python=3.6 anaconda
conda activate yolo2coreml
conda install tensorflow=1.14.0
conda install keras=2.3.1
conda install coremltools=4.1
python convertv4.py yolov4.cfg yolov4.weights yolov4.mlmodel
```

6. You can use ios project as reference. Copy yolov4.mlmodel to project folder. Check anchors in yolov4.cfg and swift code. Change classes names and count, anchors, network size if you use your owns.

## Performance
YOLOv3-Tiny 224x224 (SCRATCH) network takes about 40 detections per second on iPhone X.
YOLOv4 416x416 (COCO) network takes about 0.27 detections per second on iPhone 6.


## References
* [YOLO](http://pjreddie.com/darknet/yolo)
* [AlexeyAB](https://github.com/AlexeyAB/darknet.git)
* [allanzelener](https://github.com/allanzelener/YAD2K)
* [muyiguangda](https://github.com/muyiguangda/tensorflow-keras-yolov3)

---
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)
