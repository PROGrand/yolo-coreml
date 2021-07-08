# yolo-coreml

## Quick Start
I will demonstrate, how to create and use realtime object detection engine using [YOLOv3](http://pjreddie.com/darknet/yolo/) and iOS.
For network creation i use Ubuntu 19.04 with NVidia GPU.
For iOS compilation i use Catalina and Xcode 11.
Also we need two virtualenvs in ubuntu - for python2 and python3.

1. Get and compile darknet, i recommend [AlexeyAB fork](https://github.com/AlexeyAB/darknet.git). Enable CUDA and OpenCV support.

2. Prepare image dataset. My network is for detection of SCRATCHES on 224x224 input. Refer to darknet docs if you need your own objects. Split images into scratch/positives and scratch/negatives. Positives must contain images with objects and txt files with boxes. Negatives must contain images without objects and empty txt files. You can use [Yolo_mark](https://github.com/AlexeyAB/Yolo_mark).

3. Create yolov3 darknet model. You can use my config and scripts from [scratch](scratch) folder for reference. 

4 Instal Anaconda
```
wget https://repo.continuum.io/archive/Anaconda3-5.3.1-Linux-x86_64.sh
bash ./Anaconda3-5.3.1-Linux-x86_64.sh
```

5. yolov3 -> keras (requires python3) using anaconda:
```
conda create -n yolo2keras python=3.6 anaconda
conda activate yolo2keras
conda install tensorflow=1.14.0
conda install keras=2.3.1
```

6. keras -> coreml (requires python2.7):
```
virtualenv -p /usr/bin/python2.7 keras2coreml
source keras2coreml/bin/activate
pip install tensorflow==1.14.0 keras==2.3.1 coremltools==3.1
python coreml.py scratch
```

7. See [how to use created mlmodel](https://github.com/Mrlawrance/yolov3-ios/tree/master/ios). Change classes names and count, anchors, network size if you use your owns.

## Performance
SCRATCH network gives about 40 scratch detections per second on iPhone X.

## References
* [YOLOv3](http://pjreddie.com/darknet/yolo)
* [AlexeyAB](https://github.com/AlexeyAB/darknet.git)
* [allanzelener](https://github.com/allanzelener/YAD2K)
* [muyiguangda](https://github.com/muyiguangda/tensorflow-keras-yolov3)

---
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)
