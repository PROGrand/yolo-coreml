#!/bin/bash

rm yolov3-tiny.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights
../darknet/darknet partial yolov3-tiny.cfg yolov3-tiny.weights yolov3-tiny.conv.15 15
