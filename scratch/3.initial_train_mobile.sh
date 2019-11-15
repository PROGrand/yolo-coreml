#!/bin/bash

mkdir dataset/backup

../darknet/darknet detector train dataset/scratch.data scratch.cfg yolov3-tiny.conv.15 -dont_show -mjpeg_port 8090 -json_port 8070 -map

