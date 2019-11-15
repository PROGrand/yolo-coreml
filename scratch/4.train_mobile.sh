#!/bin/bash

mkdir dataset/backup

../darknet/darknet detector train dataset/scratch.data scratch.cfg scratch_last.weights -dont_show -mjpeg_port 8090 -json_port 8070 -map

