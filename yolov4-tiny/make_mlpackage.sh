#!/bin/sh

sh ../prepare_cfg.sh yolov4-tiny.cfg yolov4-tiny_t.cfg

python ../convert_v4.py -n coco.names -c yolov4-tiny_t.cfg -w yolov4-tiny.weights -m yolov4-tiny.mlmodel -l RGB
