#!/bin/sh

sh ../prepare_cfg.sh scratch.cfg scratch_t.cfg

python ../convert_v4.py -n names.txt -c scratch_t.cfg -w scratch.weights -m scratch.mlpackage -l BGR	
