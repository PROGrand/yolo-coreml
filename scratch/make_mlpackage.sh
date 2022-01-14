#!/bin/sh

#sh ../prepare_cfg.sh scratch.cfg scratch_t.cfg
sh ../prepare_cfg.sh scratch_m7.cfg scratch_m7_t.cfg

python ../convert_v4.py -n names.txt -c scratch_m7_t.cfg -w scratch_m7.weights -m scratch.mlpackage -l BGR	
