#!/bin/bash
SCRIPT=$1
export PATH=$PATH:/usr/local/cuda-7.0/bin
export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64
python $SCRIPT
