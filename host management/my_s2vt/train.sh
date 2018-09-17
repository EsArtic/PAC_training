#!/usr/bin/env sh

#gdb --args 
./build/tools/caffe train --solver=examples/my_s2vt/s2vt_solver.prototxt --gpu=1 2>&1 | tee -a s2vt_train.log
