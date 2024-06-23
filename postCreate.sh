#!/bin/bash
cd accel-sim-framework

# Accel sim tracer
./util/tracer_nvbit/install_nvbit.sh  
make -C ./util/tracer_nvbit/

# install Accel-Sim SASS Frontend and Simulation Engine
pip3 install -r requirements.txt
source gpu-simulator/setup_environment.sh
make -j -C gpu-simulator/

# Copy the SM7_QV100 configuration file to the GPU app collection
cp gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100 /accel-sim/gpu-app-collection/bin/11.7/release/ -r

# Accel-Sim Tuner