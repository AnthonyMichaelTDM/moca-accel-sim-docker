#!/bin/bash

# This script is used to download the necessary context for the Dockerfile

# Clone the Accel-Sim-Framework repository (if it doesn't already exist)
if [ -d "accel-sim-framework" ]; then
    echo "Directory accel-sim-framework already exists. Skipping git clone."
else
    git clone https://github.com/accel-sim/accel-sim-framework.git
fi

# get the cuda 11.7 installer (if it doesn't already exist)
if [ -f "cuda_11.7.0_515.43.04_linux.run" ]; then
    echo "File cuda_11.7.0_515.43.04_linux.run already exists. Skipping download."
else
    wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
fi

# get some stuff for the traces (if they don't already exist)
if [ -f "1.1.0.trace.summary.txt" ]; then
    echo "File 1.1.0.trace.summary.txt already exists. Skipping download."
else
    wget ftp://ftp.ecn.purdue.edu/tgrogers/accel-sim/traces/1.1.0.trace.summary.txt
fi
if [ -d "rodinia_2.0-ft" ]; then
    echo "Directory rodinia_2.0-ft already exists. Skipping download."
else
    wget ftp://ftp.ecn.purdue.edu/tgrogers/accel-sim/traces/tesla-v100/1.1.0.latest/rodinia_2.0-ft.tgz
    tar -xvzf rodinia_2.0-ft.tgz
    rm rodinia_2.0-ft.tgz
fi

# Clone the Accel-Sim gpu-app-collection repository
if [ -d "gpu-app-collection" ]; then
    echo "Directory gpu-app-collection already exists. Skipping git clone."
else
    git clone https://github.com/accel-sim/gpu-app-collection
fi