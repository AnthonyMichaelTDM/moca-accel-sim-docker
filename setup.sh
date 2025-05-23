#!/bin/bash

# This script is used to download the necessary context for the Dockerfile

# Clone the Accel-Sim-Framework repository (if it doesn't already exist)
if [ -d "accel-sim-framework" ]; then
    echo "Directory accel-sim-framework already exists. Skipping git clone."
else
    git clone https://github.com/AnthonyMichaelTDM/accel-sim-framework.git
    if ! [ -d "accel-sim-framework/gpu-simulator/gpgpu-sim" ]; then
        cd accel-sim-framework/gpu-simulator
        git clone https://github.com/accel-sim/gpgpu-sim_distribution.git gpgpu-sim
        cd -
    fi
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