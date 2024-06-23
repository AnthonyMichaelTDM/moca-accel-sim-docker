#!/bin/bash

# This script is used to download the necessary context for the Dockerfile

# Clone the Accel-Sim-Framework repository
git clone https://github.com/accel-sim/accel-sim-framework.git

# get the cuda 11.7 installer
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run

# get some stuff for the traces
wget ftp://ftp.ecn.purdue.edu/tgrogers/accel-sim/traces/1.1.0.trace.summary.txt
wget ftp://ftp.ecn.purdue.edu/tgrogers/accel-sim/traces/tesla-v100/1.1.0.latest/rodinia_2.0-ft.tgz
tar -xvzf rodinia_2.0-ft.tgz
rm rodinia_2.0-ft.tgz

# Clone the Accel-Sim gpu-app-collection repository
git clone https://github.com/accel-sim/gpu-app-collection