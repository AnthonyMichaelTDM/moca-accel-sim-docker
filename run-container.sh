#!/bin/bash

./setup.sh
# check if the docker image exists
if [ "$(docker image ls -q moca-accel-sim)" == "" ]; then
    echo "Building the docker image"
    docker build -t moca-accel-sim .
else
    echo "Docker image already exists"
fi

# if nvidia-container-toolkit is installed, run the container with GPU support
if [ "$(command -v nvidia-container-toolkit)" != "" ]; then
    echo "Running the container with GPU support"
    docker run -it --gpus all -v ./shared:/shared moca-accel-sim
else
    echo "Running the container without GPU support"
    docker run -it -v ./shared:/shared moca-accel-sim
fi