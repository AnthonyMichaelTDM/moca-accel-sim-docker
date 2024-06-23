#!/bin/bash

./setup.sh
# check if the docker image exists
if [ "$(docker image ls -q moca-accel-sim)" == "" ]; then
    echo "Building the docker image"
    docker build -t moca-accel-sim .
else
    echo "Docker image already exists"
fi
docker run -it -v shared:/shared moca-accel-sim