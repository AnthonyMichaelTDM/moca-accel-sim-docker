# the context for this dockerfile can be generated with the following commands, which are
# conveniently provided in setup.sh
# ```sh
# # Clone the Accel-Sim-Framework repository
# git clone https://github.com/accel-sim/accel-sim-framework.git
#
# # get some stuff for the traces
# wget ftp://ftp.ecn.purdue.edu/tgrogers/accel-sim/traces/1.1.0.trace.summary.txt
# wget ftp://ftp.ecn.purdue.edu/tgrogers/accel-sim/traces/tesla-v100/1.1.0.latest/rodinia_2.0-ft.tgz
# tar -xvzf rodinia_2.0-ft.tgz
# rm rodinia_2.0-ft.tgz
#
# # Clone the Accel-Sim gpu-app-collection repository
# git clone https://github.com/accel-sim/gpu-app-collection
# ```

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

SHELL ["/bin/bash", "-c"]

WORKDIR /accel-sim

ENV CUDA_INSTALL_PATH /usr/local/cuda
ENV PTXAS_CUDA_INSTALL_PATH /usr/local/cuda
ENV BOOST_ROOT=/usr/local/boost
ENV PATH=$CUDA_INSTALL_PATH/bin:$PATH
ENV GPUAPPS_ROOT /accel-sim/gpu-app-collection

# install prerequisites
RUN apt-get update
RUN apt-get install -y wget build-essential xutils-dev bison zlib1g-dev flex \
      libglu1-mesa-dev git g++ libssl-dev libxml2-dev libboost-all-dev git g++ \
      libxml2-dev vim python3-setuptools python3 python3-pip python3-venv cmake \
      libfreeimage3 libfreeimage-dev freeglut3-dev pkg-config \
      python3-doc python3-tk python3.12-venv python3.12-doc binfmt-support psmisc apt-utils \
      bc gdb 
RUN apt-get clean

# Create and activate a virtual environment, venv is needed because of PEP 668
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip3 install --upgrade pip
RUN pip3 install pyyaml plotly psutil

#get Nsys
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update --allow-insecure-repositories && apt update && apt install -y --no-install-recommends gnupg wget \
    && mkdir -p /etc/apt/keyrings 
RUN wget -qO - https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/7fa2af80.pub | tee /etc/apt/keyrings/nvidia.asc
RUN echo "deb [signed-by=/etc/apt/keyrings/nvidia.asc] http://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64 /" | tee /etc/apt/sources.list.d/nvidia.list
RUN apt-get update --allow-insecure-repositories
RUN apt install  -y nsight-systems-cli --allow-unauthenticated

# clone accel-sim-framework
# this comes from "git clone https://github.com/accel-sim/accel-sim-framework.git"
ADD accel-sim-framework /accel-sim/accel-sim-framework

# add some environment variables to .bashrc
ADD .bashrc /root/.bashrc

WORKDIR /accel-sim/accel-sim-framework

# download benchmark applications traces
RUN mkdir hw_run
# download traces summary file to hw_run (this comes from "wget ftp://ftp.ecn.purdue.edu/tgrogers/accel-sim/traces/1.1.0.trace.summary.txt")
ADD 1.1.0.trace.summary.txt hw_run/1.1.0.trace.summary.txt
# download and extract rodinia traces to hw_run/traces/device-{device_id}/{cuda version}/rodinia_2.0-ft
# these come from the following commands:
# wget ftp://ftp.ecn.purdue.edu/tgrogers/accel-sim/traces/tesla-v100/1.1.0.latest/rodinia_2.0-ft.tgz
# tar -xvzf rodinia_2.0-ft.tgz
ADD rodinia_2.0-ft hw_run/traces/device-0/12.8/rodinia_2.0-ft

# # Compile simulator
RUN pip3 install -r requirements.txt \
&& source gpu-simulator/setup_environment.sh \
&& make -j -C gpu-simulator

WORKDIR /accel-sim

# install gpu-app-collection
ADD gpu-app-collection /accel-sim/gpu-app-collection
# RUN export PATH=$CUDA_INSTALL_PATH/bin:$PATH \
# && source ./gpu-app-collection/src/setup_environment \
# && make -j -C ./gpu-app-collection/src rodinia_2.0-ft \
# && make -j -C ./gpu-app-collection/src GPU_Microbenchmark \
# && make -j -C ./gpu-app-collection/src data \
# && rm gpucomputingsdk_4.2.9_linux.run \
# && rm -rf 4.2
