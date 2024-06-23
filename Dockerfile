# the context for this dockerfile can be generated with the following commands, which are
# conveniently provided in setup.sh
# ```sh
# # Clone the Accel-Sim-Framework repository
# git clone https://github.com/accel-sim/accel-sim-framework.git
#
# # get the cuda 11.7 installer
# wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
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

FROM ubuntu:22.04

SHELL ["/bin/bash", "-c"]

WORKDIR /accel-sim

ENV CUDA_INSTALL_PATH /usr/local/cuda-11.7
ENV PTXAS_CUDA_INSTALL_PATH /usr/local/cuda-11.7
ENV GPUAPPS_ROOT /accel-sim/gpu-app-collection

# install prerequisites
RUN apt-get update \
&& apt-get install -y wget build-essential xutils-dev bison zlib1g-dev flex \
      libglu1-mesa-dev git g++ libssl-dev libxml2-dev libboost-all-dev git g++ \
      libxml2-dev vim python-setuptools python3 python3-pip cmake \
&& apt-get clean \
&& pip3 install pyyaml plotly psutil
# install cuda 11.7
## this comes from "wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run"
ADD cuda_11.7.0_515.43.04_linux.run /accel-sim/cuda_11.7.0_515.43.04_linux.run
RUN sh cuda_11.7.0_515.43.04_linux.run --silent --toolkit \
&& rm cuda_11.7.0_515.43.04_linux.run \
&& rm -rf /usr/local/cuda-11.7/nsight-compute-2022.2.0 \
&& rm -rf /usr/local/cuda-11.7/nsight-systems-2022.1.3

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
ADD rodinia_2.0-ft hw_run/traces/device-0/11.7/rodinia_2.0-ft

# Compile simulator
RUN pip3 install -r requirements.txt \
&& source gpu-simulator/setup_environment.sh \
&& export LD_LIBRARY_PATH=$CUDA_INSTALL_PATH/lib64:$LD_LIBRARY_PATH \
&& make -j -C gpu-simulator

WORKDIR /accel-sim

# install gpu-app-collection
ADD gpu-app-collection /accel-sim/gpu-app-collection
RUN export PATH=$CUDA_INSTALL_PATH/bin:$PATH \
&& source ./gpu-app-collection/src/setup_environment \
&& make -j -C ./gpu-app-collection/src rodinia_2.0-ft \
&& make -j -C ./gpu-app-collection/src GPU_Microbenchmark \
&& make -j -C ./gpu-app-collection/src data \
&& rm gpucomputingsdk_4.2.9_linux.run \
&& rm -rf 4.2
