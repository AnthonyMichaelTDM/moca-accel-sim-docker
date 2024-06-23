source /accel-sim/accel-sim-framework/gpu-simulator/setup_environment.sh
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_INSTALL_PATH/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim/lib/gcc-/cuda-11070/release/:$LD_LIBRARY_PATH