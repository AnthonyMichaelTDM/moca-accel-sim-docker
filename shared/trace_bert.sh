#!/bin/bash


# set environment variables
export CUDA_VISIBLE_DEVICES=0
export CUDA_VERSION="11.7"
# export DYNAMIC_KERNEL_LIMIT_END="10"

# export KERNEL_NAME_FILTER="_ZN2at6native44_GLOBAL__N__50c743a2_11_Indexing_cu_89862edb21indexSelectSmallIndexIfljLi2ELi2ELin2EEEvNS_4cuda6detail10TensorInfoIT_T1_EENS5_IKS6_S7_EENS5_IKT0_S7_EEiiS7_l"
export DYNAMIC_KERNEL_REGIONS="1-2,4-4"

# export KERNEL_NAME_FILTER="_ZN7cutlass7Kernel2I50cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4EEvNT_6ParamsE,_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_13AUnaryFunctorIllbNS0_51_GLOBAL__N__28ce311f_18_CompareEQKernel_cu_d8008c9616CompareEqFunctorIlEEEENS_6detail5ArrayIPcLi2EEEEEviT0_T1_"
# export DYNAMIC_KERNEL_REGIONS="4-38"

# run tracer
LD_PRELOAD=/accel-sim/accel-sim-framework/util/tracer_nvbit/tracer_tool/tracer_tool.so python3 /shared/model_pool_finetuned.py --pre_train_name=bert-base-uncased --finetune_name=victoraavila/bert-base-uncased-finetuned-squad --sentence=1

# post-process traces
/accel-sim/accel-sim-framework/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing traces/kernelslist

# remove unprocessed trace files
rm -f traces/*.trace
rm -f traces/kernelslist