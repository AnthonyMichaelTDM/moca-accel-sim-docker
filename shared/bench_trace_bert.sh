#!/bin/bash

# run the tracer many times with and without KERNEL_NAME_FILTER set, timing each run and comparing the results

# set environment variables
export CUDA_VISIBLE_DEVICES=0
export CUDA_VERSION="11.7"
# export DYNAMIC_KERNEL_LIMIT_END="10"

export KERNEL_NAME_FILTER="_ZN7cutlass7Kernel2I50cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4EEvNT_6ParamsE,_ZN8cublasLt19splitKreduce_kernelILi32ELi16EiffffLb1ELb1ELb0EEEvNS_18cublasSplitKParamsIT4_EEPKT2_PKT3_PS7_PKS2_SC_PKT5_S6_PSD_PvlPS2_Pi"
export DYNAMIC_KERNEL_REGIONS="9-14,26-31"

# check if time is installed
if ! command -v time &> /dev/null
then
    echo "time could not be found, installing"
    apt install time
fi

# remove the temp benchmark results file if it already exists
if [ -f temp_benchmark_results.txt ]; then
    rm temp_benchmark_results.txt
fi

# run the tracer many times with KERNEL_NAME_FILTER set
for i in {1..20}
do
    echo "Run $i with KERNEL_NAME_FILTER" | tee -a temp_benchmark_results.txt
    # run the tracer
    /usr/bin/time -v env LD_PRELOAD=/accel-sim/accel-sim-framework/util/tracer_nvbit/tracer_tool/tracer_tool.so python3 /shared/model_pool_finetuned.py --pre_train_name=bert-base-uncased --finetune_name=victoraavila/bert-base-uncased-finetuned-squad --sentence=1 > /dev/null 2>> temp_benchmark_results.txt
done

# determine the average time taken to run the tracer with KERNEL_NAME_FILTER set
grep "Elapsed (wall clock) time (h:mm:ss or m:ss): " temp_benchmark_results.txt | awk '{print $8}' | awk -F: '{print $2}' | awk '{ sum += $1; n++ } END { if (n > 0) print "Average time with KERNEL_NAME_FILTER set: " sum / n; }' | tee -a temp_benchmark_results.txt

# unset environment variables
unset KERNEL_NAME_FILTER
cat temp_benchmark_results.txt >> benchmark_results.txt
rm temp_benchmark_results.txt

# run the tracer many times without KERNEL_NAME_FILTER set
for i in {1..20}
do
    echo "Run $i without KERNEL_NAME_FILTER" | tee -a temp_benchmark_results.txt
    # run the tracer
    /usr/bin/time -v env LD_PRELOAD=/accel-sim/accel-sim-framework/util/tracer_nvbit/tracer_tool/tracer_tool.so python3 /shared/model_pool_finetuned.py --pre_train_name=bert-base-uncased --finetune_name=victoraavila/bert-base-uncased-finetuned-squad --sentence=1 > /dev/null 2>> temp_benchmark_results.txt
done

# determine the average time taken to run the tracer without KERNEL_NAME_FILTER set
grep "Elapsed (wall clock) time (h:mm:ss or m:ss): " temp_benchmark_results.txt | awk '{print $8}' | awk -F: '{print $2}' | awk '{ sum += $1; n++ } END { if (n > 0) print "Average time without KERNEL_NAME_FILTER set: " sum / n; }' | tee -a temp_benchmark_results.txt
cat temp_benchmark_results.txt >> benchmark_results.txt
rm temp_benchmark_results.txt
