import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv(
    "/home/anthony/Data/moca-accel-sim/shared/organized_sim_results.csv", delimiter="\t"
)

# map column names to more readable names
metrics = {
    "Kernel Name": "kernel_name",
    "demangled name": "demangled_name",
    "demangled name suffix pruned": "suffix_pruned",
    "gpu_tot_sim_insn\\s*=\\s*(.*)": "gpu_tot_sim_insn",
    "gpgpu_simulation_time\\s*=.*\\(([0-9]+) sec\\).*": "gpgpu_simulation_time",
    "gpu_tot_sim_cycle\\s*=\\s*(.*)": "gpu_tot_sim_cycle",
    "\\s+L2_cache_stats_breakdown\\[GLOBAL_ACC_R\\]\\[HIT\\]\\s*=\\s*(.*)": "L2_cache_global_read_hit",
    "\\s+L2_cache_stats_breakdown\\[GLOBAL_ACC_R\\]\\[TOTAL_ACCESS\\]\\s*=\\s*(.*)": "L2_cache_global_read_total_access",
    "\\s+L2_cache_stats_breakdown\\[GLOBAL_ACC_W\\]\\[HIT\\]\\s*=\\s*(.*)": "L2_cache_global_write_hit",
    "\\s+L2_cache_stats_breakdown\\[GLOBAL_ACC_W\\]\\[TOTAL_ACCESS\\]\\s*=\\s*(.*)": "L2_cache_global_write_total_access",
    "\\s+Total_core_cache_stats_breakdown\\[GLOBAL_ACC_R\\]\\[TOTAL_ACCESS\\]\\s*=\\s*(.*)": "Total_core_cache_stats_breakdown_GLOBAL_ACC_R_TOTAL_ACCESS",
    "\\s+Total_core_cache_stats_breakdown\\[GLOBAL_ACC_R\\]\\[HIT\\]\\s*=\\s*(.*)": "Total_core_cache_stats_breakdown_GLOBAL_ACC_R_HIT",
    "\\s+Total_core_cache_stats_breakdown\\[GLOBAL_ACC_W\\]\\[HIT\\]\\s*=\\s*(.*)": "Total_core_cache_stats_breakdown_GLOBAL_ACC_W_HIT",
    "\\s+Total_core_cache_stats_breakdown\\[GLOBAL_ACC_W\\]\\[TOTAL_ACCESS\\]\\s*=\\s*(.*)": "Total_core_cache_stats_breakdown_GLOBAL_ACC_W_TOTAL_ACCESS",
    "\\s+Total_core_cache_stats_breakdown\\[GLOBAL_ACC_R\\]\\[MSHR_HIT\\]\\s*=\\s*(.*)": "Total_core_cache_stats_breakdown_GLOBAL_ACC_R_MSHR_HIT",
    "gpgpu_n_tot_w_icount\\s*=\\s*(.*)": "gpgpu_n_tot_w_icount",
    "total dram reads\\s*=\\s*(.*)": "total_dram_reads",
    "total dram writes\\s*=\\s*(.*)": "total_dram_writes",
    "kernel_launch_uid\\s*=\\s*(.*)": "kernel_launch_uid",
    "gpgpu_n_shmem_bkconflict\\s*=\\s*(.*)": "gpgpu_n_shmem_bkconflict",
    "gpgpu_n_l1cache_bkconflict\\s*=\\s*(.*)": "gpgpu_n_l1cache_bkconflict",
    "gpu_ipc\\s*=\\s*(.*)": "gpu_ipc",
    "gpu_occupancy\\s*=\\s*(.*)%": "gpu_occupancy",
    "L2_BW\\s*=\\s*(.*)+GB\\/Sec": "L2_BW",
    "gpgpu_simulation_rate\\s+=\\s+(.*)\\s+\\(inst\\/sec\\)": "gpgpu_simulation_rate_instructions",
    "gpgpu_simulation_rate\\s+=\\s+(.*)\\s+\\(cycle\\/sec\\)": "gpgpu_simulation_rate_cycles",
    "gpgpu_silicon_slowdown\\s*=\\s*(.*)x": "gpgpu_silicon_slowdown",
    "gpu_tot_ipc\\s*=\\s*(.*)": "gpu_tot_ipc",
}


# maps kernel names to more readable names
kernel_names = {
    "fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params)": "fmha_cutlassF_f32_aligned_64x64_rf_sm80",
    "model_pool_finetuned.py/__pre_train_name_bert_base_uncased___finetune_name_victoraavila_bert_base_uncased_finetuned_squad___sentence_1--ampere_sgemm_32x32_sliced1x4_tn": "model_pool_finetuned.py/…ampere_sgemm_32x32_sliced1x4_tn",
    "void at::native::(anonymous namespace)::indexSelectSmallIndex<float, long, unsigned int, 2, 2, -2>(at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<float const, unsigned int>, at::cuda::detail::TensorInfo<long const, unsigned int>, int, int, unsigned int, long)": "indexSelectSmallIndex",
    "void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float>(int, float, float const*, float const*, float const*, float*, float*, float*)": "vectorized_layer_norm_kernel",
    "void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1})": "elementwise_kernel<128, 2, …>",
    "void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<long, long, bool, at::native::(anonymous namespace)::CompareEqFunctor<long> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<long, long, bool, at::native::(anonymous namespace)::CompareEqFunctor<long> >, std::array<char*, 2ul>)": "vectorized_elementwise_kernel<4, AunaryFunctor<…>, _>",
    "void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)": "vectorized_elementwise_kernel<4, CUDAFunctor_add<…>, _>",
    "void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>)": "vectorized_elementwise_kernel<4, GeluCUDAKernelImpl(…), _>",
    "void cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, true, true, false>(cublasLt::cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*, float const*, float const*, float*, void*, long, float*, int*)": "splitKreduce_kernel<32, 16, …>",
    "void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4>(cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4::Params)": "cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4",
    "void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4>(cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4::Params)": "cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4",
}

# Select interesting metrics to plot
metrics_to_plot = [
    "gpu_tot_sim_insn",
    "gpgpu_simulation_time",
    "gpu_tot_sim_cycle",
    "L2_cache_global_read_hit",
    "L2_cache_global_read_total_access",
    "L2_cache_global_write_hit",
    "L2_cache_global_write_total_access",
    "Total_core_cache_stats_breakdown_GLOBAL_ACC_R_TOTAL_ACCESS",
    "Total_core_cache_stats_breakdown_GLOBAL_ACC_R_HIT",
    "Total_core_cache_stats_breakdown_GLOBAL_ACC_W_HIT",
    "Total_core_cache_stats_breakdown_GLOBAL_ACC_W_TOTAL_ACCESS",
    "Total_core_cache_stats_breakdown_GLOBAL_ACC_R_MSHR_HIT",
    "gpgpu_n_tot_w_icount",
    "total_dram_reads",
    "total_dram_writes",
    "gpgpu_n_shmem_bkconflict",
    "gpgpu_n_l1cache_bkconflict",
    "gpu_ipc",
    "gpgpu_simulation_rate_instructions",
    "gpgpu_simulation_rate_cycles",
    "gpu_tot_ipc",
]

# which kernels to keep
kernels_to_keep = [
    "fmha_cutlassF_f32_aligned_64x64_rf_sm80",
    # "model_pool_finetuned.py/…ampere_sgemm_32x32_sliced1x4_tn",
    "indexSelectSmallIndex",
    "elementwise_kernel<128, 2, …>",
    # "vectorized_elementwise_kernel<4, AunaryFunctor<…>, _>",
    "vectorized_elementwise_kernel<4, CUDAFunctor_add<…>, _>",
    "vectorized_elementwise_kernel<4, GeluCUDAKernelImpl(…), _>",
    "splitKreduce_kernel<32, 16, …>",
    "cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4",
    "cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4",
]


# map readable names to column names
df = df.rename(columns=lambda x: metrics[x])

# map kernel names to more readable names
df["clean_names"] = df["suffix_pruned"].map(kernel_names)

# remove the unwanted kernels
df = df[df["clean_names"].isin(kernels_to_keep)]

# # print the columns and kernels
# print("\nColumns:")
# print(df.columns)
# print("\nKernels:")
# print(df["clean_names"].unique())

# # Print summary statistics
# print("\nSummary Statistics:")
# for metric in metrics_to_plot:
#     print(f"\n{metric}:")
#     print(df.groupby("clean_names")[metric].describe())


def violin_plot(df, metric):
    sns.violinplot(
        y="clean_names",
        x=metric,
        data=df,
        density_norm="width",
        inner="quartile",
        linewidth=1,
    )
    plt.xlabel("")
    plt.ylabel("")
    plt.title(f"{metric} Distribution")
    plt.tight_layout()


# def show_violin_plot(df, metric):
#     plt.figure(figsize=(20, 10))
#     violin_plot(df, metric)
#     plt.show()
#     plt.close("all")


# show_violin_plot(df, "gpu_tot_sim_insn")


def save_violin_plot(df, metric, filename):
    plt.figure(figsize=(20, 10))
    violin_plot(df, metric)
    plt.savefig(filename)
    plt.close("all")


for metric in metrics_to_plot:
    save_violin_plot(df, metric, f"plots/{metric}_violin_plot.png")
