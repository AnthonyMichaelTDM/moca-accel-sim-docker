import argparse
import os
from attr import dataclass
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# map column names to more readable names
metrics = {
    "Kernel Name": "kernel_name",
    "Kernel Name (Demangled)": "demangled_name",
    "demangled name with suffix": "demangled_name_with_suffix",
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
    # model_pool_finetuned.py
    "fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params)": "fmha_cutlassF_f32_aligned_64x64_rf_sm80",
    "ampere_sgemm_32x32_sliced1x4_tn": "ampere_sgemm_32x32_sliced1x4_tn",
    "void at::native::(anonymous namespace)::indexSelectSmallIndex<float, long, unsigned int, 2, 2, -2>(at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<float const, unsigned int>, at::cuda::detail::TensorInfo<long const, unsigned int>, int, int, unsigned int, long)": "indexSelectSmallIndex",
    "void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float>(int, float, float const*, float const*, float const*, float*, float*, float*)": "vectorized_layer_norm_kernel",
    "void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1})": "elementwise_kernel<128, 2, …>",
    "void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<long, long, bool, at::native::(anonymous namespace)::CompareEqFunctor<long> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<long, long, bool, at::native::(anonymous namespace)::CompareEqFunctor<long> >, std::array<char*, 2ul>)": "vectorized_elementwise_kernel<4, AunaryFunctor<…>, _>",
    "void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)": "vectorized_elementwise_kernel<4, CUDAFunctor_add<…>, _>",
    "void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>)": "vectorized_elementwise_kernel<4, GeluCUDAKernelImpl(…), _>",
    "void cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, true, true, false>(cublasLt::cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*, float const*, float const*, float*, void*, long, float*, int*)": "splitKreduce_kernel<32, 16, …>",
    "void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4>(cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4::Params)": "cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4",
    "void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4>(cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4::Params)": "cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4",
    # mnist
    "std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, float, float, float, float, false, true, true, false, 6, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)": "gemvx kernel type 1",
    "void cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, true, false, false>(cublasLt::cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*, float const*, float const*, float*, void*, long, float*, int*)": "splitKreduce_kernel<32, 16, …>",
    "void cublasLt::epilogue::impl::globalKernel<8, 32, float, float, float, true, true, 1>(int, int, long, float*, cublasLtEpilogue_t, int, float*, long, void*, long, long, long, float*, long, int*)": "epilogue::impl::globalKernel<8, 32, …>",
    "void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>)": "vectorized_elementwise_kernel<4, launch_clamp_scalar(…), _>",
    "void gemv2T_kernel_val<int, int, float, float, float, float, 128, 16, 2, 2, false, true, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>, float, float)": "gemv2T_kernel_val",
    "void (anonymous namespace)::softmax_warp_forward<float, float, float, 4, true, false>(float*, float const*, int, int, int, bool const*, int, bool)": "softmax_warp_forward",
    "void at::native::(anonymous namespace)::nll_loss_forward_reduce_cuda_kernel_2d<float, float, long>(float*, float*, float const*, long const*, float const*, bool, long, long, long, long)": "nll_loss_forward_reduce_cuda_kernel_2d",
    "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)": "vectorized_elementwise_kernel<4, FillFunctor<float>, _>",
    "void at::native::(anonymous namespace)::nll_loss_backward_reduce_cuda_kernel_2d<float, long>(float*, float const*, long const*, float const*, float const*, bool, int, int, long, long)": "nll_loss_backward_reduce_cuda_kernel_2d",
    "void (anonymous namespace)::softmax_warp_backward<float, float, float, 4, true, false>(float*, float const*, float const*, int, int, int, bool const*)": "softmax_warp_backward",
    "void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::(anonymous namespace)::threshold_kernel_impl<float>(at::TensorIteratorBase&, float, float)::{lambda(float, float)#1}>, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<float, float, float, at::native::(anonymous namespace)::threshold_kernel_impl<float>(at::TensorIteratorBase&, float, float)::{lambda(float, float)#1}>, std::array<char*, 3ul>)": "vectorized_elementwise_kernel<4, BinaryFunctor<…>, …>",
    "std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, float, float, float, float, false, true, false, false, 6, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)": "gemvx kernel type 2",
    "void gemmk1_kernel<int, float, 256, 5, false, false, false, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, 0>(cublasGemmk1Params<float, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, biasType<cublasGemvTensorStridedBatched<float>::value_type, float>::type>)": "gemmk1_kernel",
    "void at::native::reduce_kernel<256, 2, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4>)": "reduce_kernel<256, 2, …>",
    "std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, float, float, float, float, false, true, false, false, 9, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)": "gemvx kernel type 3",
    "void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4>)": "reduce_kernel<128, 4, …>",
    "void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymous namespace)::TensorListMetadata<2>, at::native::(anonymous namespace)::BinaryOpListAlphaFunctor<float, 2, 2, 0>, std::plus<float>, float>(at::native::(anonymous namespace)::TensorListMetadata<2>, at::native::(anonymous namespace)::BinaryOpListAlphaFunctor<float, 2, 2, 0>, std::plus<float>, float)": "multi_tensor_apply_kernel<TensorListMetadata<2>, …>",
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
    # model_pool_finetuned.py #
    "fmha_cutlassF_f32_aligned_64x64_rf_sm80",
    # "ampere_sgemm_32x32_sliced1x4_tn",
    "indexSelectSmallIndex",
    "vectorized_layer_norm_kernel",
    "elementwise_kernel<128, 2, …>",
    # "vectorized_elementwise_kernel<4, AunaryFunctor<…>, _>",
    "vectorized_elementwise_kernel<4, CUDAFunctor_add<…>, _>",
    "vectorized_elementwise_kernel<4, GeluCUDAKernelImpl(…), _>",
    "splitKreduce_kernel<32, 16, …>",
    "cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4",
    "cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4",
    # mnist #
    "gemvx kernel type 1",
    "splitKreduce_kernel<32, 16, …>",
    "epilogue::impl::globalKernel<8, 32, …>",
    "vectorized_elementwise_kernel<4, launch_clamp_scalar(…), _>",
    "gemv2T_kernel_val",
    "softmax_warp_forward",
    "nll_loss_forward_reduce_cuda_kernel_2d",
    "vectorized_elementwise_kernel<4, FillFunctor<float>, _>",
    "nll_loss_backward_reduce_cuda_kernel_2d",
    "softmax_warp_backward",
    "vectorized_elementwise_kernel<4, BinaryFunctor<…>, …>",
    "gemvx kernel type 2",
    "gemmk1_kernel",
    "reduce_kernel<256, 2, …>",
    "gemvx kernel type 3",
    "reduce_kernel<128, 4, …>",
    "multi_tensor_apply_kernel<TensorListMetadata<2>, …>",
]


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


def save_violin_plot(df, metric, filename):
    plt.figure(figsize=(20, 10))
    violin_plot(df, metric)
    plt.savefig(filename)
    plt.close("all")


def line_plot(df, metric) -> None:
    """
    Plot the performance of the kernels over time
    """
    for clean_kernel_name in df["clean_names"].unique():
        df2 = df[df["clean_names"] == clean_kernel_name].copy()
        df2["id"] = (
            0
            if (split := df2["kernel_name"].str.split("--")).empty
            else split.str[1].astype(int)
        )
        # Normalize the id column to range [0,1]
        df2["normalized_id"] = (df2["id"] - df2["id"].min()) / (
            df2["id"].max() - df2["id"].min()
        )

        df2 = df2.sort_values("id")
        sns.lineplot(
            x="normalized_id",
            y=metric,
            data=df2,
            legend="brief",
            label=f"{clean_kernel_name} ({len(df2)} launches)",
        )
    plt.xlabel("Kernel Launch")
    plt.ylabel(metric)
    plt.title(f"{metric} Over Time")
    # hide x-axis ticks
    plt.xticks([])
    plt.tight_layout()


def save_line_plot(df, metric, filename):
    plt.figure(figsize=(20, 10))
    line_plot(df, metric)
    plt.savefig(filename)
    plt.close("all")


def correlation_matrix(df, output_dir: str = "plots") -> None:
    plt.figure(figsize=(20, 20))
    correlation_matrix = df[metrics_to_plot].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close("all")


def correlation_matrix_kernel(df, kernel_name, output_dir: str = "plots"):
    plt.figure(figsize=(20, 20))
    df_kernel = df[df["clean_names"] == kernel_name]
    correlation_matrix = df_kernel[metrics_to_plot].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title(f"Correlation Matrix for {kernel_name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix_{kernel_name}.png")
    plt.close("all")


@dataclass
class Config:
    input_file: str
    output_dir: str

    @classmethod
    def from_args(cls):
        """
        Create a Config object from command line arguments.
        """
        parser = argparse.ArgumentParser(
            description="Create plots from the organized simulation results."
        )
        parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
        parser.add_argument(
            "--output_dir",
            type=str,
            default="plots",
            help="Path to the output directory.",
        )
        args = parser.parse_args()
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file {args.input_file} does not exist.")
        if not os.path.isfile(args.input_file):
            raise ValueError(f"Input file {args.input_file} is not a file.")
        if args.output_dir and not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        if args.output_dir and not os.path.isdir(args.output_dir):
            raise ValueError(f"Output directory {args.output_dir} is not a directory.")
        return cls(args.input_file, args.output_dir)


def main():
    config = Config.from_args()

    # Read the CSV file
    df = pd.read_csv(
        config.input_file,
        delimiter="\t",
    )

    # map readable names to column names
    df = df.rename(columns=lambda x: metrics[x])

    # map kernel names to more readable names
    df["clean_names"] = df["demangled_name"].map(kernel_names)

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

    correlation_matrix(df, config.output_dir)

    for kernel in kernels_to_keep:
        if kernel in df["clean_names"].unique():
            correlation_matrix_kernel(df, kernel, config.output_dir)

    for metric in metrics_to_plot:
        save_violin_plot(df, metric, f"{config.output_dir}/{metric}_violin_plot.png")
        save_line_plot(df, metric, f"{config.output_dir}/{metric}_line_plot.png")


if __name__ == "__main__":
    main()
