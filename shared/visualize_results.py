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
    # from old tests (on newer nvbit version)
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
    # from newer tests (on older library versions)
    "void gemv2T_kernel_val<int, int, float, float, float, 128, 16, 4, 4, false, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>, float, float)": "gemv2T_kernel_val",
    "void at::native::vectorized_elementwise_kernel<4, at::native::threshold_kernel_impl<float>(at::TensorIterator&, float, float)::{lambda(float, float)#1}, at::detail::Array<char*, 3> >(int, at::native::threshold_kernel_impl<float>(at::TensorIterator&, float, float)::{lambda(float, float)#1}, at::detail::Array<char*, 3>)": "vectorized_elementwise_kernel<4, threshold_kernel_impl<…>, _>",
    "void dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float> > >(cublasDotParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float> >)": "dot_kernel",
    "void reduce_1Block_kernel<float, 128, 7, cublasGemvTensorStridedBatched<float>, cublasGemvTensorStridedBatched<float> >(float const*, float, cublasGemvTensorStridedBatched<float>, int, float const*, float, cublasGemvTensorStridedBatched<float>, cublasPointerMode_t, cublasLtEpilogue_t, cublasGemvTensorStridedBatched<biasType<cublasGemvTensorStridedBatched<float>::value_type, float>::type const>)": "reduce_1Block_kernel",
    "void (anonymous namespace)::softmax_warp_forward<float, float, float, 4, true>(float*, float const*, int, int, int)": "softmax_warp_forward",
    "void cunn_ClassNLLCriterion_updateOutput_kernel<float, float>(float*, float*, float*, long*, float*, int, int, int, int, long)": "cunn_ClassNLLCriterion_updateOutput_kernel",
    "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>)": "vectorized_elementwise_kernel<4, FillFunctor<float>, _>",
    "void cunn_ClassNLLCriterion_updateGradInput_kernel<float>(float*, float*, long*, float*, float*, int, int, int, int, long)": "cunn_ClassNLLCriterion_updateGradInput_kernel",
    "void (anonymous namespace)::softmax_warp_backward<float, float, float, 4, true>(float*, float const*, float const*, int, int, int)": "softmax_warp_backward",
    "void gemvNSP_kernel<float, float, float, 1, 16, 4, 1024, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)": "gemvNSP_kernel",
    "void gemmk1_kernel<float, 256, 5, false, false, true, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>(cublasGemmk1Params<float, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, biasType<cublasGemvTensorStridedBatched<float>::value_type, float>::type>)": "gemmk1_kernel",
    "void at::native::reduce_kernel<256, 2, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4>)": "reduce_kernel<256, 2, …>",
    "void splitKreduce_kernel<float, float, float, float>(cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*, float const*)": "splitKreduce_kernel",
    "void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4>)": "reduce_kernel<128, 4, …>",
    "void at::native::vectorized_elementwise_kernel<4, at::native::AddFunctor<float>, at::detail::Array<char*, 3> >(int, at::native::AddFunctor<float>, at::detail::Array<char*, 3>)": "vectorized_elementwise_kernel<4, AddFunctor<float>, _>",
    # alexnet
    "void kernelPointwiseApply1<TensorFillOp<float>, float, unsigned int, 1>(OffsetInfo<float, unsigned int, 1>, unsigned int, TensorFillOp<float>)": "kernelPointwiseApply1",
    "void gemmk1_kernel<float, 256, 5, true, false, false, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>(cublasGemmk1Params<float, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, biasType<cublasGemvTensorStridedBatched<float>::value_type, float>::type>)": "gemmk1_kernel",
    "void at::native::im2col_kernel<float>(long, float const*, long, long, long, long, long, long, long, long, long, long, long, long, float*)": "im2col_kernel",
    "volta_sgemm_128x64_nn": "volta_sgemm_128x64_nn",
    "void at::native::vectorized_elementwise_kernel<4, at::native::threshold_kernel_impl<float>(at::TensorIterator&, float, float)::{lambda(float, float)#1}, at::detail::Array<char*, 3> >(int, at::native::threshold_kernel_impl<float>(at::TensorIterator&, float, float)::{lambda(float, float)#1}, at::detail::Array<char*, 3>)": "vectorized_elementwise_kernel<4, threshold_kernel_impl<…>, _>",
    "void at::native::(anonymous namespace)::max_pool_forward_nchw<float, float>(int, float const*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, float*, long*)": "max_pool_forward_nchw",
    "volta_sgemm_128x32_tn": "volta_sgemm_128x32_tn",
    "volta_sgemm_128x32_nn": "volta_sgemm_128x32_nn",
    "volta_sgemm_128x32_sliced1x4_nn": "volta_sgemm_128x32_sliced1x4_nn",
    "void at::native::(anonymous namespace)::adaptive_average_pool<float>(float*, float*, int, int, int, int, long, long, long)": "adaptive_average_pool",
    "void at::native::(anonymous namespace)::fused_dropout_kernel_vec<float, float, unsigned int, 1, 4>(at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<unsigned char, unsigned int>, unsigned int, float, std::pair<unsigned long, unsigned long>)": "fused_dropout_kernel_vec",
    "void gemv2T_kernel_val<int, int, float, float, float, 128, 16, 4, 4, false, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>, float, float)": "gemv2T_kernel_val",
    "void dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float> > >(cublasDotParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float> >)": "dot_kernel",
    "void reduce_1Block_kernel<float, 128, 7, cublasGemvTensorStridedBatched<float>, cublasGemvTensorStridedBatched<float> >(float const*, float, cublasGemvTensorStridedBatched<float>, int, float const*, float, cublasGemvTensorStridedBatched<float>, cublasPointerMode_t, cublasLtEpilogue_t, cublasGemvTensorStridedBatched<biasType<cublasGemvTensorStridedBatched<float>::value_type, float>::type const>)": "reduce_1Block_kernel",
    "void (anonymous namespace)::softmax_warp_forward<float, float, float, 4, true>(float*, float const*, int, int, int)": "softmax_warp_forward",
    # bert
    "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<long>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<long>, at::detail::Array<char*, 1>)": "vectorized_elementwise_kernel<4, FillFunctor<long>, _>",
    "void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<at::native::CompareGTFunctor<long> >, at::detail::Array<char*, 2> >(int, at::native::BUnaryFunctor<at::native::CompareGTFunctor<long> >, at::detail::Array<char*, 2>)": "vectorized_elementwise_kernel<4, BUnaryFunctor<CompareGTFunctor<long>,_>, _>",
    "void at::native::unrolled_elementwise_kernel<at::native::copy_device_to_device(at::TensorIterator&, bool)::{lambda()#2}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool)#1}, at::detail::Array<char*, 2>, OffsetCalculator<1, unsigned int>, OffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, at::native::copy_device_to_device(at::TensorIterator&, bool)::{lambda()#2}::operator()() const::{lambda()#11}::operator()() const::{lambda(bool)#1}, at::detail::Array<char*, 2>, OffsetCalculator<1, unsigned int>, OffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast)": "unrolled_elementwise_kernel<copy_device_to_device(…), _> (bool)",
    "void at::native::(anonymous namespace)::indexSelectLargeIndex<float, unsigned int, 2, 2, -2, true>(at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<long, unsigned int>, int, int, unsigned int, unsigned int, long)": "indexSelectLargeIndex",
    "void at::native::vectorized_elementwise_kernel<4, at::native::AddFunctor<float>, at::detail::Array<char*, 3> >(int, at::native::AddFunctor<float>, at::detail::Array<char*, 3>)": "vectorized_elementwise_kernel<4, AddFunctor<float>, _>",
    "void at::native::(anonymous namespace)::fused_dropout_kernel_vec<float, float, unsigned int, 1, 4>(at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<unsigned char, unsigned int>, unsigned int, float, std::pair<unsigned long, unsigned long>)": "fused_dropout_kernel_vec",
    "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float>, unsigned int, float, 4>)": "reduce_kernel<512, 1, MeanOps<…> >",
    "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::WelfordOps<float, float, int, float, thrust::pair<float, float> >, unsigned int, float, 2> >(at::native::ReduceOp<float, at::native::WelfordOps<float, float, int, float, thrust::pair<float, float> >, unsigned int, float, 2>)": "reduce_kernel<512, 1, WelfordOps<…> >",
    "void at::native::unrolled_elementwise_kernel<at::native::AddFunctor<float>, at::detail::Array<char*, 3>, OffsetCalculator<2, unsigned int>, OffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, at::native::AddFunctor<float>, at::detail::Array<char*, 3>, OffsetCalculator<2, unsigned int>, OffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast)": "unrolled_elementwise_kernel<AddFunctor<float>, _>",
    "void at::native::unrolled_elementwise_kernel<at::native::MulFunctor<float>, at::detail::Array<char*, 3>, OffsetCalculator<2, unsigned int>, OffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, at::native::MulFunctor<float>, at::detail::Array<char*, 3>, OffsetCalculator<2, unsigned int>, OffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast)": "unrolled_elementwise_kernel<MulFunctor<float>, _>",
    "void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<at::native::AddFunctor<float> >, at::detail::Array<char*, 2> >(int, at::native::BUnaryFunctor<at::native::AddFunctor<float> >, at::detail::Array<char*, 2>)": "vectorized_elementwise_kernel<4, BUnaryFunctor<AddFunctor<float>,_>, _>",
    "void at::native::unrolled_elementwise_kernel<at::native::DivFunctor<float>, at::detail::Array<char*, 3>, OffsetCalculator<2, unsigned int>, OffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, at::native::DivFunctor<float>, at::detail::Array<char*, 3>, OffsetCalculator<2, unsigned int>, OffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast)": "unrolled_elementwise_kernel<DivFunctor<float>, _>",
    "volta_sgemm_64x32_sliced1x4_tn": "volta_sgemm_64x32_sliced1x4_tn",
    "void at::native::unrolled_elementwise_kernel<at::native::copy_device_to_device(at::TensorIterator&, bool)::{lambda()#2}::operator()() const::{lambda()#4}::operator()() const::{lambda(float)#1}, at::detail::Array<char*, 2>, OffsetCalculator<1, unsigned int>, OffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, at::native::copy_device_to_device(at::TensorIterator&, bool)::{lambda()#2}::operator()() const::{lambda()#4}::operator()() const::{lambda(float)#1}, at::detail::Array<char*, 2>, OffsetCalculator<1, unsigned int>, OffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast)": "unrolled_elementwise_kernel<copy_device_to_device(…), _> (float)",
    "volta_sgemm_64x64_nn": "volta_sgemm_64x64_nn",
    "void at::native::vectorized_elementwise_kernel<4, at::native::MulScalarFunctor<float, float>, at::detail::Array<char*, 2> >(int, at::native::MulScalarFunctor<float, float>, at::detail::Array<char*, 2>)": "vectorized_elementwise_kernel<4, MulScalarFunctor<float, float>, _>",
    "void at::native::unrolled_elementwise_kernel<at::native::BUnaryFunctor<at::native::CompareEqFunctor<long> >, at::detail::Array<char*, 2>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast>(int, at::native::BUnaryFunctor<at::native::CompareEqFunctor<long> >, at::detail::Array<char*, 2>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast)": "unrolled_elementwise_kernel<BUnaryFunctor<CompareEqFunctor<long>,_>, _>",
    "void kernelPointwiseApply2<TensorMaskedFillOp<float, bool>, float, bool, unsigned int, 1, 2>(OffsetInfo<float, unsigned int, 1>, OffsetInfo<bool, unsigned int, 2>, unsigned int, TensorMaskedFillOp<float, bool>)": "kernelPointwiseApply2<TensorMaskedFillOp<float, bool>, …>",
    "void (anonymous namespace)::softmax_warp_forward<float, float, float, 9, false>(float*, float const*, int, int, int)": "softmax_warp_forward",
    "volta_sgemm_128x64_tn": "volta_sgemm_128x64_tn",
    "void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIterator&, float)::{lambda(float)#3}, at::detail::Array<char*, 2> >(int, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIterator&, float)::{lambda(float)#3}, at::detail::Array<char*, 2>)": "vectorized_elementwise_kernel<4, pow_tensor_scalar_kernel_impl(…), _>",
    "void at::native::vectorized_elementwise_kernel<4, at::native::tanh_kernel_cuda(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, at::detail::Array<char*, 2> >(int, at::native::tanh_kernel_cuda(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, at::detail::Array<char*, 2>)": "vectorized_elementwise_kernel<4, tanh_kernel_cuda(…), _>",
    "void at::native::vectorized_elementwise_kernel<4, at::native::MulFunctor<float>, at::detail::Array<char*, 3> >(int, at::native::MulFunctor<float>, at::detail::Array<char*, 3>)": "vectorized_elementwise_kernel<4, MulFunctor<float>, _>",
    "volta_sgemm_32x128_tn": "volta_sgemm_32x128_tn",
    "void at::native::(anonymous namespace)::cunn_SpatialSoftMaxForward<float, float, float, at::native::(anonymous namespace)::LogSoftMaxForwardEpilogue>(float*, float*, unsigned int, unsigned int, unsigned int)": "cunn_SpatialSoftMaxForward",
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
    # "gpgpu_n_shmem_bkconflict",
    # "gpgpu_n_l1cache_bkconflict",
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
    # from newer tests (on older library versions) #
    "gemv2T_kernel_val",
    "vectorized_elementwise_kernel<4, threshold_kernel_impl<…>, _>",
    "dot_kernel",
    "reduce_1Block_kernel",
    "softmax_warp_forward",
    "cunn_ClassNLLCriterion_updateOutput_kernel",
    "vectorized_elementwise_kernel<4, FillFunctor<float>, _>",
    "cunn_ClassNLLCriterion_updateGradInput_kernel",
    "softmax_warp_backward",
    "gemvNSP_kernel",
    "gemmk1_kernel",
    "reduce_kernel<256, 2, …>",
    "splitKreduce_kernel",
    "reduce_kernel<128, 4, …>",
    "vectorized_elementwise_kernel<4, AddFunctor<float>, _>",
    # alexnet #
    "kernelPointwiseApply1",
    "gemmk1_kernel",
    "im2col_kernel",
    "volta_sgemm_128x64_nn",
    "vectorized_elementwise_kernel<4, threshold_kernel_impl<…>, _>",
    "max_pool_forward_nchw",
    "volta_sgemm_128x32_tn",
    "volta_sgemm_128x32_nn",
    "volta_sgemm_128x32_sliced1x4_nn",
    "adaptive_average_pool",
    "fused_dropout_kernel_vec",
    "gemv2T_kernel_val",
    "dot_kernel",
    "reduce_1Block_kernel",
    "softmax_warp_forward",
    # bert #
    "vectorized_elementwise_kernel<4, FillFunctor<long>, _>",
    "vectorized_elementwise_kernel<4, BUnaryFunctor<CompareGTFunctor<long>,_>, _>",
    "unrolled_elementwise_kernel<copy_device_to_device(…), _> (bool)",
    "indexSelectLargeIndex",
    "vectorized_elementwise_kernel<4, AddFunctor<float>, _>",
    "fused_dropout_kernel_vec",
    "reduce_kernel<512, 1, MeanOps<…> >",
    "reduce_kernel<512, 1, WelfordOps<…> >",
    "unrolled_elementwise_kernel<AddFunctor<float>, _>",
    "unrolled_elementwise_kernel<MulFunctor<float>, _>",
    "vectorized_elementwise_kernel<4, BUnaryFunctor<AddFunctor<float>,_>, _>",
    "unrolled_elementwise_kernel<DivFunctor<float>, _>",
    "volta_sgemm_64x32_sliced1x4_tn",
    "unrolled_elementwise_kernel<copy_device_to_device(…), _> (float)",
    "volta_sgemm_64x64_nn",
    "vectorized_elementwise_kernel<4, MulScalarFunctor<float, float>, _>",
    "unrolled_elementwise_kernel<BUnaryFunctor<CompareEqFunctor<long>,_>, _>",
    "kernelPointwiseApply2<TensorMaskedFillOp<float, bool>, …>",
    "softmax_warp_forward",
    "volta_sgemm_128x64_tn",
    "vectorized_elementwise_kernel<4, pow_tensor_scalar_kernel_impl(…), _>",
    "vectorized_elementwise_kernel<4, tanh_kernel_cuda(…), _>",
    "vectorized_elementwise_kernel<4, MulFunctor<float>, _>",
    "volta_sgemm_32x128_tn",
    "cunn_SpatialSoftMaxForward",
]

# groups of kernels to keep together in the plots
kernel_groups: dict[str, list[str]] = {
    # gemm kernels
    "group-gemm": [
        "ampere_sgemm_32x32_sliced1x4_tn",
        "volta_sgemm_32x128_tn",
        "volta_sgemm_64x32_sliced1x4_tn",
        "volta_sgemm_64x64_nn",
        "volta_sgemm_128x32_tn",
        "volta_sgemm_128x32_nn",
        "volta_sgemm_128x32_sliced1x4_nn",
        "volta_sgemm_128x64_tn",
        "volta_sgemm_128x64_nn",
        "gemmk1_kernel",
    ],
    # cutlass kernels
    "group-cutlass": [
        "fmha_cutlassF_f32_aligned_64x64_rf_sm80",
        "cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4",
        "cutlass_80_tensorop_s1688gemm_128x64_16x6_tn_align4",
    ],
    # gemv kernels
    "group-gemv": [
        "gemv2T_kernel_val",
        "gemvNSP_kernel",
        "gemvx kernel type 1",
        "gemvx kernel type 2",
        "gemvx kernel type 3",
    ],
    # vectorized elementwise kernels
    "group-vectorized_elementwise": [
        "vectorized_elementwise_kernel<4, AunaryFunctor<…>, _>",
        "vectorized_elementwise_kernel<4, CUDAFunctor_add<…>, _>",
        "vectorized_elementwise_kernel<4, GeluCUDAKernelImpl(…), _>",
        "vectorized_elementwise_kernel<4, launch_clamp_scalar(…), _>",
        "vectorized_elementwise_kernel<4, FillFunctor<float>, _>",
        "vectorized_elementwise_kernel<4, FillFunctor<long>, _>",
        "vectorized_elementwise_kernel<4, BinaryFunctor<…>, …>",
        "vectorized_elementwise_kernel<4, AddFunctor<float>, _>",
        "vectorized_elementwise_kernel<4, threshold_kernel_impl<…>, _>",
        "vectorized_elementwise_kernel<4, BUnaryFunctor<CompareGTFunctor<long>,_>, _>",
        "vectorized_elementwise_kernel<4, BUnaryFunctor<AddFunctor<float>,_>, _>",
        "vectorized_elementwise_kernel<4, MulScalarFunctor<float, float>, _>",
        "vectorized_elementwise_kernel<4, pow_tensor_scalar_kernel_impl(…), _>",
        "vectorized_elementwise_kernel<4, tanh_kernel_cuda(…), _>",
        "vectorized_elementwise_kernel<4, MulFunctor<float>, _>",
    ],
    # unrolled elementwise kernels
    "group-unrolled_elementwise": [
        "unrolled_elementwise_kernel<copy_device_to_device(…), _> (bool)",
        "unrolled_elementwise_kernel<copy_device_to_device(…), _> (float)",
        "unrolled_elementwise_kernel<AddFunctor<float>, _>",
        "unrolled_elementwise_kernel<MulFunctor<float>, _>",
        "unrolled_elementwise_kernel<DivFunctor<float>, _>",
        "unrolled_elementwise_kernel<BUnaryFunctor<CompareEqFunctor<long>,_>, _>",
    ],
    # reduction kernels
    "group-reduction": [
        "reduce_kernel<256, 2, …>",
        "reduce_kernel<128, 4, …>",
        "reduce_kernel<512, 1, MeanOps<…> >",
        "reduce_kernel<512, 1, WelfordOps<…> >",
        "reduce_1Block_kernel",
        "splitKreduce_kernel<32, 16, …>",
        "splitKreduce_kernel",
    ],
}


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
    group: bool = False

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
        parser.add_argument(
            "--group",
            action="store_true",
            help="Group similar kernels in the plots.",
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
        return cls(args.input_file, args.output_dir, args.group)


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

    # optionally group similar kernels
    if config.group:

        def map_to_group(name: str) -> str:
            for group_name, group_kernels in kernel_groups.items():
                if name in group_kernels:
                    return group_name
            return "other"

        df["clean_names"] = df["clean_names"].apply(map_to_group)

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

    for kernel in df["clean_names"].unique():
        if kernel in kernels_to_keep or (
            config.group and kernel in kernel_groups.keys()
        ):
            correlation_matrix_kernel(df, kernel, config.output_dir)

    for metric in metrics_to_plot:
        save_violin_plot(df, metric, f"{config.output_dir}/{metric}_violin_plot.png")
        save_line_plot(df, metric, f"{config.output_dir}/{metric}_line_plot.png")


if __name__ == "__main__":
    main()
