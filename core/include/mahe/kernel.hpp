// MAHE 计算内核：汇编加速（AVX2）+ 标量回退，CPUID 运行时分发
// 只做 SIMD 计算，不碰网络/系统逻辑。
#pragma once

#include <cstddef>

namespace mahe {
namespace kernel {

// dst[i] += src[i]（梯度/参数差值累加）
void reduce_add_f32(float* dst, const float* src, size_t n);

// param[i] -= lr * grad[i]（融合更新，减少内存往返）
void sgd_update_f32(float* param, const float* grad, float lr, size_t n);

// 当前激活的实现名称（"avx2" / "scalar"），供诊断
const char* active_impl();

// 标量参考实现（测试对照用）
void reduce_add_f32_scalar(float* dst, const float* src, size_t n);
void sgd_update_f32_scalar(float* param, const float* grad, float lr, size_t n);

} // namespace kernel
} // namespace mahe
