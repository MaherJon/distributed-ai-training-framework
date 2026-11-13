#include "mahe/kernel.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#define MAHE_X86_64 1
#if defined(_MSC_VER)
#include <immintrin.h>
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace mahe {
namespace kernel {

// ---- 标量回退（任何平台可用，也是正确性参考） ----
void reduce_add_f32_scalar(float* dst, const float* src, size_t n) {
  for (size_t i = 0; i < n; ++i)
    dst[i] += src[i];
}

void sgd_update_f32_scalar(float* param, const float* grad, float lr,
                           size_t n) {
  for (size_t i = 0; i < n; ++i)
    param[i] -= lr * grad[i];
}

#ifdef MAHE_X86_64

// 汇编实现（NASM），C 签名见 asm/reduce_avx2.asm
// 只有 build.py 成功汇编出目标文件时才定义 MAHE_HAS_AVX2_ASM；
// 否则必须走标量回退，不能引用不存在的符号（否则链接失败）。
#ifdef MAHE_HAS_AVX2_ASM
extern "C" void mahe_reduce_add_f32_avx2(float* dst, const float* src,
                                         size_t n);
extern "C" void mahe_sgd_update_f32_avx2(float* param, const float* grad,
                                         size_t n, float lr);

// 检测 OS 是否启用 YMM 状态保存 + CPU 支持 AVX2 & FMA
static bool detect_avx2_fma() {
#if defined(_MSC_VER)
  int regs[4];
  __cpuid(regs, 1);
  const bool osxsave = (regs[2] & (1 << 27)) != 0;
  const bool avx = (regs[2] & (1 << 28)) != 0;
  const bool fma = (regs[2] & (1 << 12)) != 0;
  if (!osxsave || !avx || !fma)
    return false;
  const unsigned long long xcr0 = _xgetbv(0);
  if ((xcr0 & 0x6) != 0x6)
    return false;
  __cpuidex(regs, 7, 0);
  return (regs[1] & (1 << 5)) != 0;
#else
  unsigned int eax, ebx, ecx, edx;

  // leaf 1: OSXSAVE(ECX.27), AVX(ECX.28), FMA(ECX.12)
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx))
    return false;
  const bool osxsave = (ecx & (1u << 27)) != 0;
  const bool avx = (ecx & (1u << 28)) != 0;
  const bool fma = (ecx & (1u << 12)) != 0;
  if (!osxsave || !avx || !fma)
    return false;

  // XCR0: SSE(bit1) 与 YMM(bit2) 均由 OS 保存（内联汇编避免编译器 target 限制）
  unsigned int xcr0_lo, xcr0_hi;
  __asm__ __volatile__("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
  if ((xcr0_lo & 0x6) != 0x6)
    return false;
  (void)xcr0_hi;

  // leaf 7: AVX2(EBX.5)
  if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
    return false;
  return (ebx & (1u << 5)) != 0;
#endif
}

static const bool g_has_avx2 = detect_avx2_fma();

void reduce_add_f32(float* dst, const float* src, size_t n) {
  if (g_has_avx2) {
    mahe_reduce_add_f32_avx2(dst, src, n);
  } else {
    reduce_add_f32_scalar(dst, src, n);
  }
}

void sgd_update_f32(float* param, const float* grad, float lr, size_t n) {
  if (g_has_avx2) {
    mahe_sgd_update_f32_avx2(param, grad, n, lr);
  } else {
    sgd_update_f32_scalar(param, grad, lr, n);
  }
}

const char* active_impl() { return g_has_avx2 ? "avx2" : "scalar"; }

#else // 未汇编 AVX2（缺 NASM 或 --no-asm）：标量回退

void reduce_add_f32(float* dst, const float* src, size_t n) {
  reduce_add_f32_scalar(dst, src, n);
}
void sgd_update_f32(float* param, const float* grad, float lr, size_t n) {
  sgd_update_f32_scalar(param, grad, lr, n);
}
const char* active_impl() { return "scalar"; }

#endif

#else // 非 x86-64

#if defined(__aarch64__) || defined(_M_ARM64)

// ARM64 NEON 实现（见 asm/reduce_neon.S）
// aarch64 上 NEON 是基础特性，无条件可用（Apple Silicon / AArch64 Linux）
extern "C" void mahe_reduce_add_f32_neon(float* dst, const float* src,
                                         size_t n);
extern "C" void mahe_sgd_update_f32_neon(float* param, const float* grad,
                                         size_t n, float lr);

void reduce_add_f32(float* dst, const float* src, size_t n) {
  mahe_reduce_add_f32_neon(dst, src, n);
}
void sgd_update_f32(float* param, const float* grad, float lr, size_t n) {
  mahe_sgd_update_f32_neon(param, grad, n, lr);
}
const char* active_impl() { return "neon"; }

#else // 其它架构：标量回退

void reduce_add_f32(float* dst, const float* src, size_t n) {
  reduce_add_f32_scalar(dst, src, n);
}
void sgd_update_f32(float* param, const float* grad, float lr, size_t n) {
  sgd_update_f32_scalar(param, grad, lr, n);
}
const char* active_impl() { return "scalar"; }

#endif

#endif

} // namespace kernel
} // namespace mahe
