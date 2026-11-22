// MAHE ARM64 NEON 内核（Apple Silicon / AArch64 Linux）
// C 原型（与 kernel.cpp 一致）：
//   void mahe_reduce_add_f32_neon(float* dst, const float* src, size_t n)
//   void mahe_sgd_update_f32_neon(float* param, const float* grad, size_t n, float lr)
//
// AArch64 调用约定：x0/dst, x1/src, x2/n, s0/lr
// NEON 在 aarch64 是基础特性，无条件可用。

// Mach-O（macOS/iOS）的 C 符号带前导下划线，ELF 不带；.S 会先过 C 预处理器
#ifdef __APPLE__
#define MAHE_SYM(x) _##x
#else
#define MAHE_SYM(x) x
#endif

.text

.align 4
.global MAHE_SYM(mahe_reduce_add_f32_neon)
MAHE_SYM(mahe_reduce_add_f32_neon):
    mov     x3, x2
    lsr     x3, x3, #4              // 每轮 16 个 float（4 个 q 寄存器）
    cbz     x3, .Ltail8
1:
    ld1     {v0.4s - v3.4s}, [x0]    // dst 16
    ld1     {v4.4s - v7.4s}, [x1]    // src 16
    fadd    v0.4s, v0.4s, v4.4s
    fadd    v1.4s, v1.4s, v5.4s
    fadd    v2.4s, v2.4s, v6.4s
    fadd    v3.4s, v3.4s, v7.4s
    st1     {v0.4s - v3.4s}, [x0]
    add     x0, x0, #64
    add     x1, x1, #64
    subs    x3, x3, #1
    bne     1b
.Ltail8:
    and     x3, x2, #15             // 剩余 <16 个，标量处理
    cbz     x3, .Ldone
2:
    ldr     s0, [x0]
    ldr     s1, [x1]
    fadd    s0, s0, s1
    str     s0, [x0]
    add     x0, x0, #4
    add     x1, x1, #4
    subs    x3, x3, #1
    bne     2b
.Ldone:
    ret

.align 4
.global MAHE_SYM(mahe_sgd_update_f32_neon)
// param[i] -= lr * grad[i]
// 用 fmla：把 lr 取负广播到 v4，param += (-lr)*grad
MAHE_SYM(mahe_sgd_update_f32_neon):
    fmov    w9, s0                  // 保存 lr 的位模式（x9 是调用者保存寄存器）
    dup     v4.4s, v0.s[0]          // 广播 lr（s0 是 v0 的低 lane）
    fneg    v4.4s, v4.4s            // -lr
    mov     x3, x2
    lsr     x3, x3, #4
    cbz     x3, .Sstail
1:
    ld1     {v0.4s - v3.4s}, [x0]
    ld1     {v5.4s - v8.4s}, [x1]
    fmla    v0.4s, v5.4s, v4.4s
    fmla    v1.4s, v6.4s, v4.4s
    fmla    v2.4s, v7.4s, v4.4s
    fmla    v3.4s, v8.4s, v4.4s
    st1     {v0.4s - v3.4s}, [x0]
    add     x0, x0, #64
    add     x1, x1, #64
    subs    x3, x3, #1
    bne     1b
.Sstail:
    and     x3, x2, #15
    cbz     x3, .Sdone
2:
    ldr     s0, [x0]
    ldr     s1, [x1]
    fmov    s2, w9                  // 取回 lr（向量循环里 v5 已被覆盖，不能用它）
    fmul    s2, s1, s2              // lr * grad
    fsub    s0, s0, s2              // param -= lr*grad
    str     s0, [x0]
    add     x0, x0, #4
    add     x1, x1, #4
    subs    x3, x3, #1
    bne     2b
.Sdone:
    ret
