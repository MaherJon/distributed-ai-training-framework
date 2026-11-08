; MAHE AVX2 内核（NASM, x86-64）
; C 原型（与 kernel.cpp 中声明一致）：
;   void mahe_reduce_add_f32_avx2(float* dst, const float* src, size_t n)
;   void mahe_sgd_update_f32_avx2(float* param, const float* grad, size_t n, float lr)
;
; 参数布局：
;   SysV AMD64 : rdi, rsi, rdx, 浮点第1参 lr -> xmm0
;   Win64      : rcx, rdx, r8, 浮点第4参 lr -> xmm3
; 均为叶子函数。

%ifdef WIN64
  %define A1 rcx
  %define A2 rdx
  %define A3 r8
%else
  %define A1 rdi
  %define A2 rsi
  %define A3 rdx
%endif

section .text

global mahe_reduce_add_f32_avx2
global mahe_sgd_update_f32_avx2

; ---------------------------------------------------------------------------
; void mahe_reduce_add_f32_avx2(float* dst, const float* src, size_t n)
; dst[i] += src[i]
; ---------------------------------------------------------------------------
align 16
mahe_reduce_add_f32_avx2:
    mov     rax, A3
    shr     rax, 5              ; 主循环每次 32 个 float（4×ymm）
    jz      .tail8
.loop32:
    vmovups ymm0, [A1]
    vmovups ymm1, [A1+32]
    vmovups ymm2, [A1+64]
    vmovups ymm3, [A1+96]
    vaddps  ymm0, ymm0, [A2]
    vaddps  ymm1, ymm1, [A2+32]
    vaddps  ymm2, ymm2, [A2+64]
    vaddps  ymm3, ymm3, [A2+96]
    vmovups [A1],    ymm0
    vmovups [A1+32], ymm1
    vmovups [A1+64], ymm2
    vmovups [A1+96], ymm3
    add     A1, 128
    add     A2, 128
    dec     rax
    jnz     .loop32
.tail8:
    mov     rax, A3
    and     rax, 31
    shr     rax, 3              ; 每次 8 个
    jz      .tail1
.loop8:
    vmovups ymm0, [A1]
    vaddps  ymm0, ymm0, [A2]
    vmovups [A1], ymm0
    add     A1, 32
    add     A2, 32
    dec     rax
    jnz     .loop8
.tail1:
    and     A3, 7               ; 剩余 <8 个，标量
    jz      .done
.loop1:
    vmovss  xmm0, [A1]
    vaddss  xmm0, xmm0, [A2]
    vmovss  [A1], xmm0
    add     A1, 4
    add     A2, 4
    dec     A3
    jnz     .loop1
.done:
    vzeroupper
    ret

; ---------------------------------------------------------------------------
; void mahe_sgd_update_f32_avx2(float* param, const float* grad, size_t n, float lr)
; param[i] -= lr * grad[i]
; ---------------------------------------------------------------------------
align 16
mahe_sgd_update_f32_avx2:
%ifdef WIN64
    vbroadcastss ymm4, xmm3     ; lr（向量）
    vmovss  xmm5, xmm3, xmm3    ; lr（标量尾部备份；主循环会覆写 xmm0）
%else
    vbroadcastss ymm4, xmm0
    vmovss  xmm5, xmm0, xmm0
%endif
    mov     rax, A3
    shr     rax, 3
    jz      .stail1
.sloop8:
    vmovups ymm0, [A1]          ; param
    vmovups ymm1, [A2]          ; grad
    vfnmadd231ps ymm0, ymm1, ymm4   ; param -= grad*lr
    vmovups [A1], ymm0
    add     A1, 32
    add     A2, 32
    dec     rax
    jnz     .sloop8
.stail1:
    and     A3, 7
    jz      .sdone
.sloop1:
    vmovss  xmm0, [A1]
    vmovss  xmm1, [A2]
    vfnmadd231ss xmm0, xmm1, xmm5
    vmovss  [A1], xmm0
    add     A1, 4
    add     A2, 4
    dec     A3
    jnz     .sloop1
.sdone:
    vzeroupper
    ret

; ELF：标记无需可执行栈
%ifdef ELF
section .note.GNU-stack noalloc noexec nowrite progbits
%endif
