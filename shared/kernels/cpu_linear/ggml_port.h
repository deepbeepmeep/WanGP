// What the ggml code ported in ggml_x86_port.c expects (ggml, MIT License, see LICENSE_GGML.txt): block layouts of ggml-common.h, macros
// and the ported dot products. Shared by the C port and linear.cpp.
#pragma once
#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_RESTRICT __restrict
#define UNUSED(x) (void)(x)
#define GGML_UNUSED(x) (void)(x)
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
#define GGML_TABLE_BEGIN(type, name, size) static const type name[size] = {
#define GGML_TABLE_END() };
#define QK_K 256
#define K_SCALE_SIZE 12
#define QK8_0 32
#define IQ3S_N_SCALE (QK_K / 64)
#define QK4_NL 32
#define QK2_0 64
#ifndef __cplusplus
#define static_assert(condition, message)
#endif
typedef uint16_t ggml_half;

static inline float ggml_port_fp16_to_fp32(ggml_half h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))); }
static inline ggml_half ggml_port_fp32_to_fp16(float f) { return (ggml_half)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), 0), 0); }
#define GGML_CPU_FP16_TO_FP32(x) ggml_port_fp16_to_fp32(x)
#define GGML_CPU_FP32_TO_FP16(x) ggml_port_fp32_to_fp16(x)
#define GGML_FP16_TO_FP32(x) ggml_port_fp16_to_fp32(x)
#define GGML_FP32_TO_FP16(x) ggml_port_fp32_to_fp16(x)

typedef struct { ggml_half d; ggml_half dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K / 2]; } block_q4_K;
typedef struct { ggml_half d; ggml_half dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qh[QK_K / 8]; uint8_t qs[QK_K / 2]; } block_q5_K;
typedef struct { uint8_t ql[QK_K / 2]; uint8_t qh[QK_K / 4]; int8_t scales[QK_K / 16]; ggml_half d; } block_q6_K;
typedef struct { ggml_half d; int8_t qs[QK8_0]; } block_q8_0;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K / 16]; } block_q8_K;
typedef struct { ggml_half d; uint8_t qs[QK_K / 4]; uint8_t qh[QK_K / 32]; uint8_t scales[QK_K / 32]; } block_iq2_s;
typedef struct { ggml_half d; uint8_t qs[QK_K / 4]; uint8_t qh[QK_K / 32]; uint8_t signs[QK_K / 8]; uint8_t scales[IQ3S_N_SCALE]; } block_iq3_s;
typedef struct { ggml_half d; uint8_t qs[3 * QK_K / 8]; } block_iq3_xxs;
typedef struct { ggml_half d; uint8_t qs[QK4_NL / 2]; } block_iq4_nl;
typedef struct { ggml_half d; uint16_t scales_h; uint8_t scales_l[QK_K / 64]; uint8_t qs[QK_K / 2]; } block_iq4_xs;
typedef struct { ggml_half d; uint8_t qs[QK2_0 / 4]; } block_q2_0;

void ggml_vec_dot_q4_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q8_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_s_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_s_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_nl_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_xs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void quantize_row_q8_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);

#ifdef __cplusplus
}
#endif
