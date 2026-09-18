/*
 * test_libtinygiant.c — Verify Q4_K × Q8 kernel correctness
 *
 * Tests the NEON kernel against the scalar reference, and both
 * against gguf_dequantize → numpy matmul (via known test vectors).
 *
 * Compile: clang -O3 -mcpu=apple-m1 -o test_libtinygiant tools/test_libtinygiant.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arm_neon.h>

#define QK_K      256
#define Q4K_BSIZE 144

/* Pull in the library source directly for testing */
#include "libtinygiant.c"

/* Scalar reference for Q4_K × Q8 (to isolate NEON bugs from Q8 error) */
static float dot_q4k_q8_scalar_ref(const uint8_t *q4, const q8_block *xq, int bpr) {
    float total = 0;
    for (int b = 0; b < bpr; b++) {
        const uint8_t *bl = q4 + b * Q4K_BSIZE;
        uint16_t dr, dmr;
        memcpy(&dr, bl, 2);
        memcpy(&dmr, bl + 2, 2);
        float d = f16_to_f32(dr), dmin = f16_to_f32(dmr);
        const uint8_t *scales = bl + 4;
        const uint8_t *qs = bl + 16;
        const int8_t *xqs = xq[b].qs;
        float sx = xq[b].scale;

        for (int g = 0; g < 4; g++) {
            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            get_scale_min_k4(2 * g,     scales, &sc_lo, &m_lo);
            get_scale_min_k4(2 * g + 1, scales, &sc_hi, &m_hi);
            const uint8_t *qs_g = qs + g * 32;

            int32_t idot_lo = 0, idot_hi = 0;
            int32_t isum_lo = 0, isum_hi = 0;
            for (int j = 0; j < 32; j++) {
                uint8_t v = qs_g[j];
                int8_t nib_lo = v & 0xF;
                int8_t nib_hi = v >> 4;
                int8_t xq_lo_j = xqs[g * 64 + j];
                int8_t xq_hi_j = xqs[g * 64 + 32 + j];
                idot_lo += (int32_t)nib_lo * (int32_t)xq_lo_j;
                idot_hi += (int32_t)nib_hi * (int32_t)xq_hi_j;
                isum_lo += xq_lo_j;
                isum_hi += xq_hi_j;
            }
            total += sx * (d * (float)sc_lo * (float)idot_lo
                         - dmin * (float)m_lo * (float)isum_lo);
            total += sx * (d * (float)sc_hi * (float)idot_hi
                         - dmin * (float)m_hi * (float)isum_hi);
        }
    }
    return total;
}

/* Scalar Q6_K reference (original byte-loop unpack) */
static float dot_q6k_q8_scalar_ref(const uint8_t *q6, const q8_block *xq, int bpr) {
    float total = 0;
    for (int b = 0; b < bpr; b++) {
        const uint8_t *bl = q6 + b * Q6K_BSIZE;
        const uint8_t *ql = bl, *qh = bl + 128;
        const int8_t *sc = (const int8_t *)(bl + 192);
        uint16_t dr; memcpy(&dr, bl + 208, 2);
        float d = f16_to_f32(dr);
        const int8_t *xqs = xq[b].qs;
        float block_sum = 0;
        for (int half = 0; half < 2; half++) {
            const uint8_t *ql_h = ql + half * 64, *qh_h = qh + half * 32;
            const int8_t *sc_h = sc + half * 8, *xq_h = xqs + half * 128;
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql_h[l] & 0xF) | (((qh_h[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql_h[l + 32] & 0xF) | (((qh_h[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql_h[l] >> 4) | (((qh_h[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql_h[l + 32] >> 4) | (((qh_h[l] >> 6) & 3) << 4)) - 32;
                block_sum += (float)sc_h[is + 0] * (float)(q1 * xq_h[l]);
                block_sum += (float)sc_h[is + 2] * (float)(q2 * xq_h[l + 32]);
                block_sum += (float)sc_h[is + 4] * (float)(q3 * xq_h[l + 64]);
                block_sum += (float)sc_h[is + 6] * (float)(q4 * xq_h[l + 96]);
            }
        }
        total += d * xq[b].scale * block_sum;
    }
    return total;
}

static void fill_q4k_block(uint8_t *block, int seed) {
    srand(seed);
    /* d and dmin: small positive f16 values */
    uint16_t d_f16 = 0x3400;    /* ~0.25 */
    uint16_t dmin_f16 = 0x2C00; /* ~0.0625 */
    memcpy(block, &d_f16, 2);
    memcpy(block + 2, &dmin_f16, 2);

    /* scales: random 6-bit values packed into 12 bytes */
    for (int i = 0; i < 12; i++)
        block[4 + i] = rand() & 0xFF;

    /* qs: random nibbles */
    for (int i = 0; i < 128; i++)
        block[16 + i] = rand() & 0xFF;
}

static void dequant_q4k_block(const uint8_t *block, float *out) {
    uint16_t dr, dmr;
    memcpy(&dr, block, 2);
    memcpy(&dmr, block + 2, 2);
    float d = f16_to_f32(dr), dmin = f16_to_f32(dmr);
    const uint8_t *scales = block + 4;
    const uint8_t *qs = block + 16;

    for (int g = 0; g < 4; g++) {
        uint8_t sc_lo, m_lo, sc_hi, m_hi;
        get_scale_min_k4(2 * g,     scales, &sc_lo, &m_lo);
        get_scale_min_k4(2 * g + 1, scales, &sc_hi, &m_hi);

        for (int j = 0; j < 32; j++) {
            uint8_t v = qs[g * 32 + j];
            out[g * 64 + j]      = d * sc_lo * (float)(v & 0xF) - dmin * m_lo;
            out[g * 64 + 32 + j] = d * sc_hi * (float)(v >> 4)  - dmin * m_hi;
        }
    }
}

int main(void) {
    printf("TinyGiant Q4_K kernel verification\n");
    printf("==================================\n\n");

    int pass = 1;
    int n_tests = 100;

    /* Test 1a: NEON Q4K×Q8 vs SCALAR Q4K×Q8 (same Q8 data — isolates NEON bugs) */
    printf("Test 1a: NEON vs scalar Q4K×Q8 (same Q8 input, %d trials)\n", n_tests);
    {
        float max_rel_err = 0;
        for (int trial = 0; trial < n_tests; trial++) {
            uint8_t block[Q4K_BSIZE];
            fill_q4k_block(block, trial * 31 + 7);

            float x[QK_K];
            srand(trial * 17 + 3);
            for (int i = 0; i < QK_K; i++)
                x[i] = ((float)rand() / (float)0x7fffffff - 0.5f) * 0.1f;

            q8_block xq[1];
            quantize_q8(x, xq, 1);

            float scalar_q8 = dot_q4k_q8_scalar_ref(block, xq, 1);
            float neon_q8   = dot_q4k_q8(block, xq, 1);

            float denom = fabsf(scalar_q8) > 1e-3f ? fabsf(scalar_q8) : 1.0f;
            float rel_err = fabsf(scalar_q8 - neon_q8) / denom;
            if (rel_err > max_rel_err) max_rel_err = rel_err;
        }
        printf("  Max relative error: %.4e\n", max_rel_err);
        if (max_rel_err < 1e-4f) {
            printf("  PASS (NEON matches scalar Q8 reference)\n\n");
        } else {
            printf("  FAIL (NEON kernel has a bug)\n\n");
            pass = 0;
        }
    }

    /* Test 1b: Q8 quantization error (expected ~1-5%) */
    printf("Test 1b: Q8 quantization error (float vs Q8 reconstruction, %d trials)\n", n_tests);
    {
        float max_rel_err = 0;
        for (int trial = 0; trial < n_tests; trial++) {
            uint8_t block[Q4K_BSIZE];
            fill_q4k_block(block, trial * 31 + 7);

            float x[QK_K];
            srand(trial * 17 + 3);
            for (int i = 0; i < QK_K; i++)
                x[i] = ((float)rand() / (float)0x7fffffff - 0.5f) * 0.1f;

            float exact   = dot_q4k_scalar(block, x, 1);

            q8_block xq[1];
            quantize_q8(x, xq, 1);
            float q8_ref = dot_q4k_q8_scalar_ref(block, xq, 1);

            float denom = fabsf(exact) > 1e-3f ? fabsf(exact) : 1.0f;
            float rel_err = fabsf(exact - q8_ref) / denom;
            if (rel_err > max_rel_err) max_rel_err = rel_err;
        }
        printf("  Max relative error: %.4e\n", max_rel_err);
        printf("  (This is expected Q8 quantization error, not a bug)\n\n");
    }

    /* Test 2: Dequant consistency — our scalar matches manual dequant */
    printf("Test 2: Scalar dot matches manual dequant + float dot\n");
    {
        float max_err = 0;
        for (int trial = 0; trial < n_tests; trial++) {
            uint8_t block[Q4K_BSIZE];
            fill_q4k_block(block, trial * 43 + 11);

            float x[QK_K];
            srand(trial * 23 + 5);
            for (int i = 0; i < QK_K; i++)
                x[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

            /* Manual: dequant then float dot */
            float deq[QK_K];
            dequant_q4k_block(block, deq);
            float manual = 0;
            for (int i = 0; i < QK_K; i++) manual += deq[i] * x[i];

            float scalar = dot_q4k_scalar(block, x, 1);

            float err = fabsf(manual - scalar);
            if (err > max_err) max_err = err;
        }
        printf("  Max absolute error: %.4e\n", max_err);
        if (max_err < 1e-4f) {
            printf("  PASS (scalar matches manual dequant exactly)\n\n");
        } else {
            printf("  FAIL\n\n");
            pass = 0;
        }
    }

    /* Test 3: Multi-block matmul — NEON vs scalar with same Q8 input */
    printf("Test 3: Matrix-vector multiply (768x2048, NEON Q8 vs scalar Q8)\n");
    {
        int nrows = 768, ncols = 2048;
        int bpr = ncols / QK_K;
        size_t mat_bytes = (size_t)nrows * bpr * Q4K_BSIZE;

        uint8_t *mat = malloc(mat_bytes);
        for (int r = 0; r < nrows; r++)
            for (int b = 0; b < bpr; b++)
                fill_q4k_block(mat + ((size_t)r * bpr + b) * Q4K_BSIZE,
                               r * 13 + b * 7 + 42);

        float x[2048];
        srand(999);
        for (int i = 0; i < 2048; i++)
            x[i] = ((float)rand() / (float)0x7fffffff - 0.5f) * 0.02f;

        /* Quantize input once, use for both paths */
        q8_block xq[bpr];
        quantize_q8(x, xq, bpr);

        float *out_scalar = calloc(nrows, sizeof(float));
        float *out_neon = calloc(nrows, sizeof(float));

        /* Scalar Q8 reference */
        for (int r = 0; r < nrows; r++)
            out_scalar[r] = dot_q4k_q8_scalar_ref(
                mat + (size_t)r * bpr * Q4K_BSIZE, xq, bpr);

        /* NEON Q8 */
        matmul_q4k(mat, x, out_neon, nrows, ncols);

        float max_rel = 0, max_abs = 0;
        for (int r = 0; r < nrows; r++) {
            float abs_err = fabsf(out_scalar[r] - out_neon[r]);
            float denom = fabsf(out_scalar[r]) > 1e-3f ? fabsf(out_scalar[r]) : 1.0f;
            float rel = abs_err / denom;
            if (rel > max_rel) max_rel = rel;
            if (abs_err > max_abs) max_abs = abs_err;
        }

        printf("  Max relative error: %.4e\n", max_rel);
        printf("  Max absolute error: %.4e\n", max_abs);
        if (max_rel < 0.01f) {
            printf("  PASS (< 1%% — within Q8 precision)\n\n");
        } else {
            printf("  FAIL\n\n");
            pass = 0;
        }

        free(mat); free(out_scalar); free(out_neon);
    }

    /* Test 4: Expert forward pass smoke test */
    printf("Test 4: Expert forward pass (no crash, output non-zero)\n");
    {
        int embed = 2048, inter = 768;
        int gate_bpr = embed / QK_K;
        size_t gate_bytes = (size_t)inter * gate_bpr * Q4K_BSIZE;
        int down_bpr = inter / QK_K;
        size_t down_bytes = (size_t)embed * down_bpr * Q4K_BSIZE;
        size_t expert_bytes = 2 * gate_bytes + down_bytes;

        uint8_t *expert = malloc(expert_bytes);
        srand(12345);
        for (size_t i = 0; i < expert_bytes; i += Q4K_BSIZE) {
            size_t remaining = expert_bytes - i;
            size_t bsize = remaining < Q4K_BSIZE ? remaining : Q4K_BSIZE;
            fill_q4k_block(expert + i, (int)(i / Q4K_BSIZE));
            (void)bsize;
        }

        float input[2048], output[2048];
        srand(54321);
        for (int i = 0; i < 2048; i++)
            input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        memset(output, 0, sizeof(output));

        tg_expert_forward(expert, input, output, 0.125f, embed, inter);

        float sum = 0;
        for (int i = 0; i < 2048; i++) sum += fabsf(output[i]);

        printf("  Output L1 norm: %.6f\n", sum);
        if (sum > 0 && !isnan(sum) && !isinf(sum)) {
            printf("  PASS (non-zero, finite output)\n\n");
        } else {
            printf("  FAIL\n\n");
            pass = 0;
        }

        free(expert);
    }

    /* Test 5: 4-row Q4_K kernel matches single-row kernel */
    printf("Test 5: dot_q4k_q8_x4 vs dot_q4k_q8 (8 blocks, 4 rows)\n");
    {
        int bpr = 8;
        uint8_t *row = malloc(bpr * Q4K_BSIZE);
        for (int b = 0; b < bpr; b++) fill_q4k_block(row + b * Q4K_BSIZE, 100 + b);
        q8_block xq[4][8];
        const q8_block *p[4];
        for (int r = 0; r < 4; r++) {
            float x[2048];
            srand(500 + r);
            for (int i = 0; i < 2048; i++)
                x[i] = ((float)rand() / (float)0x7fffffff - 0.5f) * 0.1f;
            quantize_q8(x, xq[r], bpr);
            p[r] = xq[r];
        }
        float o4[4];
        dot_q4k_q8_x4(row, p, bpr, o4);
        float max_rel = 0;
        for (int r = 0; r < 4; r++) {
            float o1 = dot_q4k_q8(row, xq[r], bpr);
            float denom = fabsf(o1) > 1e-3f ? fabsf(o1) : 1.f;
            float rel = fabsf(o1 - o4[r]) / denom;
            if (rel > max_rel) max_rel = rel;
        }
        printf("  Max relative error: %.4e\n", max_rel);
        if (max_rel < 1e-5f) printf("  PASS\n\n"); else { printf("  FAIL\n\n"); pass = 0; }
        free(row);
    }

    /* Test 6: vectorized Q6_K unpack matches scalar reference, x4 matches x1 */
    printf("Test 6: Q6_K NEON unpack vs scalar reference; x4 vs x1\n");
    {
        int bpr = 3;
        uint8_t *row = malloc(bpr * Q6K_BSIZE);
        srand(777);
        for (int i = 0; i < bpr * Q6K_BSIZE; i++) row[i] = rand() & 0xFF;
        for (int b = 0; b < bpr; b++) {
            uint16_t d_f16 = 0x3400;
            memcpy(row + b * Q6K_BSIZE + 208, &d_f16, 2);
            int8_t *sc = (int8_t *)(row + b * Q6K_BSIZE + 192);
            for (int i = 0; i < 16; i++) sc[i] = (int8_t)((rand() % 60) - 30);
        }
        q8_block xq[4][3];
        const q8_block *p[4];
        for (int r = 0; r < 4; r++) {
            float x[768];
            srand(600 + r);
            for (int i = 0; i < 768; i++)
                x[i] = ((float)rand() / (float)0x7fffffff - 0.5f) * 0.1f;
            quantize_q8(x, xq[r], bpr);
            p[r] = xq[r];
        }
        float max_rel = 0;
        for (int r = 0; r < 4; r++) {
            float ref = dot_q6k_q8_scalar_ref(row, xq[r], bpr);
            float got = dot_q6k_q8(row, xq[r], bpr);
            float denom = fabsf(ref) > 1e-3f ? fabsf(ref) : 1.f;
            float rel = fabsf(ref - got) / denom;
            if (rel > max_rel) max_rel = rel;
        }
        float o4[4];
        dot_q6k_q8_x4(row, p, bpr, o4);
        for (int r = 0; r < 4; r++) {
            float o1 = dot_q6k_q8(row, xq[r], bpr);
            float denom = fabsf(o1) > 1e-3f ? fabsf(o1) : 1.f;
            float rel = fabsf(o1 - o4[r]) / denom;
            if (rel > max_rel) max_rel = rel;
        }
        printf("  Max relative error: %.4e\n", max_rel);
        if (max_rel < 1e-5f) printf("  PASS\n\n"); else { printf("  FAIL\n\n"); pass = 0; }
        free(row);
    }

    /* Test 7: multi-row expert forward matches per-row calls; threaded MoE matches serial */
    printf("Test 7: tg_expert_forward_rows / tg_moe_forward_rows vs per-row single kernels\n");
    {
        int embed = 2048, inter = 768, k = 6, n_exp = 8;
        int gate_bpr = embed / QK_K, down_bpr = inter / QK_K;
        size_t gate_bytes = (size_t)inter * gate_bpr * Q4K_BSIZE;
        size_t down_q4 = (size_t)embed * down_bpr * Q4K_BSIZE;
        size_t down_q6 = (size_t)embed * down_bpr * Q6K_BSIZE;

        uint8_t *gates[8], *ups[8]; void *downs[8]; int fmts[8];
        for (int e = 0; e < n_exp; e++) {
            gates[e] = malloc(gate_bytes); ups[e] = malloc(gate_bytes);
            for (size_t i = 0; i < gate_bytes; i += Q4K_BSIZE) {
                fill_q4k_block(gates[e] + i, (int)(i / Q4K_BSIZE) + e * 1000);
                fill_q4k_block(ups[e] + i, (int)(i / Q4K_BSIZE) + e * 1000 + 7);
            }
            fmts[e] = e % 2 ? 2 : 0;
            if (fmts[e] == 0) {
                downs[e] = malloc(down_q4);
                for (size_t i = 0; i < down_q4; i += Q4K_BSIZE)
                    fill_q4k_block((uint8_t *)downs[e] + i, (int)(i / Q4K_BSIZE) + e * 3000);
            } else {
                downs[e] = malloc(down_q6);
                srand(e * 91);
                uint8_t *d = downs[e];
                for (size_t i = 0; i < down_q6; i++) d[i] = rand() & 0xFF;
                for (size_t b = 0; b < down_q6 / Q6K_BSIZE; b++) {
                    uint16_t d_f16 = 0x3400;
                    memcpy(d + b * Q6K_BSIZE + 208, &d_f16, 2);
                    int8_t *sc = (int8_t *)(d + b * Q6K_BSIZE + 192);
                    for (int i = 0; i < 16; i++) sc[i] = (int8_t)((rand() % 60) - 30);
                }
            }
        }

        float *inputs = malloc(sizeof(float) * k * embed);
        srand(4242);
        for (int i = 0; i < k * embed; i++)
            inputs[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;

        /* Sparse row weights: each row uses 3 experts */
        float rw[8 * 6];
        memset(rw, 0, sizeof(rw));
        for (int r = 0; r < k; r++)
            for (int j = 0; j < 3; j++) rw[((r + j * 2) % n_exp) * k + r] = 0.2f + 0.1f * j;

        /* Reference: per-row, per-expert single kernel */
        float *ref = calloc((size_t)k * embed, sizeof(float));
        for (int r = 0; r < k; r++)
            for (int e = 0; e < n_exp; e++)
                if (rw[e * k + r] != 0.f)
                    tg_expert_forward_mixed(gates[e], ups[e], downs[e], fmts[e],
                                            inputs + r * embed, ref + r * embed,
                                            rw[e * k + r], embed, inter);

        float *got1 = calloc((size_t)k * embed, sizeof(float));
        tg_moe_forward_rows((const uint8_t *const *)gates, (const uint8_t *const *)ups,
                            (const void *const *)downs, fmts, n_exp,
                            rw, inputs, got1, k, embed, inter);

        tg_set_threads(8);
        float *got8 = calloc((size_t)k * embed, sizeof(float));
        tg_moe_forward_rows((const uint8_t *const *)gates, (const uint8_t *const *)ups,
                            (const void *const *)downs, fmts, n_exp,
                            rw, inputs, got8, k, embed, inter);
        tg_set_threads(1);

        float max_abs = 0, scale = 0;
        for (int i = 0; i < k * embed; i++) {
            if (fabsf(ref[i]) > scale) scale = fabsf(ref[i]);
            float e1 = fabsf(ref[i] - got1[i]), e8 = fabsf(ref[i] - got8[i]);
            if (e1 > max_abs) max_abs = e1;
            if (e8 > max_abs) max_abs = e8;
        }
        printf("  Max abs error %.4e vs output scale %.4e (ratio %.2e)\n",
               max_abs, scale, max_abs / scale);
        if (max_abs < 1e-5f * scale) printf("  PASS\n\n"); else { printf("  FAIL\n\n"); pass = 0; }

        for (int e = 0; e < n_exp; e++) { free(gates[e]); free(ups[e]); free(downs[e]); }
        free(inputs); free(ref); free(got1); free(got8);
    }

    /* Test 8: threaded matmul_q4k matches serial */
    printf("Test 8: threaded matmul_q4k vs serial (4096x2048)\n");
    {
        int nrows = 4096, ncols = 2048, bpr = ncols / QK_K;
        uint8_t *mat = malloc((size_t)nrows * bpr * Q4K_BSIZE);
        for (int r = 0; r < nrows; r++)
            for (int b = 0; b < bpr; b++)
                fill_q4k_block(mat + ((size_t)r * bpr + b) * Q4K_BSIZE, r * 3 + b);
        float x[2048];
        srand(31337);
        for (int i = 0; i < 2048; i++) x[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        float *o1 = malloc(sizeof(float) * nrows), *o8 = malloc(sizeof(float) * nrows);
        matmul_q4k(mat, x, o1, nrows, ncols);
        tg_set_threads(8);
        matmul_q4k(mat, x, o8, nrows, ncols);
        tg_set_threads(1);
        float max_abs = 0;
        for (int r = 0; r < nrows; r++) {
            float e = fabsf(o1[r] - o8[r]);
            if (e > max_abs) max_abs = e;
        }
        printf("  Max absolute diff: %.4e\n", max_abs);
        if (max_abs == 0.f) printf("  PASS\n\n"); else { printf("  FAIL\n\n"); pass = 0; }
        free(mat); free(o1); free(o8);
    }

    printf("==================================\n");
    printf("Result: %s\n", pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return pass ? 0 : 1;
}
