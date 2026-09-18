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
            quantize_q8(x, xq, QK_K);

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
            quantize_q8(x, xq, QK_K);
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
        quantize_q8(x, xq, ncols);

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
            quantize_q8(x, xq[r], 2048);
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
            quantize_q8(x, xq[r], 768);
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

    /* Test 9: MXFP4 and Q8_0 kernels vs scalar dequantized references (ncols = 2880) */
    printf("Test 9: MXFP4 / Q8_0 kernels vs dequantized float reference (ncols 2880)\n");
    {
        int ncols = 2880, n_sub = ncols / 32;
        uint8_t *wm = malloc((size_t)n_sub * MXFP4_BSIZE);
        uint8_t *w8 = malloc((size_t)n_sub * Q80_BSIZE);
        float *fm = malloc(sizeof(float) * ncols), *f8 = malloc(sizeof(float) * ncols);
        srand(2025);
        for (int s = 0; s < n_sub; s++) {
            uint8_t *bl = wm + s * MXFP4_BSIZE;
            bl[0] = (uint8_t)(120 + rand() % 8);
            for (int i = 0; i < 16; i++) bl[1 + i] = rand() & 0xFF;
            float d = e8m0_to_fp32_half(bl[0]);
            for (int i = 0; i < 16; i++) {
                fm[s * 32 + i]      = d * mxfp4_kvalues[bl[1 + i] & 0xF];
                fm[s * 32 + 16 + i] = d * mxfp4_kvalues[bl[1 + i] >> 4];
            }
            uint8_t *b8 = w8 + s * Q80_BSIZE;
            uint16_t d16 = 0x2C00;   /* 0.0625 */
            memcpy(b8, &d16, 2);
            for (int i = 0; i < 32; i++) {
                int8_t q = (int8_t)(rand() % 255 - 127);
                b8[2 + i] = (uint8_t)q;
                f8[s * 32 + i] = 0.0625f * q;
            }
        }
        float *x = malloc(sizeof(float) * ncols);
        float *x2 = malloc(sizeof(float) * ncols);
        for (int i = 0; i < ncols; i++) {
            x[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
            x2[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
        }
        q8_block xq[Q8_NBLOCKS(2880)], xq2[Q8_NBLOCKS(2880)];
        quantize_q8(x, xq, ncols);
        quantize_q8(x2, xq2, ncols);
        float ref_m = 0, ref_8 = 0, ref_m2 = 0;
        for (int i = 0; i < ncols; i++) { ref_m += fm[i] * x[i]; ref_8 += f8[i] * x[i]; ref_m2 += fm[i] * x2[i]; }
        float got_m = dot_mxfp4_q8(wm, xq, ncols);
        float got_8 = dot_q80_q8(w8, xq, ncols);
        const q8_block *px[4] = { xq, xq2, xq, xq2 };
        float o4[4];
        dot_mxfp4_q8_x4(wm, px, ncols, o4);
        float e1 = fabsf(got_m - ref_m) / fabsf(ref_m), e2 = fabsf(got_8 - ref_8) / fabsf(ref_8);
        float e3 = fabsf(o4[0] - got_m) / fabsf(got_m), e4 = fabsf(o4[1] - ref_m2) / fabsf(ref_m2);
        printf("  mxfp4 rel err %.3e, q8_0 rel err %.3e, x4 vs x1 %.3e, x4 row2 vs ref %.3e\n", e1, e2, e3, e4);
        if (e1 < 0.02f && e2 < 0.02f && e3 < 1e-5f && e4 < 0.02f) printf("  PASS\n\n");
        else { printf("  FAIL\n\n"); pass = 0; }
        free(wm); free(w8); free(fm); free(f8); free(x); free(x2);
    }

    /* Test 10: generalized MoE (MXFP4 experts, gpt-oss activation, biases) vs float reference */
    printf("Test 10: tg_moe_forward_rows_v2 with MXFP4 + gpt-oss activation + biases\n");
    {
        int embed = 2880, inter = 2880, k = 3, n_exp = 2;
        int n_sub_e = embed / 32, n_sub_i = inter / 32;
        size_t gate_bytes = (size_t)inter * n_sub_e * MXFP4_BSIZE;
        size_t down_bytes = (size_t)embed * n_sub_i * MXFP4_BSIZE;
        uint8_t *gates[2], *ups[2], *downs[2];
        float *fg[2], *fu[2], *fd[2], *bg[2], *bu[2], *bd[2];
        int fmts[2] = { TG_FMT_MXFP4, TG_FMT_MXFP4 };
        srand(77);
        for (int e = 0; e < n_exp; e++) {
            gates[e] = malloc(gate_bytes); ups[e] = malloc(gate_bytes); downs[e] = malloc(down_bytes);
            fg[e] = malloc(sizeof(float) * inter * embed); fu[e] = malloc(sizeof(float) * inter * embed);
            fd[e] = malloc(sizeof(float) * embed * inter);
            bg[e] = malloc(sizeof(float) * inter); bu[e] = malloc(sizeof(float) * inter); bd[e] = malloc(sizeof(float) * embed);
            uint8_t *mats[3] = { gates[e], ups[e], downs[e] };
            float *fl[3] = { fg[e], fu[e], fd[e] };
            size_t nsub[3] = { (size_t)inter * n_sub_e, (size_t)inter * n_sub_e, (size_t)embed * n_sub_i };
            for (int m = 0; m < 3; m++) {
                for (size_t s = 0; s < nsub[m]; s++) {
                    uint8_t *bl = mats[m] + s * MXFP4_BSIZE;
                    bl[0] = (uint8_t)(112 + rand() % 6);
                    for (int i = 0; i < 16; i++) bl[1 + i] = rand() & 0xFF;
                    float d = e8m0_to_fp32_half(bl[0]);
                    for (int i = 0; i < 16; i++) {
                        fl[m][s * 32 + i]      = d * mxfp4_kvalues[bl[1 + i] & 0xF];
                        fl[m][s * 32 + 16 + i] = d * mxfp4_kvalues[bl[1 + i] >> 4];
                    }
                }
            }
            for (int i = 0; i < inter; i++) { bg[e][i] = ((float)rand() / RAND_MAX - 0.5f); bu[e][i] = ((float)rand() / RAND_MAX - 0.5f); }
            for (int i = 0; i < embed; i++) bd[e][i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
        float *inputs = malloc(sizeof(float) * k * embed);
        for (int i = 0; i < k * embed; i++) inputs[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.f;
        float rw[2 * 3] = { 0.6f, 0.f, 0.3f,  0.4f, 1.0f, 0.7f };

        /* float reference */
        float *ref = calloc((size_t)k * embed, sizeof(float));
        float *h = malloc(sizeof(float) * inter);
        for (int e = 0; e < n_exp; e++) for (int r = 0; r < k; r++) {
            float w = rw[e * k + r];
            if (w == 0.f) continue;
            const float *x = inputs + (size_t)r * embed;
            for (int i = 0; i < inter; i++) {
                float g = bg[e][i], u = bu[e][i];
                for (int j = 0; j < embed; j++) { g += fg[e][(size_t)i * embed + j] * x[j]; u += fu[e][(size_t)i * embed + j] * x[j]; }
                if (g > 7.f) g = 7.f;
                if (u > 7.f) u = 7.f; else if (u < -7.f) u = -7.f;
                h[i] = g / (1.f + expf(-1.702f * g)) * (u + 1.f);
            }
            for (int i = 0; i < embed; i++) {
                float d = bd[e][i];
                for (int j = 0; j < inter; j++) d += fd[e][(size_t)i * inter + j] * h[j];
                ref[(size_t)r * embed + i] += w * d;
            }
        }
        float *got = calloc((size_t)k * embed, sizeof(float));
        tg_set_threads(6);
        tg_moe_forward_rows_v2((const uint8_t *const *)gates, (const uint8_t *const *)ups,
                               (const void *const *)downs, TG_FMT_MXFP4, fmts,
                               (const float *const *)bg, (const float *const *)bu, (const float *const *)bd,
                               TG_ACT_GPTOSS, n_exp, rw, inputs, got, k, embed, inter);
        tg_set_threads(1);
        float max_abs = 0, scale = 0;
        for (int i = 0; i < k * embed; i++) {
            if (fabsf(ref[i]) > scale) scale = fabsf(ref[i]);
            float e = fabsf(ref[i] - got[i]);
            if (e > max_abs) max_abs = e;
        }
        printf("  Max abs error %.4e vs output scale %.4e (ratio %.2e)\n", max_abs, scale, max_abs / scale);
        if (max_abs < 0.03f * scale) printf("  PASS\n\n"); else { printf("  FAIL\n\n"); pass = 0; }
        for (int e = 0; e < n_exp; e++) { free(gates[e]); free(ups[e]); free(downs[e]); free(fg[e]); free(fu[e]); free(fd[e]); free(bg[e]); free(bu[e]); free(bd[e]); }
        free(inputs); free(ref); free(h); free(got);
    }

    /* Test 11: tg_attention_v2 equals the Q4 kernel, and window/sink behave */
    printf("Test 11: tg_attention_v2 vs tg_attention_decode_q4; sliding window; sinks\n");
    {
        int embed = 2048, n_heads = 32, n_kv = 4, hd = 128, q_dim = n_heads * hd, kv_dim = n_kv * hd;
        int kv_max = 64, bpr = embed / QK_K;
        uint8_t *wq = malloc((size_t)q_dim * bpr * Q4K_BSIZE), *wk = malloc((size_t)kv_dim * bpr * Q4K_BSIZE);
        uint8_t *wo = malloc((size_t)embed * (q_dim / QK_K) * Q4K_BSIZE);
        uint16_t *wv = malloc(sizeof(uint16_t) * kv_dim * embed);
        for (int r = 0; r < q_dim; r++) for (int b = 0; b < bpr; b++) fill_q4k_block(wq + ((size_t)r * bpr + b) * Q4K_BSIZE, r + b);
        for (int r = 0; r < kv_dim; r++) for (int b = 0; b < bpr; b++) fill_q4k_block(wk + ((size_t)r * bpr + b) * Q4K_BSIZE, 5 * r + b);
        for (int r = 0; r < embed; r++) for (int b = 0; b < q_dim / QK_K; b++) fill_q4k_block(wo + ((size_t)r * (q_dim / QK_K) + b) * Q4K_BSIZE, 7 * r + b);
        srand(11);
        for (size_t i = 0; i < (size_t)kv_dim * embed; i++) wv[i] = (uint16_t)(0x2000 + (rand() % 0x800));
        float qn[128], kn[128];
        for (int i = 0; i < hd; i++) { qn[i] = 1.f + 0.01f * i; kn[i] = 1.f - 0.001f * i; }
        float *cosv = malloc(sizeof(float) * kv_max * hd / 2), *sinv = malloc(sizeof(float) * kv_max * hd / 2);
        for (int p = 0; p < kv_max; p++) for (int i = 0; i < hd / 2; i++) {
            float f = p / powf(10000.f, 2.f * i / hd); cosv[p * hd / 2 + i] = cosf(f); sinv[p * hd / 2 + i] = sinf(f);
        }
        float *kv1k = calloc((size_t)n_kv * kv_max * hd, sizeof(float)), *kv1v = calloc((size_t)n_kv * kv_max * hd, sizeof(float));
        float *kv2k = calloc((size_t)n_kv * kv_max * hd, sizeof(float)), *kv2v = calloc((size_t)n_kv * kv_max * hd, sizeof(float));
        float *kv3k = calloc((size_t)n_kv * kv_max * hd, sizeof(float)), *kv3v = calloc((size_t)n_kv * kv_max * hd, sizeof(float));
        float in[2048], o1[2048], o2[2048], o3[2048], o4[2048];
        float sinks_low[32];
        for (int h = 0; h < n_heads; h++) sinks_low[h] = -1e9f;
        float max_d = 0, max_win = 0, max_sink = 0;
        int n_steps = 12, window = 5;
        for (int p = 0; p < n_steps; p++) {
            srand(100 + p);
            for (int i = 0; i < embed; i++) in[i] = ((float)rand() / RAND_MAX - 0.5f);
            tg_attention_decode_q4(wq, wk, wv, wo, qn, kn, kv1k, kv1v, p, kv_max, cosv, sinv, in, o1,
                                   embed, n_heads, n_kv, hd, p);
            tg_attention_v2(TG_FMT_Q4K, TG_FMT_Q4K, TG_FMT_F16, TG_FMT_Q4K, wq, wk, wv, wo,
                            NULL, NULL, NULL, NULL, qn, kn, NULL, kv2k, kv2v, p, kv_max, 0,
                            cosv, sinv, in, o2, embed, n_heads, n_kv, hd, p, 1e-6f);
            tg_attention_v2(TG_FMT_Q4K, TG_FMT_Q4K, TG_FMT_F16, TG_FMT_Q4K, wq, wk, wv, wo,
                            NULL, NULL, NULL, NULL, qn, kn, sinks_low, kv3k, kv3v, p, kv_max, 0,
                            cosv, sinv, in, o3, embed, n_heads, n_kv, hd, p, 1e-6f);
            for (int i = 0; i < embed; i++) {
                float d = fabsf(o1[i] - o2[i]); if (d > max_d) max_d = d;
                float s = fabsf(o2[i] - o3[i]); if (s > max_sink) max_sink = s;
            }
            /* window: result must equal full attention over a KV cache holding only the last `window` positions */
            if (p >= window) {
                float *kvwk = calloc((size_t)n_kv * kv_max * hd, sizeof(float)), *kvwv = calloc((size_t)n_kv * kv_max * hd, sizeof(float));
                int start = p + 1 - window;
                for (int h = 0; h < n_kv; h++)
                    for (int t = start; t < p; t++) {
                        memcpy(kvwk + ((size_t)h * kv_max + t - start) * hd, kv2k + ((size_t)h * kv_max + t) * hd, hd * sizeof(float));
                        memcpy(kvwv + ((size_t)h * kv_max + t - start) * hd, kv2v + ((size_t)h * kv_max + t) * hd, hd * sizeof(float));
                    }
                float o_win[2048];
                tg_attention_v2(TG_FMT_Q4K, TG_FMT_Q4K, TG_FMT_F16, TG_FMT_Q4K, wq, wk, wv, wo,
                                NULL, NULL, NULL, NULL, qn, kn, NULL, kv2k, kv2v, p, kv_max, window,
                                cosv, sinv, in, o4, embed, n_heads, n_kv, hd, p, 1e-6f);
                tg_attention_v2(TG_FMT_Q4K, TG_FMT_Q4K, TG_FMT_F16, TG_FMT_Q4K, wq, wk, wv, wo,
                                NULL, NULL, NULL, NULL, qn, kn, NULL, kvwk, kvwv, p - start, kv_max, 0,
                                cosv, sinv, in, o_win, embed, n_heads, n_kv, hd, p, 1e-6f);
                for (int i = 0; i < embed; i++) { float d = fabsf(o4[i] - o_win[i]); if (d > max_win) max_win = d; }
                free(kvwk); free(kvwv);
            }
        }
        float scale = 0;
        for (int i = 0; i < embed; i++) if (fabsf(o1[i]) > scale) scale = fabsf(o1[i]);
        printf("  v2 vs q4 kernel max diff %.3e, window vs truncated cache %.3e, huge-negative sink vs none %.3e (scale %.3e)\n",
               max_d, max_win, max_sink, scale);
        if (max_d < 1e-4f * scale && max_win < 1e-4f * scale && max_sink < 1e-4f * scale) printf("  PASS\n\n");
        else { printf("  FAIL\n\n"); pass = 0; }
        free(wq); free(wk); free(wo); free(wv); free(cosv); free(sinv);
        free(kv1k); free(kv1v); free(kv2k); free(kv2v); free(kv3k); free(kv3v);
    }

    printf("==================================\n");
    printf("Result: %s\n", pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return pass ? 0 : 1;
}
