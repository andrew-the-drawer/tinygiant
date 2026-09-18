/*
 * libtinygiant.c — Fused Q4_K × Q8 inference library
 *
 * Correct Q4_K_M handling with sub-block scales (8 sub-blocks per 256-value block).
 * ARM NEON + vdotq_s32 for the hot path.
 *
 * Build (macOS):
 *   clang -shared -O3 -mcpu=apple-m1 -o libtinygiant.dylib tools/libtinygiant.c
 *
 * Build (Linux aarch64):
 *   gcc -shared -fPIC -O3 -march=armv8.2-a+dotprod -o libtinygiant.so tools/libtinygiant.c
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sys/mman.h>
#include <arm_neon.h>

#define QK_K      256
#define Q4K_BSIZE 144   /* 2(d) + 2(dmin) + 12(scales) + 128(qs) */
#define Q6K_BSIZE 210   /* 128(ql) + 64(qh) + 16(scales) + 2(d) */

/* ═══ Thread pool ═══ */

typedef void (*tg_task_fn)(void *ctx, size_t i);

/* Workers spin briefly before sleeping so that the ~200 dispatches per decoded
 * token don't each pay a condvar wake-up. */
#define POOL_SPIN_ITERS 200000

static struct {
    pthread_t *threads;
    int n_workers;
    pthread_mutex_t mu;
    pthread_cond_t cv_job;
    atomic_ulong generation;
    atomic_int stop;
    tg_task_fn fn;
    void *ctx;
    size_t n_items;
    atomic_size_t next;
    atomic_int remaining;
} pool = { .mu = PTHREAD_MUTEX_INITIALIZER,
           .cv_job = PTHREAD_COND_INITIALIZER };

static inline void cpu_relax(void) {
#if defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
}

static void pool_run_items(void) {
    for (;;) {
        size_t i = atomic_fetch_add(&pool.next, 1);
        if (i >= pool.n_items) break;
        pool.fn(pool.ctx, i);
    }
}

static void *pool_worker(void *arg) {
    (void)arg;
    unsigned long seen = 0;
    for (;;) {
        int spins = 0;
        while (atomic_load_explicit(&pool.generation, memory_order_acquire) == seen &&
               !atomic_load(&pool.stop)) {
            if (++spins < POOL_SPIN_ITERS) { cpu_relax(); continue; }
            pthread_mutex_lock(&pool.mu);
            while (atomic_load_explicit(&pool.generation, memory_order_acquire) == seen &&
                   !atomic_load(&pool.stop))
                pthread_cond_wait(&pool.cv_job, &pool.mu);
            pthread_mutex_unlock(&pool.mu);
        }
        if (atomic_load(&pool.stop)) return NULL;
        seen = atomic_load_explicit(&pool.generation, memory_order_acquire);

        pool_run_items();
        atomic_fetch_sub_explicit(&pool.remaining, 1, memory_order_release);
    }
}

void tg_set_threads(int n) {
    if (pool.n_workers > 0) {
        pthread_mutex_lock(&pool.mu);
        atomic_store(&pool.stop, 1);
        pthread_cond_broadcast(&pool.cv_job);
        pthread_mutex_unlock(&pool.mu);
        for (int i = 0; i < pool.n_workers; i++) pthread_join(pool.threads[i], NULL);
        free(pool.threads);
        pool.threads = NULL;
        pool.n_workers = 0;
        atomic_store(&pool.stop, 0);
    }
    if (n <= 1) return;
    pool.n_workers = n - 1;
    pool.threads = malloc(sizeof(pthread_t) * pool.n_workers);
    for (int i = 0; i < pool.n_workers; i++)
        pthread_create(&pool.threads[i], NULL, pool_worker, NULL);
}

int tg_get_threads(void) { return pool.n_workers + 1; }

static void tg_parallel_for(size_t n, tg_task_fn fn, void *ctx) {
    if (pool.n_workers == 0 || n <= 1) {
        for (size_t i = 0; i < n; i++) fn(ctx, i);
        return;
    }
    pool.fn = fn;
    pool.ctx = ctx;
    pool.n_items = n;
    atomic_store(&pool.next, 0);
    atomic_store(&pool.remaining, pool.n_workers);
    pthread_mutex_lock(&pool.mu);
    atomic_fetch_add_explicit(&pool.generation, 1, memory_order_release);
    pthread_cond_broadcast(&pool.cv_job);
    pthread_mutex_unlock(&pool.mu);

    pool_run_items();

    while (atomic_load_explicit(&pool.remaining, memory_order_acquire) > 0) cpu_relax();
}

/* ═══ f16 → f32 conversion ═══ */

static inline float f16_to_f32(uint16_t h) {
    uint32_t s = (h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    if (e == 0) {
        if (m == 0) { float r; uint32_t v = s; memcpy(&r, &v, 4); return r; }
        while (!(m & 0x400)) { m <<= 1; e--; }
        e++; m &= ~0x400;
    } else if (e == 31) { e = 255; }
    uint32_t f = s | ((e + 112) << 23) | (m << 13);
    float r; memcpy(&r, &f, 4); return r;
}

/* ═══ Q4_K scale extraction ═══
 *
 * 12-byte scales array encodes 8 (scale, min) pairs, each 6 bits.
 * Bytes 0-3: lower 6 bits = sc[0..3], upper 2 bits = extra for sc[4..7]
 * Bytes 4-7: lower 6 bits = m[0..3],  upper 2 bits = extra for m[4..7]
 * Bytes 8-11: lower 4 bits = sc[4..7] low4, upper 4 bits = m[4..7] low4
 */
static inline void get_scale_min_k4(int j, const uint8_t *scales,
                                     uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = scales[j] & 63;
        *m  = scales[j + 4] & 63;
    } else {
        *sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        *m  = (scales[j + 4] >>  4) | ((scales[j    ] >> 6) << 4);
    }
}

/* ═══ Q8 quantization ═══ */

typedef struct {
    float   scale;
    int32_t sum;
    int32_t bsums[8];   /* sum of each 32-value sub-block */
    int8_t  qs[QK_K];
} q8_block;

#define Q8_NBLOCKS(n) (((n) + QK_K - 1) / QK_K)

/* Quantizes ncols floats into 256-wide Q8 blocks; a partial last block is zero padded. */
static void quantize_q8(const float *x, q8_block *out, int ncols) {
    int n_blocks = Q8_NBLOCKS(ncols);
    for (int b = 0; b < n_blocks; b++) {
        const float *xb = x + b * QK_K;
        int n = ncols - b * QK_K < QK_K ? ncols - b * QK_K : QK_K;   /* multiple of 32 */

        float32x4_t vmax = vdupq_n_f32(0);
        for (int i = 0; i < n; i += 4) {
            float32x4_t v = vld1q_f32(xb + i);
            vmax = vmaxq_f32(vmax, vabsq_f32(v));
        }
        float amax = vmaxvq_f32(vmax);

        float sc = amax / 127.f;
        float inv = amax > 0 ? 127.f / amax : 0;
        out[b].scale = sc;

        float32x4_t vinv = vdupq_n_f32(inv);
        int32_t total = 0;
        for (int j = 0; j < 8; j++) {
            int32x4_t vsum = vdupq_n_s32(0);
            if (j * 32 < n) {
                for (int i = j * 32; i < (j + 1) * 32; i += 4) {
                    float32x4_t v = vld1q_f32(xb + i);
                    float32x4_t scaled = vmulq_f32(v, vinv);
                    int32x4_t rounded = vcvtnq_s32_f32(scaled);
                    int16x4_t n16 = vmovn_s32(rounded);
                    int8x8_t n8 = vmovn_s16(vcombine_s16(n16, n16));
                    vst1_lane_s32((int32_t *)(out[b].qs + i), vreinterpret_s32_s8(n8), 0);
                    vsum = vaddq_s32(vsum, rounded);
                }
            } else {
                memset(out[b].qs + j * 32, 0, 32);
            }
            out[b].bsums[j] = vaddvq_s32(vsum);
            total += out[b].bsums[j];
        }
        out[b].sum = total;
    }
}

/* ═══ Q4_K × Q8 dot product (correct sub-block scales) ═══
 *
 * Each Q4_K block: 256 values in 8 sub-blocks of 32.
 * Layout of qs[128]:
 *   bytes 0-31:   lo nibbles → values   0-31  (sub-block 0)
 *                 hi nibbles → values  32-63  (sub-block 1)
 *   bytes 32-63:  lo nibbles → values  64-95  (sub-block 2)
 *                 hi nibbles → values  96-127 (sub-block 3)
 *   bytes 64-95:  lo nibbles → values 128-159 (sub-block 4)
 *                 hi nibbles → values 160-191 (sub-block 5)
 *   bytes 96-127: lo nibbles → values 192-223 (sub-block 6)
 *                 hi nibbles → values 224-255 (sub-block 7)
 */
static float dot_q4k_q8(const uint8_t *q4, const q8_block *xq, int bpr) {
    float total = 0;
    const uint8x16_t m0f = vdupq_n_u8(0x0F);

    for (int b = 0; b < bpr; b++) {
        const uint8_t *bl = q4 + b * Q4K_BSIZE;
        uint16_t dr, dmr;
        memcpy(&dr, bl, 2);
        memcpy(&dmr, bl + 2, 2);
        float d    = f16_to_f32(dr);
        float dmin = f16_to_f32(dmr);
        const uint8_t *scales = bl + 4;
        const uint8_t *qs = bl + 16;
        const int8_t *xqs = xq[b].qs;
        const int32_t *bsums = xq[b].bsums;
        float sx = xq[b].scale;

        float block_sum = 0;

        /* 4 groups of 32 bytes, each producing 2 sub-blocks of 32 values */
        for (int g = 0; g < 4; g++) {
            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            get_scale_min_k4(2 * g,     scales, &sc_lo, &m_lo);
            get_scale_min_k4(2 * g + 1, scales, &sc_hi, &m_hi);

            const uint8_t *qs_g = qs + g * 32;
            const int8_t *xq_lo = xqs + g * 64;
            const int8_t *xq_hi = xqs + g * 64 + 32;

            int32x4_t idot_lo = vdupq_n_s32(0);
            int32x4_t idot_hi = vdupq_n_s32(0);

            for (int j = 0; j < 32; j += 16) {
                uint8x16_t raw = vld1q_u8(qs_g + j);
                int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(raw, m0f));
                int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(raw, 4));
                idot_lo = vdotq_s32(idot_lo, lo, vld1q_s8(xq_lo + j));
                idot_hi = vdotq_s32(idot_hi, hi, vld1q_s8(xq_hi + j));
            }

            block_sum += d * (float)sc_lo * (float)vaddvq_s32(idot_lo)
                       - dmin * (float)m_lo * (float)bsums[2 * g];
            block_sum += d * (float)sc_hi * (float)vaddvq_s32(idot_hi)
                       - dmin * (float)m_hi * (float)bsums[2 * g + 1];
        }

        total += sx * block_sum;
    }
    return total;
}

/* Same as dot_q4k_q8 but one weight row against 4 input rows: the weight
 * nibbles are decoded once and reused for all 4 rows. */
static void dot_q4k_q8_x4(const uint8_t *q4, const q8_block *const xq[4],
                          int bpr, float out[4]) {
    float total0 = 0, total1 = 0, total2 = 0, total3 = 0;
    const uint8x16_t m0f = vdupq_n_u8(0x0F);

    for (int b = 0; b < bpr; b++) {
        const uint8_t *bl = q4 + b * Q4K_BSIZE;
        uint16_t dr, dmr;
        memcpy(&dr, bl, 2);
        memcpy(&dmr, bl + 2, 2);
        float d    = f16_to_f32(dr);
        float dmin = f16_to_f32(dmr);
        const uint8_t *scales = bl + 4;
        const uint8_t *qs = bl + 16;
        const q8_block *x0 = &xq[0][b], *x1 = &xq[1][b], *x2 = &xq[2][b], *x3 = &xq[3][b];

        float bs0 = 0, bs1 = 0, bs2 = 0, bs3 = 0;

        for (int g = 0; g < 4; g++) {
            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            get_scale_min_k4(2 * g,     scales, &sc_lo, &m_lo);
            get_scale_min_k4(2 * g + 1, scales, &sc_hi, &m_hi);

            const uint8_t *qs_g = qs + g * 32;
            int off_lo = g * 64, off_hi = g * 64 + 32;

            int32x4_t l0 = vdupq_n_s32(0), l1 = vdupq_n_s32(0), l2 = vdupq_n_s32(0), l3 = vdupq_n_s32(0);
            int32x4_t h0 = vdupq_n_s32(0), h1 = vdupq_n_s32(0), h2 = vdupq_n_s32(0), h3 = vdupq_n_s32(0);

            for (int j = 0; j < 32; j += 16) {
                uint8x16_t raw = vld1q_u8(qs_g + j);
                int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(raw, m0f));
                int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(raw, 4));
                l0 = vdotq_s32(l0, lo, vld1q_s8(x0->qs + off_lo + j));
                l1 = vdotq_s32(l1, lo, vld1q_s8(x1->qs + off_lo + j));
                l2 = vdotq_s32(l2, lo, vld1q_s8(x2->qs + off_lo + j));
                l3 = vdotq_s32(l3, lo, vld1q_s8(x3->qs + off_lo + j));
                h0 = vdotq_s32(h0, hi, vld1q_s8(x0->qs + off_hi + j));
                h1 = vdotq_s32(h1, hi, vld1q_s8(x1->qs + off_hi + j));
                h2 = vdotq_s32(h2, hi, vld1q_s8(x2->qs + off_hi + j));
                h3 = vdotq_s32(h3, hi, vld1q_s8(x3->qs + off_hi + j));
            }

            float dsl = d * (float)sc_lo, dml = dmin * (float)m_lo;
            float dsh = d * (float)sc_hi, dmh = dmin * (float)m_hi;
            bs0 += dsl * (float)vaddvq_s32(l0) - dml * (float)x0->bsums[2 * g]
                 + dsh * (float)vaddvq_s32(h0) - dmh * (float)x0->bsums[2 * g + 1];
            bs1 += dsl * (float)vaddvq_s32(l1) - dml * (float)x1->bsums[2 * g]
                 + dsh * (float)vaddvq_s32(h1) - dmh * (float)x1->bsums[2 * g + 1];
            bs2 += dsl * (float)vaddvq_s32(l2) - dml * (float)x2->bsums[2 * g]
                 + dsh * (float)vaddvq_s32(h2) - dmh * (float)x2->bsums[2 * g + 1];
            bs3 += dsl * (float)vaddvq_s32(l3) - dml * (float)x3->bsums[2 * g]
                 + dsh * (float)vaddvq_s32(h3) - dmh * (float)x3->bsums[2 * g + 1];
        }

        total0 += x0->scale * bs0;
        total1 += x1->scale * bs1;
        total2 += x2->scale * bs2;
        total3 += x3->scale * bs3;
    }
    out[0] = total0; out[1] = total1; out[2] = total2; out[3] = total3;
}

/* ═══ Matrix-vector multiply: Q4_K matrix × float vector ═══ */

#define MM_CHUNK 64

static float dot_q6k_q8(const uint8_t *q6, const q8_block *xq, int bpr);

typedef struct {
    const uint8_t *mat;
    const q8_block *xq;
    float *out;
    int nrows, bpr;
    int q6;
} mm_ctx;

static void mm_task(void *p, size_t i) {
    mm_ctx *c = p;
    int r0 = (int)(i * MM_CHUNK);
    int r1 = r0 + MM_CHUNK < c->nrows ? r0 + MM_CHUNK : c->nrows;
    if (c->q6) {
        for (int r = r0; r < r1; r++)
            c->out[r] = dot_q6k_q8(c->mat + (size_t)r * c->bpr * Q6K_BSIZE, c->xq, c->bpr);
    } else {
        for (int r = r0; r < r1; r++)
            c->out[r] = dot_q4k_q8(c->mat + (size_t)r * c->bpr * Q4K_BSIZE, c->xq, c->bpr);
    }
}

static void matmul_q4k(const uint8_t *q4_matrix, const float *input,
                        float *output, int nrows, int ncols) {
    int bpr = ncols / QK_K;
    q8_block xq[bpr];
    quantize_q8(input, xq, ncols);
    mm_ctx c = { q4_matrix, xq, output, nrows, bpr, 0 };
    tg_parallel_for((nrows + MM_CHUNK - 1) / MM_CHUNK, mm_task, &c);
}

void tg_matmul_q4k(const uint8_t *q4_matrix, const float *input,
                    float *output, int nrows, int ncols) {
    matmul_q4k(q4_matrix, input, output, nrows, ncols);
}

/* ═══ Q6_K × Q8 fused kernel ═══ */

/* Decode 16 bytes of a Q6_K half-block at offset l into the four 16-lane
 * signed vectors that pair with input offsets l, l+32, l+64, l+96. */
static inline void unpack_q6k_16(const uint8_t *ql_h, const uint8_t *qh_h, int l,
                                 int8x16_t *v1, int8x16_t *v2,
                                 int8x16_t *v3, int8x16_t *v4) {
    const uint8x16_t m0f = vdupq_n_u8(0x0F);
    const uint8x16_t m03 = vdupq_n_u8(0x03);
    const int8x16_t off = vdupq_n_s8(32);
    uint8x16_t q0 = vld1q_u8(ql_h + l);
    uint8x16_t q1 = vld1q_u8(ql_h + l + 32);
    uint8x16_t h  = vld1q_u8(qh_h + l);
    uint8x16_t b1 = vorrq_u8(vandq_u8(q0, m0f), vshlq_n_u8(vandq_u8(h, m03), 4));
    uint8x16_t b2 = vorrq_u8(vandq_u8(q1, m0f), vshlq_n_u8(vandq_u8(vshrq_n_u8(h, 2), m03), 4));
    uint8x16_t b3 = vorrq_u8(vshrq_n_u8(q0, 4),  vshlq_n_u8(vandq_u8(vshrq_n_u8(h, 4), m03), 4));
    uint8x16_t b4 = vorrq_u8(vshrq_n_u8(q1, 4),  vshlq_n_u8(vshrq_n_u8(h, 6), 4));
    *v1 = vsubq_s8(vreinterpretq_s8_u8(b1), off);
    *v2 = vsubq_s8(vreinterpretq_s8_u8(b2), off);
    *v3 = vsubq_s8(vreinterpretq_s8_u8(b3), off);
    *v4 = vsubq_s8(vreinterpretq_s8_u8(b4), off);
}

static float dot_q6k_q8(const uint8_t *q6, const q8_block *xq, int bpr) {
    float total = 0;
    for (int b = 0; b < bpr; b++) {
        const uint8_t *bl = q6 + b * Q6K_BSIZE;
        const uint8_t *ql = bl;
        const uint8_t *qh = bl + 128;
        const int8_t *sc = (const int8_t *)(bl + 192);
        uint16_t dr;
        memcpy(&dr, bl + 208, 2);
        float d = f16_to_f32(dr);
        float sx = xq[b].scale;
        const int8_t *xqs = xq[b].qs;

        float block_sum = 0;

        for (int half = 0; half < 2; half++) {
            const uint8_t *ql_h = ql + half * 64;
            const uint8_t *qh_h = qh + half * 32;
            const int8_t *sc_h = sc + half * 8;
            const int8_t *xq_h = xqs + half * 128;

            for (int l = 0; l < 32; l += 16) {
                int is_base = l / 16;
                int8x16_t vv1, vv2, vv3, vv4;
                unpack_q6k_16(ql_h, qh_h, l, &vv1, &vv2, &vv3, &vv4);

                int32x4_t d1 = vdotq_s32(vdupq_n_s32(0), vv1, vld1q_s8(xq_h + l));
                int32x4_t d2 = vdotq_s32(vdupq_n_s32(0), vv2, vld1q_s8(xq_h + l + 32));
                int32x4_t d3 = vdotq_s32(vdupq_n_s32(0), vv3, vld1q_s8(xq_h + l + 64));
                int32x4_t d4 = vdotq_s32(vdupq_n_s32(0), vv4, vld1q_s8(xq_h + l + 96));

                block_sum += (float)sc_h[is_base + 0] * (float)vaddvq_s32(d1);
                block_sum += (float)sc_h[is_base + 2] * (float)vaddvq_s32(d2);
                block_sum += (float)sc_h[is_base + 4] * (float)vaddvq_s32(d3);
                block_sum += (float)sc_h[is_base + 6] * (float)vaddvq_s32(d4);
            }
        }
        total += d * sx * block_sum;
    }
    return total;
}

static void dot_q6k_q8_x4(const uint8_t *q6, const q8_block *const xq[4],
                          int bpr, float out[4]) {
    float total[4] = {0, 0, 0, 0};
    for (int b = 0; b < bpr; b++) {
        const uint8_t *bl = q6 + b * Q6K_BSIZE;
        const uint8_t *ql = bl;
        const uint8_t *qh = bl + 128;
        const int8_t *sc = (const int8_t *)(bl + 192);
        uint16_t dr;
        memcpy(&dr, bl + 208, 2);
        float d = f16_to_f32(dr);

        float bs[4] = {0, 0, 0, 0};

        for (int half = 0; half < 2; half++) {
            const uint8_t *ql_h = ql + half * 64;
            const uint8_t *qh_h = qh + half * 32;
            const int8_t *sc_h = sc + half * 8;

            for (int l = 0; l < 32; l += 16) {
                int is_base = l / 16;
                int8x16_t vv1, vv2, vv3, vv4;
                unpack_q6k_16(ql_h, qh_h, l, &vv1, &vv2, &vv3, &vv4);
                float s1 = (float)sc_h[is_base + 0], s2 = (float)sc_h[is_base + 2];
                float s3 = (float)sc_h[is_base + 4], s4 = (float)sc_h[is_base + 6];

                for (int r = 0; r < 4; r++) {
                    const int8_t *xq_h = xq[r][b].qs + half * 128;
                    int32x4_t d1 = vdotq_s32(vdupq_n_s32(0), vv1, vld1q_s8(xq_h + l));
                    int32x4_t d2 = vdotq_s32(vdupq_n_s32(0), vv2, vld1q_s8(xq_h + l + 32));
                    int32x4_t d3 = vdotq_s32(vdupq_n_s32(0), vv3, vld1q_s8(xq_h + l + 64));
                    int32x4_t d4 = vdotq_s32(vdupq_n_s32(0), vv4, vld1q_s8(xq_h + l + 96));
                    bs[r] += s1 * (float)vaddvq_s32(d1);
                    bs[r] += s2 * (float)vaddvq_s32(d2);
                    bs[r] += s3 * (float)vaddvq_s32(d3);
                    bs[r] += s4 * (float)vaddvq_s32(d4);
                }
            }
        }
        for (int r = 0; r < 4; r++) total[r] += d * xq[r][b].scale * bs[r];
    }
    for (int r = 0; r < 4; r++) out[r] = total[r];
}

static void matmul_q6k(const uint8_t *q6_matrix, const float *input,
                        float *output, int nrows, int ncols) {
    int bpr = ncols / QK_K;
    q8_block xq[bpr];
    quantize_q8(input, xq, ncols);
    mm_ctx c = { q6_matrix, xq, output, nrows, bpr, 1 };
    tg_parallel_for((nrows + MM_CHUNK - 1) / MM_CHUNK, mm_task, &c);
}

void tg_matmul_q6k(const uint8_t *q6_matrix, const float *input,
                    float *output, int nrows, int ncols) {
    matmul_q6k(q6_matrix, input, output, nrows, ncols);
}

/* ═══ MXFP4 × Q8 fused kernel ═══
 *
 * Block of 32 values: 1 byte e8m0 shared exponent, then 16 bytes of e2m1
 * nibbles (low nibbles = values 0-15, high nibbles = values 16-31).
 * value = 2^(e-127) * kvalue[nibble] / 2, kvalue = {0,1,2,3,4,6,8,12,0,-1,...,-12}.
 * Our q8 blocks cover 256 values, i.e. 8 MXFP4 blocks each.
 */

#define MXFP4_BSIZE 17
#define MXFP4_PER_Q8 8

static const int8_t mxfp4_kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

static inline float e8m0_to_fp32_half(uint8_t e) {
    uint32_t bits = e < 2 ? (0x00200000u << e) : ((uint32_t)(e - 1) << 23);
    float f; memcpy(&f, &bits, 4); return f;
}

/* Decodes one MXFP4 block into two 16-lane int8 vectors (doubled e2m1 values). */
static inline void unpack_mxfp4(const uint8_t *qs, int8x16_t *lo, int8x16_t *hi) {
    const uint8x16_t m0f = vdupq_n_u8(0x0F);
    const int8x16_t table = vld1q_s8(mxfp4_kvalues);
    uint8x16_t raw = vld1q_u8(qs);
    *lo = vqtbl1q_s8(table, vandq_u8(raw, m0f));
    *hi = vqtbl1q_s8(table, vshrq_n_u8(raw, 4));
}

/* 32-wide-block kernels index sub-blocks directly so ncols only needs to be a
 * multiple of 32 (gpt-oss uses 2880). Sub-block s lives in q8 block s/8. */

static float dot_mxfp4_q8(const uint8_t *w, const q8_block *xq, int ncols) {
    int n_sub = ncols / 32;
    float total = 0;
    for (int s = 0; s < n_sub; s++) {
        const q8_block *xb = &xq[s >> 3];
        const int8_t *xqs = xb->qs + (s & 7) * 32;
        const uint8_t *bl = w + (size_t)s * MXFP4_BSIZE;
        int8x16_t lo, hi;
        unpack_mxfp4(bl + 1, &lo, &hi);
        int32x4_t acc = vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(xqs));
        acc = vdotq_s32(acc, hi, vld1q_s8(xqs + 16));
        total += xb->scale * e8m0_to_fp32_half(bl[0]) * (float)vaddvq_s32(acc);
    }
    return total;
}

static void dot_mxfp4_q8_x4(const uint8_t *w, const q8_block *const xq[4], int ncols, float out[4]) {
    int n_sub = ncols / 32;
    float t0 = 0, t1 = 0, t2 = 0, t3 = 0;
    for (int s = 0; s < n_sub; s++) {
        int b = s >> 3, o = (s & 7) * 32;
        const int8_t *x0 = xq[0][b].qs + o, *x1 = xq[1][b].qs + o;
        const int8_t *x2 = xq[2][b].qs + o, *x3 = xq[3][b].qs + o;
        const uint8_t *bl = w + (size_t)s * MXFP4_BSIZE;
        int8x16_t lo, hi;
        unpack_mxfp4(bl + 1, &lo, &hi);
        float d = e8m0_to_fp32_half(bl[0]);
        int32x4_t a0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(x0)), hi, vld1q_s8(x0 + 16));
        int32x4_t a1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(x1)), hi, vld1q_s8(x1 + 16));
        int32x4_t a2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(x2)), hi, vld1q_s8(x2 + 16));
        int32x4_t a3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(x3)), hi, vld1q_s8(x3 + 16));
        t0 += xq[0][b].scale * d * (float)vaddvq_s32(a0);
        t1 += xq[1][b].scale * d * (float)vaddvq_s32(a1);
        t2 += xq[2][b].scale * d * (float)vaddvq_s32(a2);
        t3 += xq[3][b].scale * d * (float)vaddvq_s32(a3);
    }
    out[0] = t0; out[1] = t1; out[2] = t2; out[3] = t3;
}

/* ═══ Q8_0 × Q8 fused kernel (34-byte blocks: f16 scale + 32 int8) ═══ */

#define Q80_BSIZE 34

static float dot_q80_q8(const uint8_t *w, const q8_block *xq, int ncols) {
    int n_sub = ncols / 32;
    float total = 0;
    for (int s = 0; s < n_sub; s++) {
        const q8_block *xb = &xq[s >> 3];
        const int8_t *xqs = xb->qs + (s & 7) * 32;
        const uint8_t *bl = w + (size_t)s * Q80_BSIZE;
        uint16_t dr; memcpy(&dr, bl, 2);
        const int8_t *q = (const int8_t *)(bl + 2);
        int32x4_t acc = vdotq_s32(vdupq_n_s32(0), vld1q_s8(q), vld1q_s8(xqs));
        acc = vdotq_s32(acc, vld1q_s8(q + 16), vld1q_s8(xqs + 16));
        total += xb->scale * f16_to_f32(dr) * (float)vaddvq_s32(acc);
    }
    return total;
}

/* ═══ Format dispatch ═══ */

enum { TG_FMT_Q4K = 0, TG_FMT_F16 = 1, TG_FMT_Q6K = 2, TG_FMT_MXFP4 = 3, TG_FMT_Q80 = 4 };

static inline size_t fmt_row_bytes(int fmt, int ncols) {
    switch (fmt) {
        case TG_FMT_Q4K:   return (size_t)(ncols / QK_K) * Q4K_BSIZE;
        case TG_FMT_Q6K:   return (size_t)(ncols / QK_K) * Q6K_BSIZE;
        case TG_FMT_MXFP4: return (size_t)(ncols / 32) * MXFP4_BSIZE;
        case TG_FMT_Q80:   return (size_t)(ncols / 32) * Q80_BSIZE;
        default:           return (size_t)ncols * 2;
    }
}

size_t tg_row_bytes(int fmt, int ncols) { return fmt_row_bytes(fmt, ncols); }

static void matvec_f16(const uint16_t *A, const float *x, float *out, int nrows, int ncols);

/* One weight row against one quantized input (x_f32 only used by the f16 format). */
static inline float dot_fmt(int fmt, const uint8_t *row, const q8_block *xq,
                            const float *x_f32, int ncols) {
    switch (fmt) {
        case TG_FMT_Q4K:   return dot_q4k_q8(row, xq, ncols / QK_K);
        case TG_FMT_Q6K:   return dot_q6k_q8(row, xq, ncols / QK_K);
        case TG_FMT_MXFP4: return dot_mxfp4_q8(row, xq, ncols);
        case TG_FMT_Q80:   return dot_q80_q8(row, xq, ncols);
        default: { float o; matvec_f16((const uint16_t *)row, x_f32, &o, 1, ncols); return o; }
    }
}

static inline void dot_fmt_x4(int fmt, const uint8_t *row, const q8_block *const xq[4],
                              const float *const x_f32[4], int ncols, float out[4]) {
    switch (fmt) {
        case TG_FMT_Q4K:   dot_q4k_q8_x4(row, xq, ncols / QK_K, out); return;
        case TG_FMT_Q6K:   dot_q6k_q8_x4(row, xq, ncols / QK_K, out); return;
        case TG_FMT_MXFP4: dot_mxfp4_q8_x4(row, xq, ncols, out); return;
        default:
            for (int r = 0; r < 4; r++) out[r] = dot_fmt(fmt, row, xq[r], x_f32 ? x_f32[r] : NULL, ncols);
    }
}

/* ═══ f16 matrix-vector multiply ═══ */

static void matvec_f16(const uint16_t *A, const float *x,
                        float *out, int nrows, int ncols) {
    for (int r = 0; r < nrows; r++) {
        const uint16_t *row = A + (size_t)r * ncols;
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        int i = 0;
        for (; i + 7 < ncols; i += 8) {
            uint16x4_t h0 = vld1_u16(row + i);
            uint16x4_t h1 = vld1_u16(row + i + 4);
            float32x4_t a0 = vcvt_f32_f16(vreinterpret_f16_u16(h0));
            float32x4_t a1 = vcvt_f32_f16(vreinterpret_f16_u16(h1));
            acc0 = vfmaq_f32(acc0, a0, vld1q_f32(x + i));
            acc1 = vfmaq_f32(acc1, a1, vld1q_f32(x + i + 4));
        }
        float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
        for (; i < ncols; i++) sum += f16_to_f32(row[i]) * x[i];
        out[r] = sum;
    }
}

void tg_f16_matvec(const uint16_t *A, const float *x,
                    float *out, int nrows, int ncols) {
    matvec_f16(A, x, out, nrows, ncols);
}

/* ═══ f32 matrix-vector multiply ═══ */

void tg_f32_matvec(const float *A, const float *x,
                    float *out, int nrows, int ncols) {
    for (int r = 0; r < nrows; r++) {
        const float *row = A + (size_t)r * ncols;
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        int i = 0;
        for (; i + 7 < ncols; i += 8) {
            acc0 = vfmaq_f32(acc0, vld1q_f32(row + i),     vld1q_f32(x + i));
            acc1 = vfmaq_f32(acc1, vld1q_f32(row + i + 4), vld1q_f32(x + i + 4));
        }
        float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
        for (; i < ncols; i++) sum += row[i] * x[i];
        out[r] = sum;
    }
}

/* ═══ RMS Norm ═══ */

void tg_rms_norm(const float *x, const float *weight, float *out,
                  int dim, float eps) {
    float32x4_t vsum = vdupq_n_f32(0);
    int i = 0;
    for (; i + 3 < dim; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        vsum = vfmaq_f32(vsum, v, v);
    }
    float ss = vaddvq_f32(vsum);
    for (; i < dim; i++) ss += x[i] * x[i];
    float scale = 1.f / sqrtf(ss / dim + eps);
    float32x4_t vs = vdupq_n_f32(scale);
    for (i = 0; i + 3 < dim; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t w = vld1q_f32(weight + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(v, vs), w));
    }
    for (; i < dim; i++) out[i] = x[i] * scale * weight[i];
}

/* ═══ RoPE ═══ */

static void apply_rope_inplace(float *x, int n_heads, int head_dim,
                                const float *cos_table, const float *sin_table,
                                int pos) {
    int half = head_dim / 2;
    const float *c = cos_table + pos * half;
    const float *s = sin_table + pos * half;
    for (int h = 0; h < n_heads; h++) {
        float *xh = x + h * head_dim;
        for (int i = 0; i < half; i += 4) {
            float32x4_t x1 = vld1q_f32(xh + i);
            float32x4_t x2 = vld1q_f32(xh + half + i);
            float32x4_t cv = vld1q_f32(c + i);
            float32x4_t sv = vld1q_f32(s + i);
            vst1q_f32(xh + i,        vmlsq_f32(vmulq_f32(x1, cv), x2, sv));
            vst1q_f32(xh + half + i, vfmaq_f32(vmulq_f32(x2, cv), x1, sv));
        }
    }
}

/* ═══ GQA Attention Decode Step ═══ */

void tg_attention_decode(
    const float *wq, const float *wk, const float *wv, const float *wo,
    const float *q_norm_w, const float *k_norm_w,
    float *kv_k, float *kv_v, int kv_len, int kv_max,
    const float *rope_cos, const float *rope_sin,
    const float *input, float *output,
    int embed_dim, int n_heads, int n_kv_heads, int head_dim, int pos)
{
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int gqa_ratio = n_heads / n_kv_heads;

    float q[q_dim], k[kv_dim], v[kv_dim];

    /* Q/K/V projections */
    tg_f32_matvec(wq, input, q, q_dim, embed_dim);
    tg_f32_matvec(wk, input, k, kv_dim, embed_dim);
    tg_f32_matvec(wv, input, v, kv_dim, embed_dim);

    /* QK norms */
    float q_normed[q_dim], k_normed[kv_dim];
    for (int h = 0; h < n_heads; h++)
        tg_rms_norm(q + h * head_dim, q_norm_w, q_normed + h * head_dim, head_dim, 1e-6f);
    for (int h = 0; h < n_kv_heads; h++)
        tg_rms_norm(k + h * head_dim, k_norm_w, k_normed + h * head_dim, head_dim, 1e-6f);

    /* RoPE */
    apply_rope_inplace(q_normed, n_heads, head_dim, rope_cos, rope_sin, pos);
    apply_rope_inplace(k_normed, n_kv_heads, head_dim, rope_cos, rope_sin, pos);

    /* Append to KV cache: kv_k/kv_v are [n_kv_heads, kv_max, head_dim] */
    for (int h = 0; h < n_kv_heads; h++) {
        memcpy(kv_k + ((size_t)h * kv_max + kv_len) * head_dim,
               k_normed + h * head_dim, head_dim * sizeof(float));
        memcpy(kv_v + ((size_t)h * kv_max + kv_len) * head_dim,
               v + h * head_dim, head_dim * sizeof(float));
    }

    /* GQA attention */
    float attn_out[q_dim];
    int seq_len = kv_len + 1;
    float inv_sqrt = 1.f / sqrtf((float)head_dim);

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / gqa_ratio;
        const float *q_h = q_normed + h * head_dim;
        const float *k_cache = kv_k + (size_t)kv_h * kv_max * head_dim;
        const float *v_cache = kv_v + (size_t)kv_h * kv_max * head_dim;

        /* Compute attention scores */
        float scores[seq_len];
        float max_score = -1e30f;
        for (int t = 0; t < seq_len; t++) {
            float32x4_t acc = vdupq_n_f32(0);
            const float *kt = k_cache + t * head_dim;
            int d = 0;
            for (; d + 3 < head_dim; d += 4)
                acc = vfmaq_f32(acc, vld1q_f32(q_h + d), vld1q_f32(kt + d));
            float dot = vaddvq_f32(acc);
            for (; d < head_dim; d++) dot += q_h[d] * kt[d];
            scores[t] = dot * inv_sqrt;
            if (scores[t] > max_score) max_score = scores[t];
        }

        /* Softmax */
        float sum_exp = 0;
        for (int t = 0; t < seq_len; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp += scores[t];
        }
        float inv_sum = 1.f / sum_exp;

        /* Weighted value sum */
        float *out_h = attn_out + h * head_dim;
        memset(out_h, 0, head_dim * sizeof(float));
        for (int t = 0; t < seq_len; t++) {
            float w = scores[t] * inv_sum;
            const float *vt = v_cache + t * head_dim;
            float32x4_t vw = vdupq_n_f32(w);
            int d = 0;
            for (; d + 3 < head_dim; d += 4) {
                float32x4_t cur = vld1q_f32(out_h + d);
                cur = vfmaq_f32(cur, vw, vld1q_f32(vt + d));
                vst1q_f32(out_h + d, cur);
            }
            for (; d < head_dim; d++) out_h[d] += w * vt[d];
        }
    }

    /* Output projection */
    tg_f32_matvec(wo, attn_out, output, embed_dim, q_dim);
}

/* ═══ GQA Attention Decode Step (Q4_K weights) ═══ */

typedef struct {
    const uint8_t *wq, *wk, *wv;
    int fmt_q, fmt_k, fmt_v;
    const float *bq, *bk, *bv;     /* nullable */
    const q8_block *xq;
    const float *input;
    float *q, *k, *v;
    int q_dim, kv_dim, embed_dim;
    int n_q_chunks, n_k_chunks;
} qkv_ctx;

static void qkv_rows(int fmt, const uint8_t *w, const float *bias, const q8_block *xq,
                     const float *input, float *out, int r0, int r1, int ncols) {
    size_t rb = fmt_row_bytes(fmt, ncols);
    for (int r = r0; r < r1; r++)
        out[r] = dot_fmt(fmt, w + (size_t)r * rb, xq, input, ncols) + (bias ? bias[r] : 0.f);
}

static void qkv_task(void *p, size_t i) {
    qkv_ctx *c = p;
    if ((int)i < c->n_q_chunks) {
        int r0 = (int)i * MM_CHUNK, r1 = r0 + MM_CHUNK < c->q_dim ? r0 + MM_CHUNK : c->q_dim;
        qkv_rows(c->fmt_q, c->wq, c->bq, c->xq, c->input, c->q, r0, r1, c->embed_dim);
    } else if ((int)i < c->n_q_chunks + c->n_k_chunks) {
        int r0 = ((int)i - c->n_q_chunks) * MM_CHUNK;
        int r1 = r0 + MM_CHUNK < c->kv_dim ? r0 + MM_CHUNK : c->kv_dim;
        qkv_rows(c->fmt_k, c->wk, c->bk, c->xq, c->input, c->k, r0, r1, c->embed_dim);
    } else {
        int r0 = ((int)i - c->n_q_chunks - c->n_k_chunks) * MM_CHUNK;
        int r1 = r0 + MM_CHUNK < c->kv_dim ? r0 + MM_CHUNK : c->kv_dim;
        qkv_rows(c->fmt_v, c->wv, c->bv, c->xq, c->input, c->v, r0, r1, c->embed_dim);
    }
}

typedef struct {
    const float *q;
    const float *kv_k, *kv_v;
    float *out;
    int seq_len, kv_max, head_dim, gqa_ratio;
    float inv_sqrt;
    int start;             /* first attended position (sliding window) */
    const float *sinks;    /* per-head attention sink logits, or NULL */
} head_ctx;

static void head_task(void *p, size_t h) {
    head_ctx *c = p;
    int head_dim = c->head_dim, seq_len = c->seq_len, start = c->start;
    int kv_h = (int)h / c->gqa_ratio;
    const float *q_h = c->q + h * head_dim;
    const float *k_cache = c->kv_k + (size_t)kv_h * c->kv_max * head_dim;
    const float *v_cache = c->kv_v + (size_t)kv_h * c->kv_max * head_dim;

    float scores[seq_len - start];
    float max_score = c->sinks ? c->sinks[h] : -1e30f;
    for (int t = start; t < seq_len; t++) {
        float32x4_t acc = vdupq_n_f32(0);
        const float *kt = k_cache + (size_t)t * head_dim;
        int d = 0;
        for (; d + 3 < head_dim; d += 4)
            acc = vfmaq_f32(acc, vld1q_f32(q_h + d), vld1q_f32(kt + d));
        float dot = vaddvq_f32(acc);
        for (; d < head_dim; d++) dot += q_h[d] * kt[d];
        scores[t - start] = dot * c->inv_sqrt;
        if (scores[t - start] > max_score) max_score = scores[t - start];
    }
    float sum_exp = c->sinks ? expf(c->sinks[h] - max_score) : 0.f;
    for (int t = start; t < seq_len; t++) {
        scores[t - start] = expf(scores[t - start] - max_score);
        sum_exp += scores[t - start];
    }
    float inv_sum = 1.f / sum_exp;

    float *out_h = c->out + h * head_dim;
    memset(out_h, 0, head_dim * sizeof(float));
    for (int t = start; t < seq_len; t++) {
        float w = scores[t - start] * inv_sum;
        const float *vt = v_cache + (size_t)t * head_dim;
        float32x4_t vw = vdupq_n_f32(w);
        int d = 0;
        for (; d + 3 < head_dim; d += 4) {
            float32x4_t cur = vld1q_f32(out_h + d);
            cur = vfmaq_f32(cur, vw, vld1q_f32(vt + d));
            vst1q_f32(out_h + d, cur);
        }
        for (; d < head_dim; d++) out_h[d] += w * vt[d];
    }
}

/* ═══ Generic threaded matvec (any weight format) ═══ */

typedef struct {
    int fmt;
    const uint8_t *mat;
    const q8_block *xq;
    const float *input;
    const float *bias;
    float *out;
    int nrows, ncols;
} gmm_ctx;

static void gmm_task(void *p, size_t i) {
    gmm_ctx *c = p;
    int r0 = (int)(i * MM_CHUNK);
    int r1 = r0 + MM_CHUNK < c->nrows ? r0 + MM_CHUNK : c->nrows;
    qkv_rows(c->fmt, c->mat, c->bias, c->xq, c->input, c->out, r0, r1, c->ncols);
}

void tg_matmul(int fmt, const void *mat, const float *bias, const float *input,
               float *output, int nrows, int ncols) {
    q8_block xq[Q8_NBLOCKS(ncols)];
    if (fmt != TG_FMT_F16) quantize_q8(input, xq, ncols);
    gmm_ctx c = { fmt, mat, xq, input, bias, output, nrows, ncols };
    tg_parallel_for((nrows + MM_CHUNK - 1) / MM_CHUNK, gmm_task, &c);
}

/* ═══ Generic attention decode step ═══
 *
 * Any weight format per projection, optional biases, optional QK RMS norm
 * (pass NULL norm weights to skip), optional attention sinks, and a sliding
 * window (0 = full attention).
 */
void tg_attention_v2(
    int fmt_q, int fmt_k, int fmt_v, int fmt_o,
    const void *wq, const void *wk, const void *wv, const void *wo,
    const float *bq, const float *bk, const float *bv, const float *bo,
    const float *q_norm_w, const float *k_norm_w, const float *sinks,
    float *kv_k, float *kv_v, int kv_len, int kv_max, int window,
    const float *rope_cos, const float *rope_sin,
    const float *input, float *output,
    int embed_dim, int n_heads, int n_kv_heads, int head_dim, int pos, float eps)
{
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int gqa_ratio = n_heads / n_kv_heads;

    float q[q_dim], k[kv_dim], v[kv_dim];
    q8_block xq[Q8_NBLOCKS(embed_dim)];
    quantize_q8(input, xq, embed_dim);
    qkv_ctx qc = { wq, wk, wv, fmt_q, fmt_k, fmt_v, bq, bk, bv, xq, input, q, k, v,
                   q_dim, kv_dim, embed_dim,
                   (q_dim + MM_CHUNK - 1) / MM_CHUNK, (kv_dim + MM_CHUNK - 1) / MM_CHUNK };
    tg_parallel_for((size_t)qc.n_q_chunks + 2 * qc.n_k_chunks, qkv_task, &qc);

    if (q_norm_w) {
        for (int h = 0; h < n_heads; h++)
            tg_rms_norm(q + h * head_dim, q_norm_w, q + h * head_dim, head_dim, eps);
        for (int h = 0; h < n_kv_heads; h++)
            tg_rms_norm(k + h * head_dim, k_norm_w, k + h * head_dim, head_dim, eps);
    }
    apply_rope_inplace(q, n_heads, head_dim, rope_cos, rope_sin, pos);
    apply_rope_inplace(k, n_kv_heads, head_dim, rope_cos, rope_sin, pos);

    for (int h = 0; h < n_kv_heads; h++) {
        memcpy(kv_k + ((size_t)h * kv_max + kv_len) * head_dim, k + h * head_dim, head_dim * sizeof(float));
        memcpy(kv_v + ((size_t)h * kv_max + kv_len) * head_dim, v + h * head_dim, head_dim * sizeof(float));
    }

    int seq_len = kv_len + 1;
    int start = (window > 0 && seq_len > window) ? seq_len - window : 0;
    float attn_out[q_dim];
    head_ctx hc = { q, kv_k, kv_v, attn_out, seq_len, kv_max, head_dim, gqa_ratio,
                    1.f / sqrtf((float)head_dim), start, sinks };
    tg_parallel_for(n_heads, head_task, &hc);

    tg_matmul(fmt_o, wo, bo, attn_out, output, embed_dim, q_dim);
}

void tg_attention_decode_q4(
    const uint8_t *wq_q4, const uint8_t *wk_q4,
    const uint16_t *wv_f16, const uint8_t *wo_q4,
    const float *q_norm_w, const float *k_norm_w,
    float *kv_k, float *kv_v, int kv_len, int kv_max,
    const float *rope_cos, const float *rope_sin,
    const float *input, float *output,
    int embed_dim, int n_heads, int n_kv_heads, int head_dim, int pos)
{
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int gqa_ratio = n_heads / n_kv_heads;
    int bpr = embed_dim / QK_K;

    float q[q_dim], k[kv_dim], v[kv_dim];

    /* Q/K/V projections in one dispatch (shared Q8 input) */
    q8_block xq[bpr];
    quantize_q8(input, xq, embed_dim);
    qkv_ctx qc = { wq_q4, wk_q4, (const uint8_t *)wv_f16, TG_FMT_Q4K, TG_FMT_Q4K, TG_FMT_F16,
                   NULL, NULL, NULL, xq, input, q, k, v, q_dim, kv_dim, embed_dim,
                   (q_dim + MM_CHUNK - 1) / MM_CHUNK, (kv_dim + MM_CHUNK - 1) / MM_CHUNK };
    tg_parallel_for((size_t)qc.n_q_chunks + 2 * qc.n_k_chunks, qkv_task, &qc);

    /* QK norms */
    float q_normed[q_dim], k_normed[kv_dim];
    for (int h = 0; h < n_heads; h++)
        tg_rms_norm(q + h * head_dim, q_norm_w, q_normed + h * head_dim, head_dim, 1e-6f);
    for (int h = 0; h < n_kv_heads; h++)
        tg_rms_norm(k + h * head_dim, k_norm_w, k_normed + h * head_dim, head_dim, 1e-6f);

    /* RoPE */
    apply_rope_inplace(q_normed, n_heads, head_dim, rope_cos, rope_sin, pos);
    apply_rope_inplace(k_normed, n_kv_heads, head_dim, rope_cos, rope_sin, pos);

    /* Append to KV cache */
    for (int h = 0; h < n_kv_heads; h++) {
        memcpy(kv_k + ((size_t)h * kv_max + kv_len) * head_dim,
               k_normed + h * head_dim, head_dim * sizeof(float));
        memcpy(kv_v + ((size_t)h * kv_max + kv_len) * head_dim,
               v + h * head_dim, head_dim * sizeof(float));
    }

    /* GQA attention, one task per head */
    float attn_out[q_dim];
    head_ctx hc = { q_normed, kv_k, kv_v, attn_out, kv_len + 1, kv_max, head_dim, gqa_ratio,
                    1.f / sqrtf((float)head_dim), 0, NULL };
    tg_parallel_for(n_heads, head_task, &hc);

    /* Output projection via fused Q4×Q8 */
    matmul_q4k(wo_q4, attn_out, output, embed_dim, q_dim);
}

/* ═══ Expert forward pass (original, all-Q4_K) ═══ */

void tg_expert_forward(const uint8_t *q4_data, const float *input,
                       float *output, float weight,
                       int embed_dim, int inter_dim) {
    int gate_bpr = embed_dim / QK_K;
    size_t gate_bytes = (size_t)inter_dim * gate_bpr * Q4K_BSIZE;
    int down_bpr = inter_dim / QK_K;

    const uint8_t *gate = q4_data;
    const uint8_t *up   = q4_data + gate_bytes;
    const uint8_t *down = q4_data + 2 * gate_bytes;

    q8_block xq_embed[gate_bpr];
    quantize_q8(input, xq_embed, embed_dim);

    float gate_out[inter_dim];
    for (int r = 0; r < inter_dim; r++)
        gate_out[r] = dot_q4k_q8(gate + (size_t)r * gate_bpr * Q4K_BSIZE,
                                  xq_embed, gate_bpr);

    float up_out[inter_dim];
    for (int r = 0; r < inter_dim; r++)
        up_out[r] = dot_q4k_q8(up + (size_t)r * gate_bpr * Q4K_BSIZE,
                                xq_embed, gate_bpr);

    float hidden[inter_dim];
    for (int i = 0; i < inter_dim; i++) {
        float silu = gate_out[i] / (1.f + expf(-gate_out[i]));
        hidden[i] = silu * up_out[i];
    }

    q8_block xq_inter[down_bpr];
    quantize_q8(hidden, xq_inter, inter_dim);

    float down_out[embed_dim];
    for (int r = 0; r < embed_dim; r++)
        down_out[r] = dot_q4k_q8(down + (size_t)r * down_bpr * Q4K_BSIZE,
                                  xq_inter, down_bpr);

    for (int i = 0; i < embed_dim; i++)
        output[i] += weight * down_out[i];
}

/* ═══ Expert forward pass (mixed Q4_K gate/up + f16 or Q4_K down) ═══ */

void tg_expert_forward_mixed(
    const uint8_t *gate_q4, const uint8_t *up_q4,
    const void *down_data, int down_format,
    const float *input, float *output, float weight,
    int embed_dim, int inter_dim)
{
    int gate_bpr = embed_dim / QK_K;
    int down_bpr = inter_dim / QK_K;

    q8_block xq_embed[gate_bpr];
    quantize_q8(input, xq_embed, embed_dim);

    float gate_out[inter_dim];
    for (int r = 0; r < inter_dim; r++)
        gate_out[r] = dot_q4k_q8(gate_q4 + (size_t)r * gate_bpr * Q4K_BSIZE,
                                  xq_embed, gate_bpr);

    float up_out[inter_dim];
    for (int r = 0; r < inter_dim; r++)
        up_out[r] = dot_q4k_q8(up_q4 + (size_t)r * gate_bpr * Q4K_BSIZE,
                                xq_embed, gate_bpr);

    float hidden[inter_dim];
    for (int i = 0; i < inter_dim; i++) {
        float silu = gate_out[i] / (1.f + expf(-gate_out[i]));
        hidden[i] = silu * up_out[i];
    }

    float down_out[embed_dim];
    if (down_format == 1) {
        matvec_f16((const uint16_t *)down_data, hidden, down_out, embed_dim, inter_dim);
    } else if (down_format == 2) {
        matmul_q6k((const uint8_t *)down_data, hidden, down_out, embed_dim, inter_dim);
    } else {
        q8_block xq_inter[down_bpr];
        quantize_q8(hidden, xq_inter, inter_dim);
        for (int r = 0; r < embed_dim; r++)
            down_out[r] = dot_q4k_q8((const uint8_t *)down_data +
                                      (size_t)r * down_bpr * Q4K_BSIZE,
                                      xq_inter, down_bpr);
    }

    for (int i = 0; i < embed_dim; i++)
        output[i] += weight * down_out[i];
}

/* ═══ Multi-row expert forward ═══
 *
 * inputs: [k][embed_dim], output: [k][embed_dim] (accumulated, += weight[r] * expert(x_r)).
 * Rows are processed in groups of 4 so each weight block is decoded once per group.
 */

static void q4k_rows(const uint8_t *mat, int nrows, int bpr,
                     const q8_block *xq, int k, int xq_stride,
                     float *out, int out_stride) {
    if (k == 1) {
        for (int row = 0; row < nrows; row++)
            out[row] = dot_q4k_q8(mat + (size_t)row * bpr * Q4K_BSIZE, xq, bpr);
        return;
    }
    for (int r0 = 0; r0 < k; r0 += 4) {
        const q8_block *p[4];
        for (int r = 0; r < 4; r++) {
            int rr = r0 + r < k ? r0 + r : k - 1;
            p[r] = xq + (size_t)rr * xq_stride;
        }
        int n = k - r0 < 4 ? k - r0 : 4;
        for (int row = 0; row < nrows; row++) {
            float o[4];
            dot_q4k_q8_x4(mat + (size_t)row * bpr * Q4K_BSIZE, p, bpr, o);
            for (int r = 0; r < n; r++) out[(size_t)(r0 + r) * out_stride + row] = o[r];
        }
    }
}

static void q6k_rows(const uint8_t *mat, int nrows, int bpr,
                     const q8_block *xq, int k, int xq_stride,
                     float *out, int out_stride) {
    if (k == 1) {
        for (int row = 0; row < nrows; row++)
            out[row] = dot_q6k_q8(mat + (size_t)row * bpr * Q6K_BSIZE, xq, bpr);
        return;
    }
    for (int r0 = 0; r0 < k; r0 += 4) {
        const q8_block *p[4];
        for (int r = 0; r < 4; r++) {
            int rr = r0 + r < k ? r0 + r : k - 1;
            p[r] = xq + (size_t)rr * xq_stride;
        }
        int n = k - r0 < 4 ? k - r0 : 4;
        for (int row = 0; row < nrows; row++) {
            float o[4];
            dot_q6k_q8_x4(mat + (size_t)row * bpr * Q6K_BSIZE, p, bpr, o);
            for (int r = 0; r < n; r++) out[(size_t)(r0 + r) * out_stride + row] = o[r];
        }
    }
}

void tg_expert_forward_rows(
    const uint8_t *gate_q4, const uint8_t *up_q4,
    const void *down_data, int down_format,
    const float *inputs, float *output, const float *weights, int k,
    int embed_dim, int inter_dim)
{
    int gate_bpr = embed_dim / QK_K;
    int down_bpr = inter_dim / QK_K;

    q8_block *xq_embed = malloc(sizeof(q8_block) * gate_bpr * k);
    float *gate_out = malloc(sizeof(float) * inter_dim * k);
    float *up_out   = malloc(sizeof(float) * inter_dim * k);
    float *hidden   = gate_out;
    float *down_out = malloc(sizeof(float) * embed_dim * k);

    for (int r = 0; r < k; r++)
        quantize_q8(inputs + (size_t)r * embed_dim, xq_embed + (size_t)r * gate_bpr, embed_dim);

    q4k_rows(gate_q4, inter_dim, gate_bpr, xq_embed, k, gate_bpr, gate_out, inter_dim);
    q4k_rows(up_q4,   inter_dim, gate_bpr, xq_embed, k, gate_bpr, up_out,   inter_dim);

    for (size_t i = 0; i < (size_t)inter_dim * k; i++) {
        float g = gate_out[i];
        hidden[i] = g / (1.f + expf(-g)) * up_out[i];
    }

    if (down_format == 1) {
        for (int r = 0; r < k; r++)
            matvec_f16((const uint16_t *)down_data, hidden + (size_t)r * inter_dim,
                       down_out + (size_t)r * embed_dim, embed_dim, inter_dim);
    } else {
        q8_block *xq_inter = malloc(sizeof(q8_block) * down_bpr * k);
        for (int r = 0; r < k; r++)
            quantize_q8(hidden + (size_t)r * inter_dim, xq_inter + (size_t)r * down_bpr, inter_dim);
        if (down_format == 2)
            q6k_rows(down_data, embed_dim, down_bpr, xq_inter, k, down_bpr, down_out, embed_dim);
        else
            q4k_rows(down_data, embed_dim, down_bpr, xq_inter, k, down_bpr, down_out, embed_dim);
        free(xq_inter);
    }

    for (int r = 0; r < k; r++) {
        float w = weights[r];
        float *o = output + (size_t)r * embed_dim;
        const float *d = down_out + (size_t)r * embed_dim;
        for (int i = 0; i < embed_dim; i++) o[i] += w * d[i];
    }

    free(xq_embed); free(gate_out); free(up_out); free(down_out);
}

/* ═══ Batched MoE layer: n_experts experts over k rows, threaded over experts ═══
 *
 * row_weights: [n_experts][k]; zero means the row does not use that expert.
 * Each expert gathers its active rows, runs the multi-row forward, and writes
 * a private partial; partials are reduced into output afterwards.
 */

/* Two phases so work is split finely across threads even for a single token:
 *   A: (expert, chunk of inter rows)  -> hidden = silu(gate x) * (up x)
 *   B: (expert, chunk of embed rows)  -> partial = w * down(hidden)
 * Within a task, the active rows of the batch are processed 4 at a time. */

#define MOE_INTER_CHUNK 96
#define MOE_EMBED_CHUNK 256

enum { TG_ACT_SILU = 0, TG_ACT_GPTOSS = 1 };

static inline float act_fn(int act, float g, float u) {
    if (act == TG_ACT_GPTOSS) {
        if (g > 7.f) g = 7.f;
        if (u > 7.f) u = 7.f; else if (u < -7.f) u = -7.f;
        return g / (1.f + expf(-1.702f * g)) * (u + 1.f);
    }
    return g / (1.f + expf(-g)) * u;
}

typedef struct {
    const uint8_t *const *gate_ptrs;
    const uint8_t *const *up_ptrs;
    const void *const *down_ptrs;
    int gate_fmt;
    const int *down_fmts;
    const float *const *gate_bias;   /* per expert [inter_dim] or NULL */
    const float *const *up_bias;
    const float *const *down_bias;   /* per expert [embed_dim] or NULL */
    int act;
    int n_experts, k, embed_dim, inter_dim, gate_bpr, down_bpr;
    int *n_rows;        /* [n_experts] */
    int *rows;          /* [n_experts][k] */
    float *wts;         /* [n_experts][k] */
    q8_block *xq_in;    /* [n_experts][k][gate_bpr] */
    float *hidden;      /* [n_experts][k][inter_dim] */
    float *partials;    /* [n_experts][k][embed_dim] */
    int n_inter_chunks, n_embed_chunks;
} moe_ctx;

static void moe_phase_a(void *p, size_t task) {
    moe_ctx *c = p;
    int e = (int)(task / c->n_inter_chunks);
    int ch = (int)(task % c->n_inter_chunks);
    int n = c->n_rows[e];
    if (n == 0) return;
    int r0 = ch * MOE_INTER_CHUNK;
    int r1 = r0 + MOE_INTER_CHUNK < c->inter_dim ? r0 + MOE_INTER_CHUNK : c->inter_dim;
    int fmt = c->gate_fmt;
    size_t gate_row = fmt_row_bytes(fmt, c->embed_dim);
    const uint8_t *gate = c->gate_ptrs[e] + r0 * gate_row;
    const uint8_t *up   = c->up_ptrs[e] + r0 * gate_row;
    const q8_block *xq = c->xq_in + (size_t)e * c->k * c->gate_bpr;
    const float *gb = c->gate_bias ? c->gate_bias[e] : NULL;
    const float *ub = c->up_bias ? c->up_bias[e] : NULL;
    float *hid = c->hidden + (size_t)e * c->k * c->inter_dim;
    int D = c->embed_dim;

    float g[MOE_INTER_CHUNK * 4], u[MOE_INTER_CHUNK * 4];
    for (int b = 0; b < n; b += 4) {
        int m = n - b < 4 ? n - b : 4;
        if (m == 1) {
            const q8_block *x = xq + (size_t)b * c->gate_bpr;
            float *h = hid + (size_t)b * c->inter_dim;
            for (int r = r0; r < r1; r++) {
                float gv = dot_fmt(fmt, gate + (size_t)(r - r0) * gate_row, x, NULL, D);
                float uv = dot_fmt(fmt, up + (size_t)(r - r0) * gate_row, x, NULL, D);
                if (gb) { gv += gb[r]; uv += ub[r]; }
                h[r] = act_fn(c->act, gv, uv);
            }
            continue;
        }
        const q8_block *px[4];
        for (int j = 0; j < 4; j++)
            px[j] = xq + (size_t)(b + (j < m ? j : m - 1)) * c->gate_bpr;
        for (int r = r0; r < r1; r++) {
            dot_fmt_x4(fmt, gate + (size_t)(r - r0) * gate_row, px, NULL, D, g + (r - r0) * 4);
            dot_fmt_x4(fmt, up + (size_t)(r - r0) * gate_row, px, NULL, D, u + (r - r0) * 4);
        }
        for (int j = 0; j < m; j++) {
            float *h = hid + (size_t)(b + j) * c->inter_dim;
            for (int r = r0; r < r1; r++) {
                float gv = g[(r - r0) * 4 + j], uv = u[(r - r0) * 4 + j];
                if (gb) { gv += gb[r]; uv += ub[r]; }
                h[r] = act_fn(c->act, gv, uv);
            }
        }
    }
}

static void moe_phase_b(void *p, size_t task) {
    moe_ctx *c = p;
    int e = (int)(task / c->n_embed_chunks);
    int ch = (int)(task % c->n_embed_chunks);
    int n = c->n_rows[e];
    if (n == 0) return;
    int r0 = ch * MOE_EMBED_CHUNK;
    int r1 = r0 + MOE_EMBED_CHUNK < c->embed_dim ? r0 + MOE_EMBED_CHUNK : c->embed_dim;
    const float *hid = c->hidden + (size_t)e * c->k * c->inter_dim;
    float *part = c->partials + (size_t)e * c->k * c->embed_dim;
    const float *wt = c->wts + (size_t)e * c->k;
    const float *db = c->down_bias ? c->down_bias[e] : NULL;
    int fmt = c->down_fmts[e];
    const void *down = c->down_ptrs[e];
    int I = c->inter_dim;

    if (fmt == TG_FMT_F16) {
        const uint16_t *w = (const uint16_t *)down;
        for (int j = 0; j < n; j++)
            matvec_f16(w + (size_t)r0 * I, hid + (size_t)j * I,
                       part + (size_t)j * c->embed_dim + r0, r1 - r0, I);
        for (int j = 0; j < n; j++)
            for (int r = r0; r < r1; r++)
                part[(size_t)j * c->embed_dim + r] = wt[j] * (part[(size_t)j * c->embed_dim + r] + (db ? db[r] : 0.f));
        return;
    }

    size_t row_bytes = fmt_row_bytes(fmt, I);
    const uint8_t *mat = (const uint8_t *)down + r0 * row_bytes;
    q8_block xq[n * c->down_bpr];
    for (int j = 0; j < n; j++)
        quantize_q8(hid + (size_t)j * I, xq + (size_t)j * c->down_bpr, I);

    for (int b = 0; b < n; b += 4) {
        int m = n - b < 4 ? n - b : 4;
        if (m == 1) {
            const q8_block *x = xq + (size_t)b * c->down_bpr;
            float *o = part + (size_t)b * c->embed_dim;
            for (int r = r0; r < r1; r++) {
                float v = dot_fmt(fmt, mat + (size_t)(r - r0) * row_bytes, x, NULL, I);
                o[r] = wt[b] * (v + (db ? db[r] : 0.f));
            }
            continue;
        }
        const q8_block *px[4];
        for (int j = 0; j < 4; j++)
            px[j] = xq + (size_t)(b + (j < m ? j : m - 1)) * c->down_bpr;
        for (int r = r0; r < r1; r++) {
            float o[4];
            dot_fmt_x4(fmt, mat + (size_t)(r - r0) * row_bytes, px, NULL, I, o);
            float bias = db ? db[r] : 0.f;
            for (int j = 0; j < m; j++)
                part[(size_t)(b + j) * c->embed_dim + r] = wt[b + j] * (o[j] + bias);
        }
    }
}

void tg_moe_forward_rows_v2(
    const uint8_t *const *gate_ptrs, const uint8_t *const *up_ptrs,
    const void *const *down_ptrs, int gate_fmt, const int *down_fmts,
    const float *const *gate_bias, const float *const *up_bias, const float *const *down_bias,
    int act, int n_experts,
    const float *row_weights, const float *inputs, float *output, int k,
    int embed_dim, int inter_dim)
{
    int gate_bpr = Q8_NBLOCKS(embed_dim), down_bpr = Q8_NBLOCKS(inter_dim);
    moe_ctx c = { gate_ptrs, up_ptrs, down_ptrs, gate_fmt, down_fmts,
                  gate_bias, up_bias, down_bias, act,
                  n_experts, k, embed_dim, inter_dim, gate_bpr, down_bpr };
    c.n_rows = calloc(n_experts, sizeof(int));
    c.rows = malloc(sizeof(int) * n_experts * k);
    c.wts = malloc(sizeof(float) * n_experts * k);
    c.xq_in = malloc(sizeof(q8_block) * (size_t)n_experts * k * gate_bpr);
    c.hidden = malloc(sizeof(float) * (size_t)n_experts * k * inter_dim);
    c.partials = calloc((size_t)n_experts * k * embed_dim, sizeof(float));
    c.n_inter_chunks = (inter_dim + MOE_INTER_CHUNK - 1) / MOE_INTER_CHUNK;
    c.n_embed_chunks = (embed_dim + MOE_EMBED_CHUNK - 1) / MOE_EMBED_CHUNK;

    /* Quantize each batch row once, then point each expert's active rows at it. */
    q8_block *xq_rows = malloc(sizeof(q8_block) * (size_t)k * gate_bpr);
    for (int r = 0; r < k; r++)
        quantize_q8(inputs + (size_t)r * embed_dim, xq_rows + (size_t)r * gate_bpr, embed_dim);
    for (int e = 0; e < n_experts; e++) {
        const float *w = row_weights + (size_t)e * k;
        int n = 0;
        for (int r = 0; r < k; r++) {
            if (w[r] == 0.f) continue;
            c.rows[e * k + n] = r;
            c.wts[e * k + n] = w[r];
            memcpy(c.xq_in + ((size_t)e * k + n) * gate_bpr, xq_rows + (size_t)r * gate_bpr,
                   sizeof(q8_block) * gate_bpr);
            n++;
        }
        c.n_rows[e] = n;
    }

    tg_parallel_for((size_t)n_experts * c.n_inter_chunks, moe_phase_a, &c);
    tg_parallel_for((size_t)n_experts * c.n_embed_chunks, moe_phase_b, &c);

    for (int e = 0; e < n_experts; e++) {
        for (int j = 0; j < c.n_rows[e]; j++) {
            float *o = output + (size_t)c.rows[e * k + j] * embed_dim;
            const float *p = c.partials + ((size_t)e * k + j) * embed_dim;
            for (int i = 0; i < embed_dim; i++) o[i] += p[i];
        }
    }
    free(xq_rows); free(c.n_rows); free(c.rows); free(c.wts);
    free(c.xq_in); free(c.hidden); free(c.partials);
}

void tg_moe_forward_rows(
    const uint8_t *const *gate_ptrs, const uint8_t *const *up_ptrs,
    const void *const *down_ptrs, const int *down_fmts, int n_experts,
    const float *row_weights, const float *inputs, float *output, int k,
    int embed_dim, int inter_dim)
{
    tg_moe_forward_rows_v2(gate_ptrs, up_ptrs, down_ptrs, TG_FMT_Q4K, down_fmts,
                           NULL, NULL, NULL, TG_ACT_SILU, n_experts,
                           row_weights, inputs, output, k, embed_dim, inter_dim);
}

/* ═══ Batch expert forward ═══ */

void tg_moe_forward(const uint8_t **expert_ptrs, const float *weights,
                     int n_experts, const float *input, float *output,
                     int embed_dim, int inter_dim) {
    memset(output, 0, embed_dim * sizeof(float));
    for (int e = 0; e < n_experts; e++) {
        tg_expert_forward(expert_ptrs[e], input, output, weights[e],
                          embed_dim, inter_dim);
    }
}

/* ═══ Full transformer layer ═══
 *
 * Routing is done in Python (to manage expert cache lookups).
 * This function takes pre-loaded expert pointers and weights.
 */
void tg_transformer_layer(
    /* Attention weights (f32) */
    const float *wq, const float *wk, const float *wv, const float *wo,
    const float *attn_norm_w, const float *q_norm_w, const float *k_norm_w,
    const float *ffn_norm_w,
    /* KV cache */
    float *kv_k, float *kv_v, int kv_len, int kv_max,
    const float *rope_cos, const float *rope_sin,
    /* Pre-routed expert data */
    const uint8_t **gate_ptrs, const uint8_t **up_ptrs,
    const void **down_ptrs, const int *down_is_f16,
    const float *expert_weights, int n_active,
    /* Input/output (x modified in place) */
    float *x,
    int embed_dim, int inter_dim,
    int n_heads, int n_kv_heads, int head_dim, int pos)
{
    float normed[embed_dim];

    /* Attention block */
    tg_rms_norm(x, attn_norm_w, normed, embed_dim, 1e-6f);
    float attn_out[embed_dim];
    tg_attention_decode(wq, wk, wv, wo, q_norm_w, k_norm_w,
                        kv_k, kv_v, kv_len, kv_max,
                        rope_cos, rope_sin,
                        normed, attn_out,
                        embed_dim, n_heads, n_kv_heads, head_dim, pos);
    for (int i = 0; i < embed_dim; i++) x[i] += attn_out[i];

    /* MoE block */
    tg_rms_norm(x, ffn_norm_w, normed, embed_dim, 1e-6f);
    float moe_out[embed_dim];
    memset(moe_out, 0, embed_dim * sizeof(float));
    for (int e = 0; e < n_active; e++) {
        tg_expert_forward_mixed(gate_ptrs[e], up_ptrs[e],
                                down_ptrs[e], down_is_f16[e],
                                normed, moe_out, expert_weights[e],
                                embed_dim, inter_dim);
    }
    for (int i = 0; i < embed_dim; i++) x[i] += moe_out[i];
}

/* ═══ Verification: scalar Q4_K dot product ═══ */

static float dot_q4k_scalar(const uint8_t *q4, const float *x, int bpr) {
    float total = 0;
    for (int b = 0; b < bpr; b++) {
        const uint8_t *bl = q4 + b * Q4K_BSIZE;
        uint16_t dr, dmr;
        memcpy(&dr, bl, 2);
        memcpy(&dmr, bl + 2, 2);
        float d = f16_to_f32(dr), dmin = f16_to_f32(dmr);
        const uint8_t *scales = bl + 4;
        const uint8_t *qs = bl + 16;
        const float *xb = x + b * QK_K;

        for (int g = 0; g < 4; g++) {
            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            get_scale_min_k4(2 * g,     scales, &sc_lo, &m_lo);
            get_scale_min_k4(2 * g + 1, scales, &sc_hi, &m_hi);
            const uint8_t *qs_g = qs + g * 32;
            for (int j = 0; j < 32; j++) {
                uint8_t v = qs_g[j];
                float val_lo = d * sc_lo * (float)(v & 0xF) - dmin * m_lo;
                float val_hi = d * sc_hi * (float)(v >> 4)  - dmin * m_hi;
                total += val_lo * xb[g * 64 + j]
                       + val_hi * xb[g * 64 + 32 + j];
            }
        }
    }
    return total;
}

void tg_matmul_q4k_scalar(const uint8_t *q4_matrix, const float *input,
                           float *output, int nrows, int ncols) {
    int bpr = ncols / QK_K;
    for (int r = 0; r < nrows; r++)
        output[r] = dot_q4k_scalar(q4_matrix + (size_t)r * bpr * Q4K_BSIZE,
                                    input, bpr);
}

/* ═══ Prefetch (madvise WILLNEED) ═══ */

void tg_prefetch(const void *addr, size_t len) {
    uintptr_t page_mask = ~((uintptr_t)16384 - 1);
    void *aligned = (void *)(((uintptr_t)addr) & page_mask);
    size_t aligned_len = len + ((uintptr_t)addr - (uintptr_t)aligned);
    posix_madvise(aligned, aligned_len, POSIX_MADV_WILLNEED);
}

void tg_prefetch_batch(const void **addrs, const size_t *lens, int count) {
    uintptr_t page_mask = ~((uintptr_t)16384 - 1);
    for (int i = 0; i < count; i++) {
        void *aligned = (void *)(((uintptr_t)addrs[i]) & page_mask);
        size_t aligned_len = lens[i] + ((uintptr_t)addrs[i] - (uintptr_t)aligned);
        posix_madvise(aligned, aligned_len, POSIX_MADV_WILLNEED);
    }
}

int tg_mlock(const void *addr, size_t len) {
    uintptr_t page_mask = ~((uintptr_t)16384 - 1);
    void *aligned = (void *)(((uintptr_t)addr) & page_mask);
    size_t aligned_len = len + ((uintptr_t)addr - (uintptr_t)aligned);
    return mlock(aligned, aligned_len);
}
