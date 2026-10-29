// SIMT kernels for the PLASTIC GPT trainer. All tensors are fp32, row-major.
#define _USE_MATH_DEFINES
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stdout, "ERROR CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); fflush(stdout); exit(1);} } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { fprintf(stdout, "ERROR cuBLAS %d at %s:%d\n", (int)s_, __FILE__, __LINE__); fflush(stdout); exit(1);} } while (0)

static inline int cdiv(long long a, long long b) { return (int)((a + b - 1) / b); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
// block-wide reductions (blockDim.x multiple of 32, <= 1024)
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float sh[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    v = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.f;
    if (wid == 0) v = warp_sum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}
__device__ __forceinline__ float block_max(float v) {
    __shared__ float shm[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) shm[wid] = v;
    __syncthreads();
    v = (threadIdx.x < nw) ? shm[threadIdx.x] : -FLT_MAX;
    if (wid == 0) v = warp_max(v);
    if (threadIdx.x == 0) shm[0] = v;
    __syncthreads();
    return shm[0];
}

// ---------------------------------------------------------------- embedding
__global__ void embedding_fwd(float* out, const int* tok, const float* wte, int N, int d) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)N * d) return;
    int n = (int)(idx / d), c = (int)(idx % d);
    out[idx] = wte[(size_t)tok[n] * d + c];
}
__global__ void embedding_bwd(float* dwte, const int* tok, const float* dout, int N, int d) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)N * d) return;
    int n = (int)(idx / d), c = (int)(idx % d);
    atomicAdd(&dwte[(size_t)tok[n] * d + c], dout[idx]);
}

// ---------------------------------------------------------------- doc-start (for document-masked attention)
// docstart[b*T+i] = largest j<=i with tok[b*T+j]==eot, or 0.
__global__ void docstart_kernel(int* ds, const int* tok, int B, int T, int eot, int use_mask) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    int cur = 0;
    for (int i = 0; i < T; i++) {
        if (use_mask && tok[b * T + i] == eot) cur = i;
        ds[b * T + i] = cur;
    }
}

// ---------------------------------------------------------------- RMSNorm (no learned gain)
__global__ void rmsnorm_fwd(float* out, float* rstd, const float* inp, int N, int d) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= N) return;
    const float* x = inp + (size_t)row * d;
    float ss = 0.f;
    for (int i = lane; i < d; i += 32) ss += x[i] * x[i];
    ss = warp_sum(ss);
    float r = rsqrtf(ss / d + 1e-6f);
    float* y = out + (size_t)row * d;
    for (int i = lane; i < d; i += 32) y[i] = x[i] * r;
    if (lane == 0) rstd[row] = r;
}
// dinp += rstd * (dout - y * mean(dout*y))
__global__ void rmsnorm_bwd(float* dinp, const float* dout, const float* y, const float* rstd, int N, int d) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= N) return;
    const float* g = dout + (size_t)row * d;
    const float* yy = y + (size_t)row * d;
    float dot = 0.f;
    for (int i = lane; i < d; i += 32) dot += g[i] * yy[i];
    dot = warp_sum(dot) / d;
    float r = rstd[row];
    float* dx = dinp + (size_t)row * d;
    for (int i = lane; i < d; i += 32) dx[i] += r * (g[i] - yy[i] * dot);
}

// ---------------------------------------------------------------- QKV split + RoPE (NeoX pairing i, i+hd/2)
// qkv: (B,T,3,H,hd) -> q,k,v: (B,H,T,hd)
__global__ void qkv_rope_fwd(float* q, float* k, float* v, const float* qkv, const float* cosT, const float* sinT,
                             int B, int T, int H, int hd) {
    int h2 = hd / 2;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)B * T * H * h2) return;
    int i = (int)(idx % h2); long long r = idx / h2;
    int h = (int)(r % H); r /= H;
    int t = (int)(r % T); int b = (int)(r / T);
    const float* row = qkv + (size_t)(b * T + t) * 3 * H * hd;
    float c = cosT[t * h2 + i], s = sinT[t * h2 + i];
    size_t o = (((size_t)b * H + h) * T + t) * hd;
    float q0 = row[h * hd + i], q1 = row[h * hd + i + h2];
    q[o + i] = q0 * c - q1 * s; q[o + i + h2] = q0 * s + q1 * c;
    float k0 = row[H * hd + h * hd + i], k1 = row[H * hd + h * hd + i + h2];
    k[o + i] = k0 * c - k1 * s; k[o + i + h2] = k0 * s + k1 * c;
    v[o + i] = row[2 * H * hd + h * hd + i]; v[o + i + h2] = row[2 * H * hd + h * hd + i + h2];
}
__global__ void qkv_rope_bwd(float* dqkv, const float* dq, const float* dk, const float* dv, const float* cosT,
                             const float* sinT, int B, int T, int H, int hd) {
    int h2 = hd / 2;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)B * T * H * h2) return;
    int i = (int)(idx % h2); long long r = idx / h2;
    int h = (int)(r % H); r /= H;
    int t = (int)(r % T); int b = (int)(r / T);
    float* row = dqkv + (size_t)(b * T + t) * 3 * H * hd;
    float c = cosT[t * h2 + i], s = sinT[t * h2 + i];
    size_t o = (((size_t)b * H + h) * T + t) * hd;
    float a0 = dq[o + i], a1 = dq[o + i + h2];
    row[h * hd + i] = a0 * c + a1 * s; row[h * hd + i + h2] = -a0 * s + a1 * c;
    float b0 = dk[o + i], b1 = dk[o + i + h2];
    row[H * hd + h * hd + i] = b0 * c + b1 * s; row[H * hd + h * hd + i + h2] = -b0 * s + b1 * c;
    row[2 * H * hd + h * hd + i] = dv[o + i]; row[2 * H * hd + h * hd + i + h2] = dv[o + i + h2];
}
// (B,H,T,hd) <-> (B,T,H*hd)
__global__ void unpermute_fwd(float* out, const float* inp, int B, int T, int H, int hd) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)B * T * H * hd) return;
    int i = (int)(idx % hd); long long r = idx / hd;
    int t = (int)(r % T); r /= T;
    int h = (int)(r % H); int b = (int)(r / H);
    out[((size_t)(b * T + t) * H + h) * hd + i] = inp[idx];
}
__global__ void unpermute_bwd(float* dinp, const float* dout, int B, int T, int H, int hd) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)B * T * H * hd) return;
    int i = (int)(idx % hd); long long r = idx / hd;
    int t = (int)(r % T); r /= T;
    int h = (int)(r % H); int b = (int)(r / H);
    dinp[idx] = dout[((size_t)(b * T + t) * H + h) * hd + i];
}

// ---------------------------------------------------------------- causal (doc-masked) softmax, one warp per row
// preatt/att: (B*H, T, T). Row (bh, i) attends keys j in [docstart[b,i], i].
__global__ void softmax_fwd(float* att, const float* preatt, const int* ds, int BH, int T, int H, float scale) {
    long long row = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (row >= (long long)BH * T) return;
    int i = (int)(row % T); int bh = (int)(row / T); int b = bh / H;
    int js = ds[b * T + i];
    const float* x = preatt + (size_t)row * T;
    float* y = att + (size_t)row * T;
    float mx = -FLT_MAX;
    for (int j = js + lane; j <= i; j += 32) mx = fmaxf(mx, x[j] * scale);
    mx = warp_max(mx);
    float s = 0.f;
    for (int j = js + lane; j <= i; j += 32) s += __expf(x[j] * scale - mx);
    s = warp_sum(s);
    float inv = 1.f / s;
    for (int j = lane; j < T; j += 32) y[j] = (j >= js && j <= i) ? __expf(x[j] * scale - mx) * inv : 0.f;
}
// in place: d (holds datt) -> dpreatt = scale * att * (datt - sum(att*datt))
__global__ void softmax_bwd(float* d, const float* att, const int* ds, int BH, int T, int H, float scale) {
    long long row = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (row >= (long long)BH * T) return;
    int i = (int)(row % T); int bh = (int)(row / T); int b = bh / H;
    int js = ds[b * T + i];
    float* g = d + (size_t)row * T;
    const float* a = att + (size_t)row * T;
    float dot = 0.f;
    for (int j = js + lane; j <= i; j += 32) dot += a[j] * g[j];
    dot = warp_sum(dot);
    for (int j = lane; j < T; j += 32) g[j] = (j >= js && j <= i) ? scale * a[j] * (g[j] - dot) : 0.f;
}

// ---------------------------------------------------------------- MLP activation: a += b1; h = relu(a)^2
__global__ void bias_relu2_fwd(float* h, float* a, const float* b1, int N, int m) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)N * m) return;
    float z = a[idx] + b1[idx % m];
    a[idx] = z;
    h[idx] = z > 0.f ? z * z : 0.f;
}
// in place: d (holds dh) -> da
__global__ void relu2_bwd(float* d, const float* a, long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float z = a[idx];
    d[idx] = z > 0.f ? d[idx] * 2.f * z : 0.f;
}

// ---------------------------------------------------------------- fused softmax cross-entropy over the vocab
// logits (N, Vp) in -> dlogits out (scaled by dscale). bias (Vp) is added on the fly. tgt<0 => ignored.
__global__ void xent_fwd_bwd(float* logits, float* losses, const int* tgt, const float* bias, int V, int Vp,
                             float dscale, int write_grad) {
    int row = blockIdx.x;
    float* x = logits + (size_t)row * Vp;
    float mx = -FLT_MAX;
    for (int j = threadIdx.x; j < V; j += blockDim.x) mx = fmaxf(mx, x[j] + bias[j]);
    mx = block_max(mx);
    float s = 0.f;
    for (int j = threadIdx.x; j < V; j += blockDim.x) s += __expf(x[j] + bias[j] - mx);
    s = block_sum(s);
    float lse = mx + __logf(s);
    int t = tgt[row];
    if (threadIdx.x == 0) losses[row] = (t >= 0) ? (lse - (x[t] + bias[t])) : 0.f;
    __syncthreads();
    if (!write_grad) return;
    for (int j = threadIdx.x; j < Vp; j += blockDim.x) {
        float p = (j < V && t >= 0) ? __expf(x[j] + bias[j] - lse) : 0.f;
        x[j] = (p - (j == t ? 1.f : 0.f)) * dscale;
    }
}

// ---------------------------------------------------------------- concept-supervision loss (functional righting force)
// For labeled rows: positions [T-len, T) of batch row b carry label y (Ktot floats; facets concatenated).
// Neuron f_off[f]+c of layer lc's MLP pre-activation represents concept c of facet f.
// loss = sum_f CE(softmax(a_f / tau), y_f) per position;  da += w * (softmax - y) / tau
// grid: (positions_total, nfacets), block 256. pos_row/pos_t map flat position index -> (row, t).
__global__ void concept_loss_kernel(float* da, const float* a, const float* labels, const int* pos_lab, const int* pos_n,
                                    const int* f_off, const int* f_size, int m, float tau, float w, float* loss_out) {
    int p = blockIdx.x, f = blockIdx.y;
    int n = pos_n[p];                // flat token index into (N, m)
    const float* y = labels + (size_t)pos_lab[p];   // label vector start (already offset by label row)
    int off = f_off[f], K = f_size[f];
    const float* ar = a + (size_t)n * m + off;
    float mx = -FLT_MAX;
    for (int c = threadIdx.x; c < K; c += blockDim.x) mx = fmaxf(mx, ar[c] / tau);
    mx = block_max(mx);
    float s = 0.f;
    for (int c = threadIdx.x; c < K; c += blockDim.x) s += __expf(ar[c] / tau - mx);
    s = block_sum(s);
    float lse = mx + __logf(s);
    float ysum = 0.f, ce = 0.f;
    for (int c = threadIdx.x; c < K; c += blockDim.x) { float yc = y[off + c]; ysum += yc; ce += yc * (lse - ar[c] / tau); }
    ysum = block_sum(ysum); ce = block_sum(ce);
    float iy = ysum > 0.f ? 1.f / ysum : 0.f;
    if (threadIdx.x == 0) atomicAdd(loss_out, ce * iy * w);
    float* dr = da + (size_t)n * m + off;
    for (int c = threadIdx.x; c < K; c += blockDim.x) {
        float pc = __expf(ar[c] / tau - lse);
        dr[c] += w * (pc - y[off + c] * iy) / tau;
    }
}

// ---------------------------------------------------------------- activation aggregation for probes (mean / max over a span)
// act: (B, T, m). For row b, aggregate positions [T-len[b], T). out_mean/out_max: (B, m)
__global__ void aggregate_kernel(float* out_mean, float* out_max, const float* act, const int* len, int B, int T, int m) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)B * m) return;
    int c = (int)(idx % m), b = (int)(idx / m);
    int L = len[b]; if (L < 1) L = 1; if (L > T) L = T;
    float s = 0.f, mx = -FLT_MAX;
    for (int t = T - L; t < T; t++) { float v = act[((size_t)b * T + t) * m + c]; s += v; mx = fmaxf(mx, v); }
    out_mean[idx] = s / L; out_max[idx] = mx;
}

// ---------------------------------------------------------------- optimizer & misc
__global__ void sumsq_kernel(float* out, const float* x, long long n) {
    float s = 0.f;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) s += x[i] * x[i];
    s = block_sum(s);
    if (threadIdx.x == 0) atomicAdd(out, s);
}
// decoupled AdamW with an optional decoupled pull toward an anchor p0 (per-element strength anc * anc_scale)
__global__ void adamw_kernel(float* p, const float* g, float* m, float* v, const float* p0, const float* anc,
                             long long n, float lr, float b1, float b2, float bc1, float bc2, float eps, float wd,
                             float gscale, float anc_scale) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gr = g[i] * gscale;
    float mi = b1 * m[i] + (1.f - b1) * gr;
    float vi = b2 * v[i] + (1.f - b2) * gr * gr;
    m[i] = mi; v[i] = vi;
    float pi = p[i];
    float upd = (mi / bc1) / (sqrtf(vi / bc2) + eps) + wd * pi;
    float pn = pi - lr * upd;
    // righting force: relax toward the anchor by a fraction min(lr*lambda, 1) per step (stable for any lambda)
    if (anc != nullptr) { float c = fminf(lr * anc[i] * anc_scale, 1.f); pn -= c * (pn - p0[i]); }
    p[i] = pn;
}
__global__ void ema_kernel(float* p0, const float* p, const float* anc, long long n, float beta) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (anc[i] != 0.f) p0[i] += beta * (p[i] - p0[i]);
}
__global__ void sqdiff_kernel(float* out, const float* p, const float* p0, const float* anc, long long n) {
    float s = 0.f;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        if (anc[i] != 0.f) { float dd = p[i] - p0[i]; s += dd * dd; }
    s = block_sum(s);
    if (threadIdx.x == 0) atomicAdd(out, s);
}
__global__ void fill_kernel(float* x, float v, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}
