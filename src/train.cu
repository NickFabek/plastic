// PLASTIC: a small GPT trainer in CUDA C++ (SIMT kernels + cuBLAS TF32 GEMMs).
// Features: doc-masked causal attention, RoPE, param-free RMSNorm, ReLU^2 MLP with bias, untied lm_head + bias,
// AdamW with decoupled anchor pull (weight-space righting force, fixed or EMA anchor), concept-supervision loss on
// MLP neurons (functional righting force), probe activation dumps, finite-difference gradient check.
// Usage: train.exe key=value ...   (see Args usage below; unknown keys are an error)
#include "kernels.cuh"
#include <vector>
#include <string>
#include <map>
#include <set>
#include <random>
#include <chrono>
#include <algorithm>
#include <cstring>

// ------------------------------------------------------------------------------------------------ args
struct Args {
    std::map<std::string, std::string> kv; std::set<std::string> used;
    Args(int argc, char** argv) {
        for (int i = 1; i < argc; i++) {
            const char* e = strchr(argv[i], '=');
            if (!e) { printf("ERROR bad arg %s (want key=value)\n", argv[i]); exit(1); }
            kv[std::string(argv[i], e - argv[i])] = std::string(e + 1);
        }
    }
    bool has(const char* k) { used.insert(k); return kv.count(k) > 0; }
    std::string s(const char* k, const char* def) { used.insert(k); auto it = kv.find(k); return it == kv.end() ? std::string(def) : it->second; }
    long long i(const char* k, long long def) { used.insert(k); auto it = kv.find(k); return it == kv.end() ? def : atoll(it->second.c_str()); }
    double f(const char* k, double def) { used.insert(k); auto it = kv.find(k); return it == kv.end() ? def : atof(it->second.c_str()); }
    void check() { for (auto& p : kv) if (!used.count(p.first)) { printf("ERROR unknown arg %s\n", p.first.c_str()); exit(1); } }
};

// ------------------------------------------------------------------------------------------------ model layout
struct Cfg { int V, Vp, d, L, H, hd, m, T; };
struct TInfo { std::string name; size_t off, n; bool wd; int group; };   // group: 0 matrix, 1 embedding, 2 head, 3 bias
struct Layout {
    Cfg c; std::vector<TInfo> t; size_t total = 0;
    size_t wte, lm_head, lm_bias; std::vector<size_t> wqkv, wo, w1, b1, w2;
    size_t add(const std::string& nm, size_t n, bool wd, int g) { t.push_back({nm, total, n, wd, g}); size_t o = total; total += (n + 63) / 64 * 64; return o; }
    void build(Cfg c_) {
        c = c_; size_t d = c.d, m = c.m;
        wte = add("wte", (size_t)c.Vp * d, false, 1);
        for (int l = 0; l < c.L; l++) {
            wqkv.push_back(add("wqkv" + std::to_string(l), 3 * d * d, true, 0));
            wo.push_back(add("wo" + std::to_string(l), d * d, true, 0));
            w1.push_back(add("w1_" + std::to_string(l), m * d, true, 0));
            b1.push_back(add("b1_" + std::to_string(l), m, false, 3));
            w2.push_back(add("w2_" + std::to_string(l), d * m, true, 0));
        }
        lm_head = add("lm_head", (size_t)c.Vp * d, true, 2);
        lm_bias = add("lm_bias", (size_t)c.Vp, false, 3);
    }
};

static cublasHandle_t g_cublas;
static cublasComputeType_t g_compute = CUBLAS_COMPUTE_32F_FAST_TF32;
static float* g_ones = nullptr;   // ones vector for bias-gradient gemv

// out(N,OC) = inp(N,C) @ w(OC,C)^T + beta*out
static void mm_fwd(float* out, const float* inp, const float* w, int N, int C, int OC, float beta) {
    float alpha = 1.f;
    CB(cublasGemmEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, OC, N, C, &alpha, w, CUDA_R_32F, C, inp, CUDA_R_32F, C, &beta,
                    out, CUDA_R_32F, OC, g_compute, CUBLAS_GEMM_DEFAULT));
}
// dinp(N,C) = dout(N,OC) @ w(OC,C) + beta*dinp
static void mm_bwd_inp(float* dinp, const float* dout, const float* w, int N, int C, int OC, float beta) {
    float alpha = 1.f;
    CB(cublasGemmEx(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, C, N, OC, &alpha, w, CUDA_R_32F, C, dout, CUDA_R_32F, OC, &beta,
                    dinp, CUDA_R_32F, C, g_compute, CUBLAS_GEMM_DEFAULT));
}
// dw(OC,C) += dout(N,OC)^T @ inp(N,C)
static void mm_bwd_w(float* dw, const float* dout, const float* inp, int N, int C, int OC) {
    float alpha = 1.f, beta = 1.f;
    CB(cublasGemmEx(g_cublas, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, N, &alpha, inp, CUDA_R_32F, C, dout, CUDA_R_32F, OC, &beta,
                    dw, CUDA_R_32F, C, g_compute, CUBLAS_GEMM_DEFAULT));
}
// db(OC) += sum_n d(N,OC)
static void colsum_acc(float* db, const float* d, int N, int OC) {
    float alpha = 1.f, beta = 1.f;
    CB(cublasSgemv(g_cublas, CUBLAS_OP_N, OC, N, &alpha, d, OC, g_ones, 1, &beta, db, 1));
}
// strided-batched helpers for attention, all (BH, T, *) row-major
static void sbgemm(cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, const float* A, int lda, long long sA,
                   const float* B, int ldb, long long sB, float* C, int ldc, long long sC, int batch) {
    float alpha = 1.f, beta = 0.f;
    CB(cublasGemmStridedBatchedEx(g_cublas, ta, tb, m, n, k, &alpha, A, CUDA_R_32F, lda, sA, B, CUDA_R_32F, ldb, sB, &beta,
                                  C, CUDA_R_32F, ldc, sC, batch, g_compute, CUBLAS_GEMM_DEFAULT));
}

// ------------------------------------------------------------------------------------------------ activations
struct Acts {
    int B = 0, T = 0, N = 0;
    float *res, *res_mid, *ln1, *ln2, *rstd1, *rstd2, *q, *k, *v, *att, *y, *a, *h, *lnf, *rstdf, *logits, *losses;
    float *qkv, *preatt, *yh, *dres, *dln, *dy, *dyh, *dq, *dk, *dv, *dqkv, *dh;
    int *tok, *tgt, *ds;
    size_t bytes = 0;
    float* al(size_t n) { float* p; CK(cudaMalloc(&p, n * sizeof(float))); bytes += n * sizeof(float); return p; }
    void alloc(const Cfg& c, int B_, bool train) {
        B = B_; T = c.T; N = B * T;
        size_t Nd = (size_t)N * c.d, Nm = (size_t)N * c.m, att_n = (size_t)B * c.H * T * T;
        res = al(Nd * (c.L + 1)); res_mid = al(Nd * c.L); ln1 = al(Nd * c.L); ln2 = al(Nd * c.L);
        rstd1 = al((size_t)N * c.L); rstd2 = al((size_t)N * c.L);
        q = al(Nd * c.L); k = al(Nd * c.L); v = al(Nd * c.L); att = al(att_n * c.L); y = al(Nd * c.L);
        a = al(Nm * c.L); h = al(Nm * c.L); lnf = al(Nd); rstdf = al(N);
        logits = al((size_t)N * c.Vp); losses = al(N);
        qkv = al(3 * Nd); preatt = al(att_n); yh = al(Nd);
        if (train) { dres = al(Nd); dln = al(Nd); dy = al(Nd); dyh = al(Nd); dq = al(Nd); dk = al(Nd); dv = al(Nd); dqkv = al(3 * Nd); dh = al(Nm); }
        CK(cudaMalloc(&tok, N * sizeof(int))); CK(cudaMalloc(&tgt, N * sizeof(int))); CK(cudaMalloc(&ds, N * sizeof(int)));
    }
};

struct Model {
    Cfg c; Layout lay;
    float *p = nullptr, *g = nullptr, *mo = nullptr, *vo = nullptr, *p0 = nullptr, *anc = nullptr;
    float *cosT, *sinT; int eot = 50256; int docmask = 1;
    void alloc_params() {
        CK(cudaMalloc(&p, lay.total * sizeof(float))); CK(cudaMemset(p, 0, lay.total * sizeof(float)));
        int h2 = c.hd / 2; std::vector<float> co((size_t)c.T * h2), si((size_t)c.T * h2);
        for (int t = 0; t < c.T; t++) for (int i = 0; i < h2; i++) {
            double fr = pow(10000.0, -2.0 * i / c.hd); co[t * h2 + i] = (float)cos(t * fr); si[t * h2 + i] = (float)sin(t * fr);
        }
        CK(cudaMalloc(&cosT, co.size() * 4)); CK(cudaMalloc(&sinT, si.size() * 4));
        CK(cudaMemcpy(cosT, co.data(), co.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(sinT, si.data(), si.size() * 4, cudaMemcpyHostToDevice));
    }
    void alloc_train() {
        size_t b = lay.total * sizeof(float);
        CK(cudaMalloc(&g, b)); CK(cudaMalloc(&mo, b)); CK(cudaMalloc(&vo, b));
        CK(cudaMemset(g, 0, b)); CK(cudaMemset(mo, 0, b)); CK(cudaMemset(vo, 0, b));
    }
};

// random init (host RNG -> device)
static void init_random(Model& M, uint64_t seed, double emb_std, double head_std, double out_std, double in_scale) {
    std::mt19937_64 rng(seed); std::vector<float> hp(M.lay.total, 0.f);
    auto fill = [&](size_t off, size_t n, double std) { std::normal_distribution<double> nd(0.0, std); for (size_t i = 0; i < n; i++) hp[off + i] = (float)nd(rng); };
    const Cfg& c = M.c; double sd = in_scale / sqrt((double)c.d);
    fill(M.lay.wte, (size_t)c.Vp * c.d, emb_std);
    for (int l = 0; l < c.L; l++) {
        fill(M.lay.wqkv[l], 3ull * c.d * c.d, sd); fill(M.lay.w1[l], (size_t)c.m * c.d, sd);
        if (out_std > 0) { fill(M.lay.wo[l], (size_t)c.d * c.d, out_std / sqrt((double)c.d)); fill(M.lay.w2[l], (size_t)c.d * c.m, out_std / sqrt((double)c.m)); }
    }
    if (head_std > 0) fill(M.lay.lm_head, (size_t)c.Vp * c.d, head_std / sqrt((double)c.d));
    CK(cudaMemcpy(M.p, hp.data(), M.lay.total * 4, cudaMemcpyHostToDevice));
}

// checkpoint: int32 hdr[16] {magic, ver, Vp, d, L, H, m, T, V, 0..} then each tensor's floats in layout order (unpadded)
static const int CKPT_MAGIC = 0x504c4153;
static void save_ckpt(Model& M, const std::string& path, const float* src) {
    FILE* f = fopen(path.c_str(), "wb"); if (!f) { printf("ERROR cannot write %s\n", path.c_str()); exit(1); }
    int hdr[16] = {CKPT_MAGIC, 1, M.c.Vp, M.c.d, M.c.L, M.c.H, M.c.m, M.c.T, M.c.V};
    fwrite(hdr, 4, 16, f);
    std::vector<float> hp(M.lay.total); CK(cudaMemcpy(hp.data(), src, M.lay.total * 4, cudaMemcpyDeviceToHost));
    for (auto& t : M.lay.t) fwrite(hp.data() + t.off, 4, t.n, f);
    fclose(f);
}
static void load_ckpt(Model& M, const std::string& path, float* dst) {
    FILE* f = fopen(path.c_str(), "rb"); if (!f) { printf("ERROR cannot read %s\n", path.c_str()); exit(1); }
    int hdr[16]; if (fread(hdr, 4, 16, f) != 16 || hdr[0] != CKPT_MAGIC) { printf("ERROR bad ckpt %s\n", path.c_str()); exit(1); }
    if (hdr[2] != M.c.Vp || hdr[3] != M.c.d || hdr[4] != M.c.L || hdr[5] != M.c.H || hdr[6] != M.c.m) {
        printf("ERROR ckpt %s config mismatch (Vp=%d d=%d L=%d H=%d m=%d)\n", path.c_str(), hdr[2], hdr[3], hdr[4], hdr[5], hdr[6]); exit(1);
    }
    std::vector<float> hp(M.lay.total, 0.f);
    for (auto& t : M.lay.t) if (fread(hp.data() + t.off, 4, t.n, f) != t.n) { printf("ERROR short ckpt %s at %s\n", path.c_str(), t.name.c_str()); exit(1); }
    fclose(f);
    CK(cudaMemcpy(dst, hp.data(), M.lay.total * 4, cudaMemcpyHostToDevice));
}

// ------------------------------------------------------------------------------------------------ forward / backward
static void forward(Model& M, Acts& A, int write_grad, float dscale) {
    const Cfg& c = M.c; int B = A.B, T = A.T, N = A.N, d = c.d, m = c.m, H = c.H, hd = c.hd;
    size_t Nd = (size_t)N * d, Nm = (size_t)N * m, attn = (size_t)B * H * T * T;
    const float* P = M.p; const Layout& L = M.lay;
    docstart_kernel<<<cdiv(B, 64), 64>>>(A.ds, A.tok, B, T, M.eot, M.docmask);
    embedding_fwd<<<cdiv((long long)N * d, 256), 256>>>(A.res, A.tok, P + L.wte, N, d);
    float scale = 1.f / sqrtf((float)hd);
    for (int l = 0; l < c.L; l++) {
        float* res = A.res + l * Nd; float* rmid = A.res_mid + l * Nd; float* ln1 = A.ln1 + l * Nd; float* ln2 = A.ln2 + l * Nd;
        float *q = A.q + l * Nd, *k = A.k + l * Nd, *v = A.v + l * Nd, *att = A.att + l * attn, *y = A.y + l * Nd;
        float *a = A.a + l * Nm, *h = A.h + l * Nm;
        rmsnorm_fwd<<<cdiv((long long)N * 32, 256), 256>>>(ln1, A.rstd1 + (size_t)l * N, res, N, d);
        mm_fwd(A.qkv, ln1, P + L.wqkv[l], N, d, 3 * d, 0.f);
        qkv_rope_fwd<<<cdiv((long long)B * T * H * (hd / 2), 256), 256>>>(q, k, v, A.qkv, M.cosT, M.sinT, B, T, H, hd);
        sbgemm(CUBLAS_OP_T, CUBLAS_OP_N, T, T, hd, k, hd, (long long)T * hd, q, hd, (long long)T * hd, A.preatt, T, (long long)T * T, B * H);
        softmax_fwd<<<cdiv((long long)B * H * T * 32, 256), 256>>>(att, A.preatt, A.ds, B * H, T, H, scale);
        sbgemm(CUBLAS_OP_N, CUBLAS_OP_N, hd, T, T, v, hd, (long long)T * hd, att, T, (long long)T * T, A.yh, hd, (long long)T * hd, B * H);
        unpermute_fwd<<<cdiv((long long)Nd, 256), 256>>>(y, A.yh, B, T, H, hd);
        CK(cudaMemcpyAsync(rmid, res, Nd * 4, cudaMemcpyDeviceToDevice));
        mm_fwd(rmid, y, P + L.wo[l], N, d, d, 1.f);
        rmsnorm_fwd<<<cdiv((long long)N * 32, 256), 256>>>(ln2, A.rstd2 + (size_t)l * N, rmid, N, d);
        mm_fwd(a, ln2, P + L.w1[l], N, d, m, 0.f);
        bias_relu2_fwd<<<cdiv((long long)Nm, 256), 256>>>(h, a, P + L.b1[l], N, m);
        float* rnext = A.res + (l + 1) * Nd;
        CK(cudaMemcpyAsync(rnext, rmid, Nd * 4, cudaMemcpyDeviceToDevice));
        mm_fwd(rnext, h, P + L.w2[l], N, m, d, 1.f);
    }
    rmsnorm_fwd<<<cdiv((long long)N * 32, 256), 256>>>(A.lnf, A.rstdf, A.res + c.L * Nd, N, d);
    mm_fwd(A.logits, A.lnf, P + L.lm_head, N, d, c.Vp, 0.f);
    xent_fwd_bwd<<<N, 512>>>(A.logits, A.losses, A.tgt, P + L.lm_bias, c.V, c.Vp, dscale, write_grad);
}

struct ConceptBatch {   // device-side description of concept-supervised positions for one micro-batch
    int npos = 0; int *pos_n = nullptr, *pos_lab = nullptr; const float* labels = nullptr;
    int nf = 0; int *f_off = nullptr, *f_size = nullptr; int layer = -1; float tau = 1.f, w = 0.f; float* loss = nullptr;
};

static void backward(Model& M, Acts& A, const ConceptBatch* cb) {
    const Cfg& c = M.c; int B = A.B, T = A.T, N = A.N, d = c.d, m = c.m, H = c.H, hd = c.hd;
    size_t Nd = (size_t)N * d, Nm = (size_t)N * m, attn = (size_t)B * H * T * T;
    const float* P = M.p; float* G = M.g; const Layout& L = M.lay;
    float scale = 1.f / sqrtf((float)hd);
    colsum_acc(G + L.lm_bias, A.logits, N, c.Vp);
    mm_bwd_inp(A.dln, A.logits, P + L.lm_head, N, d, c.Vp, 0.f);
    mm_bwd_w(G + L.lm_head, A.logits, A.lnf, N, d, c.Vp);
    CK(cudaMemsetAsync(A.dres, 0, Nd * 4));
    rmsnorm_bwd<<<cdiv((long long)N * 32, 256), 256>>>(A.dres, A.dln, A.lnf, A.rstdf, N, d);
    for (int l = c.L - 1; l >= 0; l--) {
        float* ln1 = A.ln1 + l * Nd; float* ln2 = A.ln2 + l * Nd;
        float *q = A.q + l * Nd, *k = A.k + l * Nd, *v = A.v + l * Nd, *att = A.att + l * attn, *y = A.y + l * Nd;
        float *a = A.a + l * Nm, *h = A.h + l * Nm;
        // MLP
        mm_bwd_inp(A.dh, A.dres, P + L.w2[l], N, m, d, 0.f);
        mm_bwd_w(G + L.w2[l], A.dres, h, N, m, d);
        relu2_bwd<<<cdiv((long long)Nm, 256), 256>>>(A.dh, a, (long long)Nm);
        if (cb && cb->w != 0.f && cb->layer == l && cb->npos > 0) {
            dim3 grid(cb->npos, cb->nf);
            concept_loss_kernel<<<grid, 256>>>(A.dh, a, cb->labels, cb->pos_lab, cb->pos_n, cb->f_off, cb->f_size, m, cb->tau, cb->w, cb->loss);
        }
        colsum_acc(G + L.b1[l], A.dh, N, m);
        mm_bwd_inp(A.dln, A.dh, P + L.w1[l], N, d, m, 0.f);
        mm_bwd_w(G + L.w1[l], A.dh, ln2, N, d, m);
        rmsnorm_bwd<<<cdiv((long long)N * 32, 256), 256>>>(A.dres, A.dln, ln2, A.rstd2 + (size_t)l * N, N, d);
        // attention
        mm_bwd_inp(A.dy, A.dres, P + L.wo[l], N, d, d, 0.f);
        mm_bwd_w(G + L.wo[l], A.dres, y, N, d, d);
        unpermute_bwd<<<cdiv((long long)Nd, 256), 256>>>(A.dyh, A.dy, B, T, H, hd);
        sbgemm(CUBLAS_OP_T, CUBLAS_OP_N, T, T, hd, v, hd, (long long)T * hd, A.dyh, hd, (long long)T * hd, A.preatt, T, (long long)T * T, B * H);
        sbgemm(CUBLAS_OP_N, CUBLAS_OP_T, hd, T, T, A.dyh, hd, (long long)T * hd, att, T, (long long)T * T, A.dv, hd, (long long)T * hd, B * H);
        softmax_bwd<<<cdiv((long long)B * H * T * 32, 256), 256>>>(A.preatt, att, A.ds, B * H, T, H, scale);
        sbgemm(CUBLAS_OP_N, CUBLAS_OP_N, hd, T, T, k, hd, (long long)T * hd, A.preatt, T, (long long)T * T, A.dq, hd, (long long)T * hd, B * H);
        sbgemm(CUBLAS_OP_N, CUBLAS_OP_T, hd, T, T, q, hd, (long long)T * hd, A.preatt, T, (long long)T * T, A.dk, hd, (long long)T * hd, B * H);
        qkv_rope_bwd<<<cdiv((long long)B * T * H * (hd / 2), 256), 256>>>(A.dqkv, A.dq, A.dk, A.dv, M.cosT, M.sinT, B, T, H, hd);
        mm_bwd_inp(A.dln, A.dqkv, P + L.wqkv[l], N, d, 3 * d, 0.f);
        mm_bwd_w(G + L.wqkv[l], A.dqkv, ln1, N, d, 3 * d);
        rmsnorm_bwd<<<cdiv((long long)N * 32, 256), 256>>>(A.dres, A.dln, ln1, A.rstd1 + (size_t)l * N, N, d);
    }
    embedding_bwd<<<cdiv((long long)Nd, 256), 256>>>(G + L.wte, A.tok, A.dres, N, d);
}

// ------------------------------------------------------------------------------------------------ data
static std::vector<uint16_t> read_u16(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb"); if (!f) { printf("ERROR cannot read %s\n", path.c_str()); exit(1); }
    _fseeki64(f, 0, SEEK_END); long long n = _ftelli64(f) / 2; _fseeki64(f, 0, SEEK_SET);
    std::vector<uint16_t> v(n); if ((long long)fread(v.data(), 2, n, f) != n) { printf("ERROR short read %s\n", path.c_str()); exit(1); }
    fclose(f); return v;
}
struct Labels {   // concept labels for passages: window inputs = stream[end-T, end), labels on the last len positions
    int n = 0, K = 0, nf = 0, split = 0; std::vector<int> fsize, foff; std::vector<long long> end; std::vector<int> len; std::vector<float> y;
    void load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb"); if (!f) { printf("ERROR cannot read %s\n", path.c_str()); exit(1); }
        int hdr[8]; fread(hdr, 4, 8, f);
        if (hdr[0] != 0x4c41424c) { printf("ERROR bad labels file %s\n", path.c_str()); exit(1); }
        n = hdr[2]; K = hdr[3]; nf = hdr[4]; split = hdr[5];
        fsize.resize(nf); fread(fsize.data(), 4, nf, f); foff.resize(nf); int o = 0; for (int i = 0; i < nf; i++) { foff[i] = o; o += fsize[i]; }
        end.resize(n); len.resize(n); y.resize((size_t)n * K);
        fread(end.data(), 8, n, f); fread(len.data(), 4, n, f); fread(y.data(), 4, (size_t)n * K, f); fclose(f);
    }
};
static void fill_window(const std::vector<uint16_t>& s, long long start, int T, int* tok, int* tgt) {
    for (int t = 0; t < T; t++) { tok[t] = s[start + t]; tgt[t] = s[start + t + 1]; }
}

// ------------------------------------------------------------------------------------------------ helpers
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static float dev_sum(const float* x, long long n, float* scratch) {
    CK(cudaMemsetAsync(scratch, 0, 4)); sumsq_kernel<<<256, 512>>>(scratch, x, n); float r; CK(cudaMemcpy(&r, scratch, 4, cudaMemcpyDeviceToHost)); return r;
}
static double losses_mean(Acts& A, std::vector<float>& tmp) {
    tmp.resize(A.N); CK(cudaMemcpy(tmp.data(), A.losses, A.N * 4, cudaMemcpyDeviceToHost));
    double s = 0; for (float x : tmp) s += x; return s / A.N;
}
static std::string jnum(double x) { char b[64]; snprintf(b, 64, "%.6g", x); return b; }

static float sched(const std::string& kind, double frac, double endf) {   // multiplier in [0,1] at training fraction frac
    if (kind == "const") return 1.f;
    if (kind == "step") return frac < endf ? 1.f : 0.f;
    if (kind == "linear") return frac < endf ? (float)(1.0 - frac / endf) : 0.f;
    if (kind == "cosine") return frac < endf ? (float)(0.5 * (1 + cos(3.14159265358979 * frac / endf))) : 0.f;
    printf("ERROR unknown schedule %s\n", kind.c_str()); exit(1);
}

// ------------------------------------------------------------------------------------------------ gradient check
static int gradcheck() {
    g_compute = CUBLAS_COMPUTE_32F;
    Model M; M.c = {97, 128, 32, 2, 2, 16, 64, 8}; M.eot = 5; M.lay.build(M.c); M.alloc_params(); M.alloc_train();
    init_random(M, 1, 1.0, 1.0, 1.0, 1.0);
    { std::mt19937_64 r(2); std::normal_distribution<float> nd(0, 0.3f); std::vector<float> hp(M.lay.total);
      CK(cudaMemcpy(hp.data(), M.p, hp.size() * 4, cudaMemcpyDeviceToHost));
      for (int l = 0; l < M.c.L; l++) for (int i = 0; i < M.c.m; i++) hp[M.lay.b1[l] + i] = nd(r);
      for (int i = 0; i < M.c.Vp; i++) hp[M.lay.lm_bias + i] = nd(r);
      CK(cudaMemcpy(M.p, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice)); }
    int B = 2; Acts A; A.alloc(M.c, B, true);
    CK(cudaMalloc(&g_ones, A.N * 4)); { std::vector<float> o(A.N, 1.f); CK(cudaMemcpy(g_ones, o.data(), A.N * 4, cudaMemcpyHostToDevice)); }
    std::mt19937 r(3); std::vector<int> tok(A.N), tgt(A.N);
    for (int i = 0; i < A.N; i++) { tok[i] = r() % 97; tgt[i] = r() % 97; } tok[3] = 5; tok[8 + 5] = 5;
    CK(cudaMemcpy(A.tok, tok.data(), A.N * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(A.tgt, tgt.data(), A.N * 4, cudaMemcpyHostToDevice));
    // concept supervision on layer 1, two facets (5 + 3 neurons), row 0 positions 4..7
    ConceptBatch cb; int K = 8; std::vector<float> lab = {0.1f, 0.5f, 0.2f, 0.1f, 0.1f, 0.6f, 0.3f, 0.1f};
    std::vector<int> pn = {4, 5, 6, 7}, pl = {0, 0, 0, 0}, fo = {0, 5}, fs = {5, 3};
    float* dlab; CK(cudaMalloc(&dlab, K * 4)); CK(cudaMemcpy(dlab, lab.data(), K * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&cb.pos_n, 16)); CK(cudaMalloc(&cb.pos_lab, 16)); CK(cudaMalloc(&cb.f_off, 8)); CK(cudaMalloc(&cb.f_size, 8)); CK(cudaMalloc(&cb.loss, 4));
    CK(cudaMemcpy(cb.pos_n, pn.data(), 16, cudaMemcpyHostToDevice)); CK(cudaMemcpy(cb.pos_lab, pl.data(), 16, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cb.f_off, fo.data(), 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(cb.f_size, fs.data(), 8, cudaMemcpyHostToDevice));
    cb.npos = 4; cb.labels = dlab; cb.nf = 2; cb.layer = 1; cb.tau = 0.7f; cb.w = 0.5f;
    std::vector<float> tmp;
    // concept loss value (forward only) evaluated via a backward call into scratch
    auto total_loss = [&]() -> double {
        forward(M, A, 1, 1.f / A.N); double lm = losses_mean(A, tmp);
        CK(cudaMemset(cb.loss, 0, 4)); CK(cudaMemset(M.g, 0, M.lay.total * 4)); backward(M, A, &cb);
        float cl; CK(cudaMemcpy(&cl, cb.loss, 4, cudaMemcpyDeviceToHost)); return lm + cl;
    };
    total_loss();
    std::vector<float> grad(M.lay.total); CK(cudaMemcpy(grad.data(), M.g, M.lay.total * 4, cudaMemcpyDeviceToHost));
    std::vector<float> hp(M.lay.total); CK(cudaMemcpy(hp.data(), M.p, M.lay.total * 4, cudaMemcpyDeviceToHost));
    double worst = 0; std::mt19937 pr(5);
    for (auto& t : M.lay.t) {
        double tw = 0;
        for (int s = 0; s < 6; s++) {
            size_t i = t.off + pr() % t.n;
            if (t.name == "wte" || t.name == "lm_head") { int row = (s % 2 == 0) ? tok[pr() % A.N] : tgt[pr() % A.N]; i = t.off + (size_t)row * M.c.d + pr() % M.c.d; }
            float h = 1e-2f, orig = hp[i];
            hp[i] = orig + h; CK(cudaMemcpy(M.p + i, &hp[i], 4, cudaMemcpyHostToDevice)); double lp = total_loss();
            hp[i] = orig - h; CK(cudaMemcpy(M.p + i, &hp[i], 4, cudaMemcpyHostToDevice)); double lmn = total_loss();
            hp[i] = orig; CK(cudaMemcpy(M.p + i, &hp[i], 4, cudaMemcpyHostToDevice));
            double num = (lp - lmn) / (2 * h), an = grad[i];
            double rel = fabs(num - an) / fmax(1e-3, fabs(num) + fabs(an));
            tw = fmax(tw, rel);
        }
        printf("GRADCHECK %-10s max_rel_err=%.2e\n", t.name.c_str(), tw); worst = fmax(worst, tw);
    }
    printf("GRADCHECK worst=%.2e %s\n", worst, worst < 2e-2 ? "PASS" : "FAIL");
    return worst < 2e-2 ? 0 : 1;
}

// ------------------------------------------------------------------------------------------------ main
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    Args a(argc, argv);
    std::string mode = a.s("mode", "train");
    CB(cublasCreate(&g_cublas));
    if (mode == "gradcheck") { a.check(); return gradcheck(); }
    Cfg c; c.V = 50257; c.Vp = 50304;
    c.d = (int)a.i("d", 256); c.L = (int)a.i("L", 4); c.H = (int)a.i("H", c.d / 64); c.hd = c.d / c.H; c.m = (int)a.i("m", 4 * c.d); c.T = (int)a.i("T", 256);
    Model M; M.c = c; M.lay.build(c); M.alloc_params(); M.docmask = (int)a.i("docmask", 1);
    std::string run_dir = a.s("run_dir", "runs/tmp");
    std::string init_from = a.s("init_from", "");
    uint64_t seed = (uint64_t)a.i("seed", 1);
    if (init_from.empty()) init_random(M, seed, a.f("emb_std", 1.0), a.f("head_std", 0.0), a.f("out_std", 0.0), a.f("in_scale", 1.0));
    else load_ckpt(M, init_from, M.p);
    if (mode == "init") {   // write the random init (so constructed models share every non-constructed weight)
        std::string out = a.s("save", (run_dir + "/init.bin").c_str()); a.check();
        save_ckpt(M, out, M.p); printf("DONE wrote init %s\n", out.c_str()); return 0;
    }
    int B = (int)a.i("B", 32);
    long long eval_tokens = a.i("eval_tokens", 1 << 20);
    std::vector<float> tmp; float* scratch; CK(cudaMalloc(&scratch, 64));

    if (mode == "dump") {   // probe activations: per passage mean of h and a over the passage positions, all layers
        std::string probe = a.s("probe", "labels/probe.bin"), out = a.s("out", (run_dir + "/dump.bin").c_str());
        a.check();
        Labels P; P.load(probe);
        std::vector<uint16_t> stream = read_u16(P.split ? "data_bin/val.u16" : "data_bin/train.u16");
        Acts A; A.alloc(c, B, false);
        CK(cudaMalloc(&g_ones, A.N * 4));
        int* dlen; CK(cudaMalloc(&dlen, B * 4)); float *om, *ox; CK(cudaMalloc(&om, (size_t)B * c.m * 4)); CK(cudaMalloc(&ox, (size_t)B * c.m * 4));
        FILE* f = fopen(out.c_str(), "wb"); int hdr[8] = {0x504d5544, 1, P.n, c.L, c.m, 2, 0, 0}; fwrite(hdr, 4, 8, f);
        std::vector<float> hm((size_t)P.n * c.L * c.m), am((size_t)P.n * c.L * c.m), buf((size_t)B * c.m);
        std::vector<int> tok(A.N), tgt(A.N), len(B);
        for (int i0 = 0; i0 < P.n; i0 += B) {
            int nb = std::min(B, P.n - i0);
            for (int b = 0; b < B; b++) { int j = i0 + std::min(b, nb - 1); fill_window(stream, P.end[j] - c.T, c.T, &tok[b * c.T], &tgt[b * c.T]); len[b] = P.len[j]; }
            CK(cudaMemcpy(A.tok, tok.data(), A.N * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(A.tgt, tgt.data(), A.N * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dlen, len.data(), B * 4, cudaMemcpyHostToDevice));
            forward(M, A, 0, 0.f);
            for (int l = 0; l < c.L; l++) for (int which = 0; which < 2; which++) {
                const float* src = (which == 0 ? A.h : A.a) + (size_t)l * A.N * c.m;
                aggregate_kernel<<<cdiv((long long)B * c.m, 256), 256>>>(om, ox, src, dlen, B, c.T, c.m);
                CK(cudaMemcpy(buf.data(), om, buf.size() * 4, cudaMemcpyDeviceToHost));
                std::vector<float>& dst = which == 0 ? hm : am;
                for (int b = 0; b < nb; b++) memcpy(&dst[((size_t)(i0 + b) * c.L + l) * c.m], &buf[(size_t)b * c.m], c.m * 4);
            }
        }
        fwrite(hm.data(), 4, hm.size(), f); fwrite(am.data(), 4, am.size(), f); fclose(f);
        printf("DONE dumped %d passages x %d layers x %d neurons to %s\n", P.n, c.L, c.m, out.c_str());
        return 0;
    }

    if (mode == "tokdump") {   // per-token post-activations (fp16) of every layer for the first n val windows
        int nwin = (int)a.i("n_windows", 64); std::string out = a.s("out", (run_dir + "/tokdump.bin").c_str()); a.check();
        nwin = nwin / B * B; std::vector<uint16_t> val = read_u16("data_bin/val.u16");
        Acts A; A.alloc(c, B, false); CK(cudaMalloc(&g_ones, A.N * 4));
        FILE* f = fopen(out.c_str(), "wb"); int hdr[8] = {0x4b4f5454, 1, nwin, c.T, c.L, c.m, 0, 0}; fwrite(hdr, 4, 8, f);
        std::vector<int> tok(A.N), tgt(A.N); std::vector<float> buf((size_t)A.N * c.m); std::vector<uint16_t> half((size_t)A.N * c.m);
        std::vector<int> alltok; std::vector<std::vector<uint16_t>> acts(c.L);
        for (int w0 = 0; w0 < nwin; w0 += B) {
            for (int b = 0; b < B; b++) fill_window(val, (long long)(w0 + b) * c.T, c.T, &tok[b * c.T], &tgt[b * c.T]);
            CK(cudaMemcpy(A.tok, tok.data(), A.N * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(A.tgt, tgt.data(), A.N * 4, cudaMemcpyHostToDevice));
            forward(M, A, 0, 0.f);
            alltok.insert(alltok.end(), tok.begin(), tok.end());
            for (int l = 0; l < c.L; l++) {
                CK(cudaMemcpy(buf.data(), A.h + (size_t)l * A.N * c.m, buf.size() * 4, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < buf.size(); i++) { float v = buf[i]; uint32_t x; memcpy(&x, &v, 4);   // fp32 -> fp16 (round-to-nearest, no denormals)
                    uint32_t s = (x >> 16) & 0x8000; int e = (int)((x >> 23) & 0xff) - 127 + 15; uint32_t mnt = (x >> 13) & 0x3ff;
                    half[i] = (uint16_t)(e <= 0 ? s : e >= 31 ? (s | 0x7bff) : (s | (e << 10) | mnt)); }
                acts[l].insert(acts[l].end(), half.begin(), half.end());
            }
        }
        fwrite(alltok.data(), 4, alltok.size(), f); for (int l = 0; l < c.L; l++) fwrite(acts[l].data(), 2, acts[l].size(), f); fclose(f);
        printf("DONE tokdump %d windows x %d tokens x %d layers x %d neurons -> %s\n", nwin, c.T, c.L, c.m, out.c_str()); return 0;
    }

    // ---------------- training / eval setup
    long long max_tokens = a.i("tokens", 50000000);
    int accum = (int)a.i("accum", 1);
    double lr = a.f("lr", 3e-3), lr_emb = a.f("lr_emb_mult", 10.0), lr_head = a.f("lr_head_mult", 1.0), lr_bias = a.f("lr_bias_mult", 1.0);
    double wd = a.f("wd", 0.1), b1 = a.f("beta1", 0.9), b2 = a.f("beta2", 0.95), clip = a.f("clip", 1.0);
    long long warmup = a.i("warmup", 200); double warmdown = a.f("warmdown", 0.3), final_lr = a.f("final_lr_frac", 0.0);
    int eval_every = (int)a.i("eval_every", 250); int save_every = (int)a.i("save_every", 0);
    int eval_early = (int)a.i("eval_early", 1);
    long long stop_step = a.i("stop_step", 0);   // end early without changing the LR schedule (for early-phase curves)   // extra log-spaced evals early in training, where inits differ most
    uint64_t data_seed = (uint64_t)a.i("data_seed", 1234);
    // weight-space righting force
    std::string anchor_file = a.s("anchor", ""); double anc_lambda = a.f("anchor_lambda", 0.0);
    std::string anc_sched = a.s("anchor_sched", "const"); double anc_end = a.f("anchor_end", 1.0), anc_ema = a.f("anchor_ema", 0.0);
    // functional righting force (concept supervision)
    std::string lab_file = a.s("labels", ""); double cw = a.f("concept_w", 0.0), ctau = a.f("concept_tau", 1.0);
    int clayer = (int)a.i("concept_layer", 0); int n_lab = (int)a.i("n_lab", 0); std::string c_sched = a.s("concept_sched", "const"); double c_end = a.f("concept_end", 1.0);
    std::string probe_file = a.s("probe", "");   // labeled val passages for concept-fidelity metrics
    int fid_layer = (int)a.i("fid_layer", clayer);
    std::string save_path = a.s("save", (run_dir + "/final.bin").c_str());
    a.check();

    long long tok_per_step = (long long)B * c.T * accum;
    long long steps = max_tokens / tok_per_step;
    Acts A; A.alloc(c, B, mode == "train");
    CK(cudaMalloc(&g_ones, std::max(A.N, c.Vp) * 4)); { std::vector<float> o(std::max(A.N, c.Vp), 1.f); CK(cudaMemcpy(g_ones, o.data(), o.size() * 4, cudaMemcpyHostToDevice)); }
    std::vector<uint16_t> val = read_u16("data_bin/val.u16");
    std::vector<uint16_t> train; if (mode == "train") train = read_u16("data_bin/train.u16");
    size_t pbytes = M.lay.total * 4;
    printf("INFO mode=%s d=%d L=%d H=%d m=%d T=%d B=%d accum=%d params=%.2fM (non-emb %.2fM) act_mem=%.2fGB steps=%lld tok/step=%lld\n",
           mode.c_str(), c.d, c.L, c.H, c.m, c.T, B, accum, M.lay.total / 1e6, (M.lay.total - 2.0 * c.Vp * c.d - c.Vp) / 1e6, A.bytes / 1e9, steps, tok_per_step);

    // eval windows (non-overlapping, from start of val)
    long long n_eval = std::min<long long>(eval_tokens / c.T, ((long long)val.size() - 1) / c.T);
    n_eval = n_eval / B * B;
    std::vector<int> htok(A.N), htgt(A.N);
    auto eval_loss = [&]() -> double {
        double s = 0; long long cnt = 0;
        for (long long w0 = 0; w0 < n_eval; w0 += B) {
            for (int b = 0; b < B; b++) fill_window(val, (w0 + b) * c.T, c.T, &htok[b * c.T], &htgt[b * c.T]);
            CK(cudaMemcpy(A.tok, htok.data(), A.N * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(A.tgt, htgt.data(), A.N * 4, cudaMemcpyHostToDevice));
            forward(M, A, 0, 0.f); s += losses_mean(A, tmp) * A.N; cnt += A.N;
        }
        return s / cnt;
    };
    // concept fidelity on probe passages: top-1 agreement between argmax over facet neurons (mean pre-activation
    // over the passage at fid_layer) and Jev's argmax label, plus soft CE
    Labels PR; bool have_probe = !probe_file.empty(); if (have_probe) PR.load(probe_file);
    int* dlen = nullptr; float *om = nullptr, *ox = nullptr;
    if (have_probe) { CK(cudaMalloc(&dlen, B * 4)); CK(cudaMalloc(&om, (size_t)B * c.m * 4)); CK(cudaMalloc(&ox, (size_t)B * c.m * 4)); }
    auto eval_fidelity = [&](std::vector<double>& acc, std::vector<double>& ce) {
        acc.assign(PR.nf, 0); ce.assign(PR.nf, 0); std::vector<float> buf((size_t)B * c.m); std::vector<int> len(B);
        const std::vector<uint16_t>& st = PR.split ? val : train;
        for (int i0 = 0; i0 < PR.n; i0 += B) {
            int nb = std::min(B, PR.n - i0);
            for (int b = 0; b < B; b++) { int j = i0 + std::min(b, nb - 1); fill_window(st, PR.end[j] - c.T, c.T, &htok[b * c.T], &htgt[b * c.T]); len[b] = PR.len[j]; }
            CK(cudaMemcpy(A.tok, htok.data(), A.N * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(A.tgt, htgt.data(), A.N * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dlen, len.data(), B * 4, cudaMemcpyHostToDevice));
            forward(M, A, 0, 0.f);
            aggregate_kernel<<<cdiv((long long)B * c.m, 256), 256>>>(om, ox, A.a + (size_t)fid_layer * A.N * c.m, dlen, B, c.T, c.m);
            CK(cudaMemcpy(buf.data(), om, buf.size() * 4, cudaMemcpyDeviceToHost));
            for (int b = 0; b < nb; b++) for (int f = 0; f < PR.nf; f++) {
                const float* av = &buf[(size_t)b * c.m + PR.foff[f]]; const float* yv = &PR.y[(size_t)(i0 + b) * PR.K + PR.foff[f]];
                int am = 0, ym = 0; float mx = -1e30f; for (int k = 0; k < PR.fsize[f]; k++) { if (av[k] > av[am]) am = k; if (yv[k] > yv[ym]) ym = k; mx = fmaxf(mx, av[k] / (float)ctau); }
                double se = 0; for (int k = 0; k < PR.fsize[f]; k++) se += exp(av[k] / ctau - mx); double lse = mx + log(se);
                double cc = 0, ys = 0; for (int k = 0; k < PR.fsize[f]; k++) { cc += yv[k] * (lse - av[k] / ctau); ys += yv[k]; }
                acc[f] += (am == ym); ce[f] += cc / std::max(ys, 1e-9);
            }
        }
        for (int f = 0; f < PR.nf; f++) { acc[f] /= PR.n; ce[f] /= PR.n; }
    };

    if (mode == "eval") {
        double vl = eval_loss(); printf("EVAL val_loss=%.4f\n", vl);
        if (have_probe) { std::vector<double> ac, ce; eval_fidelity(ac, ce); for (int f = 0; f < PR.nf; f++) printf("FID facet=%d acc=%.4f ce=%.4f\n", f, ac[f], ce[f]); }
        printf("DONE val_loss=%.4f\n", vl); return 0;
    }

    // ---------------- training
    M.alloc_train();
    bool use_anchor = !anchor_file.empty() && anc_lambda > 0;
    if (use_anchor) {
        CK(cudaMalloc(&M.p0, pbytes)); CK(cudaMemcpy(M.p0, M.p, pbytes, cudaMemcpyDeviceToDevice));
        CK(cudaMalloc(&M.anc, pbytes)); CK(cudaMemset(M.anc, 0, pbytes)); load_ckpt(M, anchor_file, M.anc);
    }
    Labels LB; ConceptBatch cb; std::mt19937_64 lrng(seed * 7919 + 13);
    bool use_concept = !lab_file.empty() && cw > 0 && n_lab > 0;
    if (use_concept) {
        LB.load(lab_file);
        if (LB.split != 0) { printf("ERROR concept labels must come from the train split\n"); exit(1); }
        float* dl; CK(cudaMalloc(&dl, LB.y.size() * 4)); CK(cudaMemcpy(dl, LB.y.data(), LB.y.size() * 4, cudaMemcpyHostToDevice));
        cb.labels = dl; cb.nf = LB.nf; cb.layer = clayer; cb.tau = (float)ctau;
        CK(cudaMalloc(&cb.f_off, LB.nf * 4)); CK(cudaMalloc(&cb.f_size, LB.nf * 4)); CK(cudaMalloc(&cb.loss, 4));
        CK(cudaMemcpy(cb.f_off, LB.foff.data(), LB.nf * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(cb.f_size, LB.fsize.data(), LB.nf * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&cb.pos_n, (size_t)n_lab * c.T * 4)); CK(cudaMalloc(&cb.pos_lab, (size_t)n_lab * c.T * 4));
    }
    // data order: fixed permutation of non-overlapping windows
    long long n_win = ((long long)train.size() - 1) / c.T;
    std::vector<int> perm(n_win); for (long long i = 0; i < n_win; i++) perm[i] = (int)i;
    { std::mt19937_64 dr(data_seed); for (long long i = n_win - 1; i > 0; i--) std::swap(perm[i], perm[dr() % (i + 1)]); }
    long long cursor = 0;
    // per-tensor lr multipliers, contiguous by construction
    struct Seg { size_t off, n; float lrm, wd; };
    std::vector<Seg> segs; for (auto& t : M.lay.t) segs.push_back({t.off, t.n, (float)(t.group == 1 ? lr_emb : t.group == 2 ? lr_head : t.group == 3 ? lr_bias : 1.0), (float)(t.wd ? wd : 0.0)});

    std::string mpath = run_dir + "/metrics.json";
    std::string curve = ""; double best = 1e9, last_val = 0, t_train = 0, ema_loss = -1, ema_closs = -1; double t0 = now();
    std::vector<int> pn, pl;
    auto log_eval = [&](long long step) {
        double vl = eval_loss(); last_val = vl; best = std::min(best, vl);
        std::string extra = "", jextra = "";
        if (have_probe) {
            std::vector<double> ac, ce; eval_fidelity(ac, ce);
            for (int f = 0; f < PR.nf; f++) {
                std::string fs = std::to_string(f);
                extra += " fid" + fs + "=" + jnum(ac[f]) + " fce" + fs + "=" + jnum(ce[f]);
                jextra += ",\"fid" + fs + "\":" + jnum(ac[f]) + ",\"fce" + fs + "\":" + jnum(ce[f]);
            }
        }
        if (use_concept) { extra += " concept_ce=" + jnum(ema_closs); jextra += ",\"concept_ce\":" + jnum(ema_closs); }
        if (use_anchor) {
            CK(cudaMemsetAsync(scratch, 0, 4)); sqdiff_kernel<<<256, 512>>>(scratch, M.p, M.p0, M.anc, (long long)M.lay.total);
            float dv; CK(cudaMemcpy(&dv, scratch, 4, cudaMemcpyDeviceToHost)); extra += " anchor_dev=" + jnum(sqrt(dv)); jextra += ",\"anchor_dev\":" + jnum(sqrt(dv));
        }
        long long toks = step * tok_per_step;
        printf("EVAL step=%lld tokens=%lld val_loss=%.4f train_loss=%.4f tok/s=%.0f t=%.0f%s\n", step, toks, vl, ema_loss, step > 0 ? toks / t_train : 0.0, now() - t0, extra.c_str());
        if (!curve.empty()) curve += ",";
        curve += "{\"step\":" + std::to_string(step) + ",\"tokens\":" + std::to_string(toks) + ",\"val_loss\":" + jnum(vl) + ",\"train_loss\":" + jnum(ema_loss) + ",\"time\":" + jnum(t_train) + jextra + "}";
        std::string cfg = "{"; bool first = true; for (auto& p : a.kv) { cfg += (first ? "" : ",") + std::string("\"") + p.first + "\":\"" + p.second + "\""; first = false; } cfg += "}";
        FILE* f = fopen(mpath.c_str(), "w");
        if (f) { fprintf(f, "{\"config\":%s,\"params\":%zu,\"steps\":%lld,\"tok_per_step\":%lld,\"final_val_loss\":%s,\"best_val_loss\":%s,\"wall_time\":%s,\"train_time\":%s,\"tok_per_s\":%s,\"curve\":[%s]}\n",
                         cfg.c_str(), M.lay.total, step, tok_per_step, jnum(vl).c_str(), jnum(best).c_str(), jnum(now() - t0).c_str(), jnum(t_train).c_str(), jnum(step > 0 ? toks / t_train : 0).c_str(), curve.c_str()); fclose(f); }
    };
    log_eval(0);
    std::vector<int> tokh(A.N), tgth(A.N);
    for (long long step = 0; step < steps; step++) {
        double ts = now();
        double frac = (double)step / steps;
        float lr_mult = (step < warmup) ? (float)(step + 1) / warmup : (frac < 1 - warmdown ? 1.f : (float)(1.0 - (frac - (1 - warmdown)) / warmdown * (1 - final_lr)));
        CK(cudaMemsetAsync(M.g, 0, pbytes));
        double lsum = 0; float closs_h = 0;
        for (int mi = 0; mi < accum; mi++) {
            int nl = use_concept ? n_lab : 0; pn.clear(); pl.clear();
            for (int b = 0; b < B; b++) {
                if (b < nl) {
                    int j = (int)(lrng() % LB.n);
                    fill_window(train, LB.end[j] - c.T, c.T, &tokh[b * c.T], &tgth[b * c.T]);
                    for (int t = c.T - LB.len[j]; t < c.T; t++) { pn.push_back(b * c.T + t); pl.push_back(j * LB.K); }
                } else {
                    if (cursor >= n_win) cursor = 0;
                    fill_window(train, (long long)perm[cursor++] * c.T, c.T, &tokh[b * c.T], &tgth[b * c.T]);
                }
            }
            CK(cudaMemcpy(A.tok, tokh.data(), A.N * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(A.tgt, tgth.data(), A.N * 4, cudaMemcpyHostToDevice));
            forward(M, A, 1, 1.f / (A.N * accum));
            lsum += losses_mean(A, tmp);
            ConceptBatch* cbp = nullptr;
            if (use_concept) {
                cb.npos = (int)pn.size(); cb.w = (float)(cw * sched(c_sched, frac, c_end) / std::max<size_t>(1, pn.size()) / accum);
                CK(cudaMemcpy(cb.pos_n, pn.data(), pn.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(cb.pos_lab, pl.data(), pl.size() * 4, cudaMemcpyHostToDevice));
                CK(cudaMemsetAsync(cb.loss, 0, 4)); cbp = &cb;
            }
            backward(M, A, cbp);
            if (use_concept && cb.w > 0 && cb.npos > 0) { float cl; CK(cudaMemcpy(&cl, cb.loss, 4, cudaMemcpyDeviceToHost)); closs_h += cl / cb.w / cb.npos / accum; }
        }
        if (use_concept && closs_h > 0) ema_closs = ema_closs < 0 ? closs_h : 0.98 * ema_closs + 0.02 * closs_h;
        // grad clip
        float gn = sqrtf(dev_sum(M.g, (long long)M.lay.total, scratch));
        if (!std::isfinite(gn)) { printf("NAN step=%lld grad_norm=%f\n", step, gn); exit(1); }
        float gscale = (clip > 0 && gn > clip) ? (float)(clip / gn) : 1.f;
        float bc1 = 1.f - (float)pow(b1, step + 1), bc2 = 1.f - (float)pow(b2, step + 1);
        float anc_scale = use_anchor ? (float)(anc_lambda * sched(anc_sched, frac, anc_end)) : 0.f;
        for (auto& s : segs)
            adamw_kernel<<<cdiv((long long)s.n, 256), 256>>>(M.p + s.off, M.g + s.off, M.mo + s.off, M.vo + s.off, use_anchor ? M.p0 + s.off : nullptr,
                                                             use_anchor ? M.anc + s.off : nullptr, (long long)s.n, (float)(lr * lr_mult * s.lrm), (float)b1, (float)b2, bc1, bc2, 1e-8f, s.wd, gscale, anc_scale);
        if (use_anchor && anc_ema > 0) ema_kernel<<<cdiv((long long)M.lay.total, 256), 256>>>(M.p0, M.p, M.anc, (long long)M.lay.total, (float)anc_ema);
        double l = lsum / accum; ema_loss = ema_loss < 0 ? l : 0.98 * ema_loss + 0.02 * l;
        if (!std::isfinite(l)) { printf("NAN step=%lld loss=%f\n", step, l); exit(1); }
        CK(cudaDeviceSynchronize()); t_train += now() - ts;
        long long s1 = step + 1;
        bool early = eval_early && s1 < eval_every && (s1 == 10 || s1 == 25 || s1 == 50 || s1 == 100 || s1 == 150);
        if (s1 % eval_every == 0 || s1 == steps || early) log_eval(s1);
        if (save_every > 0 && (step + 1) % save_every == 0) save_ckpt(M, run_dir + "/ckpt_" + std::to_string(step + 1) + ".bin", M.p);
        if (stop_step > 0 && s1 >= stop_step) { if (s1 % eval_every != 0) log_eval(s1); steps = s1; break; }
    }
    save_ckpt(M, save_path, M.p);
    printf("DONE final_val_loss=%.4f best_val_loss=%.4f steps=%lld tokens=%lld train_time=%.0f tok/s=%.0f\n", last_val, best, steps, steps * tok_per_step, t_train, steps * tok_per_step / t_train);
    return 0;
}
