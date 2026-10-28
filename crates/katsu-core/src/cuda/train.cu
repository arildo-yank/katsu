// Treino do Katsu LM inteiro na GPU: forward, backward e Adam sem passar pelo host.
// FP32, mesmas fórmulas do backend CPU (ops.rs): LayerNorm com eps 1e-5 e 1/sqrt, GELU tanh,
// atenção causal com escala 1/sqrt(dh) e softmax por log-sum-exp.
// Ponteiros chegam como endereços crus da arena; índices em long long para não estourar int.

#define NEG_INF (-3.402823466e+38F)
#define GELU_C 0.7978846f
#define GELU_K 0.044715f
#define LN_EPS 1e-5f

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// Redução no bloco inteiro (blockDim múltiplo de 32); `sh` tem 32 floats.
__device__ float block_reduce(float v, int is_max, float* sh) {
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = is_max ? warp_max(v) : warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float r = threadIdx.x < nw ? sh[threadIdx.x] : (is_max ? NEG_INF : 0.0f);
    if (w == 0) r = is_max ? warp_max(r) : warp_sum(r);
    if (threadIdx.x == 0) sh[0] = r;
    __syncthreads();
    return sh[0];
}
__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(GELU_C * (x + GELU_K * x * x * x)));
}
__device__ __forceinline__ float gelu_grad(float x) {
    float t = tanhf(GELU_C * (x + GELU_K * x * x * x));
    float du = GELU_C * (1.0f + 3.0f * GELU_K * x * x);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * du;
}

extern "C" __global__ void fill0(float* x, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.0f;
}
extern "C" __global__ void add2(const float* a, const float* b, float* out, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
extern "C" __global__ void copy(const float* a, float* out, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i];
}

// x = wte[token] + wpe[posição], posição = linha % t.
extern "C" __global__ void embed_fwd(const int* tok, const float* wte, const float* wpe, float* out,
                                     int rows, int t, int d) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)rows * d) return;
    int r = (int)(i / d), c = (int)(i % d);
    out[i] = wte[(long long)tok[r] * d + c] + wpe[(long long)(r % t) * d + c];
}
extern "C" __global__ void embed_bwd(const float* dx, const int* tok, float* dwte, float* dwpe,
                                     int rows, int t, int d) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)rows * d) return;
    int r = (int)(i / d), c = (int)(i % d);
    float g = dx[i];
    if (g == 0.0f) return;
    atomicAdd(dwte + (long long)tok[r] * d + c, g);
    atomicAdd(dwpe + (long long)(r % t) * d + c, g);
}

// Arandu-Ve: as n primeiras posições de cada sequência recebem vetores (prefixo, [b·n, d]) no
// lugar do embedding de token: x = pre[sequência, p] + wpe[p]. Roda depois do embed_fwd.
extern "C" __global__ void prefix_fwd(const float* pre, const float* wpe, float* out, int rows,
                                      int t, int n, int d) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)rows * d) return;
    int r = (int)(i / d), c = (int)(i % d), p = r % t;
    if (p >= n) return;
    out[i] = pre[((long long)(r / t) * n + p) * d + c] + wpe[(long long)p * d + c];
}
// embed_bwd com prefixo: nas posições do prefixo o gradiente vai para o vetor (dpre, escrito
// por inteiro a cada lote) e para o embedding de posição, nunca para a linha do token de
// preenchimento; nas demais, como no embed_bwd.
extern "C" __global__ void embed_bwd_prefix(const float* dx, const int* tok, float* dwte,
                                            float* dwpe, float* dpre, int rows, int t, int n,
                                            int d) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)rows * d) return;
    int r = (int)(i / d), c = (int)(i % d), p = r % t;
    float g = dx[i];
    if (p < n) {
        dpre[((long long)(r / t) * n + p) * d + c] = g;
        if (g != 0.0f) atomicAdd(dwpe + (long long)p * d + c, g);
        return;
    }
    if (g == 0.0f) return;
    atomicAdd(dwte + (long long)tok[r] * d + c, g);
    atomicAdd(dwpe + (long long)p * d + c, g);
}

// Um warp por linha.
extern "C" __global__ void layernorm_fwd(const float* x, const float* g, const float* b, float* out,
                                         float* mean, float* rstd, int rows, int d) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= rows) return;
    const float* xr = x + (long long)row * d;
    float s = 0.0f;
    for (int c = lane; c < d; c += 32) s += xr[c];
    float m = warp_sum(s) / d;
    float v = 0.0f;
    for (int c = lane; c < d; c += 32) { float z = xr[c] - m; v += z * z; }
    float rs = 1.0f / sqrtf(warp_sum(v) / d + LN_EPS);
    if (lane == 0) { mean[row] = m; rstd[row] = rs; }
    float* o = out + (long long)row * d;
    for (int c = lane; c < d; c += 32) o[c] = (xr[c] - m) * rs * g[c] + b[c];
}
// dx += rstd·(dx̂ − média(dx̂) − x̂·média(dx̂·x̂)), dx̂ = dout·γ; dγ += dout·x̂, dβ += dout.
// Um warp por linha; γ e β acumulam em memória compartilhada e vão uma vez por bloco.
extern "C" __global__ void layernorm_bwd(const float* dout, const float* x, const float* mean,
                                         const float* rstd, const float* g, float* dx, float* dg,
                                         float* db, int rows, int d) {
    extern __shared__ float sh[];
    float* sdg = sh;
    float* sdb = sh + d;
    for (int c = threadIdx.x; c < d; c += blockDim.x) { sdg[c] = 0.0f; sdb[c] = 0.0f; }
    __syncthreads();
    int wpb = blockDim.x >> 5, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int r = blockIdx.x * wpb + w; r < rows; r += gridDim.x * wpb) {
        const float* xr = x + (long long)r * d;
        const float* dr = dout + (long long)r * d;
        float m = mean[r], rs = rstd[r], a = 0.0f, bx = 0.0f;
        for (int c = lane; c < d; c += 32) {
            float xh = (xr[c] - m) * rs, dxh = dr[c] * g[c];
            a += dxh;
            bx += dxh * xh;
        }
        a = warp_sum(a) / d;
        bx = warp_sum(bx) / d;
        float* o = dx + (long long)r * d;
        for (int c = lane; c < d; c += 32) {
            float xh = (xr[c] - m) * rs, dxh = dr[c] * g[c];
            o[c] += rs * (dxh - a - xh * bx);
            atomicAdd(sdg + c, dr[c] * xh);
            atomicAdd(sdb + c, dr[c]);
        }
    }
    __syncthreads();
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        atomicAdd(dg + c, sdg[c]);
        atomicAdd(db + c, sdb[c]);
    }
}

// Epílogo comum dos GEMMs (Etapa 2: fusão). v = A·B da posição o = linha·ldc + col:
//   + bias[col]; epi 1: guarda v em pre e aplica GELU; epi 2: multiplica por gelu'(pre[o]);
//   com res: v = res[o] + v (resíduo, na ordem do host: x + f(x)); acc: C[o] += v.
__device__ __forceinline__ void epilogue(float v, long long o, int col, float* C, const float* bias,
                                         const float* res, float* pre, int epi, int acc) {
    if (bias) v += bias[col];
    if (epi == 1) { pre[o] = v; v = gelu(v); }
    else if (epi == 2) v *= gelu_grad(pre[o]);
    if (res) v = res[o] + v;
    C[o] = acc ? C[o] + v : v;
}

// C[M,N] (+)= op(A)[M,K]·op(B)[K,N], row-major, com o epílogo acima.
// op(A)[i,k] = ta ? A[k·lda+i] : A[i·lda+k]. Referência em tiles 16×16 (a da Etapa 1).
extern "C" __global__ void gemm16(const float* A, const float* B, float* C, const float* bias,
                                  const float* res, float* pre, int epi, int M, int N, int K,
                                  int lda, int ldb, int ldc, int ta, int tb, int acc) {
    __shared__ float As[16][17], Bs[16][17];
    int row = blockIdx.y * 16 + threadIdx.y, col = blockIdx.x * 16 + threadIdx.x;
    float v = 0.0f;
    for (int t0 = 0; t0 < K; t0 += 16) {
        int ak = t0 + threadIdx.x, bk = t0 + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < M && ak < K) ? (ta ? A[(long long)ak * lda + row] : A[(long long)row * lda + ak]) : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (col < N && bk < K) ? (tb ? B[(long long)col * ldb + bk] : B[(long long)bk * ldb + col]) : 0.0f;
        __syncthreads();
        for (int p = 0; p < 16; ++p) v += As[threadIdx.y][p] * Bs[p][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) epilogue(v, (long long)row * ldc + col, col, C, bias, res, pre, epi, acc);
}

// Etapa 3: tiles 64×64 com K em fatias de 16; 256 threads, cada uma com 4×4 resultados em
// registradores (16 FMAs por par de leituras da memória compartilhada, contra 1 no gemm16).
extern "C" __global__ void __launch_bounds__(256)
gemm64(const float* A, const float* B, float* C, const float* bias, const float* res, float* pre,
       int epi, int M, int N, int K, int lda, int ldb, int ldc, int ta, int tb, int acc) {
    __shared__ float As[16][64 + 4];
    __shared__ float Bs[16][64 + 4];
    int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    int r0 = blockIdx.y * 64, c0 = blockIdx.x * 64;
    float sum[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c) sum[r][c] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 16) {
        // carga na ordem contígua da memória global (transposta ou não)
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            int e = tid + l * 256, i, kk;
            if (ta) { kk = e >> 6; i = e & 63; } else { i = e >> 4; kk = e & 15; }
            int gi = r0 + i, gk = k0 + kk;
            As[kk][i] = (gi < M && gk < K) ? (ta ? A[(long long)gk * lda + gi] : A[(long long)gi * lda + gk]) : 0.0f;
        }
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            int e = tid + l * 256, j, kk;
            if (tb) { j = e >> 4; kk = e & 15; } else { kk = e >> 6; j = e & 63; }
            int gj = c0 + j, gk = k0 + kk;
            Bs[kk][j] = (gj < N && gk < K) ? (tb ? B[(long long)gj * ldb + gk] : B[(long long)gk * ldb + gj]) : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            float a[4], b[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) a[r] = As[kk][ty * 4 + r];
#pragma unroll
            for (int c = 0; c < 4; ++c) b[c] = Bs[kk][tx * 4 + c];
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < 4; ++c) sum[r][c] += a[r] * b[c];
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            int gi = r0 + ty * 4 + r, gj = c0 + tx * 4 + c;
            if (gi < M && gj < N)
                epilogue(sum[r][c], (long long)gi * ldc + gj, gj, C, bias, res, pre, epi, acc);
        }
}

extern "C" __global__ void bias_add(float* x, const float* bias, long long rows, int cols) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < rows * cols) x[i] += bias[i % cols];
}
// out[c] += Σ_r x[r,c]; blocos 32×8, linhas divididas em gridDim.y fatias.
extern "C" __global__ void colsum_acc(const float* x, float* out, int rows, int cols) {
    __shared__ float s[8][33];
    int c = blockIdx.x * 32 + threadIdx.x;
    int per = (rows + gridDim.y - 1) / gridDim.y, r0 = blockIdx.y * per, r1 = min(rows, r0 + per);
    float a = 0.0f;
    if (c < cols)
        for (int r = r0 + threadIdx.y; r < r1; r += 8) a += x[(long long)r * cols + c];
    s[threadIdx.y][threadIdx.x] = a;
    __syncthreads();
    if (threadIdx.y == 0 && c < cols) {
        float t = 0.0f;
        for (int k = 0; k < 8; ++k) t += s[k][threadIdx.x];
        atomicAdd(out + c, t);
    }
}
extern "C" __global__ void gelu_fwd(const float* x, float* out, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) out[i] = gelu(x[i]);
}
// d (no lugar) ← d·gelu'(pre).
extern "C" __global__ void gelu_bwd(float* d, const float* pre, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) d[i] *= gelu_grad(pre[i]);
}

// Atenção causal. qkv por linha: [q (d) | k (d) | v (d)], cabeça h nas colunas h·dh.
// att: probabilidades [B,H,T,T]; out: [B·T, d] com as cabeças lado a lado.
// Um bloco por (linha i, cabeça, exemplo); um warp por chave j nos produtos escalares.
extern "C" __global__ void attn_fwd(const float* qkv, float* att, float* out, int T, int H, int dh,
                                    float scale) {
    extern __shared__ float sh[];
    float* red = sh;
    float* q = sh + 32;
    float* s = q + dh;
    int i = blockIdx.x, h = blockIdx.y, b = blockIdx.z, d = H * dh;
    long long rs = 3LL * d;
    const float* base = qkv + (long long)b * T * rs;
    for (int c = threadIdx.x; c < dh; c += blockDim.x) q[c] = base[i * rs + h * dh + c];
    __syncthreads();
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int j = w; j <= i; j += nw) {
        const float* k = base + j * rs + d + h * dh;
        float a = 0.0f;
        for (int c = lane; c < dh; c += 32) a += q[c] * k[c];
        a = warp_sum(a);
        if (lane == 0) s[j] = a * scale;
    }
    __syncthreads();
    float mx = NEG_INF;
    for (int j = threadIdx.x; j <= i; j += blockDim.x) mx = fmaxf(mx, s[j]);
    mx = block_reduce(mx, 1, red);
    float sum = 0.0f;
    for (int j = threadIdx.x; j <= i; j += blockDim.x) sum += expf(s[j] - mx);
    float ln_sum = logf(block_reduce(sum, 0, red));
    float* prow = att + (((long long)b * H + h) * T + i) * T;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        float p = j <= i ? expf((s[j] - mx) - ln_sum) : 0.0f;
        prow[j] = p;
        if (j <= i) s[j] = p;
    }
    __syncthreads();
    for (int c = threadIdx.x; c < dh; c += blockDim.x) {
        float a = 0.0f;
        for (int j = 0; j <= i; ++j) a += s[j] * base[j * rs + 2 * d + h * dh + c];
        out[((long long)b * T + i) * d + h * dh + c] = a;
    }
}
// Backward por linha i: dP_ij = dO_i·v_j, dS_ij = P_ij(dP_ij − Σ_j P_ij dP_ij), dQ_i = escala·Σ_j dS_ij k_j.
// Grava dS [B,H,T,T] para o kernel por coluna e dQ em dqkv.
extern "C" __global__ void attn_bwd_rows(const float* qkv, const float* att, const float* dout,
                                         float* ds, float* dqkv, int T, int H, int dh, float scale) {
    extern __shared__ float sh[];
    float* red = sh;
    float* go = sh + 32;
    float* p = go + dh;
    float* dp = p + T;
    int i = blockIdx.x, h = blockIdx.y, b = blockIdx.z, d = H * dh;
    long long rs = 3LL * d;
    const float* base = qkv + (long long)b * T * rs;
    const float* prow = att + (((long long)b * H + h) * T + i) * T;
    for (int c = threadIdx.x; c < dh; c += blockDim.x) go[c] = dout[((long long)b * T + i) * d + h * dh + c];
    for (int j = threadIdx.x; j <= i; j += blockDim.x) p[j] = prow[j];
    __syncthreads();
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int j = w; j <= i; j += nw) {
        const float* v = base + j * rs + 2 * d + h * dh;
        float a = 0.0f;
        for (int c = lane; c < dh; c += 32) a += go[c] * v[c];
        a = warp_sum(a);
        if (lane == 0) dp[j] = a;
    }
    __syncthreads();
    float dot = 0.0f;
    for (int j = threadIdx.x; j <= i; j += blockDim.x) dot += p[j] * dp[j];
    dot = block_reduce(dot, 0, red);
    float* dsrow = ds + (((long long)b * H + h) * T + i) * T;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        float v = j <= i ? p[j] * (dp[j] - dot) : 0.0f;
        dsrow[j] = v;
        if (j <= i) dp[j] = v;
    }
    __syncthreads();
    for (int c = threadIdx.x; c < dh; c += blockDim.x) {
        float a = 0.0f;
        for (int j = 0; j <= i; ++j) a += dp[j] * base[j * rs + d + h * dh + c];
        dqkv[((long long)b * T + i) * rs + h * dh + c] = a * scale;
    }
}
// Backward por coluna j: dV_j = Σ_{i≥j} P_ij dO_i, dK_j = escala·Σ_{i≥j} dS_ij q_i.
extern "C" __global__ void attn_bwd_cols(const float* qkv, const float* att, const float* ds,
                                         const float* dout, float* dqkv, int T, int H, int dh,
                                         float scale) {
    int j = blockIdx.x, h = blockIdx.y, b = blockIdx.z, d = H * dh;
    long long rs = 3LL * d;
    const float* base = qkv + (long long)b * T * rs;
    const float* pcol = att + ((long long)b * H + h) * T * T + j;
    const float* scol = ds + ((long long)b * H + h) * T * T + j;
    for (int c = threadIdx.x; c < dh; c += blockDim.x) {
        float dv = 0.0f, dk = 0.0f;
        for (int i = j; i < T; ++i) {
            dv += pcol[(long long)i * T] * dout[((long long)b * T + i) * d + h * dh + c];
            dk += scol[(long long)i * T] * base[i * rs + h * dh + c];
        }
        dqkv[((long long)b * T + j) * rs + d + h * dh + c] = dk * scale;
        dqkv[((long long)b * T + j) * rs + 2 * d + h * dh + c] = dv;
    }
}

// Entropia cruzada ponderada por linha: loss_r = w_r·(ln Σ e^(x−max) − (x_alvo − max)).
// Com grad, troca os logits por w_r·(softmax − one-hot) no lugar. Peso 0: perda 0, gradiente 0.
extern "C" __global__ void ce_fwd_bwd(float* logits, const int* tgt, const float* wt, float* losses,
                                      int V, int grad) {
    __shared__ float red[32];
    int r = blockIdx.x;
    float* x = logits + (long long)r * V;
    float w = wt[r];
    if (w == 0.0f) {
        if (threadIdx.x == 0) losses[r] = 0.0f;
        if (grad)
            for (int c = threadIdx.x; c < V; c += blockDim.x) x[c] = 0.0f;
        return;
    }
    float mx = NEG_INF;
    for (int c = threadIdx.x; c < V; c += blockDim.x) mx = fmaxf(mx, x[c]);
    mx = block_reduce(mx, 1, red);
    float sum = 0.0f;
    for (int c = threadIdx.x; c < V; c += blockDim.x) sum += expf(x[c] - mx);
    sum = block_reduce(sum, 0, red);
    int t = tgt[r];
    float target = x[t];
    __syncthreads();
    if (threadIdx.x == 0) losses[r] = w * (logf(sum) - (target - mx));
    if (grad)
        for (int c = threadIdx.x; c < V; c += blockDim.x)
            x[c] = w * (expf(x[c] - mx) / sum - (c == t ? 1.0f : 0.0f));
}

// Adam no lugar, mesma ordem de operações do AdamStep::apply; zera o gradiente.
extern "C" __global__ void adam(float* w, float* g, float* m, float* v, long long count, float beta1,
                                float beta2, float lr, float eps, float c1, float c2,
                                unsigned int* nonfinite) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= count) return;
    float gi = g[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * gi;
    float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    float mh = mi / c1, vh = vi / c2;
    float wi = w[i] - lr * mh / (sqrtf(vh) + eps);
    w[i] = wi;
    g[i] = 0.0f;
    if (!isfinite(wi)) atomicAdd(nonfinite, 1u);
}
// Adam com máscara de congelamento: peso com mask 0 não muda nem acumula momento; o
// gradiente dele é zerado, como o dos demais.
extern "C" __global__ void adam_masked(float* w, float* g, float* m, float* v,
                                       const unsigned char* mask, long long count, float beta1,
                                       float beta2, float lr, float eps, float c1, float c2,
                                       unsigned int* nonfinite) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (!mask[i]) {
        g[i] = 0.0f;
        return;
    }
    float gi = g[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * gi;
    float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    float mh = mi / c1, vh = vi / c2;
    float wi = w[i] - lr * mh / (sqrtf(vh) + eps);
    w[i] = wi;
    g[i] = 0.0f;
    if (!isfinite(wi)) atomicAdd(nonfinite, 1u);
}
// V2: Adam de 8 bits (adam8.rs), um bloco de 256 threads por bloco de quantização da arena.
// Dequantiza com a escala antiga, faz a conta em FP32 e requantiza com o máximo novo do bloco.
// As operações vão com _rn explícito para o compilador não fundir em FMA: com o mesmo bloco,
// o resultado é igual bit a bit ao do Adam8bit da CPU (conferido em gpu_train.rs).
// Com máscara, o peso congelado não muda e o momento dele só é requantizado.
__device__ __forceinline__ float dec_m(signed char q, float s) {
    float r = __fdiv_rn((float)q, 127.0f);
    return __fmul_rn(__fmul_rn(r, fabsf(r)), s);
}
// √v quantizado: u = r²·escala, v = u².
__device__ __forceinline__ float dec_u(unsigned char q, float s) {
    float r = __fdiv_rn((float)q, 255.0f);
    return __fmul_rn(__fmul_rn(r, r), s);
}
__device__ __forceinline__ signed char enc_m(float x, float s) {
    if (s == 0.0f) return 0;
    float q = roundf(__fmul_rn(__fmul_rn(127.0f, copysignf(1.0f, x)), __fsqrt_rn(__fdiv_rn(fabsf(x), s))));
    return (signed char)fminf(fmaxf(q, -127.0f), 127.0f);
}
__device__ __forceinline__ unsigned char enc_u(float u, float s) {
    if (s == 0.0f) return 0;
    float q = roundf(__fmul_rn(255.0f, __fsqrt_rn(__fdiv_rn(u, s))));
    return (unsigned char)fminf(fmaxf(q, 0.0f), 255.0f);
}
extern "C" __global__ void __launch_bounds__(256)
adam8(float* w, float* g, signed char* qm, unsigned char* qv, float* sm, float* su,
      const unsigned char* mask, long long count, float beta1, float beta2, float lr, float eps,
      float c1, float c2, unsigned int* nonfinite) {
    __shared__ float red[32];
    long long i = blockIdx.x * 256LL + threadIdx.x;
    int live = i < count;
    float om = sm[blockIdx.x], ou = su[blockIdx.x];
    float mi = 0.0f, ui = 0.0f;
    if (live) {
        float m0 = dec_m(qm[i], om), u0 = dec_u(qv[i], ou);
        float gi = g[i];
        g[i] = 0.0f;
        if (mask && !mask[i]) {
            mi = m0;
            ui = u0;
        } else {
            mi = __fadd_rn(__fmul_rn(beta1, m0), __fmul_rn(1.0f - beta1, gi));
            float vi = __fadd_rn(__fmul_rn(beta2, __fmul_rn(u0, u0)), __fmul_rn(__fmul_rn(1.0f - beta2, gi), gi));
            float step = __fdiv_rn(__fmul_rn(lr, __fdiv_rn(mi, c1)), __fadd_rn(__fsqrt_rn(__fdiv_rn(vi, c2)), eps));
            float wi = __fsub_rn(w[i], step);
            w[i] = wi;
            if (!isfinite(wi)) atomicAdd(nonfinite, 1u);
            ui = __fsqrt_rn(vi);
        }
    }
    float max_m = block_reduce(fabsf(mi), 1, red);
    float max_u = block_reduce(ui, 1, red);
    if (threadIdx.x == 0) { sm[blockIdx.x] = max_m; su[blockIdx.x] = max_u; }
    if (live) { qm[i] = enc_m(mi, max_m); qv[i] = enc_u(ui, max_u); }
}
// Soma dos quadrados do gradiente em double, por bloco (256 threads); conta não finitos.
extern "C" __global__ void sumsq(const float* g, long long count, double* partial,
                                 unsigned int* nonfinite) {
    __shared__ double s[256];
    double x = 0.0;
    for (long long j = blockIdx.x * (long long)blockDim.x + threadIdx.x; j < count;
         j += (long long)gridDim.x * blockDim.x) {
        float value = g[j];
        if (isfinite(value)) x += (double)value * (double)value;
        else atomicAdd(nonfinite, 1u);
    }
    s[threadIdx.x] = x;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = s[0];
}
