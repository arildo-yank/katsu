// Etapa 3b: GEMM nos Tensor Cores (WMMA m16n16k8 em TF32, acumulando em FP32), anexado ao
// train.cu só quando pedido (precisa de mma.h e de GPU sm_80+). Tile 64×64, 4 warps com 32×32
// cada; entradas convertidas para TF32 na carga, mesmo epílogo dos GEMMs FP32.
#include <mma.h>

extern "C" __global__ void __launch_bounds__(128)
gemm_tc(const float* A, const float* B, float* C, const float* bias, const float* res, float* pre,
        int epi, int M, int N, int K, int lda, int ldb, int ldc, int ta, int tb, int acc) {
    using namespace nvcuda;
    __shared__ __align__(32) float As[64][16 + 4];
    __shared__ __align__(32) float Bs[16][64 + 4];
    __shared__ __align__(32) float Cs[64][64 + 4];
    int tid = threadIdx.x, warp = tid >> 5;
    int wr = (warp >> 1) * 32, wc = (warp & 1) * 32;
    int r0 = blockIdx.y * 64, c0 = blockIdx.x * 64;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> sum[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(sum[i][j], 0.0f);
    for (int k0 = 0; k0 < K; k0 += 16) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            int e = tid + l * 128, i, kk;
            if (ta) { kk = e >> 6; i = e & 63; } else { i = e >> 4; kk = e & 15; }
            int gi = r0 + i, gk = k0 + kk;
            float v = (gi < M && gk < K) ? (ta ? A[(long long)gk * lda + gi] : A[(long long)gi * lda + gk]) : 0.0f;
            As[i][kk] = wmma::__float_to_tf32(v);
        }
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            int e = tid + l * 128, j, kk;
            if (tb) { j = e >> 4; kk = e & 15; } else { kk = e >> 6; j = e & 63; }
            int gj = c0 + j, gk = k0 + kk;
            float v = (gj < N && gk < K) ? (tb ? B[(long long)gj * ldb + gk] : B[(long long)gk * ldb + gj]) : 0.0f;
            Bs[kk][j] = wmma::__float_to_tf32(v);
        }
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < 16; ks += 8) {
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> bf[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(af[i], &As[wr + i * 16][ks], 16 + 4);
#pragma unroll
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(bf[j], &Bs[ks][wc + j * 16], 64 + 4);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(sum[i][j], af[i], bf[j], sum[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(&Cs[wr + i * 16][wc + j * 16], sum[i][j], 64 + 4, wmma::mem_row_major);
    __syncthreads();
    for (int e = tid; e < 64 * 64; e += 128) {
        int i = e >> 6, j = e & 63, gi = r0 + i, gj = c0 + j;
        if (gi < M && gj < N) epilogue(Cs[i][j], (long long)gi * ldc + gj, gj, C, bias, res, pre, epi, acc);
    }
}
