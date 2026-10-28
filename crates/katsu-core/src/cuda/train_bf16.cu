// V2: GEMM nos Tensor Cores em BF16 (WMMA m16n16k16, acumulando em FP32), anexado ao train.cu
// só quando pedido (precisa de mma.h e cuda_bf16.h, e de GPU sm_80+). Mesmo arranjo do gemm_tc:
// tile 64×64, 4 warps com 32×32 cada. As entradas FP32 viram BF16 na carga (par mais próximo,
// como f32_to_bf16 do host); pesos-mestre, gradientes e Adam continuam FP32 na arena.
// Mistura igual à do Bf16Backend da CPU: sem `acc` (ativação ou gradiente de ativação) o
// produto volta a BF16 antes do epílogo; com `acc` (gradiente de peso) fica FP32.
#include <cuda_bf16.h>
#include <mma.h>

extern "C" __global__ void __launch_bounds__(128)
gemm_bf16(const float* A, const float* B, float* C, const float* bias, const float* res, float* pre,
          int epi, int M, int N, int K, int lda, int ldb, int ldc, int ta, int tb, int acc) {
    using namespace nvcuda;
    __shared__ __align__(32) __nv_bfloat16 As[64][16 + 8];
    __shared__ __align__(32) __nv_bfloat16 Bs[16][64 + 8];
    __shared__ __align__(32) float Cs[64][64 + 4];
    int tid = threadIdx.x, warp = tid >> 5;
    int wr = (warp >> 1) * 32, wc = (warp & 1) * 32;
    int r0 = blockIdx.y * 64, c0 = blockIdx.x * 64;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> sum[2][2];
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
            As[i][kk] = __float2bfloat16_rn(v);
        }
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            int e = tid + l * 128, j, kk;
            if (tb) { j = e >> 4; kk = e & 15; } else { kk = e >> 6; j = e & 63; }
            int gj = c0 + j, gk = k0 + kk;
            float v = (gj < N && gk < K) ? (tb ? B[(long long)gj * ldb + gk] : B[(long long)gk * ldb + gj]) : 0.0f;
            Bs[kk][j] = __float2bfloat16_rn(v);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(af[i], &As[wr + i * 16][0], 16 + 8);
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(bf[j], &Bs[0][wc + j * 16], 64 + 8);
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j) wmma::mma_sync(sum[i][j], af[i], bf[j], sum[i][j]);
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
        if (gi >= M || gj >= N) continue;
        float v = Cs[i][j];
        if (!acc) v = __bfloat162float(__float2bfloat16_rn(v));
        epilogue(v, (long long)gi * ldc + gj, gj, C, bias, res, pre, epi, acc);
    }
}
