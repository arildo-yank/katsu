// Kernels de referência FP32. Sem fast-math; GEMM em tiles 16x16.
extern "C" __global__ void gemm(const float* a,const float* b,float* c,int m,int k,int n,int ta,int tb) {
    __shared__ float aa[16][16],bb[16][16];
    int row=blockIdx.y*16+threadIdx.y,col=blockIdx.x*16+threadIdx.x;
    float value=0.0f;
    for(int tile=0;tile<k;tile+=16) {
        int ak=tile+threadIdx.x,bk=tile+threadIdx.y;
        aa[threadIdx.y][threadIdx.x]=(row<m && ak<k)?a[ta?ak*m+row:row*k+ak]:0.0f;
        bb[threadIdx.y][threadIdx.x]=(col<n && bk<k)?b[tb?col*k+bk:bk*n+col]:0.0f;
        __syncthreads();
        for(int p=0;p<16;++p) value+=aa[threadIdx.y][p]*bb[p][threadIdx.x];
        __syncthreads();
    }
    if(row<m && col<n)c[row*n+col]=value;
}
extern "C" __global__ void add(const float* a,const float* b,float* out,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)out[i]=a[i]+b[i];
}
extern "C" __global__ void add_backward(const float* g,const unsigned long long* a,const unsigned long long* b,float* da,float* db,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count){atomicAdd(da+a[i],g[i]);atomicAdd(db+b[i],g[i]);}
}
// Uma thread por linha: baseline de correção, não uma redução CUDA otimizada.
extern "C" __global__ void cross_entropy(const float* x,const unsigned long long* labels,float* losses,int rows,int classes) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=rows)return;
    float maxval=-3.402823466e+38F;
    for(int c=0;c<classes;++c)maxval=fmaxf(maxval,x[r*classes+c]);
    float sum=0.0f;for(int c=0;c<classes;++c)sum+=expf(x[r*classes+c]-maxval);
    losses[r]=logf(sum)-(x[r*classes+labels[r]]-maxval);
}
extern "C" __global__ void ce_backward(const float* x,const unsigned long long* labels,float* out,int rows,int classes,float seed) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=rows)return;
    float maxval=-3.402823466e+38F;
    for(int c=0;c<classes;++c)maxval=fmaxf(maxval,x[r*classes+c]);
    float sum=0.0f;for(int c=0;c<classes;++c)sum+=expf(x[r*classes+c]-maxval);
    for(int c=0;c<classes;++c)out[r*classes+c]=seed*(expf(x[r*classes+c]-maxval)/sum-(c==labels[r]?1.0f:0.0f))/rows;
}
