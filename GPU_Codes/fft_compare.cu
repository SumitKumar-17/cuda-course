// fft_compare.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include <cufft.h>

#define CUDA_CHECK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d -> %s\n",#x,__FILE__,__LINE__,cudaGetErrorString(e)); \
  exit(1);}}while(0)

static inline bool is_power_of_two(size_t n){ return n && ((n&(n-1))==0); }

__device__ __forceinline__ float2 cadd(float2 a, float2 b){ return make_float2(a.x+b.x, a.y+b.y); }
__device__ __forceinline__ float2 csub(float2 a, float2 b){ return make_float2(a.x-b.x, a.y-b.y); }
__device__ __forceinline__ float2 cmul(float2 a, float2 b){
    return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

__global__ void bit_reverse_permute(float2* data, int n, int log2n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    unsigned ri = __brev((unsigned)i) >> (32 - log2n);
    if(ri > (unsigned)i){
        float2 tmp = data[i];
        data[i] = data[ri];
        data[ri] = tmp;
    }
}

__global__ void fft_stage(float2* data, int n, int m, int half){
    // Each thread handles one butterfly
    int t = blockIdx.x * blockDim.x + threadIdx.x;   // 0 .. n/2-1
    int total = n >> 1;
    if(t >= total) return;

    const float PI = 3.14159265358979323846f;
    int block = t / half;          // which group of size m
    int j     = t % half;          // butterfly index within group
    int k     = block * m;         // start index of group

    int i1 = k + j;
    int i2 = i1 + half;

    float angle = -2.0f * PI * j / m;  // forward FFT
    float s, c;
    __sincosf(angle, &s, &c);
    float2 W = make_float2(c, s);

    float2 a = data[i1];
    float2 b = data[i2];
    float2 tval = cmul(W, b);
    data[i1] = cadd(a, tval);
    data[i2] = csub(a, tval);
}

static void fill_random(std::vector<float2>& h){
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for(auto& z : h){ z.x = dist(rng); z.y = dist(rng); }
}

static double rms_error(const std::vector<float2>& A, const std::vector<float2>& B){
    long double s = 0.0L;
    for(size_t i=0;i<A.size();++i){
        long double dx = (long double)A[i].x - B[i].x;
        long double dy = (long double)A[i].y - B[i].y;
        s += dx*dx + dy*dy;
    }
    return std::sqrt((double)(s / (long double)A.size()));
}

int main(int argc, char** argv){
    int N = 1<<20; // 1,048,576 by default
    if(argc >= 2) N = std::atoi(argv[1]);
    if(!is_power_of_two(N) || N < 2){
        std::cerr << "N must be a power of two >= 2\n";
        return 1;
    }
    int log2N = (int)std::round(std::log2((double)N));

    // Host input
    std::vector<float2> h_in(N);
    fill_random(h_in);

    // Device buffers
    float2 *d_data_custom=nullptr, *d_data_cufft=nullptr;
    CUDA_CHECK(cudaMalloc(&d_data_custom, N*sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_data_cufft,  N*sizeof(float2)));
    CUDA_CHECK(cudaMemcpy(d_data_custom, h_in.data(), N*sizeof(float2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data_cufft,  h_in.data(), N*sizeof(float2), cudaMemcpyHostToDevice));

    // --- cuFFT reference ---
    cufftHandle plan;
    if(cufftPlan1d(&plan, N, CUFFT_C2C, 1) != CUFFT_SUCCESS){
        std::cerr << "cufftPlan1d failed\n"; return 1;
    }

    cudaEvent_t e0,e1,e2,e3;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2));
    CUDA_CHECK(cudaEventCreate(&e3));

    // cuFFT timing
    CUDA_CHECK(cudaEventRecord(e0));
    if(cufftExecC2C(plan, (cufftComplex*)d_data_cufft, (cufftComplex*)d_data_cufft, CUFFT_FORWARD) != CUFFT_SUCCESS){
        std::cerr << "cufftExecC2C failed\n"; return 1;
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float cufft_ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&cufft_ms, e0, e1));

    // --- Custom CUDA FFT ---
    dim3 block(256);
    dim3 grid_perm((N + block.x - 1)/block.x);
    // bit-reversal
    CUDA_CHECK(cudaEventRecord(e2));
    bit_reverse_permute<<<grid_perm, block>>>(d_data_custom, N, log2N);
    // iterative stages
    for(int s=1; s<=log2N; ++s){
        int m = 1<<s;
        int half = m>>1;
        int total = N>>1;
        dim3 grid_stage((total + block.x - 1)/block.x);
        fft_stage<<<grid_stage, block>>>(d_data_custom, N, m, half);
    }
    CUDA_CHECK(cudaEventRecord(e3));
    CUDA_CHECK(cudaEventSynchronize(e3));
    float custom_ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&custom_ms, e2, e3));

    // Copy back and compare
    std::vector<float2> h_custom(N), h_cufft(N);
    CUDA_CHECK(cudaMemcpy(h_custom.data(), d_data_custom, N*sizeof(float2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cufft.data(),  d_data_cufft,  N*sizeof(float2), cudaMemcpyDeviceToHost));

    double err = rms_error(h_custom, h_cufft);

    std::cout<<std::fixed<<std::setprecision(3);
    std::cout<<"===== 1D FFT (complex, forward) =====\n";
    std::cout<<"N="<<N<<" (log2N="<<log2N<<")\n";
    std::cout<<"cuFFT time:   "<<cufft_ms<<" ms\n";
    std::cout<<"Custom time:  "<<custom_ms<<" ms\n";
    std::cout<<"Speedup (cuFFT / custom): "<<(custom_ms>0? cufft_ms/custom_ms : 0)<<"x ( >1 means cuFFT faster )\n";
    std::cout<<"RMS error (custom vs cuFFT): "<<err<<"\n";

    // Pretty print first few bins
    int show = std::min(8, N);
    std::cout<<"\nFirst "<<show<<" output bins (cuFFT | custom):\n";
    for(int i=0;i<show;i++){
        std::cout<<"k="<<std::setw(3)<<i<<"  "
                 <<"["<<std::setw(9)<<h_cufft[i].x<<","<<std::setw(9)<<h_cufft[i].y<<"]  |  "
                 <<"["<<std::setw(9)<<h_custom[i].x<<","<<std::setw(9)<<h_custom[i].y<<"]\n";
    }

    cufftDestroy(plan);
    CUDA_CHECK(cudaFree(d_data_custom));
    CUDA_CHECK(cudaFree(d_data_cufft));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaEventDestroy(e2));
    CUDA_CHECK(cudaEventDestroy(e3));
    return 0;
}
