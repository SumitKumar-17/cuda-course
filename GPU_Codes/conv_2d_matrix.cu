// conv2d_compare.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>

#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(err)); exit(1);} } while(0)

static void fill_random(float* a, int n) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n; ++i) a[i] = dist(rng);
}

// Simple CPU 2D convolution
void conv2d_cpu(const float* img, const float* kernel, float* out, int H, int W, int K) {
    int R = K/2;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            float sum = 0;
            for (int ki = -R; ki <= R; ki++) {
                for (int kj = -R; kj <= R; kj++) {
                    int ni = i + ki;
                    int nj = j + kj;
                    if (ni >= 0 && ni < H && nj >= 0 && nj < W) {
                        sum += img[ni*W + nj] * kernel[(ki+R)*K + (kj+R)];
                    }
                }
            }
            out[i*W + j] = sum;
        }
    }
}

// GPU 2D convolution with shared memory tiling
template<int TILE, int K>
__global__ void conv2d_gpu(const float* __restrict__ img, const float* __restrict__ kernel, float* __restrict__ out, int H, int W) {
    __shared__ float tile[TILE + K-1][TILE + K-1];

    int R = K/2;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row_o = blockIdx.y * TILE + ty;
    int col_o = blockIdx.x * TILE + tx;
    int row_i = row_o - R;
    int col_i = col_o - R;

    // Load into shared memory
    if (row_i >= 0 && row_i < H && col_i >= 0 && col_i < W)
        tile[ty][tx] = img[row_i*W + col_i];
    else
        tile[ty][tx] = 0.0f;
    __syncthreads();

    if (ty < TILE && tx < TILE && row_o < H && col_o < W) {
        float val = 0;
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++)
                val += kernel[i*K + j] * tile[ty + i][tx + j];
        out[row_o*W + col_o] = val;
    }
}

static double l2_diff(const float* X, const float* Y, int n) {
    double s = 0;
    for (int i = 0; i < n; ++i) {
        double d = X[i] - Y[i];
        s += d*d;
    }
    return std::sqrt(s);
}

int main(int argc, char** argv) {
    int H = 1024, W = 1024, K = 5;
    if (argc >= 3) { H = atoi(argv[1]); W = atoi(argv[2]); }

    size_t img_bytes = H*W*sizeof(float);
    size_t ker_bytes = K*K*sizeof(float);

    float *h_img = (float*)malloc(img_bytes);
    float *h_out_cpu = (float*)calloc(H*W,sizeof(float));
    float *h_out_gpu = (float*)calloc(H*W,sizeof(float));
    float *h_kernel = (float*)malloc(ker_bytes);

    fill_random(h_img, H*W);

    // Gaussian kernel
    float sigma = 1.0f; float sum = 0;
    for (int i=0;i<K;i++){
        for(int j=0;j<K;j++){
            int x = i - K/2, y = j - K/2;
            h_kernel[i*K+j] = expf(-(x*x+y*y)/(2*sigma*sigma));
            sum += h_kernel[i*K+j];
        }
    }
    for (int i=0;i<K*K;i++) h_kernel[i]/=sum;

    // CPU
    auto t0 = std::chrono::high_resolution_clock::now();
    conv2d_cpu(h_img, h_kernel, h_out_cpu, H, W, K);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double,std::milli>(t1-t0).count();

    // GPU
    float *d_img, *d_out, *d_kernel;
    CUDA_CHECK(cudaMalloc(&d_img,img_bytes));
    CUDA_CHECK(cudaMalloc(&d_out,img_bytes));
    CUDA_CHECK(cudaMalloc(&d_kernel,ker_bytes));
    CUDA_CHECK(cudaMemcpy(d_img,h_img,img_bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel,h_kernel,ker_bytes,cudaMemcpyHostToDevice));

    cudaEvent_t e0,e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    dim3 block(16,16);
    dim3 grid((W+15)/16,(H+15)/16);

    cudaEventRecord(e0);
    conv2d_gpu<16,5><<<grid,block>>>(d_img,d_kernel,d_out,H,W);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    float gpu_ms=0; cudaEventElapsedTime(&gpu_ms,e0,e1);
    CUDA_CHECK(cudaMemcpy(h_out_gpu,d_out,img_bytes,cudaMemcpyDeviceToHost));

    double err = l2_diff(h_out_cpu,h_out_gpu,H*W);

    std::cout<<std::fixed<<std::setprecision(3);
    std::cout<<"===== 2D Convolution "<<H<<"x"<<W<<" with "<<K<<"x"<<K<<" kernel =====\n";
    std::cout<<"CPU time: "<<cpu_ms<<" ms\n";
    std::cout<<"GPU time: "<<gpu_ms<<" ms\n";
    std::cout<<"Speedup: "<<cpu_ms/gpu_ms<<"x\n";
    std::cout<<"L2 error: "<<err<<"\n";

    cudaFree(d_img); cudaFree(d_out); cudaFree(d_kernel);
    free(h_img); free(h_out_cpu); free(h_out_gpu); free(h_kernel);
}
