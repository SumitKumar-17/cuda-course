// matrix_mul_compare.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>

#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(err)); exit(1);} } while(0)

static void fill_random(float* a, int n) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (int i = 0; i < n; ++i) a[i] = dist(rng);
}

static void matmul_cpu(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            float aik = A[i*N + k];
            for (int j = 0; j < N; ++j) {
                C[i*N + j] += aik * B[k*N + j];
            }
        }
    }
}

template<int TILE>
__global__ void matmul_gpu_tiled(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int m = 0; m < (N + TILE - 1) / TILE; ++m) {
        int a_row = row;
        int a_col = m * TILE + threadIdx.x;
        int b_row = m * TILE + threadIdx.y;
        int b_col = col;

        As[threadIdx.y][threadIdx.x] = (a_row < N && a_col < N) ? A[a_row * N + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < N && b_col < N) ? B[b_row * N + b_col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int e = 0; e < TILE; ++e) acc += As[threadIdx.y][e] * Bs[e][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N) C[row * N + col] = acc;
}

static double l2_diff(const float* X, const float* Y, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = (double)X[i] - (double)Y[i];
        s += d*d;
    }
    return std::sqrt(s);
}

static void print_matrix(const char* name, const float* M, int N) {
    std::cout << name << " (" << N << "x" << N << "):\n";
    std::cout.setf(std::ios::fixed); std::cout << std::setprecision(2);
    for (int i = 0; i < N; ++i) {
        std::cout << "  [ ";
        for (int j = 0; j < N; ++j) {
            std::cout << std::setw(8) << M[i*N + j] << (j+1==N? " " : " ");
        }
        std::cout << "]\n";
    }
}

int main(int argc, char** argv) {
    int N = 512;                  // default for speed test
    bool pretty = false;          // print matrices if small
    if (argc >= 2) N = std::atoi(argv[1]);
    if (argc >= 3) pretty = std::atoi(argv[2]) != 0;
    if (N <= 0) { std::cerr << "N must be > 0\n"; return 1; }

    size_t bytes = (size_t)N * N * sizeof(float);
    float* hA = (float*)malloc(bytes);
    float* hB = (float*)malloc(bytes);
    float* hC_cpu = (float*)calloc(N*N, sizeof(float));
    float* hC_gpu = (float*)calloc(N*N, sizeof(float));
    if (!hA || !hB || !hC_cpu || !hC_gpu) { std::cerr << "Host alloc failed\n"; return 1; }

    fill_random(hA, N*N);
    fill_random(hB, N*N);

    // CPU timing
    auto t0 = std::chrono::high_resolution_clock::now();
    matmul_cpu(hA, hB, hC_cpu, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // GPU memory
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, bytes));

    // GPU launch
    const int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    CUDA_CHECK(cudaEventRecord(e0));
    matmul_gpu_tiled<TILE><<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, e0, e1));

    CUDA_CHECK(cudaMemcpy(hC_gpu, dC, bytes, cudaMemcpyDeviceToHost));

    // Check correctness
    double err = l2_diff(hC_cpu, hC_gpu, N*N);
    double rel = err / (std::max(1.0, l2_diff(hC_cpu, hC_cpu, N*N) + 1e-9)); // denominator ~0 -> 1

    // Pretty output for small N or if pretty flag true
    if (pretty || N <= 8) {
        int P = std::min(N, 8);
        if (N != P) std::cout << "(Showing top-left " << P << "x" << P << " submatrices)\n\n";
        print_matrix("A", hA, P);
        print_matrix("B", hB, P);
        print_matrix("C_cpu = A*B", hC_cpu, P);
        print_matrix("C_gpu", hC_gpu, P);
        std::cout << "\n";
    }

    // GFLOPs estimate: 2*N^3 ops
    double ops = 2.0 * (double)N * (double)N * (double)N;
    double cpu_gflops = ops / (cpu_ms/1000.0) / 1e9;
    double gpu_gflops = ops / (gpu_ms/1000.0) / 1e9;

    std::cout.setf(std::ios::fixed); std::cout << std::setprecision(3);
    std::cout << "===== Matrix Multiply " << N << "x" << N << " =====\n";
    std::cout << "CPU time: " << cpu_ms << " ms  |  Throughput: " << cpu_gflops << " GFLOP/s\n";
    std::cout << "GPU time: " << gpu_ms << " ms  |  Throughput: " << gpu_gflops << " GFLOP/s\n";
    std::cout << "Speedup (CPU/GPU): " << (cpu_ms / gpu_ms) << "x\n";
    std::cout << "L2 error (CPU vs GPU): " << err << "\n";
    if (err < 1e-2 * N) std::cout << "Result: ✅ OK (close match)\n";
    else std::cout << "Result: ⚠️  Large deviation\n";

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    free(hA); free(hB); free(hC_cpu); free(hC_gpu);
    return 0;
}
