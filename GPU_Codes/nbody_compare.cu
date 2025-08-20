#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>

#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(err)); exit(1);} } while(0)

struct Body { float x,y,z,m; }; // position + mass
struct Vel  { float vx,vy,vz,pad; };

static void init_bodies(Body* pos, Vel* vel, int N, float spread=100.f) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    for (int i=0;i<N;i++) {
        pos[i].x = d(rng)*spread; pos[i].y = d(rng)*spread; pos[i].z = d(rng)*spread; pos[i].m = fabsf(d(rng))*10.f + 1.f;
        vel[i].vx = d(rng); vel[i].vy = d(rng); vel[i].vz = d(rng); vel[i].pad=0.f;
    }
}

static void cpu_step(Body* p, Vel* v, int N, float dt, float G, float eps2) {
    // compute accelerations
    std::vector<float> ax(N,0), ay(N,0), az(N,0);
    for (int i=0;i<N;i++) {
        float aix=0, aiy=0, aiz=0;
        for (int j=0;j<N;j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float r2 = dx*dx + dy*dy + dz*dz + eps2;
            float inv = rsqrtf(r2);         // 1/sqrt(r2)
            float inv3 = inv*inv*inv;       // 1/r^3
            float s = G * p[j].m * inv3;
            aix += dx * s; aiy += dy * s; aiz += dz * s;
        }
        ax[i]=aix; ay[i]=aiy; az[i]=aiz;
    }
    // integrate (Euler)
    for (int i=0;i<N;i++) {
        v[i].vx += ax[i]*dt; v[i].vy += ay[i]*dt; v[i].vz += az[i]*dt;
        p[i].x  += v[i].vx*dt; p[i].y  += v[i].vy*dt; p[i].z  += v[i].vz*dt;
    }
}

template<int TILE>
__global__ void gpu_step(Body* p, Vel* v, int N, float dt, float G, float eps2) {
    extern __shared__ Body sh[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float3 pos_i, vel_i;
    if (i < N) {
        pos_i = make_float3(p[i].x, p[i].y, p[i].z);
        vel_i = make_float3(v[i].vx, v[i].vy, v[i].vz);
    }
    float3 acc = make_float3(0,0,0);

    for (int tile=0; tile<N; tile+=TILE) {
        int j = tile + threadIdx.x;
        if (j < N) sh[threadIdx.x] = p[j];
        __syncthreads();

        int limit = min(TILE, N - tile);
        #pragma unroll
        for (int k=0;k<limit;k++) {
            float dx = sh[k].x - pos_i.x;
            float dy = sh[k].y - pos_i.y;
            float dz = sh[k].z - pos_i.z;
            float r2 = dx*dx + dy*dy + dz*dz + eps2;
            float inv = rsqrtf(r2);
            float inv3 = inv*inv*inv;
            float s = G * sh[k].m * inv3;
            acc.x += dx*s; acc.y += dy*s; acc.z += dz*s;
        }
        __syncthreads();
    }

    if (i < N) {
        vel_i.x += acc.x * dt; vel_i.y += acc.y * dt; vel_i.z += acc.z * dt;
        pos_i.x += vel_i.x * dt; pos_i.y += vel_i.y * dt; pos_i.z += vel_i.z * dt;
        v[i].vx = vel_i.x; v[i].vy = vel_i.y; v[i].vz = vel_i.z;
        p[i].x = pos_i.x;  p[i].y = pos_i.y;  p[i].z = pos_i.z;
    }
}

static double l2_pos_diff(const Body* A, const Body* B, int N) {
    long double s=0;
    for (int i=0;i<N;i++) {
        long double dx = (long double)A[i].x - B[i].x;
        long double dy = (long double)A[i].y - B[i].y;
        long double dz = (long double)A[i].z - B[i].z;
        s += dx*dx + dy*dy + dz*dz;
    }
    return std::sqrt((double)s);
}

static void print_sample(const Body* P, const Vel* V, int N, int K=5) {
    int n = std::min(N, K);
    std::cout << "Sample bodies (x,y,z | vx,vy,vz | m):\n";
    std::cout << std::fixed << std::setprecision(3);
    for (int i=0;i<n;i++) {
        std::cout << "  " << std::setw(3) << i << ": "
                  << std::setw(8)<<P[i].x<<","<<std::setw(8)<<P[i].y<<","<<std::setw(8)<<P[i].z<<" | "
                  << std::setw(8)<<V[i].vx<<","<<std::setw(8)<<V[i].vy<<","<<std::setw(8)<<V[i].vz<<" | "
                  << "m="<<P[i].m << "\n";
    }
}

int main(int argc, char** argv) {
    // Args: N steps dt [pretty]
    int N = 8192;
    int steps = 10;
    float dt = 1e-3f;
    bool pretty=false;
    if (argc>=2) N = std::atoi(argv[1]);
    if (argc>=3) steps = std::atoi(argv[2]);
    if (argc>=4) dt = std::atof(argv[3]);
    if (argc>=5) pretty = std::atoi(argv[4])!=0;

    const float G = 6.67408e-3f; // scaled grav constant for stability
    const float eps2 = 1e-2f;    // softening to avoid singularities

    // Host allocations
    Body *hPos_cpu=(Body*)malloc(N*sizeof(Body)), *hPos_gpu=(Body*)malloc(N*sizeof(Body)), *hPos_init=(Body*)malloc(N*sizeof(Body));
    Vel  *hVel_cpu=(Vel*)malloc(N*sizeof(Vel)),   *hVel_gpu=(Vel*)malloc(N*sizeof(Vel)),   *hVel_init=(Vel*)malloc(N*sizeof(Vel));
    if(!hPos_cpu||!hPos_gpu||!hVel_cpu||!hVel_gpu||!hPos_init||!hVel_init){ std::cerr<<"Host alloc failed\n"; return 1; }

    init_bodies(hPos_init, hVel_init, N);
    std::memcpy(hPos_cpu,hPos_init,N*sizeof(Body));
    std::memcpy(hVel_cpu,hVel_init,N*sizeof(Vel));
    std::memcpy(hPos_gpu,hPos_init,N*sizeof(Body));
    std::memcpy(hVel_gpu,hVel_init,N*sizeof(Vel));

    // --- CPU run ---
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int s=0;s<steps;s++) cpu_step(hPos_cpu, hVel_cpu, N, dt, G, eps2);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double,std::milli>(t1-t0).count();

    // --- GPU run ---
    Body *dPos; Vel *dVel;
    CUDA_CHECK(cudaMalloc(&dPos, N*sizeof(Body)));
    CUDA_CHECK(cudaMalloc(&dVel, N*sizeof(Vel)));
    CUDA_CHECK(cudaMemcpy(dPos, hPos_gpu, N*sizeof(Body), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVel, hVel_gpu, N*sizeof(Vel), cudaMemcpyHostToDevice));

    const int TILE = 256;
    dim3 block(TILE);
    dim3 grid((N + TILE - 1)/TILE);
    size_t shmem = TILE * sizeof(Body);

    cudaEvent_t e0,e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    for (int s=0;s<steps;s++)
        gpu_step<TILE><<<grid, block, shmem>>>(dPos, dVel, N, dt, G, eps2);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float gpu_ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&gpu_ms,e0,e1));

    CUDA_CHECK(cudaMemcpy(hPos_gpu, dPos, N*sizeof(Body), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hVel_gpu, dVel, N*sizeof(Vel), cudaMemcpyDeviceToHost));

    // correctness (position L2)
    double pos_err = l2_pos_diff(hPos_cpu, hPos_gpu, N);

    // interactions per step ~ N^2, total ops rough (not FLOPs): 20 flops approx per pair
    long double pairs = (long double)N * (long double)N;
    long double flops_per_pair = 20.0L;
    long double total_flops = flops_per_pair * pairs * steps;
    double cpu_gflops = (double)(total_flops / ((cpu_ms/1000.0)*1e9));
    double gpu_gflops = (double)(total_flops / ((gpu_ms/1000.0)*1e9));

    std::cout<<std::fixed<<std::setprecision(3);
    std::cout<<"===== CUDA N-Body =====\n";
    std::cout<<"Bodies: "<<N<<", Steps: "<<steps<<", dt: "<<dt<<"\n";
    std::cout<<"CPU time: "<<cpu_ms<<" ms  | est throughput: "<<cpu_gflops<<" GFLOP/s\n";
    std::cout<<"GPU time: "<<gpu_ms<<" ms  | est throughput: "<<gpu_gflops<<" GFLOP/s\n";
    std::cout<<"Speedup (CPU/GPU): "<<(cpu_ms/gpu_ms)<<"x\n";
    std::cout<<"L2 position error (CPU vs GPU): "<<pos_err<<"\n";

    if (pretty || N<=16) {
        std::cout<<"\n--- CPU state sample ---\n";
        print_sample(hPos_cpu, hVel_cpu, N);
        std::cout<<"--- GPU state sample ---\n";
        print_sample(hPos_gpu, hVel_gpu, N);
    }

    CUDA_CHECK(cudaFree(dPos)); CUDA_CHECK(cudaFree(dVel));
    free(hPos_cpu); free(hVel_cpu); free(hPos_gpu); free(hVel_gpu); free(hPos_init); free(hVel_init);
    return 0;
}
