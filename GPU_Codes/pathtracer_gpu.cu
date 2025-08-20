// pathtracer_gpu.cu
// Heavy Monte Carlo GPU path tracer (spheres, many bounces, many spheres)
// Compile: nvcc -O3 pathtracer_gpu.cu -o pathtracer_gpu
// Run: ./pathtracer_gpu <W> <H> <spp> <bounces> <spheres_per_side>
// Example: ./pathtracer_gpu 1024 768 128 6 40

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)

struct Vec { float x,y,z; __host__ __device__ Vec():x(0),y(0),z(0){} __host__ __device__ Vec(float a):x(a),y(a),z(a){} __host__ __device__ Vec(float X,float Y,float Z):x(X),y(Y),z(Z){} };
__host__ __device__ inline Vec operator+(const Vec &a,const Vec &b){ return Vec(a.x+b.x,a.y+b.y,a.z+b.z); }
__host__ __device__ inline Vec operator-(const Vec &a,const Vec &b){ return Vec(a.x-b.x,a.y-b.y,a.z-b.z); }
__host__ __device__ inline Vec operator*(const Vec &a,const Vec &b){ return Vec(a.x*b.x,a.y*b.y,a.z*b.z); }
__host__ __device__ inline Vec operator*(const Vec &a,float s){ return Vec(a.x*s,a.y*s,a.z*s); }
__host__ __device__ inline Vec operator*(float s,const Vec &a){ return a*s; }
__host__ __device__ inline Vec operator/(const Vec &a,float s){ float r=1.0f/s; return a*r; }
__host__ __device__ inline float dot(const Vec &a,const Vec &b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
__host__ __device__ inline Vec cross(const Vec &a,const Vec &b){ return Vec(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x); }
__host__ __device__ inline float len(const Vec &a){ return sqrtf(dot(a,a)); }
__host__ __device__ inline Vec normalize(const Vec &a){ float L=len(a); return L>0? a/L : a; }

struct Sphere {
    Vec center;
    float radius;
    Vec emission; // emissive color
    Vec color;    // surface color (albedo)
    int is_light;
};

struct Camera {
    Vec pos, forward, right, up;
    float fov;
};

// XOROSHIRO128+ style small PRNG (deterministic, fast)
struct RNG {
    uint64_t s0, s1;
    __host__ __device__ RNG(uint64_t seed = 123456789ULL) { 
        // simple splitmix64 seeding
        uint64_t z = (seed += 0x9e3779b97f4a7c15ULL);
        auto split = [&](uint64_t &x){ x += 0x9e3779b97f4a7c15ULL; uint64_t z = x; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; return z ^ (z >> 31); };
        s0 = split(z); s1 = split(z);
        if (s0==0 && s1==0) { s0 = 0xdeadbeefULL; s1 = 0x12345678ULL; }
    }
    __host__ __device__ inline uint64_t next_u64() {
        uint64_t x = s0;
        uint64_t y = s1;
        s0 = y;
        x ^= x << 23;
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        return s1 + y;
    }
    __host__ __device__ inline float next_float() {
        // convert to [0,1)
        return (next_u64() >> 11) * (1.0f/9007199254740992.0f); // 53-bit precision
    }
};

// Ray-sphere intersection
__host__ __device__ inline bool hit_sphere(const Sphere &s, const Vec &ro, const Vec &rd, float &t, Vec &n) {
    Vec oc = ro - s.center;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - s.radius*s.radius;
    float disc = b*b - c;
    if (disc <= 0.0f) return false;
    float sq = sqrtf(disc);
    float t0 = -b - sq;
    float t1 = -b + sq;
    float tt = (t0 > 1e-4f) ? t0 : ((t1 > 1e-4f) ? t1 : -1.0f);
    if (tt < 0.0f) return false;
    t = tt;
    Vec hitp = ro + rd * t;
    n = normalize(hitp - s.center);
    return true;
}

// Cosine-weighted hemisphere sample (local frame), returns direction
__host__ __device__ inline Vec cosine_sample_hemisphere(RNG &rng) {
    float u1 = rng.next_float();
    float u2 = rng.next_float();
    float r = sqrtf(u1);
    float phi = 2.0f * 3.14159265358979323846f * u2;
    float x = r * cosf(phi), y = r * sinf(phi);
    float z = sqrtf(fmaxf(0.0f, 1.0f - u1));
    return Vec(x, y, z);
}

// Build an orthonormal basis given normal n (n is assumed normalized)
__host__ __device__ inline void make_onb(const Vec &n, Vec &u, Vec &v) {
    if (fabsf(n.x) > fabsf(n.y)) {
        float inv = 1.0f / sqrtf(n.x*n.x + n.z*n.z);
        u = Vec(-n.z * inv, 0.0f, n.x * inv);
    } else {
        float inv = 1.0f / sqrtf(n.y*n.y + n.z*n.z);
        u = Vec(0.0f, n.z * inv, -n.y * inv);
    }
    v = cross(n, u);
}

// sample hemisphere aligned to normal
__host__ __device__ inline Vec sample_hemisphere(const Vec &n, RNG &rng) {
    Vec u,v;
    make_onb(n, u, v);
    Vec s = cosine_sample_hemisphere(rng);
    return normalize(u * s.x + v * s.y + n * s.z);
}

// core path-trace per-ray function (returns radiance)
__host__ __device__ Vec trace_path(const Vec &ro, const Vec &rd, const Sphere *spheres, int nspheres, RNG &rng, int max_bounces) {
    Vec throughput = Vec(1.0f,1.0f,1.0f);
    Vec L = Vec(0.0f,0.0f,0.0f);
    Vec origin = ro;
    Vec dir = rd;
    for (int bounce=0; bounce<max_bounces; ++bounce) {
        float tmin = 1e20f; int hit_i = -1; Vec n;
        float t; Vec nn;
        for (int i=0;i<nspheres;++i) {
            if (hit_sphere(spheres[i], origin, dir, t, nn)) {
                if (t < tmin) { tmin = t; hit_i = i; n = nn; }
            }
        }
        if (hit_i == -1) {
            // environment (black)
            break;
        }
        const Sphere &s = spheres[hit_i];
        Vec hitp = origin + dir * tmin;
        // accumulate emission
        if (s.is_light) {
            // direct hit light -> add emission and terminate (light is assumed delta-ish)
            L = L + throughput * s.emission;
            break;
        }
        // diffuse lambertian: sample hemisphere
        Vec new_dir = sample_hemisphere(n, rng);
        origin = hitp + n * 1e-4f; // offset
        dir = new_dir;
        // BRDF = albedo/pi, cosine sampled PDF = cos/PI, so throughput *= albedo
        throughput = throughput * s.color;
        // Russian roulette after a few bounces
        if (bounce > 2) {
            float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            if (rng.next_float() > p) break;
            throughput = throughput / p;
        }
    }
    return L;
}

// Camera generate primary ray
__host__ __device__ inline Vec uniform_rand_in_unit() { return Vec( (float)rand()/RAND_MAX, (float)rand()/RAND_MAX, 0.0f ); }

__global__ void render_kernel(Vec *accum, int W, int H, int spp, int max_bounces, Sphere *spheres, int nspheres, Camera cam, uint64_t seed_base) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;
    RNG rng(seed_base + idx * 1315423911ULL + (blockIdx.z<<32)); // unique-ish
    Vec color_accum = Vec(0,0,0);
    // pixel center coords in NDC
    float aspect = float(W)/float(H);
    float scale = tanf(cam.fov * 0.5f * (3.14159265358979323846f/180.0f));
    for (int s=0; s<spp; ++s) {
        // stratified jitter
        float u = ( (x + rng.next_float()) / (float)W ) * 2.0f - 1.0f;
        float v = ( (y + rng.next_float()) / (float)H ) * 2.0f - 1.0f;
        u *= aspect * scale; v *= scale;
        Vec dir = normalize(cam.forward + cam.right * u + cam.up * v);
        Vec L = trace_path(cam.pos, dir, spheres, nspheres, rng, max_bounces);
        color_accum = color_accum + L;
    }
    Vec prev = accum[idx];
    // accumulate and store running average (in-place accumulation)
    Vec newval = prev + color_accum;
    accum[idx] = newval;
}

// cpu single-threaded path tracer for tiny tests (very slow)
Vec cpu_trace_path(const Vec &ro, const Vec &rd, const std::vector<Sphere> &spheres, uint64_t &seed, int max_bounces) {
    // simple xor shift RNG for CPU
    auto next = [&]()->uint64_t { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };
    auto nextf = [&]()->float { return (next() & 0xFFFFFFFFULL) / (float)0xFFFFFFFFULL; };
    Vec throughput = Vec(1.0f,1.0f,1.0f), L=Vec(0,0,0);
    Vec origin = ro, dir = rd;
    for (int bounce=0; bounce<max_bounces; ++bounce) {
        float tmin = 1e20f; int hit_i = -1; Vec n;
        float t; Vec nn;
        for (size_t i=0;i<spheres.size();++i) {
            Vec oc = origin - spheres[i].center;
            float b = dot(oc, dir);
            float c = dot(oc, oc) - spheres[i].radius*spheres[i].radius;
            float disc = b*b - c;
            if (disc <= 0.0f) continue;
            float sq = sqrtf(disc);
            float t0 = -b - sq; float t1 = -b + sq;
            float tt = (t0 > 1e-4f) ? t0 : ((t1 > 1e-4f) ? t1 : -1.0f);
            if (tt < 0.0f) continue;
            if (tt < tmin) { tmin = tt; hit_i = (int)i; Vec hitp = origin + dir * tt; n = normalize(hitp - spheres[i].center); }
        }
        if (hit_i == -1) break;
        const Sphere &s = spheres[hit_i];
        Vec hitp = origin + dir * tmin;
        if (s.is_light) { L = L + throughput * s.emission; break; }
        // sample hemisphere (cosine)
        // build basis
        Vec nnx = n;
        Vec u,v;
        if (fabsf(nnx.x) > fabsf(nnx.y)) { float inv = 1.0f / sqrtf(nnx.x*nnx.x + nnx.z*nnx.z); u = Vec(-nnx.z*inv, 0, nnx.x*inv); } else { float inv = 1.0f / sqrtf(nnx.y*nnx.y + nnx.z*nnx.z); u = Vec(0, nnx.z*inv, -nnx.y*inv); }
        v = cross(nnx, u);
        // cosine sample
        float u1 = nextf(), u2 = nextf();
        float r = sqrtf(u1); float phi = 2.0f * 3.14159265358979323846f * u2;
        float sx = r * cosf(phi), sy = r * sinf(phi), sz = sqrtf(fmaxf(0.0f, 1.0f - u1));
        Vec new_dir = normalize(u*sx + v*sy + nnx*sz);
        origin = hitp + n*1e-4f; dir = new_dir;
        throughput = throughput * s.color;
        if (bounce > 2) {
            float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            if (nextf() > p) break;
            throughput = throughput / p;
        }
    }
    return L;
}

// Helper to write PPM
void write_ppm(const char *fname, const std::vector<Vec> &accum, int W, int H, int spp) {
    FILE *f = fopen(fname, "wb");
    if (!f) { perror("fopen"); return; }
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (int y=H-1;y>=0;--y) {
        for (int x=0;x<W;++x) {
            int i = y*W + x;
            Vec c = accum[i] / (float)spp;
            // simple gamma 1/2.2
            auto tone = [](float v)->unsigned char {
                v = fmaxf(0.0f, v);
                v = powf(v, 1.0f/2.2f);
                int iv = (int)(255.0f * v + 0.5f);
                if (iv < 0) iv = 0; if (iv > 255) iv = 255;
                return (unsigned char)iv;
            };
            unsigned char rgb[3] = { tone(c.x), tone(c.y), tone(c.z) };
            fwrite(rgb, 1, 3, f);
        }
    }
    fclose(f);
}

int main(int argc, char **argv) {
    int W = 800, H = 600, spp = 64, max_bounces = 6, sside = 30;
    if (argc >= 3) { W = atoi(argv[1]); H = atoi(argv[2]); }
    if (argc >= 4) spp = atoi(argv[3]);
    if (argc >= 5) max_bounces = atoi(argv[4]);
    if (argc >= 6) sside = atoi(argv[5]);
    std::cout << "Render " << W << "x" << H << " spp=" << spp << " bounces=" << max_bounces << " spheres_side=" << sside << "\n";

    // Build scene: a grid of spheres, some emissive lights
    int nspheres = sside * sside;
    std::vector<Sphere> host_spheres;
    host_spheres.reserve(nspheres + 2);
    float spacing = 1.2f;
    float start = - (sside-1) * spacing * 0.5f;
    for (int i=0;i<sside;i++){
        for (int j=0;j<sside;j++){
            Sphere s;
            s.center = Vec(start + i*spacing, -1.0f + ( (j&1)?0.05f: -0.05f ), start + j*spacing*0.6f);
            s.radius = 0.45f + 0.15f * ( (i*j)&3 ) * 0.2f;
            s.color = Vec(0.2f + 0.8f * (i%3==0), 0.2f + 0.8f * (j%3==1), 0.2f + 0.8f * ((i+j)%5==0));
            s.emission = Vec(0,0,0);
            s.is_light = 0;
            host_spheres.push_back(s);
        }
    }
    // Add a big ground sphere
    Sphere ground; ground.center = Vec(0.0f, -1001.0f, 0.0f); ground.radius = 1000.0f; ground.color = Vec(0.8f,0.8f,0.8f); ground.emission = Vec(0,0,0); ground.is_light = 0;
    host_spheres.push_back(ground);
    // Add a few small emissive spheres (lights)
    Sphere light1; light1.center = Vec(0.0f, 6.5f, -2.0f); light1.radius = 1.0f; light1.emission = Vec(40,30,20); light1.color = Vec(1,1,1); light1.is_light = 1; host_spheres.push_back(light1);
    Sphere light2; light2.center = Vec(-6.0f, 5.0f, 6.0f); light2.radius = 1.0f; light2.emission = Vec(15,25,35); light2.color = Vec(1,1,1); light2.is_light = 1; host_spheres.push_back(light2);

    int total_spheres = (int)host_spheres.size();
    std::cout << "Total spheres: " << total_spheres << "\n";

    // Camera
    Camera cam;
    cam.pos = Vec(0.0f, 2.0f, 18.0f);
    Vec look = Vec(0.0f, 0.0f, 0.0f) - cam.pos;
    cam.forward = normalize(look);
    cam.right = normalize(cross(cam.forward, Vec(0,1,0)));
    cam.up = normalize(cross(cam.right, cam.forward));
    cam.fov = 40.0f;

    // Allocate accum buffer on GPU (store sum of radiance per pixel, then divided by spp)
    Vec *d_accum = nullptr;
    Vec *d_spheres = nullptr;
    size_t pixels = (size_t)W * (size_t)H;
    CUDA_CHECK(cudaMalloc(&d_accum, pixels * sizeof(Vec)));
    CUDA_CHECK(cudaMemset(d_accum, 0, pixels * sizeof(Vec)));
    CUDA_CHECK(cudaMalloc(&d_spheres, total_spheres * sizeof(Sphere)));
    CUDA_CHECK(cudaMemcpy(d_spheres, host_spheres.data(), total_spheres * sizeof(Sphere), cudaMemcpyHostToDevice));

    // Launch GPU kernel
    dim3 block(16,16);
    dim3 grid( (W + block.x - 1)/block.x, (H + block.y - 1)/block.y );
    // we'll do one kernel that runs spp samples per thread (so kernel cost scales with spp)
    std::cout << "Launching GPU kernel...\n";
    cudaEvent_t e0,e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    // We launch with 1 kernel that internally loops spp times (to reduce kernel-launch overhead)
    render_kernel<<<grid, block>>>(d_accum, W, H, spp, max_bounces, d_spheres, total_spheres, cam, 0x12345678abcdefULL);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float gpu_ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, e0, e1));
    std::cout << "GPU render time: " << gpu_ms << " ms\n";

    // Copy back accum buffer
    std::vector<Vec> host_accum(pixels);
    CUDA_CHECK(cudaMemcpy(host_accum.data(), d_accum, pixels * sizeof(Vec), cudaMemcpyDeviceToHost));

    // Convert accum to final image dividing by spp
    write_ppm("render.ppm", host_accum, W, H, spp);

    // Optional: tiny CPU reference for very small resolution to check qualitative output
    if (W*H <= 64*64 && total_spheres <= 256) {
        std::cout << "Running tiny CPU reference (very slow) for sanity check...\n";
        std::vector<Vec> cpu_acc(pixels, Vec(0,0,0));
        uint64_t seed = 987654321ULL;
        for (int y=0;y<H;y++) for (int x=0;x<W;x++) {
            Vec col = Vec(0,0,0);
            for (int s=0;s<spp;s++) {
                float u = (x + (rand()/(float)RAND_MAX)) / (float)W * 2.0f - 1.0f;
                float v = (y + (rand()/(float)RAND_MAX)) / (float)H * 2.0f - 1.0f;
                float aspect = float(W)/float(H);
                float scale = tanf(cam.fov * 0.5f * (3.14159265358979323846f/180.0f));
                Vec dir = normalize(cam.forward + cam.right * (u*aspect*scale) + cam.up * (v*scale));
                Vec L = cpu_trace_path(cam.pos, dir, host_spheres, seed, max_bounces);
                col = col + L;
            }
            cpu_acc[y*W + x] = col;
        }
        write_ppm("render_cpu.ppm", cpu_acc, W, H, spp);
        std::cout << "Wrote render_cpu.ppm\n";
    }

    std::cout << "Wrote render.ppm\n";
    CUDA_CHECK(cudaFree(d_accum));
    CUDA_CHECK(cudaFree(d_spheres));
    return 0;
}
