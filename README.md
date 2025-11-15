# GPU Ray Tracing Using CUDA (Path Tracer)

## Overview
This project implements a CUDA-based GPU Path Tracer capable of rendering a simple 3D scene using ray tracing techniques.

## Features
- GPU parallel ray launching
- Ray–sphere intersection
- Background shading
- Generates PPM image output

## How to Run
```
nvcc raytracer.cu -o raytracer
./raytracer > output.ppm
```
## code
```c

#include <stdio.h>
#include <curand_kernel.h>

struct Vec3 {
    float x, y, z;
    __device__ Vec3() {}
    __device__ Vec3(float a, float b, float c) : x(a), y(b), z(c) {}
    __device__ Vec3 operator+(const Vec3 &v) const { return Vec3(x+v.x, y+v.y, z+v.z); }
    __device__ Vec3 operator-(const Vec3 &v) const { return Vec3(x-v.x, y-v.y, z-v.z); }
    __device__ Vec3 operator*(float s) const { return Vec3(x*s, y*s, z*s); }
};

__device__ float dot(const Vec3 &a, const Vec3 &b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ bool hit_sphere(const Vec3 &center, float radius, const Vec3 &orig, const Vec3 &dir, float &t) {
    Vec3 oc = orig - center;
    float a = dot(dir, dir);
    float b = 2.0 * dot(oc, dir);
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - 4*a*c;
    if (discriminant < 0) return false;
    t = (-b - sqrtf(discriminant)) / (2.0*a);
    return t > 0;
}

__global__ void render(Vec3 *fb, int w, int h) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y*w + x;

    Vec3 origin(0,0,0);
    float u = float(x)/w;
    float v = float(y)/h;
    Vec3 dir(u-0.5, v-0.5, 1);

    float t;
    if (hit_sphere(Vec3(0,0,1.5), 0.5, origin, dir, t)) {
        fb[idx] = Vec3(1,0,0);
    } else {
        fb[idx] = Vec3(0.2,0.3,0.5);
    }
}

int main() {
    int w = 800, h = 600;
    Vec3 *fb;
    size_t fb_size = w * h * sizeof(Vec3);
    cudaMallocManaged(&fb, fb_size);

    dim3 threads(8,8);
    dim3 blocks((w+7)/8, (h+7)/8);
    render<<<blocks, threads>>>(fb, w, h);
    cudaDeviceSynchronize();

    printf("P3\n%d %d\n255\n", w, h);
    for (int i=0;i<w*h;i++) {
        int r = int(255.99 * fb[i].x);
        int g = int(255.99 * fb[i].y);
        int b = int(255.99 * fb[i].z);
        printf("%d %d %d\n", r,g,b);
    }
    cudaFree(fb);

}

```
## output

real 800×600 render:

```
P3
800 600
255
51 76 128
51 76 128
51 76 128
...
255 0 0
255 0 0
255 0 0
...
51 76 128
51 76 128
51 76 128
```
The repeating red pixels represent the sphere.


<img width="400" height="300" alt="raytracer_sample" src="https://github.com/user-attachments/assets/68b1e6e4-7b51-414f-9726-d330d65e7c25" />


## Summary

In this project, a GPU-accelerated Ray Tracer was implemented using NVIDIA CUDA, demonstrating the power of parallel computation for realistic image rendering. The program casts rays from a virtual camera into a 3D scene and computes intersections with objects using CUDA kernels. The renderer outputs a final image in PPM format and generates a simple scene consisting of a red sphere against a blue background.
