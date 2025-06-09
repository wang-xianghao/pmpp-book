#include <stdio.h>
#include <cuda.h>
#include "cuda_helper.h"

static int bx, by;
static int r;

__global__ void matmulKernel(float *M, float *N, float *P, int width)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < width && row < width)
    {
        float Pval = 0.0f;
        for (int k = 0; k < width; ++k)
            Pval += M[row * width + k] * N[k * width + col];
        P[row * width + col] = Pval;
    }
}

void matmul(float *M, float *N, float *P, int width)
{
    float *M_d, *N_d, *P_d;
    size_t size = width * width * sizeof(float);

    cudaMalloc((void **)&M_d, size);
    cudaMalloc((void **)&N_d, size);
    cudaMalloc((void **)&P_d, size);
    cudaMemcpy(M_d, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N, size, cudaMemcpyHostToDevice);
    cudaMemcpy(P_d, P, size, cudaMemcpyHostToDevice);

    int gx = (width + bx - 1) / bx;
    int gy = (width + by - 1) / by;
    dim3 dimBlock(bx, by, 1);
    dim3 dimGrid(gx, gy, 1);

    // Warmup
    matmulKernel<<<dimGrid, dimBlock> > >(M_d, N_d, P_d, width);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    // Benchmark
    double start_time = cpuSecond();
    for (int i = 0; i < r; ++i)
        matmulKernel<<<dimGrid, dimBlock> > >(M_d, N_d, P_d, width);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    double end_time = cpuSecond();
    double elapsed_time = (end_time - start_time) / r;
    double FLOPS = (2 * width * width * width - width * width) / elapsed_time / 1e9;

    printf("Elapsed time: %.3f s\n", elapsed_time);
    printf("Performance: %.3f GFLOPS\n", FLOPS);

    cudaMemcpy(P, P_d, size, cudaMemcpyDeviceToHost);
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        fprintf(stderr, "%s <width> <bx> <by> <r>\n", argv[0]);
        exit(1);
    }

    int width = atoi(argv[1]);
    bx = atoi(argv[2]);
    by = atoi(argv[3]);
    r = atoi(argv[4]);

    size_t size = width * width * sizeof(float);
    float *M = (float *)malloc(size);
    float *N = (float *)malloc(size);
    float *P = (float *)malloc(size);

    // Prepare matrix
    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            M[i * width + j] = rand() / (float)RAND_MAX;
            N[i * width + j] = rand() / (float)RAND_MAX;
        }
    }

    matmul(M, N, P, width);

    free(M);
    free(N);
    free(P);
    return 0;
}