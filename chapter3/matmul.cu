#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include "cuda_helper.h"

static int bx, by;
static int r;
static int mode;

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

__global__ void matmulRowKernel(float *M, float *N, float *P, int width)
{
    int block_dim = blockDim.x * blockDim.y;
    int block_id = blockIdx.y * gridDim.x + blockIdx.x;
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int row = thread_id + block_id * block_dim;
    
    if (row < width)
    {
        for (int col = 0; col < width; ++ col)
        {
            float Pval = 0.0;
            for (int k = 0; k < width; ++ k)
                Pval += M[row * width + k] * N[k * width + col];
            P[row * width + col] = Pval;
        }
    }
}

__global__ void matmulColKernel(float *M, float *N, float *P, int width)
{
    int block_dim = blockDim.x * blockDim.y;
    int block_id = blockIdx.y * gridDim.x + blockIdx.x;
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int col = thread_id + block_id * block_dim;
    
    if (col < width)
    {
        for (int row = 0; row < width; ++ row)
        {
            float Pval = 0.0;
            for (int k = 0; k < width; ++ k)
                Pval += M[row * width + k] * N[k * width + col];
            P[row * width + col] = Pval;
        }
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
    switch (mode)
    {
    case 0:
        matmulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, width);
        break;
    case 1:
        matmulRowKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, width);
        break;
    case 2:
        matmulColKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, width);
        break;
    default:
        fprintf(stderr, "Not supported mode\n");
        exit(1);
        break;
    }
    
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    // Benchmark
    double start_time = cpuSecond();
    for (int i = 0; i < r; ++i)
       switch (mode)
        {
        case 0:
            matmulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, width);
            break;
        case 1:
            matmulRowKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, width);
            break;
        case 2:
            matmulColKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, width);
            break;
        default:
            fprintf(stderr, "Not supported mode\n");
            exit(1);
            break;
        }
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    double end_time = cpuSecond();
    double elapsed_time = (end_time - start_time) / r;
    double FLOPS = (2.0 * pow(width, 3) - pow(width, 2)) / elapsed_time / 1e9;

    printf("Elapsed time: %.3f s\n", elapsed_time);
    printf("Performance: %.3f GFLOPS\n", FLOPS);

    cudaMemcpy(P, P_d, size, cudaMemcpyDeviceToHost);
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        fprintf(stderr, "%s <mode> <width> <bx> <by> <r>\n", argv[0]);
        exit(1);
    }

    mode = atoi(argv[1]);
    int width = atoi(argv[2]);
    bx = atoi(argv[3]);
    by = atoi(argv[4]);
    r = atoi(argv[5]);

    size_t size = width * width * sizeof(float);
    float *M = (float *) malloc(size);
    float *N = (float *) malloc(size);
    float *P = (float *) malloc(size);

    // Prepare matrix
    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            M[i * width + j] = rand() / (float)RAND_MAX;
            N[i * width + j] = rand() / (float)RAND_MAX;
        }
    }

    // Sequential solution
    // float *Pans = (float *) malloc(size);
    // for (int row = 0; row < width; ++ row)
    // {
    //     for (int col = 0; col < width; ++ col)
    //     {
    //         float Pval = 0.0;
    //         for (int k = 0; k < width; ++ k)
    //         {
    //             Pval += M[row * width + k] * N[k * width + col];
    //         }
    //         P[row * width + col] = Pval;
    //     }
    // }

    matmul(M, N, P, width);

    // Verify
    // for (int row = 0; row < width; ++ row)
    // {
    //     for (int col = 0; col < width; ++ col)
    //     {
    //         if (fabs(P[row * width + col] - Pans[row * width + col]) > 1e4)
    //         {
    //             fprintf(stderr, "Incorrect at (%d, %d): %.6lf should be %.6lf\n", row, col, P[row * width + col], Pans[row * width + col]);
    //             exit(1);
    //         }
    //     }
    // }
    // printf("Correct\n");

    free(M);
    free(N);
    free(P);
    return 0;
}