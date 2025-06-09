#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include "cuda_helper.h"

static int threadsPerBlock;

__global__ void matvecKernel(float *A, float *B, float *C, int width, int height)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < height)
    {
        float Aval = 0.0;
        for (int k = 0; k < width; ++ k)
            Aval += B[row * width + k] * C[k];
        A[row] = Aval;
    }
}

void matvec(float *A, float *B, float *C, int width, int height)
{
    float *A_d, *B_d, *C_d;
    size_t Asize = height * sizeof(float);
    size_t Bsize = height * width * sizeof(float);
    size_t Csize = width * sizeof(float1);
    cudaMalloc((void **) &A_d, Asize);
    cudaMalloc((void **) &B_d, Bsize);
    cudaMalloc((void **) &C_d, Csize);
    cudaMemcpy(B_d, B, Bsize, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, Csize, cudaMemcpyHostToDevice);

    double start_time = cpuSecond();
    int blocksPerGrid = (height + threadsPerBlock - 1) / threadsPerBlock;
    matvecKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, width, height);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    double end_time = cpuSecond();
    double elapsed_time = end_time - start_time;
    double GFLOPS = (2.0 * width * height - height) / elapsed_time / 1e9;  
    printf("Elapsed time: %.6lf s\n", elapsed_time);
    printf("Performance: %.3lf GFLOPS\n", GFLOPS);

    cudaMemcpy(A, A_d, Asize, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}


int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        fprintf(stderr, "%s <width> <height> <threadsPerBlock>\n", argv[0]);
        exit(1);
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    threadsPerBlock = atoi(argv[3]);

    float *A = (float *) malloc(height * sizeof(float));
    float *B = (float *) malloc(height * width * sizeof(float));
    float *C = (float *) malloc(width * sizeof(float));

    // Prepare data
    for (int i = 0; i < width; ++ i)
    {
        C[i] = rand() / (float) RAND_MAX;
        for (int j = 0; j < height; ++ j)
            B[j * width + i] = rand() / (float) RAND_MAX;
    }

    matvec(A, B, C, width, height);

    free(A);
    free(B);
    free(C);
    return 0;
}