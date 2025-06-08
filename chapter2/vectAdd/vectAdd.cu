#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <cuda_helper.h>

static int threadsPerBlock;
static int r;

__global__ void vectAddKernel(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

void vectAdd(const float *A, const float *B, float *C, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    printf("Blocks per grid: %d\n", blocksPerGrid);
    printf("Threads per block: %d\n", threadsPerBlock);

    double start_time = cpuSecond();
    for (int i = 0; i < r; ++ i)
    {
        vectAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, n);
    }
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    double end_time = cpuSecond();
    
    double elapsed_time = (end_time - start_time) * 1e6 / r;
    printf("Kernel execution time: %.3lf us\n", elapsed_time);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "%s <n> <threads_per_block> <r>\n", argv[0]);
        exit(1);
    }
    int n = atoi(argv[1]);
    threadsPerBlock = atoi(argv[2]);
    r = atoi(argv[3]);
    size_t size = n * sizeof(float);

    float *A = (float *) malloc(size);
    float *B = (float *) malloc(size);
    float *C = (float *) malloc(size);

    // Prepare data
    for (int i = 0; i < n; i++) {
        A[i] = rand() / (float)RAND_MAX;
        B[i] = rand() / (float)RAND_MAX;
    }

    // Execute
    vectAdd(A, B, C, n);

    free(A);
    free(B);
    free(C);
    return 0;
}