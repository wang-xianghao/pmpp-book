#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <cuda_helper.h>

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
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    printf("Blocks per grid: %d\n", blocksPerGrid);
    printf("Threads per block: %d\n", threadsPerBlock);
    double start_time = cpuSecond();
    vectAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, n);
    cudaDeviceSynchronize();
    double end_time = cpuSecond();
    printf("Kernel execution time: %.3lf s\n", end_time - start_time);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "%s <n>\n", argv[0]);
        exit(1);
    }
    int n = atoi(argv[1]);
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

    // Verify
    for (int i = 0; i < n; ++ i) {
        float expect = A[i] + B[i];
        if (fabs(C[i] - expect) > 1e-4) {
            printf("Wrong result at %d: %.6f should be %.6f\n", i, C[i], expect);
            break;
        }
    }
    printf("Verfied\n");

    free(A);
    free(B);
    free(C);
    return 0;
}