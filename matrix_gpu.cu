#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <time.h>
#include <string.h>

#define N 5000
#define PRINT_BLOCK_SIZE 5

void print_matrix_part(const char *name, float* matrix) {
    printf("%s:\n", name);
    for (int i = 0; i < PRINT_BLOCK_SIZE; ++i) {
        for (int j = 0; j < PRINT_BLOCK_SIZE; ++j) {
            printf("%5.1f ", matrix[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void matrixMultWithOmp(float *A, float* B, float* C_omp) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            C_omp[i * N + j] = sum;
        }
    }
}

__global__ void matrixMultWithGpu(float* A, float* B, float* C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main()
{
    int numDevices = 0;
    cudaError_t err = cudaGetDeviceCount(&numDevices);

    if (err != cudaSuccess)
    {
        printf("cudaGetDeviceCount error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Found %d CUDA device\n", numDevices);

    for (int i = 0; i < numDevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("\nDevice %d:\n", i);
        printf("  Name: %s\n", prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        double total_mem_gib = (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
        printf("  Total global memory: %.2f GB (or %zu bytes)\n", total_mem_gib, prop.totalGlobalMem);
        printf("  Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    }

    printf("\n\n\n\n\n");

    srand((unsigned int)time(NULL));
    size_t bytes = N * N * sizeof(float);

    float *A, *B, *C_cpu, *C_gpu; //Приведение типа в данном случае к float* в C автоматически? Вроде без явного даже лучше
    A = malloc(bytes);
    B = malloc(bytes);
    C_cpu = malloc(bytes);
    C_gpu = malloc(bytes);

    for (int i = 0; i < N * N; i++)
    {
        A[i] = (float)(rand() % 10);
        B[i] = (float)(rand() % 10);
    }
    print_matrix_part("Part of A", A);
    print_matrix_part("Part of B", B);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);


    printf("Launching GPU\n");
    auto gstart = std::chrono::high_resolution_clock::now();
    matrixMultWithGpu<<<grid, block>>>(dA, dB, dC);
    cudaDeviceSynchronize();
    auto gend = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double>(gend - gstart).count();
    printf("GPU time: %.3f seconds\n", gpu_time);
    // Copy back to CPU
    cudaMemcpy(C_gpu, dC, bytes, cudaMemcpyDeviceToHost);
    print_matrix_part("Part of C_gpu", C_gpu);


    printf("Calculating on CPU\n");
    auto start = std::chrono::high_resolution_clock::now();
    matmul_cpu(A, B, C_cpu);
    auto end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(end - start).count();
    printf("CPU time: %.3f seconds\n", cpu_time);
    print_matrix_part("Part of C_cpu", C_cpu);
    printf("GPU speedup: %.2fx\n", cpu_time / gpu_time);
    printf("Verification: %s\n", (memcmp(C_cpu, C_gpu, bytes) == 0) ? "SUCCESS" : "FAILURE");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(A); free(B); free(C_cpu); free(C_gpu);
    return 0;
}