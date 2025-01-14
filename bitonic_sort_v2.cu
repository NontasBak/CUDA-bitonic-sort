#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

__device__ void swap(int *arr, int i, int j) {
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}

__global__ void exchange(int *arr, int distance, int group_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (tid / distance) * distance * 2 + (tid % distance);
    int partner = i ^ distance;

    if ((i & group_size) == 0 && arr[i] > arr[partner]) {
        swap(arr, i, partner);
    }
    if ((i & group_size) != 0 && arr[i] < arr[partner]) {
        swap(arr, i, partner);
    }
}

__global__ void initialExchangeLocally(int *arr) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int t = threadIdx.x;
    int b = blockIdx.x;
    int offset = b * blockDim.x * 2;

    // Transfer data to shared memory
    __shared__ int shared_arr[2048];
    shared_arr[t] = arr[offset + t];
    shared_arr[t + blockDim.x] = arr[offset + t + blockDim.x];
    __syncthreads();

    for (int group_size = 2; group_size <= 2048; group_size <<= 1) {
        for (int distance = group_size >> 1; distance > 0; distance >>= 1) {
            int i_global = (tid / distance) * distance * 2 + (tid % distance);
            int i = (t / distance) * distance * 2 + (t % distance);
            int partner = i ^ distance;

            if ((i_global & group_size) == 0 &&
                shared_arr[i] > shared_arr[partner]) {
                swap(shared_arr, i, partner);
            }
            if ((i_global & group_size) != 0 &&
                shared_arr[i] < shared_arr[partner]) {
                swap(shared_arr, i, partner);
            }
            __syncthreads();
        }
    }

    // Transfer data back to global memory
    arr[offset + t] = shared_arr[t];
    arr[offset + t + blockDim.x] = shared_arr[t + blockDim.x];
}

__global__ void exchangeLocally(int *arr, int group_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int t = threadIdx.x;
    int b = blockIdx.x;
    int offset = b * blockDim.x * 2;

    // Transfer data to shared memory
    __shared__ int shared_arr[2048];
    shared_arr[t] = arr[offset + t];
    shared_arr[t + blockDim.x] = arr[offset + t + blockDim.x];
    __syncthreads();

    for (int distance = 1024; distance > 0; distance >>= 1) {
        int i_global = (tid / distance) * distance * 2 + (tid % distance);
        int i = (t / distance) * distance * 2 + (t % distance);
        int partner = i ^ distance;

        if ((i_global & group_size) == 0 &&
            shared_arr[i] > shared_arr[partner]) {
            swap(shared_arr, i, partner);
        }
        if ((i_global & group_size) != 0 &&
            shared_arr[i] < shared_arr[partner]) {
            swap(shared_arr, i, partner);
        }
        __syncthreads();
    }

    // Transfer data back to global memory
    arr[offset + t] = shared_arr[t];
    arr[offset + t + blockDim.x] = shared_arr[t + blockDim.x];
}

int main() {
    int n = 1 << 23;
    int num_threads = 1 << 10;
    int num_blocks = n / (2 * num_threads);

    int *arr = (int *)malloc(n * sizeof(int));
    int *d_arr;

    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100;
    }

    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    initialExchangeLocally<<<num_blocks, num_threads>>>(d_arr);

    for (int group_size = 4096; group_size <= n; group_size <<= 1) {
        for (int distance = group_size >> 1; distance > 1024; distance >>= 1) {
            exchange<<<num_blocks, num_threads>>>(d_arr, distance, group_size);
        }
        exchangeLocally<<<num_blocks, num_threads>>>(d_arr, group_size);
    }

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("Execution time: %f seconds\n", duration.count());

    // for (int i = 0; i < n; i++) {
    //     if (i % 1024 == 0) printf("\n\n");
    //     printf("%d ", out[i]);
    // }

    for (int i = 1; i < n; i++) {
        assert(arr[i - 1] <= arr[i]);
    }
    printf("PASSED\n");

    cudaFree(d_arr);
    free(arr);

    return 0;
}