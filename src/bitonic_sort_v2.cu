#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
    } else if ((i & group_size) != 0 && arr[i] < arr[partner]) {
        swap(arr, i, partner);
    }
}

__global__ void initialExchangeLocally(int *arr) {
    int t = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + t;

    int offset = blockIdx.x * blockDim.x * 2; // 0, 2048, 4096, ...

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
            } else if ((i_global & group_size) != 0 &&
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
    int t = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + t;

    int offset = blockIdx.x * blockDim.x * 2; // 0, 2048, 4096, ...

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
        } else if ((i_global & group_size) != 0 &&
                   shared_arr[i] < shared_arr[partner]) {
            swap(shared_arr, i, partner);
        }
        __syncthreads();
    }

    // Transfer data back to global memory
    arr[offset + t] = shared_arr[t];
    arr[offset + t + blockDim.x] = shared_arr[t + blockDim.x];
}

void bitonicSort(int *d_arr, int n, int num_threads, int num_blocks) {
    initialExchangeLocally<<<num_blocks, num_threads>>>(d_arr);

    for (int group_size = 4096; group_size <= n; group_size <<= 1) {
        for (int distance = group_size >> 1; distance > 1024; distance >>= 1) {
            exchange<<<num_blocks, num_threads>>>(d_arr, distance, group_size);
        }
        exchangeLocally<<<num_blocks, num_threads>>>(d_arr, group_size);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <power_of_2>\n", argv[0]);
        printf("Example: %s 27 for 2^27 elements\n", argv[0]);
        return 1;
    }

    int power = atoi(argv[1]);
    int n = 1 << power;
    int num_threads = 1 << 10;
    int num_blocks = n / (2 * num_threads);

    printf("Executing V2 with %d elements (2^%d)\n", n, power);

    int *arr = (int *)malloc(n * sizeof(int));
    int *d_arr;

    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100;
    }

    struct timespec start_total, end_total, start, end;
    double malloc_time, h2d_time, sort_time, d2h_time, total_time,
        execution_time;

    // Start total timing
    clock_gettime(CLOCK_MONOTONIC, &start_total);

    // Measure cudaMalloc
    clock_gettime(CLOCK_MONOTONIC, &start);
    cudaMalloc(&d_arr, n * sizeof(int));
    clock_gettime(CLOCK_MONOTONIC, &end);
    malloc_time =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    // Measure Host to Device copy
    clock_gettime(CLOCK_MONOTONIC, &start);
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);
    clock_gettime(CLOCK_MONOTONIC, &end);
    h2d_time =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    // Measure sort algorithm
    clock_gettime(CLOCK_MONOTONIC, &start);
    bitonicSort(d_arr, n, num_threads, num_blocks);
    clock_gettime(CLOCK_MONOTONIC, &end);
    sort_time =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    // Measure Device to Host copy
    clock_gettime(CLOCK_MONOTONIC, &start);
    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    clock_gettime(CLOCK_MONOTONIC, &end);
    d2h_time =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &end_total);
    total_time = (end_total.tv_sec - start_total.tv_sec) +
                 (end_total.tv_nsec - start_total.tv_nsec) / 1e9;
    execution_time = h2d_time + sort_time + d2h_time;

    printf("\nTiming Results:\n");
    printf("cudaMalloc time: %f seconds\n", malloc_time);
    printf("Host to Device copy time: %f seconds\n", h2d_time);
    printf("Sort algorithm time: %f seconds\n", sort_time);
    printf("Device to Host copy time: %f seconds\n", d2h_time);
    printf("Total time (including malloc): %f seconds\n", total_time);
    printf("Execution time (excluding malloc): %f seconds\n", execution_time);

    // Verification
    for (int i = 1; i < n; i++) {
        assert(arr[i - 1] <= arr[i]);
    }
    printf("PASSED\n");

    cudaFree(d_arr);
    free(arr);

    return 0;
}