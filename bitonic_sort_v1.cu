#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__device__ void swap(int *arr, int i, int j) {
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}

__global__ void exchange(int *arr, int size, int distance, int group_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int partner = tid ^ distance;
    if (partner > tid) {
        if ((tid & group_size) == 0 && arr[tid] > arr[partner]) {
            swap(arr, tid, partner);
        }
        if ((tid & group_size) != 0 && arr[tid] < arr[partner]) {
            swap(arr, tid, partner);
        }
    }
}

__global__ void initialExchangeLocally(int *arr) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int group_size = 2; group_size <= 2048; group_size <<= 1) {
        for (int distance = group_size >> 1; distance > 0; distance >>= 1) {
            int start_index_i = tid + blockDim.x * blockIdx.x;
            int end_index_i = start_index_i + blockDim.x;

            for (int i = start_index_i; i <= end_index_i; i += 1024) {
                int partner = i ^ distance;
                if (partner > i) {
                    if ((i & group_size) == 0 && arr[i] > arr[partner]) {
                        swap(arr, i, partner);
                    }
                    if ((i & group_size) != 0 && arr[i] < arr[partner]) {
                        swap(arr, i, partner);
                    }
                }
            }
            __syncthreads();
        }
    }
}

__global__ void exchangeLocally(int *arr, int group_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int distance = 1024; distance > 0; distance >>= 1) {
        int start_index_i = tid + blockDim.x * blockIdx.x;
        int end_index_i = start_index_i + blockDim.x;

        for (int i = start_index_i; i <= end_index_i; i += 1024) {
            int partner = i ^ distance;
            if (partner > i) {
                if ((i & group_size) == 0 && arr[i] > arr[partner]) {
                    swap(arr, i, partner);
                }
                if ((i & group_size) != 0 && arr[i] < arr[partner]) {
                    swap(arr, i, partner);
                }
            }
        }
        __syncthreads();
    }
}

int main() {
    int n = 1 << 25;
    int num_threads = 1 << 10;
    int num_blocks = n / (2 * num_threads);
    int num_blocks_exch = n / num_threads;

    int *arr = (int *)malloc(n * sizeof(int));
    int *out = (int *)malloc(n * sizeof(int));
    int *d_arr;

    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100;
    }

    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    initialExchangeLocally<<<num_blocks, num_threads>>>(d_arr);
    // cudaDeviceSynchronize();

    // cudaMemcpy(out, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < n; i++) {
    //     if (i % 1024 == 0) printf("\n\n");
    //     printf("%d ", out[i]);
    // }

    for (int group_size = 4096; group_size <= n; group_size <<= 1) {
        for (int distance = group_size >> 1; distance > 1024; distance >>= 1) {
            exchange<<<num_blocks_exch, num_threads>>>(d_arr, n, distance,
                                                  group_size);
        }
        exchangeLocally<<<num_blocks, num_threads>>>(d_arr, group_size);
    }

    cudaMemcpy(out, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < n; i++) {
    //     if (i % 1024 == 0) printf("\n\n");
    //     printf("%d ", out[i]);
    // }

    for (int i = 1; i < n; i++) {
        assert(out[i - 1] <= out[i]);
    }

    printf("PASSED\n");

    cudaFree(d_arr);

    free(arr);
    free(out);

    return 0;
}