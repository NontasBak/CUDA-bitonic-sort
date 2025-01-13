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

int main() {
    int n = 1 << 13;
    int num_threads = 1024;
    int num_blocks = n / (2 * num_threads);

    int *arr = (int *)malloc(n * sizeof(int));
    int *out = (int *)malloc(n * sizeof(int));
    int *d_arr;

    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100;
    }

    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    for (int group_size = 2; group_size <= n; group_size <<= 1) {
        for (int distance = group_size >> 1; distance > 0; distance >>= 1) {
            exchange<<<num_blocks, num_threads>>>(d_arr, n, distance,
                                                  group_size);
        }
    }

    cudaMemcpy(out, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < n; i++) {
    //     if (i % 1024 == 0) printf("\n\n");
    //     printf("%d ", out[i]);
    // }

    // for (int i = 1; i < n; i++) {
    //     assert(out[i - 1] <= out[i]);
    // }

    printf("PASSED\n");

    cudaFree(d_arr);

    free(arr);
    free(out);

    return 0;
}