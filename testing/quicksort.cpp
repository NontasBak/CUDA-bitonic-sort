#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int compare(const void* a, const void* b) { return (*(int*)a - *(int*)b); }

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <power_of_2>\n", argv[0]);
        return 1;
    }

    int power = atoi(argv[1]);
    int size = 1 << power;  // 2^power

    // Allocate and initialize array
    int* arr = (int*)malloc(size * sizeof(int));
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 100;
    }

    // Sort array
    clock_t start = clock();
    qsort(arr, size, sizeof(int), compare);
    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time taken: %f seconds\n", time_spent);

    // Verify sorting
    for (int i = 1; i < size; i++) {
        if (arr[i] < arr[i - 1]) {
            printf("Sort failed!\n");
            free(arr);
            return 1;
        }
    }
    printf("Sort successful!\n");

    free(arr);
    return 0;
}
