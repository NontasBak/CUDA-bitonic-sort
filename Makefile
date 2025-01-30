CC=nvcc
GCC=g++
TARGETS=bitonic_sort_v0 bitonic_sort_v1 bitonic_sort_v2 testing/quicksort_sequential testing/parallel_sort
SRC_V0=src/bitonic_sort_v0.cu
SRC_V1=src/bitonic_sort_v1.cu
SRC_V2=src/bitonic_sort_v2.cu
SEQUENTIAL=testing/quicksort.cpp
PARALLEL=testing/parallel_sort.cpp

# Default power of 2 if not specified
POWER ?= 27
THREADS ?= 8

all: $(TARGETS)

bitonic_sort_v0: $(SRC_V0)
	$(CC) -o bitonic_sort_v0 $(SRC_V0)

bitonic_sort_v1: $(SRC_V1)
	$(CC) -o bitonic_sort_v1 $(SRC_V1)

bitonic_sort_v2: $(SRC_V2)
	$(CC) -o bitonic_sort_v2 $(SRC_V2)

testing/quicksort_sequential: $(SEQUENTIAL)
	$(GCC) -o $@ $<

testing/parallel_sort: $(PARALLEL)
	$(GCC) -fopenmp -D_GLIBCXX_PARALLEL -o $@ $<

run_v0: bitonic_sort_v0
	./bitonic_sort_v0 $(POWER)

run_v1: bitonic_sort_v1
	./bitonic_sort_v1 $(POWER)

run_v2: bitonic_sort_v2
	./bitonic_sort_v2 $(POWER)

run_seq: testing/quicksort_sequential
	./testing/quicksort $(POWER)

run_par: testing/quicksort_parallel
	OMP_NUM_THREADS=$(THREADS) ./testing/parallel_sort $(POWER)

clean:
	rm -f $(TARGETS)

