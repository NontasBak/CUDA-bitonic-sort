CC=nvcc
TARGETS=bitonic_sort_v0 bitonic_sort_v1 bitonic_sort_v2
SRC_V0=bitonic_sort_v0.cu
SRC_V1=bitonic_sort_v1.cu
SRC_V2=bitonic_sort_v2.cu

all: $(TARGETS)

bitonic_sort_v0: $(SRC_V0)
	$(CC) -o bitonic_sort_v0 $(SRC_V0)

bitonic_sort_v1: $(SRC_V1)
	$(CC) -o bitonic_sort_v1 $(SRC_V1)

bitonic_sort_v2: $(SRC_V2)
	$(CC) -o bitonic_sort_v2 $(SRC_V2)

run_v0: bitonic_sort_v0
	./bitonic_sort_v0

run_v1: bitonic_sort_v1
	./bitonic_sort_v1

run_v2: bitonic_sort_v2
	./bitonic_sort_v2

clean:
	rm -f $(TARGETS)

