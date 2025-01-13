CC=nvcc
TARGETS=bitonic_sort_v0 bitonic_sort_v1
SRC_V0=bitonic_sort_v0.cu
SRC_V1=bitonic_sort_v1.cu

all: $(TARGETS)

bitonic_sort_v0: $(SRC_V0)
	$(CC) -o bitonic_sort_v0 $(SRC_V0)

bitonic_sort_v1: $(SRC_V1)
	$(CC) -o bitonic_sort_v1 $(SRC_V1)

run_v0: bitonic_sort_v0
	./bitonic_sort_v0

run_v1: bitonic_sort_v1
	./bitonic_sort_v1

clean:
	rm -f $(TARGETS)

