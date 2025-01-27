CC=nvcc
TARGETS=bitonic_sort_v0 bitonic_sort_v1 bitonic_sort_v2
SRC_V0=src/bitonic_sort_v0.cu
SRC_V1=src/bitonic_sort_v1.cu
SRC_V2=src/bitonic_sort_v2.cu

# Default power of 2 if not specified
POWER ?= 27

all: $(TARGETS)

bitonic_sort_v0: $(SRC_V0)
	$(CC) -o bitonic_sort_v0 $(SRC_V0)

bitonic_sort_v1: $(SRC_V1)
	$(CC) -o bitonic_sort_v1 $(SRC_V1)

bitonic_sort_v2: $(SRC_V2)
	$(CC) -o bitonic_sort_v2 $(SRC_V2)

run_v0: bitonic_sort_v0
	./bitonic_sort_v0 $(POWER)

run_v1: bitonic_sort_v1
	./bitonic_sort_v1 $(POWER)

run_v2: bitonic_sort_v2
	./bitonic_sort_v2 $(POWER)

clean:
	rm -f $(TARGETS)

