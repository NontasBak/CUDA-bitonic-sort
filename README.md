# CUDA-bitonic-sort

Bitonic Sort implementation using CUDA

There are 3 different versions, v0, v1 and v2. The differences between them are explain in `docs/report.pdf`.

### Compile and run

1. First, clone the repository
```
git clone git@github.com:NontasBak/CUDA-bitonic-sort.git
```

2. Make sure you have the CUDA drivers installed and run
```
make
```

3. Run any of the 3 versions while also specifying the number of elements in the array as a power of 2 ($2^{power}$)
```
./bitonic_sort_v0 <power>
./bitonic_sort_v1 <power>
./bitonic_sort_v2 <power>
```

For example running `./bitonic_sort_v0 23` will run the algorithm with $2^{23}$ elements in total

4. Once you finish, remove the binaries
```
make clean
```

### Benchmarks
The complete benchmark results can be found in `docs/report.pdf`.

TL;DR: Compared to a CPU sorting algorithm running in multiple threads, there's a considerable speedup using CUDA, reaching up to 12 times faster.

### Author
Bakoulas Epameinondas

Class: Parallel and Distributed Systems with professor Nikolaos Pitsianis

February 2025
