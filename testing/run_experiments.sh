#!/bin/bash

make all
export OMP_NUM_THREADS=8

echo "Power,Size,Version,cudaMalloc_Time,Host_to_Device_Time,Sort_Time,Device_to_Host_Time,Total_Time_with_Malloc,Total_Time_without_Malloc,QuickSort_Sequential,Parallel_Sort" > results/timing_results_rtx4060.csv

for power in {15..28}; do
    size=$((1 << power))
    echo "Running experiments for 2^$power = $size elements"
    
    # First get sequential time
    echo "  Running sequential quicksort..."
    seq_time=$(./testing/quicksort_sequential $power | grep "Time taken:" | awk '{print $3}')
    
    # Get parallel sort time
    echo "  Running parallel sort..."
    par_time=$(./testing/parallel_sort $power | grep "Time taken:" | awk '{print $3}')

    # Run bitonic sort versions
    for version in v0 v1 v2; do
        echo "  Running version $version..."
        output=$(./bitonic_sort_$version $power)
        
        # Check if the program exited successfully
        if [ $? -ne 0 ]; then
            echo "Verification failed for version $version with power $power. Exiting."
            exit 1
        fi
        
        # Extract all timing metrics using grep and awk
        malloc_time=$(echo "$output" | grep "cudaMalloc time:" | awk '{print $3}')
        h2d_time=$(echo "$output" | grep "Host to Device copy time:" | awk '{print $6}')
        sort_time=$(echo "$output" | grep "Sort algorithm time:" | awk '{print $4}')
        d2h_time=$(echo "$output" | grep "Device to Host copy time:" | awk '{print $6}')
        total_with_malloc=$(echo "$output" | grep "Total time (including malloc):" | awk '{print $5}')
        total_without_malloc=$(echo "$output" | grep "Execution time (excluding malloc):" | awk '{print $5}')
        
        # Write complete line including sequential and parallel sort time
        echo "$power,$size,$version,$malloc_time,$h2d_time,$sort_time,$d2h_time,$total_with_malloc,$total_without_malloc,$seq_time,$par_time" >> results/timing_results_rtx4060.csv
    done
done

echo "Experiments completed. Results saved in results/timing_results_rtx4060.csv"