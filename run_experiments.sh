#!/bin/bash

# Make sure executables are built
make all

# Create results directory if it doesn't exist
mkdir -p results

# Create or clear the results file with all columns
echo "Power,Size,Version,cudaMalloc_Time,Host_to_Device_Time,Sort_Time,Device_to_Host_Time,Total_Time_with_Malloc,Total_Time_without_Malloc" > results/timing_results.csv

# Run experiments for powers 15 through 29
for power in {15..29}; do
    size=$((1 << power))
    echo "Running experiments for 2^$power = $size elements"
    
    for version in v0 v1 v2; do
        echo "  Running version $version..."
        output=$(./bitonic_sort_$version $power)
        
        # Extract all timing metrics using grep and awk
        malloc_time=$(echo "$output" | grep "cudaMalloc time:" | awk '{print $3}')
        h2d_time=$(echo "$output" | grep "Host to Device copy time:" | awk '{print $6}')
        sort_time=$(echo "$output" | grep "Sort algorithm time:" | awk '{print $4}')
        d2h_time=$(echo "$output" | grep "Device to Host copy time:" | awk '{print $6}')
        total_with_malloc=$(echo "$output" | grep "Total time (including malloc):" | awk '{print $5}')
        total_without_malloc=$(echo "$output" | grep "Execution time (excluding malloc):" | awk '{print $5}')
        
        # Append to CSV with all metrics
        echo "$power,$size,$version,$malloc_time,$h2d_time,$sort_time,$d2h_time,$total_with_malloc,$total_without_malloc" >> results/timing_results.csv
    done
done

echo "Experiments completed. Results saved in results/timing_results.csv"
