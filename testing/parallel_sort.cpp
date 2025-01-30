#include <chrono>
#include <cmath>
#include <iostream>
#include <parallel/algorithm>
#include <random>
#include <vector>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <power>\n";
        cerr << "Example: " << argv[0] << " 20 (for 2^20 elements)\n";
        return 1;
    }

    // Parse the power argument
    int power = atoi(argv[1]);
    size_t size = static_cast<size_t>(pow(2, power));

    cout << "Generating " << size << " elements (2^" << power << ")\n";

    // Create and fill vector with random numbers
    vector<int> vec(size);

    // Use same random number generation as quicksort
    srand(time(NULL));
    cout << "Filling vector with random numbers...\n";
    for (size_t i = 0; i < size; i++) {
        vec[i] = rand() % 100;
    }

    // Sort and measure time
    cout << "Starting parallel sort...\n";
    auto start = chrono::high_resolution_clock::now();

    __gnu_parallel::sort(vec.begin(), vec.end());

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    // Verify sorting
    bool is_sorted = std::is_sorted(vec.begin(), vec.end());
    if (!is_sorted) {
        cerr << "Sorting failed verification!\n";
        return 1;
    }

    // Output in format consistent with other tests
    printf("Time taken: %f seconds\n", elapsed.count());
    return 0;
}