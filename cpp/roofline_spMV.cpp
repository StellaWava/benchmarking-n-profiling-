#include <chrono>
#include <cstddef>
#include <iostream>
#include <vector>
#include <numeric>
#include <omp.h>
#include <fstream>
#include <cstdlib>
#include <random>

std::string get_machine_name() {
#if defined(_WIN32) || defined(_WIN64)
    const char* env = std::getenv("COMPUTERNAME");
#else
    const char* env = std::getenv("HOSTNAME");
    if (!env) env = std::getenv("NAME");
#endif
    return env ? std::string(env) : "Unknown_Machine";
}

void run_spmv(std::size_t num_rows, std::size_t nnz_per_row, int thread_count, const std::string& machine_name, std::ofstream& csv_file) {
    omp_set_num_threads(thread_count);

    std::size_t total_nnz = num_rows * nnz_per_row;

    // CSR Format Arrays
    std::vector<std::size_t> row_ptr(num_rows + 1);
    std::vector<std::size_t> col_idx(total_nnz);
    std::vector<double> values(total_nnz, 2.5);
    std::vector<double> x(num_rows, 1.5);
    std::vector<double> y(num_rows, 0.0);

    // Initialize CSR Structure (Mocking a random sparse structure)
    row_ptr[0] = 0;
    for (std::size_t i = 0; i < num_rows; ++i) {
        row_ptr[i + 1] = row_ptr[i] + nnz_per_row;
        for (std::size_t j = 0; j < nnz_per_row; ++j) {
            std::size_t idx = row_ptr[i] + j;
            // Distribute column indices across the vector width
            col_idx[idx] = (i + j) % num_rows; 
        }
    }

    // Warm-up pass
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < num_rows; ++i) {
        double sum = 0.0;
        for (std::size_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            sum += values[j] * x[col_idx[j]];
        }
        y[i] = sum;
    }

    // Active Profile Timing Block
    const auto start = std::chrono::steady_clock::now();

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < num_rows; ++i) {
        double sum = 0.0;
        // Inner loop does irregular memory indexing: x[col_idx[j]]
        for (std::size_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            sum += values[j] * x[col_idx[j]];
        }
        y[i] = sum;
    }

    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    const double seconds = elapsed.count();

    // Verification checksum
    double checksum = 0.0;
    #pragma omp parallel for reduction(+:checksum)
    for (std::size_t i = 0; i < num_rows; ++i) {
        checksum += y[i];
    }

    // Metric Calculations
    // 1 multiply and 1 add per non-zero element
    double total_flops = 2.0 * static_cast<double>(total_nnz);
    
    // Memory footprint calculation:
    // Reads: row_ptr, values, col_idx, and indirect tracking of vector x
    // Writes: vector y
    double bytes_row_ptr = static_cast<double>((num_rows + 1) * sizeof(std::size_t));
    double bytes_col_idx = static_cast<double>(total_nnz * sizeof(std::size_t));
    double bytes_values  = static_cast<double>(total_nnz * sizeof(double));
    double bytes_vectors = static_cast<double>(num_rows * 2 * sizeof(double)); 
    double total_bytes   = bytes_row_ptr + bytes_col_idx + bytes_values + bytes_vectors;

    const double gflops = total_flops / seconds / 1.0e9;
    const double bandwidth_gb_s = total_bytes / seconds / 1.0e9;
    const double arithmetic_intensity = total_flops / total_bytes;

    std::cout << "SpMV Threads: " << thread_count 
              << " | Bandwidth: " << bandwidth_gb_s << " GB/s"
              << " | Performance: " << gflops << " GFLOP/s\n";

    // Append to your unified roofline.csv file
    csv_file << machine_name << ","
             << "sparse matrix vector" << ","
             << arithmetic_intensity << ","
             << bandwidth_gb_s << ","
             << gflops << ","
             << thread_count << ","
             << checksum << "\n";
}

int main() {
    // 50,000 rows with 500 non-zero entries per row (~25 million total matrix elements)
    constexpr std::size_t num_rows = 50000;
    constexpr std::size_t nnz_per_row = 500;
    
    std::string machine_name = get_machine_name();
    std::vector<int> thread_configs = {1, 2, 4, 6, 12, 16, 24};

    std::ofstream csv("roofline.csv", std::ios::app);

    std::cout << "Starting automated SpMV CPU sweep on machine: " << machine_name << "\n";
    for (int threads : thread_configs) {
        run_spmv(num_rows, nnz_per_row, threads, machine_name, csv);
    }
    std::cout << "SpMV Sweep completed. Output appended to 'roofline.csv'\n";

    return 0;
}
