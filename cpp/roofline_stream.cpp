#include <chrono>
#include <cstddef>
#include <iostream>
#include <vector>
#include <numeric>
#include <omp.h>
#include <fstream>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <sched.h> 

// STREAM TRIAD-style benchmark:
// C[i] = A[i] + scalar * B[i]

// fetch host name across the os architecture
std::string get_machine_name() {
#if defined(_WIN32) || defined(_WIN64)
    const char* env = std::getenv("COMPUTERNAME");
#else
    const char* env = std::getenv("HOSTNAME");
    if (!env) env = std::getenv("HOSTNAME");
#endif
    return env ? std::string(env) : "Unknown_Machine";
}

// single thread benchmark iteration
void run_benchmark(std::size_t N, int thread_count,
                   const std::string& machine_name,
                   std::ofstream& csv_file) {
    constexpr double scalar = 2.0;

    // set omp Threads
    omp_set_num_threads(thread_count);

    std::vector<double> vector_A(N, 5.0);
    std::vector<double> vector_B(N, 10.0);
    std::vector<double> vector_C(N, 0.0);

// //run thread IDs - can check cpu architecture whether it is NUMA or not. 
// #pragma omp parallel
// {
//     const int thread_id = omp_get_thread_num();
//     const int cpu_id = sched_getcpu();

//     #pragma omp critical
//     {
//         std::cout << "OpenMP thread " << thread_id
//                   << " runs on logical CPU " << cpu_id
//                   << '\n';
//     } 
// }

    // warm up
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i) {
        vector_C[i] = vector_A[i] + vector_B[i] * scalar;
    }

    // activate timing
    const auto start = std::chrono::steady_clock::now();

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i) {
        vector_C[i] = vector_A[i] + vector_B[i] * scalar;
    }

    // close timing
    const auto end = std::chrono::steady_clock::now();

    // total duration
    const std::chrono::duration<double> elapsed = end - start;
    const double seconds = elapsed.count();

    double checksum = 0.0;

#pragma omp parallel for reduction(+ : checksum)
    for (std::size_t i = 0; i < N; ++i) {
        checksum += vector_C[i];
    }



    // metric calculation
    constexpr double flops_per_element = 2.0;
    const double total_flops = static_cast<double>(N) * flops_per_element;
    constexpr double bytes_per_element = 3 * sizeof(double);
    const double total_bytes = static_cast<double>(N) * bytes_per_element;

    const double gflops = total_flops / seconds / 1.0e9;
    const double bandwidth_gb_s = total_bytes / seconds / 1.0e9;
    const double arithmetic_intensity = flops_per_element / bytes_per_element;

    // Log to standard console for live tracking
    std::cout << "Threads: " << thread_count
              << " | Bandwidth: " << bandwidth_gb_s << " GB/s"
              << " | Performance: " << gflops << " GFLOP/s\n";

    // Write row directly to CSV file
    // Columns: machine, dwarf name, AI, bandwidth, performance flops, threads, validation_checksum
    csv_file << machine_name << ","
             << "stream triad"
             << ","
             << arithmetic_intensity << ","
             << bandwidth_gb_s << ","
             << gflops << ","
             << thread_count << ","
             << checksum << "\n";
}

int main() {
    constexpr std::size_t N = 50000000;

    std::string machine_name = get_machine_name();
    std::vector<int> threads_configs = {1, 2, 4, 6, 12, 16, 24};

    // open file stream and add header
    std::ifstream check_empty("roofline.csv");
    bool add_header = !check_empty.is_open() ||
                      check_empty.peek() == std::ifstream::traits_type::eof();
    check_empty.close();

    std::ofstream csv("roofline.csv", std::ios::app);

    if (add_header) {
        csv << "machine,dwarf name,AI,bandwidth(GB/s),performance(GFLOPs/s),threads,checksum\n";
    }

    std::cout << "Start automated roofline sweep " << machine_name << "\n";

    for (int threads : threads_configs) {
        run_benchmark(N, threads, machine_name, csv);
    }

    std::cout << "Sweep completed and roofline.csv saved\n";
    return 0;
}