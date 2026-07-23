/*
Running the experiment on GPU without directly using CUDA.
Data transfer from host RAM to device VRAM occurs before timing.
*/

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <omp.h>

std::string get_machine_name() {
#if defined(_WIN32) || defined(_WIN64)
    const char* env = std::getenv("COMPUTERNAME");
#else
    const char* env = std::getenv("HOSTNAME");
    if (!env) {
        env = std::getenv("NAME");
    }
#endif

    return env ? std::string(env) : "GPU_Machine";
}

void run_gpu_benchmark(
    std::size_t N,
    const std::string& machine_name,
    std::ofstream& csv_file
) {
    constexpr double scalar = 2.0;

    std::vector<double> vector_A(N, 5.0);
    std::vector<double> vector_B(N, 10.0);
    std::vector<double> vector_C(N, 0.0);

    double* pA = vector_A.data();
    double* pB = vector_B.data();
    double* pC = vector_C.data();

    double seconds = 0.0;

    /*
    Data transfers occur when entering and leaving this region.

    A and B: host -> device
    C:       device -> host
    */
#pragma omp target data map(to : pA[0:N], pB[0:N]) map(from : pC[0:N])
    {
        // Warm-up GPU execution.
#pragma omp target teams distribute parallel for
        for (std::size_t i = 0; i < N; ++i) {
            pC[i] = pA[i] + scalar * pB[i];
        }

        const auto start = std::chrono::steady_clock::now();

#pragma omp target teams distribute parallel for
        for (std::size_t i = 0; i < N; ++i) {
            pC[i] = pA[i] + scalar * pB[i];
        }

        const auto end = std::chrono::steady_clock::now();

        const std::chrono::duration<double> elapsed = end - start;
        seconds = elapsed.count();
    }

    /*
    pC has now been copied from device memory back to vector_C
    because the target data region has ended.
    */
    const double checksum =
        std::accumulate(vector_C.begin(), vector_C.end(), 0.0);

    constexpr double flops_per_element = 2.0;
    constexpr double bytes_per_element = 3.0 * sizeof(double);

    const double total_flops =
        static_cast<double>(N) * flops_per_element;

    const double total_bytes =
        static_cast<double>(N) * bytes_per_element;

    const double gflops =
        total_flops / seconds / 1.0e9;

    const double bandwidth_gb_s =
        total_bytes / seconds / 1.0e9;

    const double arithmetic_intensity =
        flops_per_element / bytes_per_element;

    std::cout << "GPU | Bandwidth: "
              << bandwidth_gb_s << " GB/s"
              << " | Performance: "
              << gflops << " GFLOP/s"
              << " | Checksum: "
              << checksum << '\n';

    csv_file << machine_name << ','
             << "stream triad gpu" << ','
             << arithmetic_intensity << ','
             << bandwidth_gb_s << ','
             << gflops << ','
             << "GPU" << ','
             << checksum << '\n';
}

int main() {
    constexpr std::size_t N = 50000000;

    const std::string machine_name = get_machine_name();

    const int number_of_devices = omp_get_num_devices();

    std::cout << "OpenMP target devices detected: "
              << number_of_devices << '\n';

    if (number_of_devices == 0) {
        std::cerr
            << "No OpenMP target device was detected.\n"
            << "The target regions may execute on the host CPU.\n";

        return 1;
    }

    std::ifstream check_empty("roofline.csv");

    const bool add_header =
        !check_empty.is_open() ||
        check_empty.peek() == std::ifstream::traits_type::eof();

    check_empty.close();

    std::ofstream csv("roofline.csv", std::ios::app);

    if (!csv.is_open()) {
        std::cerr << "Unable to open roofline.csv\n";
        return 1;
    }

    if (add_header) {
        csv << "machine,dwarf name,AI,bandwidth (GB/s),"
               "performance (GFLOP/s),threads,checksum\n";
    }

    std::cout
        << "Starting GPU Roofline profiling on machine: "
        << machine_name << '\n';

    run_gpu_benchmark(N, machine_name, csv);

    std::cout
        << "GPU sweep completed. Output appended to roofline.csv\n";

    return 0;
}