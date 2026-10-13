#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kThreadsPerBlock = 128;
constexpr int kBlocksPerSm = 8;
constexpr int kGridIterations = 512;
constexpr int kWarmupIterations = 20;
constexpr int kMeasurementIterations = 10;
constexpr int kTrimCount = 2;

// Source - https://stackoverflow.com/a/14038590
// Posted by talonmies, modified by community. See post 'Timeline' for change history
// Retrieved 2026-10-09, License - CC BY-SA 4.0

#define gpuErrchk(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while (0)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template<int ROW_STRIDE, int THREAD_GROUP_SIZE>
__global__ void submatrix_sum(const float* input, float* output)
{
    static_assert(THREAD_GROUP_SIZE == 8 || THREAD_GROUP_SIZE == 16);
    static_assert(ROW_STRIDE >= THREAD_GROUP_SIZE);

    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int thread_group_id = thread_id / THREAD_GROUP_SIZE;
    const int element_in_row = thread_id % THREAD_GROUP_SIZE;
    const int num_threads = gridDim.x * blockDim.x;
    const int num_thread_groups = num_threads / THREAD_GROUP_SIZE;

    float sum = 0.0f;
    for (int grid_iteration = 0; grid_iteration < kGridIterations; ++grid_iteration) {
        const size_t row =
            static_cast<size_t>(grid_iteration) * num_thread_groups +
            thread_group_id;
        sum += input[row * ROW_STRIDE + element_in_row];
    }
    output[thread_id] = sum;
}

__global__ void fill(float* data, size_t count, uint32_t seed)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, index, 0, &state);
        data[index] = curand_uniform(&state);
    }
}

__global__ void flush_l2(float* data, size_t count)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) data[index] += 1.0f;
}

float trimmed_mean(std::vector<float> samples)
{
    std::sort(samples.begin(), samples.end());
    const auto first = samples.begin() + kTrimCount;
    const auto last = samples.end() - kTrimCount;
    return std::accumulate(first, last, 0.0f) /
           static_cast<float>(last - first);
}

float trimmed_stdev(const std::vector<float>& samples, float mean)
{
    std::vector<float> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const auto first = sorted.begin() + kTrimCount;
    const auto last = sorted.end() - kTrimCount;
    float squared_error = 0.0f;
    for (auto iterator = first; iterator != last; ++iterator) {
        const float difference = *iterator - mean;
        squared_error += difference * difference;
    }
    return std::sqrt(squared_error / static_cast<float>(last - first));
}

template<int ROW_STRIDE, int THREAD_GROUP_SIZE>
void run_case(
    int num_blocks, int l2_size_bytes, bool perform_global_warmup, bool ncu)
{
    const int num_threads = num_blocks * kThreadsPerBlock;
    const int num_thread_groups = num_threads / THREAD_GROUP_SIZE;
    const size_t input_elements =
        static_cast<size_t>(kGridIterations) * num_thread_groups * ROW_STRIDE;
    const size_t accessed_input_elements =
        static_cast<size_t>(kGridIterations) * num_threads;
    const size_t output_elements = num_threads;
    const size_t logical_bytes =
        (accessed_input_elements + output_elements) * sizeof(float);
    const size_t l2_flush_elements =
        static_cast<size_t>(2) * l2_size_bytes / sizeof(float);

    float* input = nullptr;
    float* output = nullptr;
    float* l2_flush = nullptr;
    gpuErrchk(cudaMalloc(&input, input_elements * sizeof(float)));
    gpuErrchk(cudaMalloc(&output, output_elements * sizeof(float)));
    gpuErrchk(cudaMalloc(&l2_flush, l2_flush_elements * sizeof(float)));

    constexpr int utility_threads = 256;
    constexpr uint32_t base_seed = 12345U;
    fill<<<static_cast<int>((input_elements + utility_threads - 1) /
                            utility_threads), utility_threads>>>(
        input, input_elements, base_seed);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemset(output, 0, output_elements * sizeof(float)));
    gpuErrchk(cudaMemset(l2_flush, 0, l2_flush_elements * sizeof(float)));
    gpuErrchk(cudaDeviceSynchronize());

    if (perform_global_warmup) {
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        gpuErrchk(cudaEventCreate(&start));
        gpuErrchk(cudaEventCreate(&stop));
        submatrix_sum<ROW_STRIDE, THREAD_GROUP_SIZE><<<num_blocks, kThreadsPerBlock>>>(
            input, output);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaEventRecord(start));
        for (int iteration = 0; iteration < 10; ++iteration) {
            submatrix_sum<ROW_STRIDE, THREAD_GROUP_SIZE><<<num_blocks, kThreadsPerBlock>>>(
                input, output);
            gpuErrchk(cudaPeekAtLastError());
        }
        gpuErrchk(cudaEventRecord(stop));
        gpuErrchk(cudaDeviceSynchronize());
        float milliseconds = 0.0f;
        gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
        if (milliseconds <= 0.0f) throw std::runtime_error("Invalid warmup duration");
        const int iterations = std::max(
            1, static_cast<int>(std::ceil(10000.0f / milliseconds)));
        for (int iteration = 0; iteration < iterations; ++iteration) {
            submatrix_sum<ROW_STRIDE, THREAD_GROUP_SIZE><<<num_blocks, kThreadsPerBlock>>>(
                input, output);
            gpuErrchk(cudaPeekAtLastError());
        }
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaEventDestroy(stop));
        gpuErrchk(cudaEventDestroy(start));
    }

    for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
        submatrix_sum<ROW_STRIDE, THREAD_GROUP_SIZE><<<num_blocks, kThreadsPerBlock>>>(
            input, output);
        gpuErrchk(cudaPeekAtLastError());
    }
    gpuErrchk(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    std::vector<float> runtimes_us;
    for (int iteration = 0;
         iteration < kMeasurementIterations;
         ++iteration) {
        fill<<<static_cast<int>((input_elements + utility_threads - 1) /
                                utility_threads), utility_threads>>>(
            input, input_elements, base_seed + iteration + 1);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        flush_l2<<<static_cast<int>((l2_flush_elements + utility_threads - 1) /
                                    utility_threads), utility_threads>>>(
            l2_flush, l2_flush_elements);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        const bool profile_launch = ncu && iteration == kMeasurementIterations - 1;
        if (profile_launch) {
            nvtxRangePushA("ProfileRange");
            const std::string range_name = "THREAD_GROUP_SIZE=" + std::to_string(THREAD_GROUP_SIZE) +
                                           ", ROW_STRIDE=" + std::to_string(ROW_STRIDE);
            nvtxRangePushA(range_name.c_str());
        }
        gpuErrchk(cudaEventRecord(start));
        submatrix_sum<ROW_STRIDE, THREAD_GROUP_SIZE><<<num_blocks, kThreadsPerBlock>>>(
            input, output);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaEventRecord(stop));
        gpuErrchk(cudaDeviceSynchronize());
        if (profile_launch) {
            nvtxRangePop();
            nvtxRangePop();
        }
        float milliseconds = 0.0f;
        gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
        runtimes_us.push_back(milliseconds * 1000.0f);
    }
    gpuErrchk(cudaEventDestroy(stop));
    gpuErrchk(cudaEventDestroy(start));

    std::vector<float> host_input(input_elements);
    std::vector<float> host_output(output_elements);
    gpuErrchk(cudaMemcpy(
        host_input.data(), input,
        input_elements * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(
        host_output.data(), output,
        output_elements * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(l2_flush));
    gpuErrchk(cudaFree(output));
    gpuErrchk(cudaFree(input));
    bool valid = true;
    double max_abs_error = 0.0;
    for (size_t thread_id = 0; thread_id < output_elements; ++thread_id) {
        float expected = 0.0f;
        for (int iteration = 0; iteration < kGridIterations; ++iteration) {
            const size_t row = static_cast<size_t>(iteration) * num_thread_groups +
                               thread_id / THREAD_GROUP_SIZE;
            expected += host_input[row * ROW_STRIDE + thread_id % THREAD_GROUP_SIZE];
        }
        const double error = std::abs(static_cast<double>(host_output[thread_id]) - expected);
        max_abs_error = std::max(max_abs_error, error);
        if (!std::isfinite(host_output[thread_id]) || error != 0.0) valid = false;
    }

    const float mean_us = trimmed_mean(runtimes_us);
    const float stdev_us = trimmed_stdev(runtimes_us, mean_us);
    const double useful_gbps =
        static_cast<double>(logical_bytes) / (mean_us * 1.0e3);

    std::cout << "THREAD_GROUP_SIZE=" << THREAD_GROUP_SIZE
              << ", ROW_STRIDE=" << ROW_STRIDE
              << ", submatrix=" << 32 / THREAD_GROUP_SIZE << "x" << THREAD_GROUP_SIZE
              << " floats per warp"
              << ", allocated_input="
              << input_elements * sizeof(float) / 1.0e6 << " MB"
              << ", useful_bytes=" << logical_bytes
              << ", runtime=" << mean_us << " +/- " << stdev_us << " us"
              << ", useful_bandwidth=" << useful_gbps << " GB/s"
              << ", validation=" << (valid ? "OK" : "FAIL")
              << ", max_abs_error=" << max_abs_error << "\n";

    if (!valid) throw std::runtime_error("Validation failed");
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        std::string row_stride = "all";
        std::string thread_group_size = "8";
        bool ncu = false;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--ncu") {
                ncu = true;
            } else if (argument == "--row-stride" && index + 1 < argc) {
                row_stride = argv[++index];
            } else if (argument == "--thread-group-size" && index + 1 < argc) {
                thread_group_size = argv[++index];
            } else {
                throw std::invalid_argument(
                    "Usage: " + std::string(argv[0]) +
                    " [--thread-group-size 8|16] [--row-stride 8|16|32|all] [--ncu]");
            }
        }
        if (row_stride != "8" && row_stride != "16" && row_stride != "32" && row_stride != "all") {
            throw std::invalid_argument("row stride must be 8, 16, 32, or all");
        }
        if (thread_group_size != "8" && thread_group_size != "16") {
            throw std::invalid_argument("thread group size must be 8 or 16");
        }
        if (thread_group_size == "16" && row_stride == "8") {
            throw std::invalid_argument("thread group size 16 supports only row strides 16 and 32");
        }

        cudaDeviceProp properties{};
        gpuErrchk(cudaGetDeviceProperties(&properties, 0));
        const int num_blocks = properties.multiProcessorCount * kBlocksPerSm;
        std::cout << "GPU=" << properties.name
                  << ", SMs=" << properties.multiProcessorCount
                  << ", blocks=" << num_blocks
                  << ", threads_per_block=" << kThreadsPerBlock
                  << ", L2=" << properties.l2CacheSize / 1024 << " KiB\n";
        std::cout << std::fixed << std::setprecision(3);

        if (thread_group_size == "8") {
            if (row_stride == "8" || row_stride == "all") {
                run_case<8, 8>(num_blocks, properties.l2CacheSize, true, ncu);
            }
            if (row_stride == "16" || row_stride == "all") {
                run_case<16, 8>(num_blocks, properties.l2CacheSize, row_stride == "16", ncu);
            }
            if (row_stride == "32" || row_stride == "all") {
                run_case<32, 8>(num_blocks, properties.l2CacheSize, row_stride == "32", ncu);
            }
        } else {
            if (row_stride == "16" || row_stride == "all") {
                run_case<16, 16>(num_blocks, properties.l2CacheSize, true, ncu);
            }
            if (row_stride == "32" || row_stride == "all") {
                run_case<32, 16>(num_blocks, properties.l2CacheSize, row_stride == "32", ncu);
            }
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
