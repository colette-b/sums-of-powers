#include <random>
#include <limits>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/binary_search.h>

#include "basic.cu"
#include "logging.cc"

/*  input: 
 *  A, B are two arrays (of int-like type),
 *  A_param, B_param are two arrays (of any types T, S),
 *  interval [L, H),
 *  output:
 *  all collisions in the interval, where we iterate over all
 *  sums A[i] + B[j] such that condition(A_param[i], B_param[j]) holds.
 */

class range_too_large_error : public std::runtime_error {
    public:
    range_too_large_error() : std::runtime_error("total_deposit_size > MAX_BATCH_SIZE") { }
};

template<typename data_t, typename Aparam_t, typename Bparam_t, typename Condition>
__global__
void precount(data_t H, size_t A_size, size_t B_size, data_t *raw_A, data_t *raw_B, int *raw_lowerbounds, 
                Aparam_t *raw_A_param, Bparam_t *raw_B_param, int *raw_prefix_sums) {
    int count = 0;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= A_size)
        return;
    for(int j = raw_lowerbounds[i]; j < B_size && raw_A[i] + raw_B[j] < H; j++) {
        if(Condition::condition(raw_A_param[i], raw_B_param[j])) {
            count++;
        }
    }
    raw_prefix_sums[i] = count;
}

template<typename data_t, typename Aparam_t, typename Bparam_t, typename Condition>
__global__
void deposit(data_t H, size_t A_size, size_t B_size, data_t *raw_A, data_t *raw_B, int *raw_lowerbounds, 
             Aparam_t *raw_A_param, Bparam_t *raw_B_param, int *raw_prefix_sums, data_t *raw_items) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= A_size)
        return;
    int count = raw_prefix_sums[i];
    for(int j = raw_lowerbounds[i]; j < B_size and raw_A[i] + raw_B[j] < H; j++) {
        if(Condition::condition(raw_A_param[i], raw_B_param[j])) {
            raw_items[count] = raw_A[i] + raw_B[j];
            count++;
        }
    }
}

template<typename data_t, typename bool_t>
__global__
void check_consecutive_eq(bool_t *raw_eq_check, data_t *raw_items) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    raw_eq_check[i] = (raw_items[i] == raw_items[i+1]);
}

template<typename data_t, typename Aparam_t, typename Bparam_t, typename Condition>
class SortedSums {
    thrust::device_vector<data_t> A, B;
    thrust::device_vector<Aparam_t> A_param;
    thrust::device_vector<Bparam_t> B_param;
    int A_size, B_size;
    int MAX_BATCH_SIZE;
    int MAX_EXPECTED_COLLISIONS;
    thrust::device_vector<int> prefix_sums;
    thrust::device_vector<data_t> items;
    thrust::device_vector<data_t> lowerbound_args;
    thrust::device_vector<int> lowerbounds;
    thrust::device_vector<int> eq_check;
    thrust::device_vector<data_t> collisions_collected;
    // raw pointers 
    data_t *raw_A, *raw_B, *raw_lowerbound_args, *raw_items;
    Aparam_t *raw_A_param;
    Bparam_t *raw_B_param;
    int *raw_lowerbounds;
    int *raw_prefix_sums;
    int *raw_eq_check;

    public:
    SortedSums(
        int _MAX_BATCH_SIZE,
        int _MAX_EXPECTED_COLLISIONS,
        thrust::host_vector<data_t> hA, 
        thrust::host_vector<data_t> hB,
        thrust::host_vector<Aparam_t> hA_param,
        thrust::host_vector<Bparam_t> hB_param
    ) : 
        MAX_BATCH_SIZE(_MAX_BATCH_SIZE),
        MAX_EXPECTED_COLLISIONS(_MAX_EXPECTED_COLLISIONS),
        A_size(hA.size()), B_size(hB.size()),
        prefix_sums(A_size),
        lowerbound_args(A_size),
        lowerbounds(A_size),
        items(MAX_BATCH_SIZE + 1),
        eq_check(MAX_BATCH_SIZE),
        collisions_collected(MAX_EXPECTED_COLLISIONS),
        raw_lowerbound_args(thrust::raw_pointer_cast(lowerbound_args.data())),
        raw_lowerbounds(thrust::raw_pointer_cast(lowerbounds.data())),
        raw_prefix_sums(thrust::raw_pointer_cast(prefix_sums.data())),
        raw_items(thrust::raw_pointer_cast(items.data())),
        raw_eq_check(thrust::raw_pointer_cast(eq_check.data()))
    {
        A = hA;
        B = hB;
        A_param = hA_param;
        B_param = hB_param;
        raw_A = thrust::raw_pointer_cast(A.data());
        raw_B = thrust::raw_pointer_cast(B.data());
        raw_A_param = thrust::raw_pointer_cast(A_param.data());
        raw_B_param = thrust::raw_pointer_cast(B_param.data());

    }

    //template<typename Logger>
    int check_range(data_t L, data_t H, SpecializedLogger& fcl) {
        fcl.time_tick();
        thrust::fill(lowerbound_args.begin(), lowerbound_args.end(), L);
        gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
        fcl.time_tick();
        thrust::transform(lowerbound_args.begin(), lowerbound_args.end(), A.begin(), lowerbound_args.begin(), thrust::minus<data_t>());
        gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
        fcl.time_tick();
        thrust::lower_bound(B.begin(), B.end(), lowerbound_args.begin(), lowerbound_args.end(), lowerbounds.begin());
        gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
        fcl.time_tick();
        precount<data_t, Aparam_t, Bparam_t, Condition>
                <<<1 + A_size/GPU_BLOCK_SIZE, GPU_BLOCK_SIZE>>>
                (
                H, A_size, B_size, raw_A, raw_B, raw_lowerbounds, raw_A_param, raw_B_param, raw_prefix_sums
        );
        gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
        int total_deposit_size = thrust::reduce(prefix_sums.begin(), prefix_sums.end());
        if(total_deposit_size > MAX_BATCH_SIZE) {
            fcl.cleanup();
            throw range_too_large_error();
        }
        thrust::exclusive_scan(prefix_sums.begin(), prefix_sums.end(), prefix_sums.begin());
        gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
        fcl.time_tick();
        deposit<data_t, Aparam_t, Bparam_t, Condition>
               <<<1 + A_size/GPU_BLOCK_SIZE, GPU_BLOCK_SIZE>>>(
            H, A_size, B_size, raw_A, raw_B, raw_lowerbounds, raw_A_param, raw_B_param, raw_prefix_sums, raw_items
        );
        gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
        fcl.time_tick();
        thrust::sort(items.begin(), items.begin() + total_deposit_size);
        gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
        fcl.time_tick();
        if(total_deposit_size > 1) {
            check_consecutive_eq<<<(total_deposit_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE, GPU_BLOCK_SIZE>>>(
                raw_eq_check, raw_items
            );
        }
        gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
        int collision_happened = (total_deposit_size > 1) ? thrust::reduce(eq_check.begin(), eq_check.begin() + total_deposit_size - 1) : 0;
        fcl.time_tick();
        gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
        thrust::host_vector<data_t> collisions(0);
        if(collision_happened > 0) {
            thrust::copy_if(
                items.begin(), 
                items.begin() + total_deposit_size - 1, 
                eq_check.begin(), 
                collisions_collected.begin(), 
                thrust::identity<bool>()
            );
            collisions = collisions_collected;
        }
        fcl.time_tick();
        fcl.set<0>(total_deposit_size);
        if(collision_happened) {
            for(int i = 0; i < collision_happened; i++) {
                std::cout << collisions[i] << "\n";
            }
        }
        gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
        return total_deposit_size;
    }

    template<typename Logger>
    size_t check_large_range(data_t L, data_t H, Logger& fcl) {
        data_t current_L = L, jump = 1 << 20;
        size_t total = 0;
        while(current_L < H) {
            try {
                data_t current_H = std::min(current_L + jump, H);
                int batch_size = check_range(current_L, current_H, fcl);
                if(batch_size < 0.1 * MAX_BATCH_SIZE) {
                    jump *= 2;
                }
                if(batch_size < 0.5 * MAX_BATCH_SIZE) {
                    jump *= 1.05;
                }
                total += batch_size;
                current_L = current_H;
            } catch (range_too_large_error& e) {
                std::cout << "too many items in range; decreasing jump\n";
                jump /= 1.3;
            }
        }
        return total;
    }
};
