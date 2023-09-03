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
 *  some bool function condition(T t, S s),
 *  output:
 *  all collisions in the interval, where we iterate over all
 *  sums A[i] + B[j] such that condition(A_param[i], B_param[j]) holds.
 */

class range_too_large_error : public std::runtime_error {
    public:
    range_too_large_error() : std::runtime_error("total_deposit_size > MAX_BATCH_SIZE") { }
};

template<typename data_t, typename Aparam_t, typename Bparam_t>
struct SortedSumsPointers {
    size_t A_size, B_size;
    data_t *A, *B;
    void *items;
    Aparam_t *A_param;
    Bparam_t *B_param;
    int *lowerbounds, *prefix_sums;
};

template<typename data_t, typename Aparam_t, typename Bparam_t, typename Condition>
__global__
void precount(data_t H, SortedSumsPointers<data_t, Aparam_t, Bparam_t> ssp) {
    int count = 0;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= ssp.A_size)
        return;
    data_t diff = H - ssp.A[i];
    for(int j = ssp.lowerbounds[i]; j < ssp.B_size && ssp.B[j] < diff && ssp.B[j] >= ssp.A[i]; j++) {
        if(Condition::condition(ssp.A_param[i], ssp.B_param[j])) {
            count++;
        }
    }
    ssp.prefix_sums[i] = count;
}

template<typename item_t, typename data_t, typename Aparam_t, typename Bparam_t, typename Condition>
__global__
void deposit(data_t H, SortedSumsPointers<data_t, Aparam_t, Bparam_t> ssp) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= ssp.A_size)
        return;
    int count = ssp.prefix_sums[i];
    int finish = ssp.prefix_sums[i + 1];
    item_t *ptr = reinterpret_cast<item_t*>(ssp.items);
    for(int j = ssp.lowerbounds[i]; count < finish; j++) {
        if(Condition::condition(ssp.A_param[i], ssp.B_param[j])) {
            ptr[count] = ssp.A[i] + ssp.B[j];
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
std::vector<std::pair<data_t, data_t>> restore(thrust::host_vector<data_t> A, thrust::host_vector<data_t> B, 
                                        thrust::host_vector<Aparam_t> Aparam, thrust::host_vector<Bparam_t> Bparam, data_t collision) {
    int l = 0, r = B.size() - 1;
    std::vector<std::pair<data_t, data_t>> values;
    while(l < A.size() and r >= 0) {
        if(A[l] + B[r] < collision) {
            l++;
            continue;
        }
        if(A[l] + B[r] > collision) {
            r--;
            continue;
        }
        if(A[l] + B[r] == collision) {
            if(Condition::condition(Aparam[l], Bparam[r])) {
                values.push_back(std::make_pair(A[l], B[r]));
            }
            l++;
            r--;
        }
    }
    return values;
}

template<typename data_t, typename Aparam_t, typename Bparam_t, typename Condition>
class SortedSums {
    int A_size, B_size;
    int MAX_BATCH_SIZE, MAX_EXPECTED_COLLISIONS;
    thrust::device_vector<data_t> &A, &B;
    thrust::device_vector<data_t> items, lowerbound_args, collisions_collected;
    thrust::device_vector<Aparam_t> &A_param;
    thrust::device_vector<Bparam_t> &B_param;
    thrust::device_vector<int> prefix_sums, lowerbounds, eq_check;

    public:
    SortedSums(
        int MAX_BATCH_SIZE,
        int MAX_EXPECTED_COLLISIONS,
        thrust::device_vector<data_t> &A, 
        thrust::device_vector<data_t> &B,
        thrust::device_vector<Aparam_t> &A_param,
        thrust::device_vector<Bparam_t> &B_param
    ) : 
        A(A), B(B), A_param(A_param), B_param(B_param),
        MAX_BATCH_SIZE(MAX_BATCH_SIZE),
        MAX_EXPECTED_COLLISIONS(MAX_EXPECTED_COLLISIONS),
        A_size(A.size()), B_size(B.size()),
        prefix_sums(A_size + 1),
        lowerbound_args(A_size),
        lowerbounds(A_size),
        items(MAX_BATCH_SIZE + 1),
        eq_check(MAX_BATCH_SIZE),
        collisions_collected(MAX_EXPECTED_COLLISIONS)
    { }

    SortedSumsPointers<data_t, Aparam_t, Bparam_t> get_ssp() {
        SortedSumsPointers<data_t, Aparam_t, Bparam_t> ssp;
        ssp.A = thrust::raw_pointer_cast(A.data());
        ssp.B = thrust::raw_pointer_cast(B.data());
        ssp.A_param = thrust::raw_pointer_cast(A_param.data());
        ssp.B_param = thrust::raw_pointer_cast(B_param.data());
        ssp.items = reinterpret_cast<void*>(thrust::raw_pointer_cast(items.data()));
        ssp.A_size = A.size();
        ssp.B_size = B.size();
        ssp.lowerbounds = thrust::raw_pointer_cast(lowerbounds.data());
        ssp.prefix_sums = thrust::raw_pointer_cast(prefix_sums.data());
        return ssp;
    }

    int make_prefix_sums(data_t L, data_t H, SpecializedLogger& fcl) {
        // prepares the prefix_sums vector
        // returns the total deposit size
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
                (H, get_ssp());
        gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
        int total_deposit_size = thrust::reduce(prefix_sums.begin(), prefix_sums.end() - 1);
        thrust::exclusive_scan(prefix_sums.begin(), prefix_sums.end(), prefix_sums.begin());
        return total_deposit_size;
    }
    
    template<typename item_t, bool tick>
    int check_collisions(data_t L, data_t H, int total_deposit_size, SpecializedLogger& fcl) {
        if(tick)
            fcl.time_tick();
        deposit<item_t, data_t, Aparam_t, Bparam_t, Condition>
               <<<1 + A_size/GPU_BLOCK_SIZE, GPU_BLOCK_SIZE>>>(H, get_ssp());
        gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
        if(tick)
            fcl.time_tick();
        item_t *items_ptr = reinterpret_cast<item_t*>(thrust::raw_pointer_cast(items.data()));
        thrust::sort(thrust::device, items_ptr, items_ptr + total_deposit_size);
        gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
        if(tick)
            fcl.time_tick();
        check_consecutive_eq<<<(total_deposit_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE, GPU_BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(eq_check.data()), 
            items_ptr
        );
        int collision_count = thrust::reduce(eq_check.begin(), eq_check.begin() + total_deposit_size - 1);
        return collision_count;
    }

    int check_range(data_t L, data_t H, SpecializedLogger& fcl) {
        int total_deposit_size = make_prefix_sums(L, H, fcl);
        if(total_deposit_size > MAX_BATCH_SIZE) {
            fcl.cleanup();
            throw range_too_large_error();
        }
        if(total_deposit_size <= 1) {
            fcl.cleanup();
            return total_deposit_size;
        }
        gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
        
/*      fcl.time_tick();
        deposit<data_t, data_t, Aparam_t, Bparam_t, Condition>
               <<<1 + A_size/GPU_BLOCK_SIZE, GPU_BLOCK_SIZE>>>(
            H, get_ssp());
        gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
        fcl.time_tick();
        thrust::sort(items.begin(), items.begin() + total_deposit_size);
        gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
        fcl.time_tick();
        check_consecutive_eq<<<(total_deposit_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE, GPU_BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(eq_check.data()), 
            thrust::raw_pointer_cast(items.data())
        );
        gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
        int collision_happened = thrust::reduce(eq_check.begin(), eq_check.begin() + total_deposit_size - 1);
*/
        //int collision_happened = check_collisions<unsigned long long, true>(L, H, total_deposit_size, fcl);
        //if(collision_happened) {
        //    std::cerr << "seen " << collision_happened << " quick collisions\n";
        //    check_collisions<data_t, false>(L, H, total_deposit_size, fcl);
        //}
        int collision_happened = check_collisions<data_t, true>(L, H, total_deposit_size, fcl);

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
        for(int iter = 0; current_L < H; iter++){
            try {
                data_t current_H = std::min(current_L + jump, H);
                int batch_size = check_range(current_L, current_H, fcl);
                if(iter % 10 == 9) {
                    std::cerr << "[" << current_L << ", " << current_H << ")\t";
                    fcl.show();
                }
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
