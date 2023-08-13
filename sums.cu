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

constexpr int N = 7 * 1024;
constexpr int E = 7;
constexpr int K = N * (N + 1) / 2;
constexpr int MAX_BATCH_SIZE = 50 << 20;
constexpr int GPU_BLOCK_SIZE = 256;
constexpr int MAX_EXPECTED_COLLISIONS = 1 << 10;   // we expect that more collisions won't happen in one segment
static_assert(K % GPU_BLOCK_SIZE == 0);

/* sums2 is sorted,
 * sums2_components[i].first <= sums2_components[i].second, and 
 * sums2[i] = sums2_components[i].first^E + sums2_components[i].second^E
 */
thrust::device_vector<data_t> sums2(K);
thrust::device_vector<Pair> sums2_components(K);     
thrust::device_vector<data_t> collisions_collected(MAX_EXPECTED_COLLISIONS);

void initialize_sums2() {
    thrust::host_vector<data_t> h_sums2(K);
    thrust::host_vector<Pair> h_sums2_components(K);
    int ctr = 0;
    for(int i = 0; i < N; i++) {
        for(int j = i; j < N; j++) {
            h_sums2_components[ctr] = {i, j};
            h_sums2[ctr] = mypow(i, E) + mypow(j, E);
            ctr++;
        }
    }
    assert(ctr == K);
    sums2 = h_sums2;
    sums2_components = h_sums2_components;
    thrust::sort_by_key(sums2.begin(), sums2.end(), sums2_components.begin());
}

thrust::device_vector<int> deposit_sizes(K);
thrust::device_vector<int> prefix_sums(K);
thrust::device_vector<data_t> items(MAX_BATCH_SIZE + 1);
thrust::device_vector<data_t> lowerbound_args(K);
thrust::device_vector<int> lowerbounds(K);
thrust::device_vector<int> eq_check(MAX_BATCH_SIZE);

__global__
void precount(data_t H, data_t *ptr_sums2, Pair *ptr_sums2_components, int *ptr_lowerbounds, int *destination) {
    int count = 0;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    for(int j = ptr_lowerbounds[i]; j < K and ptr_sums2[j] < H - ptr_sums2[i]; j++) {
        if(ptr_sums2_components[i].hi <= ptr_sums2_components[j].lo) {
            count++;
        }
    }
    destination[i] = count;
}

__global__
void deposit(data_t H, data_t *ptr_sums2, Pair *ptr_sums2_components, int *ptr_lowerbounds, int *prefix_sums, data_t* destination) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int count = prefix_sums[i];
    for(int j = ptr_lowerbounds[i]; j < K and ptr_sums2[j] < H - ptr_sums2[i]; j++) {
        if(ptr_sums2_components[i].hi <= ptr_sums2_components[j].lo) {
            destination[count] = ptr_sums2[i] + ptr_sums2[j];
            count++;
        }
    }
}

__global__
void check_consecutive_eq(data_t *ptr_items, int *destination) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    destination[i] = (ptr_items[i] == ptr_items[i+1]);
}

class range_too_large_error : public std::runtime_error {
    public:
    range_too_large_error() : std::runtime_error("total_deposit_size > MAX_BATCH_SIZE") { }
};

int check_range(data_t L, data_t H, FunctionCallLogger<9, int>& fcl) {
    fcl.time_tick();
    thrust::fill(lowerbound_args.begin(), lowerbound_args.end(), L);
    fcl.time_tick();
    thrust::transform(lowerbound_args.begin(), lowerbound_args.end(), sums2.begin(), lowerbound_args.begin(), thrust::minus<data_t>());
    fcl.time_tick();
    thrust::lower_bound(sums2.begin(), sums2.end(), lowerbound_args.begin(), lowerbound_args.end(), lowerbounds.begin());
    fcl.time_tick();
    precount<<<K/GPU_BLOCK_SIZE, GPU_BLOCK_SIZE>>>(
        H,
        thrust::raw_pointer_cast(sums2.data()),
        thrust::raw_pointer_cast(sums2_components.data()),
        thrust::raw_pointer_cast(lowerbounds.data()),
        thrust::raw_pointer_cast(deposit_sizes.data())
    );
    cudaDeviceSynchronize();
    fcl.time_tick();
    int total_deposit_size = thrust::reduce(deposit_sizes.begin(), deposit_sizes.end());
    if(total_deposit_size > MAX_BATCH_SIZE) {
        fcl.cleanup();
        throw range_too_large_error();
    }
    thrust::exclusive_scan(deposit_sizes.begin(), deposit_sizes.end(), deposit_sizes.begin());
    fcl.time_tick();
    deposit<<<K/GPU_BLOCK_SIZE, GPU_BLOCK_SIZE>>>(
        H,
        thrust::raw_pointer_cast(sums2.data()),
        thrust::raw_pointer_cast(sums2_components.data()),
        thrust::raw_pointer_cast(lowerbounds.data()),
        thrust::raw_pointer_cast(deposit_sizes.data()),
        thrust::raw_pointer_cast(items.data())
    );
    cudaDeviceSynchronize();
    fcl.time_tick();
    thrust::sort(items.begin(), items.begin() + total_deposit_size);

    fcl.time_tick();
    check_consecutive_eq<<<(total_deposit_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE, GPU_BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(items.data()),
        thrust::raw_pointer_cast(eq_check.data())
    );
    int collision_happened = thrust::reduce(eq_check.begin(), eq_check.begin() + total_deposit_size - 1);
    cudaDeviceSynchronize();
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
            std::cout << collisions[i] << " ";
        }
        std::cout << "\n";
    }
    /*
    std::cout << "fill:               " << (t2 - t1) / std::chrono::milliseconds(1) << "ms\n";
    std::cout << "transform:          " << (t3 - t2) / std::chrono::milliseconds(1) << "ms\n";
    std::cout << "lowerbound:         " << (t4 - t3) / std::chrono::milliseconds(1) << "ms\n";
    std::cout << "precount:           " << (t5 - t4) / std::chrono::milliseconds(1) << "ms\n";
    std::cout << "scan:               " << (t6 - t5) / std::chrono::milliseconds(1) << "ms\n";
    std::cout << "deposit:            " << (t7 - t6) / std::chrono::milliseconds(1) << "ms\n";
    std::cout << "sort:               " << (t8 - t7) / std::chrono::milliseconds(1) << "ms\n";
    std::cout << "check eq:           " << (t9 - t8) / std::chrono::milliseconds(1) << "ms\n";
    */
    return total_deposit_size;
}
