#include <bits/stdc++.h>
#include "sortedsums.cu"

using data_t = __int128_t;

struct HiLoCondition {
    __device__ __host__
    static bool condition(Pair x, Pair y) {
        return x.hi <= y.lo;
    }
};

constexpr int N = 5000;
constexpr int K = N * (N + 1) / 2;
constexpr int E = 7;

thrust::device_vector<data_t> sums2;
thrust::device_vector<Pair> sums2_components;

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
    gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
    thrust::sort_by_key(sums2.begin(), sums2.end(), sums2_components.begin());
    gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());
    std::cerr << "initialization done\n";
}

int main() {
    initialize_sums2();
    SortedSums<data_t, Pair, Pair, HiLoCondition> ss(
        50 << 20,
        1 << 15,
        sums2,
        sums2,
        sums2_components,
        sums2_components
    );
    SpecializedLogger fcl;
    size_t final_size = ss.check_large_range(mypow(N, E) * 0.593129, mypow(N, E), fcl);
    std::cerr << "Final size: " << final_size << std::endl;
    //restore<data_t, Pair, Pair, HiLoCondition>(h_sums2, h_sums2, h_sums2_components, h_sums2_components, 2056364173794800LL);
}