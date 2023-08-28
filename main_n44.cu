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
thrust::host_vector<data_t> h_sums2(K);
thrust::host_vector<Pair> h_sums2_components(K);

void initialize_sums2() {
    int ctr = 0;
    for(int i = 0; i < N; i++) {
        for(int j = i; j < N; j++) {
            h_sums2_components[ctr] = {i, j};
            h_sums2[ctr] = mypow(i, E) + mypow(j, E);
            ctr++;
        }
    }
    assert(ctr == K);
    thrust::device_vector<data_t> s = h_sums2;
    thrust::device_vector<Pair> p = h_sums2_components;
    thrust::sort_by_key(s.begin(), s.end(), p.begin());
    h_sums2 = s;
    h_sums2_components = p;
    std::cerr << "initialization done\n";
}

int main() {
    initialize_sums2();
    SortedSums<data_t, Pair, Pair, HiLoCondition> ss(
        30 << 20,
        1 << 15,
        h_sums2,
        h_sums2,
        h_sums2_components,
        h_sums2_components
    );
    SpecializedLogger fcl;
    size_t final_size = ss.check_large_range(0, h_sums2[K - 1], fcl);
    std::cerr << "Final size: " << final_size << std::endl;
    //restore<data_t, Pair, Pair, HiLoCondition>(h_sums2, h_sums2, h_sums2_components, h_sums2_components, 2056364173794800LL);
}