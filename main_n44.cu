#include <bits/stdc++.h>
#include "sortedsums.cu"

using data_t = __int128_t;

struct HiLoCondition {
    __device__
    static bool condition(Pair x, Pair y) {
        return x.hi <= y.lo;
    }
};

constexpr int N = 1000;
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
    thrust::sort_by_key(h_sums2.begin(), h_sums2.end(), h_sums2_components.begin());
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
}