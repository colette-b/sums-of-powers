#include <bits/stdc++.h>
#include "sortedsums.cu"

using data_t = __int128_t;

struct LeqCondition {
    __device__
    static bool condition(data_t x, data_t y) {
        return x < y;
    }
};

int main() {
    thrust::host_vector<data_t> powers;
    thrust::host_vector<bool> dummy;
    int N = 1000000;
    for(data_t i = 0; i < N; i++) {
        powers.push_back(i * i * i * i);
        dummy.push_back(true);
    }
    SortedSums<data_t, data_t, data_t, LeqCondition> ss(
        30 << 20,
        1 << 15,
        powers,
        powers,
        powers,
        powers
    );
    SpecializedLogger fcl;
    size_t final_size = ss.check_large_range(0, powers[N - 1], fcl);
    std::cerr << "Final size: " << final_size << std::endl;
}