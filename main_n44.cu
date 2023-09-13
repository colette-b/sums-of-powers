#include <bits/stdc++.h>
#include "sortedsums.cu"

struct HiLoCondition {
    __device__ __host__
    static bool condition(Pair x, Pair y) {
        return x.hi <= y.lo;
    }
};

using data_t = __int128_t;

constexpr int N = 5000;
constexpr int K = N * (N + 1) / 2;
constexpr int E = 7;

thrust::device_vector<data_t> sums2;
thrust::device_vector<Pair> sums2_components;
std::map<data_t, std::pair<int, int>> reverse_lookup;

void initialize_sums2() {
    thrust::host_vector<data_t> h_sums2(K);
    thrust::host_vector<Pair> h_sums2_components(K);
    int ctr = 0;
    for(int i = 0; i < N; i++) {
        for(int j = i; j < N; j++) {
            h_sums2_components[ctr] = {i, j};
            h_sums2[ctr] = mypow(i, E) + mypow(j, E);
            reverse_lookup[mypow(i, E) + mypow(j, E)] = std::make_pair(i, j);
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

void print_collision(data_t P, SortedSums<data_t, Pair, Pair, HiLoCondition>& ss) {
    std::vector<std::pair<data_t, data_t>> possible_sums = ss.restore_collision(P);
    std::cout << "collision: " << P;
    int gcd = 0;
    for(auto& pair : possible_sums) {
        auto xy = reverse_lookup[pair.first], zt = reverse_lookup[pair.second];
        int summands[4] = {xy.first, xy.second, zt.first, zt.second};
        fmt::print(" = {}+{}+{}+{}", summands[0], summands[1], summands[2], summands[3]);
        gcd = std::gcd(gcd, summands[0]);
        gcd = std::gcd(gcd, summands[1]);
        gcd = std::gcd(gcd, summands[2]);
        gcd = std::gcd(gcd, summands[3]);
    }
    if(gcd == 1) {
        std::cout << "\t(primitive)" << std::endl;
    } else {
        std::cout << "\tnot primitive, gcd=" << gcd << std::endl;
    }
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
    size_t final_size = ss.check_large_range(
        mypow(N, E) * 0, 
        mypow(N, E), 
        fcl,
        [&ss](data_t P) {print_collision(P, ss);}
    );
    std::cerr << "Final size: " << final_size << std::endl;
}