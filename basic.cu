#include<iostream>

using data_t = __int128_t;

__host__ __device__ 
data_t mypow(__int128_t x, __int128_t exponent) {
    __int128_t result = x;
    for(int i = 0; i < exponent - 1; i++) {
        result *= x;
    }
    return result;
}

struct Pair {
    __host__ __device__
    Pair() : lo(0), hi(0) { }
    Pair(int _lo, int _hi) : lo(_lo), hi(_hi) { }
    int lo;
    int hi;
};

std::ostream& operator<<(std::ostream& os, const Pair& p) {
    os << "(" << p.lo << ", " << p.hi << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, data_t x) {
	if(x == 0) {
        os << "0";
		return os;
    }
    if(x < 0) {
        os << "-";
        os << -x;
        return os;
    }
    while(x > 0) {
        os << char(x%10 + '0');
        x /= 10;
    }
    return os;
}
