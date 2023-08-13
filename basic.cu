#include<iostream>

__host__ __device__ 
__int128_t mypow(__int128_t x, __int128_t exponent) {
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

std::ostream& operator<<(std::ostream& os, __int128_t x) {
	if(x == 0) {
        os << "0";
		return os;
    }
    if(x < 0) {
        os << "-";
        os << -x;
        return os;
    }
    std::string decimal;
    while(x > 0) {
        decimal = char(x%10 + '0') + decimal;
        x /= 10;
    }
    os << decimal;
    return os;
}
