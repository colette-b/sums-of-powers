#include<iostream>

constexpr int GPU_BLOCK_SIZE = 256;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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
    return os << "(" << p.lo << ", " << p.hi << ")";
}

std::ostream& operator<<(std::ostream& os, __int128_t x) {
	if(x == 0) {
        return os << "0";
    }
    if(x < 0) {
        return os << "-" << -x;
    }
    std::string decimal;
    while(x > 0) {
        decimal = char(x%10 + '0') + decimal;
        x /= 10;
    }
    return os << decimal;
}

std::istream& operator>>(std::istream& is, __int128_t& x) {
    std::string s;
    is >> s;
    x = 0;
    for(int i = (s[0] == '-' ? 1 : 0); i < s.length(); i++) {
        x *= 10;
        x += (s[i] - '0');
    }
    if(s[0] == '-') {
        x *= -1;
    }
    return is;
}