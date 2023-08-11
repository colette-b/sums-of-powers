#include "sums.cu"

int main() {
    initialize_sums2();
    FunctionCallLogger<9, int> fcl;

    for(int run = 0; run < 1; run++) {
        check_range(1LL<<53, 1LL<<55, fcl);
    }
    return;
    
    //std::cout << "\n\n=============\n\n";
    int total = 0;
    for(int i = 0; i < 40; i++) {
        int curr = check_range(1LL<<i, 1LL<<(i+1), fcl);
        total += curr;
    }
    
    std::cout << "total: " << total << "\n";
    int total_check = check_range(1, 1LL<<40, fcl);
    assert(total == total_check);
}