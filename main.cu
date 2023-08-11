#include "sums.cu"

struct SpecializedLogger : FunctionCallLogger<9, int> {
    long long total_seen = 0;

    void show() override {
        FunctionCallLogger::_show("");
        duration total = std::accumulate(timings[last()].begin(), timings[last()].end(), duration(0));
        std::cout << "\tspeed=" << std::get<0>(outs[last()]) / std::chrono::duration_cast<std::chrono::milliseconds>(total).count() / 1000 << "M/s\n";
        total_seen += std::get<0>(outs[last()]);
    }
};

int main() {
    initialize_sums2();
    SpecializedLogger fcl;

    data_t L = 0;
    data_t jump = 1LL<<50;

    while(true) {
        try {
            int batch_size = check_range(L, L + jump, fcl);
            if(batch_size < (35LL << 20)) {
                jump *= 1.05;
            }
            L += jump;
        } catch (range_too_large_error& e) {
            std::cout << "too many items in range; decreasing jump\n";
            jump /= 1.1;
        }
    }
}