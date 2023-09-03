#include <chrono>
#include <vector>
#include <tuple>
#include <iostream>
#include <array>

using ::std::chrono::system_clock;
using time_point = std::chrono::time_point<system_clock, system_clock::duration>;
using duration = system_clock::duration;

template<typename... Ts>
std::ostream& operator<<(std::ostream& os, std::tuple<Ts...> const& theTuple) {
    std::apply(
        [&os](Ts const&... tupleArgs) {
            os << '[';
            std::size_t n{0};
            ((os << tupleArgs << (++n != sizeof...(Ts) ? ", " : "")), ...);
            os << ']';
        }, 
        theTuple
    );
    return os;
}

template<int timepoints_count, bool loud, typename... outputs>
struct FunctionCallLogger {
    std::vector<std::array<duration, timepoints_count - 1>> timings;
    std::vector<std::tuple<outputs...>> outs;
    std::array<time_point, timepoints_count> current_time_points;
    uint current_item_count = 0;
    uint current_time_point = 0;

    FunctionCallLogger() : outs(1) { }

    void time_tick() {
        if constexpr(loud) {
            std::cerr << "tick" << current_time_point << std::endl;
        }
        current_time_points[current_time_point++] = system_clock::now();
        check_if_completed_and_cleanup();
    }

    template<int index> 
    void set(typename std::tuple_element<index, std::tuple<outputs...>>::type item) {
        std::get<index>(outs[outs.size() - 1]) = item;
        current_item_count++;
        check_if_completed_and_cleanup();
    }

    void cleanup() {
        current_item_count = 0;
        current_time_point = 0;
    }

    void check_if_completed_and_cleanup() {
        if(current_item_count < sizeof...(outputs) or current_time_point < timepoints_count)
            return;
        outs.push_back(std::tuple<outputs...>());
        std::array<duration, timepoints_count - 1> current_timings;
        for(int i = 0; i < timepoints_count - 1; i++)
            current_timings[i] = current_time_points[i + 1] - current_time_points[i];
        timings.push_back(current_timings);
        cleanup();
        //show();
    }

    int last() {
        return timings.size() - 1;
    }

    void _show(std::string end) {
        std::cerr << "id=" << last() << "\t";
        for(duration d : timings[last()])
            std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>(d).count() << "\t";
        std::cerr << outs[last()] << end;
    }

    virtual void show() {
        _show("\n");
    }
};

struct SpecializedLogger : FunctionCallLogger<9, false, int> {
    using super_t = FunctionCallLogger<9, false, int>;
    long long total_seen = 0;

    void show() override {
        if(timings.size() == 0)
            return;
        FunctionCallLogger::_show("");
        duration total = std::accumulate(timings[last()].begin(), timings[last()].end(), duration(0));
        std::cerr << "\tspeed=" << std::get<0>(outs[last()]) / std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(total).count() / 1000 << "M/s\n";
        total_seen += std::get<0>(outs[last()]);
    }
};

template<typename T>
void print_device_vector(thrust::device_vector<T>& d_vec) {
    cudaDeviceSynchronize();
    std::cerr << "!!!";
    thrust::host_vector<T> h_vec = d_vec;
    std::cerr << "!!!";
    for(int i = 0; i < std::min(size_t(10), h_vec.size()); i++) {
        std::cerr << h_vec[i] << " ";
    }
    std::cerr << "...";
    for(int i = h_vec.size() - 10; i < h_vec.size(); i++) {
        std::cerr << h_vec[i] << " ";
    }
    std::cerr << "(" << h_vec.size() << ")\n";
}