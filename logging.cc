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

template<int timepoints_count, typename... outputs>
struct FunctionCallLogger {
    std::vector<std::array<duration, timepoints_count - 1>> timings;
    std::vector<std::tuple<outputs...>> outs;
    std::array<time_point, timepoints_count> current_time_points;
    int current_item_count = 0;
    int current_time_point = 0;

    FunctionCallLogger() : outs(1) { }

    void time_tick() {
        current_time_points[current_time_point++] = system_clock::now();
        check_if_completed_and_cleanup();
    }

    template<int index> void set(typename std::tuple_element<index, std::tuple<outputs...>>::type item) {
        std::get<index>(outs[outs.size() - 1]) = item;
        current_item_count++;
        check_if_completed_and_cleanup();
    }

    void check_if_completed_and_cleanup() {
        if(current_item_count < sizeof...(outputs) or current_time_point < timepoints_count)
            return;
        current_item_count = 0;
        outs.push_back(std::tuple<outputs...>());
        current_time_point = 0;
        std::array<duration, timepoints_count - 1> current_timings;
        for(int i = 0; i < timepoints_count - 1; i++)
            current_timings[i] = current_time_points[i + 1] - current_time_points[i];
        timings.push_back(current_timings);
        show();
    }

    void show() {
        int last = timings.size() - 1;
        std::cout << "last=" << last << "\t";
        for(duration d : timings[last])
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(d).count() << "\t";
        std::cout << outs[last] << "\n";
    }
};

/*
int main() {
    FunctionCallLogger<3, int, double> fcl;
    fcl.set<0>(1);
    fcl.set<1>(1e39);
    fcl.time_tick();
    fcl.time_tick();
    fcl.time_tick();
    fcl.show();
    //std::vector<std::tuple<int, double>> v;
    //v.push_back(std::make_tuple(1, 1.0f));
}
*/