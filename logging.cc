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
        show();
    }

    int last() {
        return timings.size() - 1;
    }

    void _show(std::string end) {
        std::cout << "id=" << last() << "\t";
        for(duration d : timings[last()])
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(d).count() << "\t";
        std::cout << outs[last()] << end;
    }

    virtual void show() {
        _show("\n");
    }
};
