#include <gtest/gtest.h>
#include "basic.cu"

TEST(mypow_test, HandlesExponent1) {
    EXPECT_EQ(mypow(10, 1), 10);
    EXPECT_EQ(mypow(1, 1), 1);
    EXPECT_EQ(mypow(123, 1), 123);
}

TEST(mypow_test, HandlesHigherExponents) {
    EXPECT_EQ(mypow(10, 2), 100);
    EXPECT_EQ(mypow(2, 30), 1<<30);
    EXPECT_EQ(mypow(3, 4), 81);
    EXPECT_EQ(mypow(-3, 5), -243);
}

TEST(pair_test, Printing) {
    Pair p(12, 34);
    std::stringstream ss;
    ss << p;
    EXPECT_EQ(ss.str(), "(12, 34)");
}

std::string get_string(__int128_t x) {
    std::ostringstream ss;
    ss << x;
    return ss.str();
}

TEST(int128_test, InsertionOp) {
    EXPECT_EQ(get_string(1), "1");
    EXPECT_EQ(get_string(mypow(10, 20)), "100000000000000000000");
    EXPECT_EQ(get_string(mypow(-9, 25)), "-717897987691852588770249");
}

__int128_t get_number(std::string s) {
    std::istringstream ss(s);
    __int128_t x;
    ss >> x;
    return x;
}
TEST(int128_test, ExtractionOp) {
    EXPECT_EQ(get_number("123"), 123);
    EXPECT_EQ(get_number("-1000"), -1000);
    EXPECT_EQ(get_number("0"), 0);
    EXPECT_EQ(get_number("1237940039285380274899124224"), mypow(2, 90));
    EXPECT_EQ(get_number("-2535301200456458802993406410752"), mypow(-2, 101));
}