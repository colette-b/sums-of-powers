#include <gtest/gtest.h>
#include "basic.cu"

TEST(mypow_test, HandlesExponent1) {
    EXPECT_EQ(mypow(10, 1), 10);
    EXPECT_EQ(mypow(1, 1), 1);
    EXPECT_EQ(mypow(123, 1), 123);
}
