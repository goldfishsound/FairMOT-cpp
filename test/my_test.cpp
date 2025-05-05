// my_test.cpp
#include <gtest/gtest.h>
#include "../src/api/include/FairMotAPI.hpp"

// A simple function to test
int Add(int a, int b) {
    return a + b;
}

TEST(LibraryTest, LoadLibrary) {
    tracker::Tracker tracker;
    tracker.SetScoreThreshold(0.5);

}

// Test case: Add
TEST(AdditionTest, HandlesPositiveNumbers) {
    EXPECT_EQ(Add(2, 3), 5);
}

TEST(AdditionTest, HandlesNegativeNumbers) {
    EXPECT_EQ(Add(-2, -3), -5);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}