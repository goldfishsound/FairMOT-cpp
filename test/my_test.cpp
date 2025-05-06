// my_test.cpp
#include <gtest/gtest.h>
#include "../src/api/include/FairMotAPI.hpp"
#include <fstream>
#include <iostream>
#include <c10/util/Exception.h>

std::string model_path = "/Users/thomas/Developer/projects/FairMOT-cpp/weights/fairmot_dla34_jit.pth";



// A simple function to test
int Add(int a, int b) {
    return a + b;
}

// AWTracker::Tracker tracker();
// AWTracker::Tracker tracker(model_path, 25.0, 50, 120);
// std::cout << "Tracker constructed successfully" << std::endl;

TEST(LibraryTest, LoadLibrary) {
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        std::cout << "Current working directory: " << cwd << std::endl;
    } else {
        perror("getcwd() error");
    }
    AWTracker::Tracker tracker;

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