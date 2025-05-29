// my_test.cpp
#include <gtest/gtest.h>
#include "../src/api/include/FairMotAPI.hpp"
#include <fstream>
#include <iostream>
#include <c10/util/Exception.h>
#include <opencv2/opencv.hpp>
#include "my_test.h"

std::string model_path = "/Users/thomas/Developer/projects/FairMOT-cpp/weights/fairmot_dla34_jit.pth";

// A simple function to test
int Add(int a, int b) {
    return a + b;
}

// AWTracker::Tracker tracker();
// AWTracker::Tracker tracker(model_path, 25.0, 50, 120);
// std::cout << "Tracker constructed successfully" << std::endl;

TEST(LibraryTest, DefaultConstructor) {
    Tracker tracker;
    // ASSERT_EQ(tracker, !nullptr);
}

TEST(LibraryTest, CopyConstructor) {
    Tracker tracker(model_path, 25.0, 50, 120);
    Tracker copy(tracker);
    // ASSERT_EQ(copy, !nullptr);
}

TEST(LibraryTest, ParameterizedConstructor) {
    Tracker tracker(model_path, 25.0, 50, 120);
    // ASSERT_EQ(tracker, !nullptr);
}

TEST(LibraryTest, SetScoreThreshold) {
    Tracker tracker(model_path, 25.0, 50, 120);
    tracker.SetScoreThreshold(0.5);
    EXPECT_EQ(tracker.GetScoreThreshold(), 0.5);
}

TEST(LibraryTest, TrackImage)
{
    std::string project_root = "/Users/thomas/Developer/projects/FairMOT-cpp"; // Replace with your project root path
    std::string image_path = project_root + "/test/assets/threePeople.jpg";
    cv::Mat image = cv::imread(image_path);
    ASSERT_FALSE(image.empty()) << "Failed to load image";

    int width = image.cols;
    int height = image.rows;
    std::cout << "Image dimensions: " << width << "x" << height << std::endl;
    Tracker tracker(model_path, 25.0, 50, 120);

    const unsigned char* image_data_BGR = image.data;
    TrackOutputVector  results_BGR = tracker.TrackImageBGR(image_data_BGR, height, width);
    PrintResults(results_BGR);

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    const unsigned char* image_data_RGB = image.data;
    TrackOutputVector results_RGB = tracker.TrackImageRGB(image_data_RGB, height, width);
    PrintResults(results_RGB);
}

TEST(LibraryTest, TestIntVector)
{
    Tracker tracker(model_path, 25.0, 50, 120);
    std::vector<float> int_vector = tracker.TestIntVector();
    std::cout << "Int vector size: " << int_vector.size() << std::endl;
    for (size_t i = 0; i < int_vector.size(); ++i) {
        std::cout << "Element " << i << ": " << int_vector[i] << std::endl;
    }
    EXPECT_EQ(int_vector.size(), 3) << "Expected vector size of 3";
    EXPECT_FLOAT_EQ(int_vector[0], 1.0f) << "Expected first element to be 1.0";
    EXPECT_FLOAT_EQ(int_vector[1], 2.0f) << "Expected second element to be 2.0";
    EXPECT_FLOAT_EQ(int_vector[2], 3.0f) << "Expected third element to be 3.0";
}
 

TEST(LibraryTest, TestOutputTrackVector)
{
    Tracker tracker(model_path, 25.0, 50, 120);
    TrackOutputVector results = tracker.TestOutputTrackVector();
    PrintResults(results);
    EXPECT_EQ(results.size(), 3) << "Expected vector size of 3";
    EXPECT_EQ(results[0].track_id, 1) << "Expected first track ID to be 1";
    EXPECT_FLOAT_EQ(results[0].score, 0.9f) << "Expected first score to be 0.9";
    EXPECT_FLOAT_EQ(results[0].tlwh[0], 100.0f) << "Expected first bounding box tlwh[0] to be 100.0";
}

void PrintResults(std::__1::vector<TrackOutput> &results)
{
    std::cout << "Number of tracking results: " << results.size() << std::endl;
    EXPECT_FALSE(results.empty()) << "Tracking results should not be empty";
    EXPECT_EQ(results.size(), 3) << "Expected 3 tracking results";
    for (const auto &result : results)
    {
        std::cout << "Track ID: " << result.track_id << ", Score: " << result.score
                  << ", Bounding Box: [" << result.tlwh[0] << ", " << result.tlwh[1]
                  << ", " << result.tlwh[2] << ", " << result.tlwh[3] << "]" << std::endl;
    }
}
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}