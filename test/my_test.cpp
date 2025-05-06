// my_test.cpp
#include <gtest/gtest.h>
#include "../src/api/include/FairMotAPI.hpp"
#include <fstream>
#include <iostream>
#include <c10/util/Exception.h>
#include <opencv2/opencv.hpp>

std::string model_path = "/Users/thomas/Developer/projects/FairMOT-cpp/weights/fairmot_dla34_jit.pth";



// A simple function to test
int Add(int a, int b) {
    return a + b;
}

TEST(LibraryTest, DefaultConstructor) {
    AWTracker::Tracker tracker;
    // ASSERT_EQ(tracker, !nullptr);
}

TEST(LibraryTest, ParameterizedConstructor) {
    AWTracker::Tracker tracker(model_path, 25.0, 50, 120);
    // ASSERT_EQ(tracker, !nullptr);
}

TEST(LibraryTest, SetScoreThreshold) {
    AWTracker::Tracker tracker(model_path, 25.0, 50, 120);
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
    // if (image.type() != CV_8UC3) {
    //     image.convertTo(image, CV_8UC3);
    // }
    const unsigned char* image_data = image.data;

    AWTracker::Tracker tracker(model_path, 25.0, 50, 120);
    std::vector<AWTracker::Tracker::TrackOutput>  results = tracker.TrackImage(image_data, height, width);
    
    std::cout << "Number of tracking results: " << results.size() << std::endl;
    EXPECT_FALSE(results.empty()) << "Tracking results should not be empty";
    EXPECT_EQ(results.size(), 3) << "Expected 3 tracking results";
    for (const auto& result : results) {
        std::cout << "Track ID: " << result.track_id << ", Score: " << result.score
                  << ", Bounding Box: [" << result.tlwh[0] << ", " << result.tlwh[1]
                  << ", " << result.tlwh[2] << ", " << result.tlwh[3] << "]" << std::endl;
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}