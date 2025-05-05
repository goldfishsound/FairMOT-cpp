// FairMotAPI.cpp

#include <string>
#include <vector> 
#include <opencv2/opencv.hpp>
#include "../include/FairMotAPI.hpp"
#include "../../FairMot.hpp"

Tracker::Tracker(const std::string &rModelPath, double frameRate,
    int maxPerImage, int trackBuffer)
: modelPointer(std::make_unique<fairmot::FairMot>(
rModelPath, frameRate, maxPerImage, trackBuffer)) {}

Tracker::Tracker(){
    // Default parameters
    std::string _modelPath = defaultModelPath;
    double _frameRate = 25.0;
    int _maxPerImage = 50;
    int _trackBuffer = 120;
    modelPointer = std::make_unique<fairmot::FairMot>(
        _modelPath, _frameRate, _maxPerImage, _trackBuffer);
    }

Tracker::~Tracker() = default;

void Tracker::SetScoreThreshold(double threshold) {
    modelPointer->SetScoreThreshold(threshold);
}

 // Default parameters
 const std::string defaultModelPath = "../../weights/fairmot_dla34.pth";
 double _frameRate = 25.0;
 int _maxPerImage = 50;
 int _trackBuffer = 120; 

 // Initialize the FairMot tracker with default parameters
 fairmot::FairMot Tracker(defaultModelPath, _frameRate, _maxPerImage, _trackBuffer);

// Function to track objects in the image
// This function takes an image and a tracker object as input and returns a vector of TrackOutput
// containing the tracking results.
std::vector<Tracker::TrackOutput>  Tracker::TrackImage(const unsigned char &rImage, const int height, const int width) {
    cv::Mat _image(height, width, CV_8UC3, const_cast<unsigned char*>(&rImage));
    std::vector<fairmot::TrackOutput> internalResult = modelPointer->Track(_image);
    std::vector<Tracker::TrackOutput> result;
    result.reserve(internalResult.size());

    for (const auto& t : internalResult) {
        result.emplace_back(Tracker::TrackOutput{t.tlwh, t.track_id, t.score});
    }

    return result;
}
