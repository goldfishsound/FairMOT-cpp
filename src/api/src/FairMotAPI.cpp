// FairMotAPI.cpp

#include <string>
#include <vector> 
#include <opencv2/opencv.hpp>
#include "../include/FairMotAPI.hpp"
#include "../../FairMot.hpp"

using namespace AWTracker;

Tracker::Tracker(const std::string &rModelPath, double frameRate,
    int maxPerImage, int trackBuffer)
: modelPointer(std::make_unique<fairmot::FairMot>(
rModelPath, frameRate, maxPerImage, trackBuffer)) {}

Tracker::Tracker(){
    // Default parameters
    if (defaultModelPath.empty()) {
        throw std::invalid_argument("Default model path is empty. Please provide a valid path.");
    }

    if (!std::filesystem::exists(defaultModelPath)) {
        throw std::invalid_argument("Default model path does not exist. Please provide a valid path.");
    }
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

double Tracker::GetScoreThreshold() const {
    return modelPointer->GetScoreThreshold();
}
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
