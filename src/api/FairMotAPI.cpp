// FairMotAPI.cpp

#include <vector> 
#include "api/FairMotAPI.hpp"
#include "FairMot.hpp"
#include "DataType.hpp"

// Constructor with default parameters
fairmot::FairMot CreateTracker(const std::string &model_path) {
    // Default parameters
    double _frameRate = 25.0;
    int _maxPerImage = 50;
    int _trackBuffer = 120; 
return fairmot::FairMot(model_path, _frameRate, _maxPerImage, _trackBuffer);
}

// Constructor with custom parameters
fairmot::FairMot CreateTracker(const std::string &model_path,
                                  double frameRate,
                                  int maxPerImage,
                                  int trackBuffer) {
    return fairmot::FairMot(model_path, frameRate, maxPerImage, trackBuffer);
}


// Function to track objects in the image
// This function takes an image and a tracker object as input and returns a vector of TrackOutput
// containing the tracking results.
std::vector<fairmot::TrackOutput>  TrackImage(const cv::Mat &rImage, fairmot::FairMot tracker) {
    return tracker.Track(rImage);
}
