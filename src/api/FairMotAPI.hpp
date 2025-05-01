// FairMotAPI.hpp
#ifndef FAIR_MOT_API_HPP_
#define FAIR_MOT_API_HPP_

#include <vector>
#include "FairMot.hpp"
#include "DataType.hpp"

// Constructor with default parameters
fairmot::FairMot CreateTracker(const char &model_path);

// Constructor with custom parameters
fairmot::FairMot CreateTracker(const char &model_path,
                                  double frameRate,
                                  int maxPerImage,
                                  int trackBuffer);

// Function to track objects in the image
// This function takes an image and a tracker object as input and returns a vector of TrackOutput
// containing the tracking results.
std::vector<fairmot::TrackOutput>  TrackImage(const cv::Mat &rImage, fairmot::FairMot tracker);

#endif  // FAIR_MOT_API_HPP_