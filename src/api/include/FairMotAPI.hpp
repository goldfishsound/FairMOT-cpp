// FairMotAPI.hpp
#ifndef FAIR_MOT_API_HPP_
#define FAIR_MOT_API_HPP_

#include <memory>
#include <string>
#include <vector>
#include <array>
#include <swift/bridging>


#ifdef _WIN32
#ifdef FAIRMOTCORE_BUILD
#define FAIRMOT_API __declspec(dllexport)
#else
#define FAIRMOT_API __declspec(dllimport)
#endif
#else
#define FAIRMOT_API __attribute__((visibility("default")))
#endif

struct FAIRMOT_API TrackOutput
{
  float tlwh[4];
  int32_t track_id;
  float score;
};


struct SWIFT_SELF_CONTAINED TrackOutputList {
    const TrackOutput* data;
    size_t size;
};

// Define a type alias TrackOutputVector for a vector of TrackOutput
using TrackOutputVector = std::vector<TrackOutput>;

class Tracker
{
public:
  // Default constructor
  FAIRMOT_API Tracker();

  // Parameterized constructor
  FAIRMOT_API Tracker(const std::string &rModelPath, double frameRate,
                      int maxPerImage, int trackBuffer);

  // Destructor
  FAIRMOT_API ~Tracker();

  // Method to track objects in the image
  // This function takes a BRG image as input and returns a vector of TrackOutput
  FAIRMOT_API TrackOutputVector TrackImageBGR(const unsigned char *rImage, const int height, const int width);

  // - ToDo: Method no longer needed - Remove
  // Method to track objects in the image
  // This function takes a BRG image as input and returns a TrackOutputList
  SWIFT_RETURNS_INDEPENDENT_VALUE
  FAIRMOT_API TrackOutputList TrackImageBGRRaw(const unsigned char *rImage, int height, int width);

  // Method to track objects in the image
  // This function takes a RGB image as input and returns a vector of TrackOutput
  FAIRMOT_API TrackOutputVector TrackImageRGB(const unsigned char *rImage, const int height, const int width);

  // - ToDo: Method no longer needed - Remove
  // Method to track objects in the image
  // This function takes a RGB image as input and returns a vector of TrackOutput
  SWIFT_RETURNS_INDEPENDENT_VALUE
   FAIRMOT_API TrackOutputList TrackImageRGBRaw(const unsigned char *rImage, const int height, const int width);

  // Method to get the score threshold
  FAIRMOT_API double GetScoreThreshold() const;

  // Method to set the score threshold
  FAIRMOT_API void SetScoreThreshold(double threshold);

  // Test methods

  // This function returns a vector of floats for testing purposes
  FAIRMOT_API std::vector<float> TestIntVector();

  // This function returns a vector of TrackOutput for testing purposes
  FAIRMOT_API TrackOutputVector TestOutputTrackVector();

private:
    class Impl;                            // Forward declaration of the implementation class
    std::shared_ptr<Impl> pImpl;           // Pointer to the implementation

  // TrackOutputList trackOutputList;       // List to hold track outputs
  
};

#endif // FAIR_MOT_API_HPP_