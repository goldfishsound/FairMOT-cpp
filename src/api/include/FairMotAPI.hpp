// FairMotAPI.hpp
#ifndef FAIR_MOT_API_HPP_
#define FAIR_MOT_API_HPP_

#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
  #ifdef AWTRACKER_BUILD
    #define AWTRACKER_API __declspec(dllexport)
  #else
    #define AWTRACKER_API __declspec(dllimport)
  #endif
#else
  #define AWTRACKER_API __attribute__((visibility("default")))
#endif

namespace fairmot
{
  class FairMot; // Forward declaration only
} // namespace fairmot

namespace AWTracker
{
  class AWTRACKER_API Tracker
  {
  private:
    std::unique_ptr<fairmot::FairMot> modelPointer;
    std::string defaultModelPath = "/Users/thomas/Developer/projects/FairMOT-cpp/weights/fairmot_dla34_jit.pth";
    static constexpr std::size_t kBBoxSize = 4;
    static constexpr std::size_t kEmbeddingSize = 128;
    typedef std::array<float, kBBoxSize> BBox;
    typedef std::array<float, kEmbeddingSize> Embedding;

  public:
    struct TrackOutput
    {
      BBox tlwh;
      int track_id;
      float score;
    };

    // Default constructor
    Tracker();

    // Parameterized constructor
    Tracker(const std::string &rModelPath, double frameRate,
            int maxPerImage, int trackBuffer);

    // Destructor
    ~Tracker();

    // Method to track objects in the image
    // This function takes an image and a tracker object as input and returns a vector of TrackOutput
    // containing the tracking results.
    std::vector<Tracker::TrackOutput> TrackImage(const unsigned char* rImage, const int height, const int width);

    // Method to get the score threshold
    double GetScoreThreshold() const;

    // Threshold is a value to determine the minimum score for a detection to be considered valid
    void SetScoreThreshold(double threshold);
  };

} // namespace AWTracker
#endif // FAIR_MOT_API_HPP_