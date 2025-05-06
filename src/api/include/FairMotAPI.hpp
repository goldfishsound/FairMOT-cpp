// FairMotAPI.hpp
#ifndef FAIR_MOT_API_HPP_
#define FAIR_MOT_API_HPP_

#include <memory>
#include <string>
#include <vector>
#include <array>

#ifdef _WIN32
#ifdef FAIRMOTCORE_BUILD
#define FAIRMOT_API __declspec(dllexport)
#else
#define FAIRMOT_API __declspec(dllimport)
#endif
#else
#define FAIRMOT_API __attribute__((visibility("default")))
#endif

class FAIRMOT_API Tracker
{
public:
    struct TrackOutput
    {
        std::array<float, 4> tlwh; // Bounding box
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

    // Copy constructor
    Tracker(const Tracker &other);

    // Method to track objects in the image
    // This function takes an image and a tracker object as input and returns a vector of TrackOutput
    // containing the tracking results.
    std::vector<Tracker::TrackOutput> TrackImage(const unsigned char* rImage, const int height, const int width);

    // Method to get the score threshold
    double GetScoreThreshold() const;

    // Method to set the score threshold
    void SetScoreThreshold(double threshold);

private:
    class Impl; // Forward declaration of the implementation class
    std::shared_ptr<Impl> pImpl; // Pointer to the implementation
};

#endif // FAIR_MOT_API_HPP_