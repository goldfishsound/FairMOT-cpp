// FairMotAPI.cpp

#include "FairMotAPI.hpp"
#include "../../FairMot.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <stdexcept>
#include <iostream>

// Implementation class definition
class Tracker::Impl
{
public:
    Impl() = default;

    Impl(const std::string &rModelPath, double frameRate, int maxPerImage, int trackBuffer)
        : modelPointer(std::make_shared<fairmot::FairMot>(rModelPath, frameRate, maxPerImage, trackBuffer)) {}

    ~Impl() = default;

    // TrackImage method for RGB image
    std::vector<Tracker::TrackOutput> TrackImageBGR(const unsigned char *rImage, int height, int width)
    {
        cv::Mat image(height, width, CV_8UC3, const_cast<unsigned char *>(rImage));
        std::vector<fairmot::TrackOutput> internalResult = modelPointer->Track(image);

        std::vector<Tracker::TrackOutput> result;
        result.reserve(internalResult.size());
        for (const auto &t : internalResult)
        {
            result.emplace_back(Tracker::TrackOutput{t.tlwh, t.track_id, t.score});
        }
        return result;
    }

    // TrackImage method for RGB image
    std::vector<Tracker::TrackOutput> TrackImageRGB(const unsigned char *rImage, int height, int width)
    {
        cv::Mat image(height, width, CV_8UC3, const_cast<unsigned char *>(rImage));
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        std::vector<fairmot::TrackOutput> internalResult = modelPointer->Track(image);

        std::vector<Tracker::TrackOutput> result;
        result.reserve(internalResult.size());
        for (const auto &t : internalResult)
        {
            result.emplace_back(Tracker::TrackOutput{t.tlwh, t.track_id, t.score});
        }
        return result;
    }

    double GetScoreThreshold() const
    {
        return modelPointer->GetScoreThreshold();
    }

    void SetScoreThreshold(double threshold)
    {
        if (threshold < 0.0 || threshold > 1.0)
        {
            throw std::out_of_range("scoreThreshold must be between 0.0 and 1.0");
        }
        modelPointer->SetScoreThreshold(threshold);
    }

    std::shared_ptr<fairmot::FairMot> modelPointer;
    std::string defaultModelPath = "/Users/thomas/Developer/projects/FairMOT-cpp/weights/fairmot_dla34_jit.pth";
};

// Tracker class implementation

// Default constructor
Tracker::Tracker()
    : pImpl(std::make_shared<Impl>())
{
    if (pImpl->defaultModelPath.empty())
    {
        throw std::invalid_argument("Default model path is empty. Please provide a valid path.");
    }

    if (!std::filesystem::exists(pImpl->defaultModelPath))
    {
        throw std::invalid_argument("Default model path does not exist. Please provide a valid path.");
    }

    pImpl->modelPointer = std::make_shared<fairmot::FairMot>(
        pImpl->defaultModelPath, 25.0, 50, 120);
}

// Parameterized constructor
Tracker::Tracker(const std::string &rModelPath, double frameRate, int maxPerImage, int trackBuffer)
    : pImpl(std::make_shared<Impl>(rModelPath, frameRate, maxPerImage, trackBuffer)) {}

// Destructor
Tracker::~Tracker() = default;

// Copy constructor
Tracker::Tracker(const Tracker &other)
    : pImpl(std::make_shared<Impl>(*other.pImpl)) {}

// TrackImage method for BRG images
std::vector<Tracker::TrackOutput> Tracker::TrackImageBGR(const unsigned char *rImage, int height, int width)
{
    return pImpl->TrackImageBGR(rImage, height, width);
}

// TrackImage method for BRG images
std::vector<Tracker::TrackOutput> Tracker::TrackImageRGB(const unsigned char *rImage, int height, int width)
{
    return pImpl->TrackImageRGB(rImage, height, width);
}

// GetScoreThreshold method
double Tracker::GetScoreThreshold() const
{
    return pImpl->GetScoreThreshold();
}

// SetScoreThreshold method
void Tracker::SetScoreThreshold(double threshold)
{
    pImpl->SetScoreThreshold(threshold);
}
