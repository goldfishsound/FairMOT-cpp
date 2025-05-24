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
    std::vector<fairmot::TrackOutput> internalStorage_;

    Impl(const std::string &rModelPath, double frameRate, int maxPerImage, int trackBuffer)
        : modelPointer(std::make_shared<fairmot::FairMot>(rModelPath, frameRate, maxPerImage, trackBuffer)) {}

    ~Impl() = default;

    // TrackImage method for RGB image
    std::vector<TrackOutput> TrackImageBGR(const unsigned char *rImage, int height, int width)
    {
        cv::Mat image(height, width, CV_8UC3, const_cast<unsigned char *>(rImage));
        std::vector<fairmot::TrackOutput> internalResult = modelPointer->Track(image);

        std::vector<TrackOutput> result;
        result.reserve(internalResult.size());
        for (const auto &t : internalResult)
        {
            TrackOutput out;
            std::memcpy(out.tlwh, t.tlwh.data(), sizeof(out.tlwh)); // ← Correct and safe
            out.track_id = t.track_id;
            out.score = t.score;
            result.emplace_back(out);
        }
        return result;
    }

    // TrackImage method for RGB image
    TrackOutputList TrackImageBGRRaw(const unsigned char *rImage, int height, int width)
    {
        cv::Mat image(height, width, CV_8UC3, const_cast<unsigned char *>(rImage));
        internalStorage_ = modelPointer->Track(image);
        const TrackOutput *trackOutputPtr = reinterpret_cast<const TrackOutput *>(internalStorage_.data());
        return TrackOutputList{
            reinterpret_cast<const TrackOutput *>(internalStorage_.data()),
            internalStorage_.size()};
    }

    // TrackImage method for RGB image
    std::vector<TrackOutput> TrackImageRGB(const unsigned char *rImage, int height, int width)
    {
        cv::Mat image(height, width, CV_8UC3, const_cast<unsigned char *>(rImage));
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        std::vector<fairmot::TrackOutput> internalResult = modelPointer->Track(image);

        std::vector<TrackOutput> result;
        result.reserve(internalResult.size());
        for (const auto &t : internalResult)
        {
            TrackOutput out;
            std::memcpy(out.tlwh, t.tlwh.data(), sizeof(out.tlwh)); // ← Correct and safe
            out.track_id = t.track_id;
            out.score = t.score;
            result.emplace_back(out);
        }
        return result;
    }

    // TrackImage method for RGB image
    TrackOutputList TrackImageRGBRaw(const unsigned char *rImage, int height, int width)
    {
        cv::Mat image(height, width, CV_8UC3, const_cast<unsigned char *>(rImage));
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        internalStorage_ = modelPointer->Track(image);
        const TrackOutput *trackOutputPtr = reinterpret_cast<const TrackOutput *>(internalStorage_.data());
        return TrackOutputList{
            reinterpret_cast<const TrackOutput *>(internalStorage_.data()),
            internalStorage_.size()};
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

// TrackImage method for BRG images
std::vector<TrackOutput> Tracker::TrackImageBGR(const unsigned char *rImage, int height, int width)
{
    return pImpl->TrackImageBGR(rImage, height, width);
}

// TrackImageRaw method for BRG images
TrackOutputList Tracker::TrackImageBGRRaw(const unsigned char *rImage, int height, int width)
{
    return pImpl->TrackImageBGRRaw(rImage, height, width);
}

// TrackImage method for BRG images
std::vector<TrackOutput> Tracker::TrackImageRGB(const unsigned char *rImage, int height, int width)
{
    return pImpl->TrackImageRGB(rImage, height, width);
}

// TrackImageRaw method for RGB images
TrackOutputList Tracker::TrackImageRGBRaw(const unsigned char *rImage, int height, int width)
{
    return pImpl->TrackImageRGBRaw(rImage, height, width);
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
