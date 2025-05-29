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
    TrackOutputVector TrackImageBGR(const unsigned char *rImage, int height, int width)
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
    
    // - ToDo: Method no longer needed - Remove
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
    TrackOutputVector TrackImageRGB(const unsigned char *rImage, int height, int width)
    {
        cv::Mat image(height, width, CV_8UC3, const_cast<unsigned char *>(rImage));
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        std::vector<fairmot::TrackOutput> internalResult = modelPointer->Track(image);

        TrackOutputVector result;
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

    // - ToDo: Method no longer needed - Remove
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
TrackOutputVector Tracker::TrackImageBGR(const unsigned char *rImage, int height, int width)
{
    return pImpl->TrackImageBGR(rImage, height, width);
}

// - ToDo: Method no longer needed - Remove
// TrackImageRaw method for BRG images
TrackOutputList Tracker::TrackImageBGRRaw(const unsigned char *rImage, int height, int width)
{
    return pImpl->TrackImageBGRRaw(rImage, height, width);
}

// TrackImage method for BRG images
TrackOutputVector Tracker::TrackImageRGB(const unsigned char *rImage, int height, int width)
{
    return pImpl->TrackImageRGB(rImage, height, width);
}

// - ToDo: Method no longer needed - Remove
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

// Test methods
std::vector<float> Tracker::TestIntVector()
{
    std::vector<float> result = {1.0f, 2.0f, 3.0f};
    return result;
}

TrackOutputVector Tracker::TestOutputTrackVector()
{
    std::vector<TrackOutput> result;
    TrackOutput output1 = {{100.0f, 200.0f, 50.0f, 80.0f}, 1, 0.9f};
    TrackOutput output2 = {{150.0f, 250.0f, 60.0f, 90.0f}, 2, 0.85f};
    TrackOutput output3 = {{200.0f, 300.0f, 70.0f, 100.0f}, 3, 0.8f};
    result.push_back(output1);
    result.push_back(output2);
    result.push_back(output3);
    return result;
}
