#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <unordered_set>

#include "FairMot.hpp"
#include "Filesystem.hpp"
#include "Utils.hpp"

// Path to the model file
std::string model_path = "../weights/fairmot_dla34_jit.pth";

fs::path results_dir;

// Initialize the FairMot tracker
fairmot::FairMot tracker(model_path, /*frameRate=*/25.0,
                             /*maxPerImage=*/50,
                             /*trackBuffer=*/120);

// Declare the function to process video                         
int ProcessVideo(const fs::path &rVideoPath);

//extern "C" void __dummy_function_from_torchvision() {
//    // Reference some symbol from torchvision
//    (void)torchvision::ops::nms;
//}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Missing input video file!" << std::endl;
        std::cerr << "Usage: ./FairMOT <path/to/video>" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::filesystem::path currentDir = std::filesystem::current_path();
    std::cout << "Current directory: " << currentDir << std::endl;
    
    // std::ifstream input_file(argv[1], std::ios::in);
    // if (!input_file.is_open()) {
    //     std::cerr << "Could not open input file: " << argv[1] << std::endl;
    //     return EXIT_FAILURE;
    // }
    
    std::vector<fs::path> video_files;
    auto input_path = fs::canonical(fs::path(argv[1]));

    // Create a directory for results
    results_dir = input_path.parent_path() / "results";
    if (!fs::exists(results_dir)) {
        fs::create_directory(results_dir);
    }

    // Define valid video file extensions
    const std::unordered_set<std::string> valid_extensions = {".mp4", ".avi", ".mov"};

    // Check if the input path is a directory or a file
    if (fs::is_directory(input_path)) {
        for (const auto &entry : fs::directory_iterator(input_path)) {
            if (entry.is_regular_file() && 
                valid_extensions.find(entry.path().extension().string()) != valid_extensions.end()) {
                video_files.push_back(entry.path());
            }
        }
        if (video_files.empty()) {
            std::cerr << "No valid video files found in directory: " << input_path << std::endl;
            return EXIT_FAILURE;
        }
    } else if (fs::is_regular_file(input_path)) {
        if (valid_extensions.find(input_path.extension().string()) != valid_extensions.end()) {
            video_files.push_back(input_path);
        } else {
            std::cerr << "Invalid file extension: " << input_path.extension() << std::endl;
            return EXIT_FAILURE;
        }
    } else {
        std::cerr << "Invalid input path: " << input_path << std::endl;
        return EXIT_FAILURE;
    }

    for (const auto &video_path : video_files) {
        std::cout << "Processing video: " << video_path << std::endl;
        if (ProcessVideo(video_path) != EXIT_SUCCESS) {
            std::cerr << "Failed to process video: " << video_path << std::endl;
            return EXIT_FAILURE;
        }
    }
    std::cout << "Processing completed." << std::endl;
    return EXIT_SUCCESS;
}

int ProcessVideo(const fs::path &rVideoPath) {
    auto output_dir = results_dir / rVideoPath.stem().string();
    std::cout << "Creating output dir: " << output_dir << std::endl;
    fs::create_directory(output_dir);
    auto image_dir = output_dir / "images";
    std::cout << "Creating frames dir: " << image_dir << std::endl;
    fs::create_directory(image_dir);
    auto video_dir = output_dir / "video";
    std::cout << "Creating video dir: " << video_dir << std::endl;
    fs::create_directory(video_dir);

    cv::VideoCapture cap(rVideoPath.native());
    if (!cap.isOpened()) {
        return EXIT_FAILURE;
    }
    auto num_frames = 0;
    std::chrono::duration<double> total_elapsed(0);
    cv::Mat image;
    while (true) {
        cap >> image;
        if (image.empty()) {
            break;
        }
        int video_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        int video_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        const auto results = tracker.Track(image);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        std::stringstream outfile_name;
        outfile_name << std::setfill('0') << std::setw(5) << num_frames << ".jpg";
        fairmot::util::Visualize(image, results, num_frames);
        cv::imwrite((image_dir / outfile_name.str()).native(), image);
        
        std::chrono::duration<double> elapsed = end_time - start_time;
        total_elapsed += elapsed;
        ++num_frames;
        if (num_frames % 20 == 0) {
            std::cout << static_cast<double>(num_frames) / total_elapsed.count()
            << " fps" << std::endl;
        }

    }
    cap.release();
    std::stringstream cmd_stream;
    cmd_stream << "ffmpeg -y -f image2 -i " << image_dir / "%05d.jpg"
    << " -c:v copy " << video_dir / "results.mp4";
    return std::system(cmd_stream.str().c_str());
}
