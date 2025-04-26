#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>

#include "FairMot.hpp"
#include "Filesystem.hpp"
#include "Utils.hpp"

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
    
    std::ifstream input_file(argv[1], std::ios::in);
    if (!input_file.is_open()) {
        std::cerr << "Could not open input file: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }
    
    auto video_path = fs::canonical(fs::path(argv[1]));
    return ProcessVideo(video_path);
}

int ProcessVideo(const fs::path &rVideoPath) {
    std::string model_path = "../weights/fairmot_dla34_jit.pth";
    fairmot::FairMot tracker(model_path, /*frameRate=*/25.0,
                             /*maxPerImage=*/50,
                             /*trackBuffer=*/120);
    
    auto output_dir = rVideoPath.parent_path().parent_path() / "results2";
    std::cout << "Creating output dir: " << output_dir << std::endl;
    fs::create_directory(output_dir);
    
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
        cv::imwrite((output_dir / outfile_name.str()).native(), image);
        
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
    cmd_stream << "ffmpeg -y -f image2 -i " << output_dir / "%05d.jpg"
    << " -c:v copy " << output_dir / "results.mp4";
    return std::system(cmd_stream.str().c_str());
}
