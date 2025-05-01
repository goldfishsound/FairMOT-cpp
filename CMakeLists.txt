cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(FairMOT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")

# Apple Silicon specific flags
if(APPLE)
    message(STATUS "Apple detected")
    # Check if the architecture is arm64 (Apple Silicon)
    set(ACCELERATE_FRAMEWORK "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Accelerate.tbd")
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        message(STATUS "Apple Silicon detected")
        set(CMAKE_OSX_ARCHITECTURES "arm64")
        set(USING_MPS 1)
        set(OMP_LIBRARY "/usr/local/lib/libomp.dylib")
    else()
        message(STATUS "Intel-based macOS detected")
        set(CMAKE_OSX_ARCHITECTURES "x86_64")
        set(USING_MPS 0)
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdeclspec -D__cplusplus=201703L")
    set(CMAKE_C_COMPILER /Library/Developer/CommandLineTools/usr/bin/clang)
    set(CMAKE_CXX_COMPILER /Library/Developer/CommandLineTools/usr/bin/clang++)

    execute_process(
        COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE MACOS_SDK_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "macOS SDK path: ${MACOS_SDK_PATH}")
    include_directories(SYSTEM "${MACOS_SDK_PATH}/usr/include/c++/v1")
    message(STATUS "ACCELERATE: ${ACCELERATE_FRAMEWORK}")
endif()

# Creates compile_commands.json file in build folder.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set paths to OpenCV, Torch, and TorchVision.
set(TORCH_DIR "${CMAKE_SOURCE_DIR}/third_party/torch")
set(VISION_DIR "${CMAKE_SOURCE_DIR}/third_party/vision")
set(OPENCV_DIR "${CMAKE_SOURCE_DIR}/third_party/opencv")


#Torch
set(TORCH_INCLUDE_DIRS "${TORCH_DIR}/include")
set(TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS} "${TORCH_DIR}/include/torch/csrc/api/include" )
foreach(include_dir ${TORCH_INCLUDE_DIRS})
    message(STATUS "Torch include dir: ${include_dir}")
endforeach(include_dir ${TORCH_INCLUDE_DIRS})

# Collect all dynamic and static libraries in the torch lib folder
file(GLOB TORCH_LIBRARIES "${TORCH_DIR}/lib/*.dylib" "${TORCH_DIR}/lib/*.a")

# Print collected libraries for debugging
foreach(lib ${TORCH_LIBRARIES})
    message(STATUS "Torch library: ${lib}")
endforeach(lib ${TORCH_LIBS})

# TorchVision
# Include dir
set(VISION_INCLUDE_DIRS "${VISION_DIR}/include")
message(STATUS "TorchVision include dirs: ${VISION_INCLUDE_DIRS}")

# library dir
file(GLOB VISION_LIBRARY "${VISION_DIR}/lib/*.dylib" "${VISION_DIR}/lib/*.a")
foreach(lib ${VISION_LIBRARY})
    message(STATUS "TorchVision library: ${lib}")
endforeach(lib ${VISION_LIBRARY})

#OMP
if(USING_MPS)
    set(OMP_LIBRARY "/usr/local/lib/libomp.dylib")
endif()

# Eigen
find_package(Eigen3 REQUIRED NO_MODULE
    PATHS ./third_party/eigen
    NO_DEFAULT_PATH
)

#openCV
set(OpenCV_INCLUDE_DIRS "${OPENCV_DIR}/include/opencv4")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

file(GLOB OpenCV_LIBRARIES "${OPENCV_DIR}/lib/*.dylib" "${OPENCV_DIR}/lib/*.a")
foreach(lib ${OpenCV_LIBRARIES})
    message(STATUS "OpenCV library: ${lib}")
endforeach(lib ${OpenCV_LIBS})

#Python
find_package(Python3 COMPONENTS Development)

# Include header files for torch, torchvision, opencv, and python.
include_directories(
    ${TORCH_DIR}/include
    ${TORCH_DIR}/include/torch/csrc/api/include
    ${VISION_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)


# Warning flags
set(FairMOT_CXX_WARNING_FLAGS
    -Wall
    -Wextra
    -Wshadow
    -Wno-unused
    -Wmissing-declarations
    -Wmissing-include-dirs
)
string(JOIN " " FairMOT_CXX_WARNING_FLAGS_STR ${FairMOT_CXX_WARNING_FLAGS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FairMOT_CXX_WARNING_FLAGS_STR}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -I/usr/local/include -ftree-vectorize -march=native")

if(FairMOT_PROFILE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
endif()

# Source files
set(SOURCES
    src/Decoder.cpp
    src/FairMot.cpp
    src/KalmanFilter.cpp
    src/Matching.cpp
    src/STrack.cpp
    src/Utils.cpp
    src/Lap.hpp
    src/Cpuid.hpp
)

if(FairMOT_EVAL)
    list(APPEND SOURCES src/main_eval.cpp)
else()
    list(APPEND SOURCES src/main.cpp)
endif()

add_executable(FairMOT ${SOURCES})

target_include_directories(FairMOT
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/third_party/mini
)

# Link libraries
target_link_libraries(FairMOT
    Eigen3::Eigen
    ${OpenCV_LIBRARIES}
    ${Python3_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${VISION_LIBRARY}
)

# Platform-specific linking
if(APPLE)
    # Try to find the libc++ library
    find_library(CXX_LIB NAMES libc++.tbd PATHS "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/lib" NO_DEFAULT_PATH)

    # Check if the library was found
    if(NOT CXX_LIB)
        message(FATAL_ERROR "Failed to find libc++.tbd on macOS. Please check the path.")
    else()
        target_link_libraries(FairMOT 
        ${CXX_LIB}
        ${OMP_LIBRARY}
        ${ACCELERATE_FRAMEWORK})
    endif()
else()
    # For non-Apple systems, link against libc++
    target_link_libraries(FairMOT 
    libc++)
endif()

# Set again for clarity
set_property(TARGET FairMOT PROPERTY CXX_STANDARD 17)