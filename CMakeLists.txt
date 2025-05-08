cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(FairMOT)

set(PACKAGE_DIR "${CMAKE_SOURCE_DIR}/product/FairMotPackage")
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Apple Silicon specific flags
if(APPLE)
    message(STATUS "Apple detected")
    find_library(ACCELERATE_FRAMEWORK
    NAMES Accelerate
    PATHS ${CMAKE_SYSTEM_FRAMEWORK_PATH}
    PATH_SUFFIXES Frameworks
)
    # Check if the architecture is arm64 (Apple Silicon)
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
endif()

# ================================================
# Include directories
# ================================================
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
find_package(Python3 COMPONENTS Development) #Seems we're not using this

# Include header files for torch, torchvision, opencv, and python.
include_directories(
    ${TORCH_DIR}/include
    ${TORCH_DIR}/include/torch/csrc/api/include
    ${VISION_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

# ================================================
# Compiler
# ================================================
# Warning flags
set(FairMOT_CXX_WARNING_FLAGS
    -Wall
    -Wextra
    -Wshadow
    -Wno-unused
    -Wno-unused-parameter
    -Wmissing-declarations
    -Wmissing-include-dirs
)

# Other flags
string(JOIN " " FairMOT_CXX_WARNING_FLAGS_STR ${FairMOT_CXX_WARNING_FLAGS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FairMOT_CXX_WARNING_FLAGS_STR}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -I/usr/local/include -ftree-vectorize -march=native")

if(FairMOT_PROFILE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
endif()

# Creates compile_commands.json file in build folder.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ================================================
# Source files
# ================================================
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

# ================================================
# Shared library
# ================================================
add_library(AWTracker SHARED 
    ${SOURCES}
    src/api/src/FairMotAPI.cpp
)

# Define the macro so that __declspec(dllexport) or visibility gets used
target_compile_definitions(AWTracker PRIVATE AWTRACKER_BUILD)

target_include_directories(AWTracker PUBLIC "${CMAKE_SOURCE_DIR}/src/api/include")

target_link_libraries(AWTracker PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBRARIES}
    ${Python3_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${VISION_LIBRARY}
    ${OMP_LIBRARY}
    ${ACCELERATE_FRAMEWORK}
)

# Optional: set version info
set_target_properties(AWTracker PROPERTIES
    VERSION 1.0
    SOVERSION 1
)

# ================================================
# Executable
# ================================================
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

# ================================================
# Linking
# ================================================
target_link_libraries(FairMOT
    Eigen3::Eigen
    ${OpenCV_LIBRARIES}
    ${Python3_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${VISION_LIBRARY}
    ${CXX_LIB}
    ${OMP_LIBRARY}
    ${ACCELERATE_FRAMEWORK}
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

# ================================================
# Installation
# ================================================
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/product/FairMotPackage")

# Install library
install(TARGETS AWTracker 
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    # RUNTIME DESTINATION bin
)

# # Install executable (if desired)
install(TARGETS FairMOT
        RUNTIME DESTINATION bin)

# Install the public header
install(FILES ${CMAKE_SOURCE_DIR}/src/api/include/FairMotAPI.hpp DESTINATION include)

# Install the model resource
install(FILES ${CMAKE_SOURCE_DIR}/weights/fairmot_dla34_jit.pth DESTINATION resources)

# ================================================
# Testing
# ================================================
add_subdirectory(test)
enable_testing()
