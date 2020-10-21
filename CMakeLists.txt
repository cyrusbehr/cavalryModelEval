cmake_minimum_required(VERSION 3.17)
project(cavalryModelEvaluation)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -std=c++17")

find_package(OpenCV REQUIRED)

include_directories(/home/cyrus/work/c-sdks/sample_code/trueface_sdk)
link_directories(/home/cyrus/work/c-sdks/sample_code/trueface_sdk)
set(CMAKE_CXX_STANDARD 14)

add_executable(eval_sdk eval_sdk.cpp)
target_link_libraries(eval_sdk tf ${OpenCV_LIBS})