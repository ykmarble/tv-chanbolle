cmake_minimum_required(VERSION 2.6)

project(tv-chanbolle)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(OpenCV REQUIRED)

add_executable(main main.cpp ctutils.cpp)
target_link_libraries(main ${OpenCV_LIBS})
