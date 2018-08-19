cmake_minimum_required(VERSION 2.6)
project(gradient_method)

add_compile_options(-std=c++11)

find_package(OpenCV 3.0 QUIET)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(gradient_method main.cpp)
target_link_libraries(gradient_method ${OpenCV_LIBS})

install(TARGETS gradient_method RUNTIME DESTINATION bin)
