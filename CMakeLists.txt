cmake_minimum_required (VERSION 3.0)

project(Fit3DCircle VERSION 1.0.0)

if (EXISTS "${CMAKE_SOURCE_DIR}/cmake")
  set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
endif ()

# SET(CMAKE_CXX_FLAGS "-std=gnu++11")

find_package (Eigen3 REQUIRED)

find_package (OpenCV REQUIRED)

find_package (PCL REQUIRED)

find_package (Glog REQUIRED)

find_package (OpenMP REQUIRED)

add_definitions (-march=native)
if(MSVC)	
  add_definitions(-D_USE_MATH_DEFINES)
endif()

find_package(Boost REQUIRED COMPONENTS filesystem)

if (OpenMP_FOUND)
    message(STATUS "found openmp")
    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS}, ${OPENMP_C_FLAGS})
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}, ${OPENMP_CXX_FLAGS})
else ()
    message(FATAL_ERROR "openmp not found")
endif()


# Include PCL directories
include_directories(
    ${CMAKE_SOURCE_DIR}/../include
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    # ${GLOG_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    )
link_directories(${PCL_LIBRARY_DIRS} ${OpenCV_LIBS})
add_definitions(${PCL_DEFINITIONS})

add_executable (demo src/main.cpp)
target_link_libraries (demo ${OPENCV_LIBRARIES} ${GLOG_LIBRARIES})

add_executable(benchmark3d benchmark_circle_fitting_noise.cpp)
target_link_libraries(benchmark3d ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES})
