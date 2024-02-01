cmake_minimum_required(VERSION 3.6)
project(PointCloudClassification)

# Find VTK
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

# Find PDAL
find_package(PDAL 2.0.0 REQUIRED CONFIG)

# Find PCL
find_package(PCL 1.14 REQUIRED COMPONENTS common io kdtree etc)

# Find OpenCV
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(test testPCL.cpp)

# Link PDAL libraries
target_link_libraries(test PRIVATE ${PDAL_LIBRARIES})

# Link PCL libraries
target_link_libraries(test PRIVATE ${PCL_LIBRARIES})

# Link OpenCV libraries
target_link_libraries(test PRIVATE ${OpenCV_LIBS})

# Include directories for PDAL, PCL, and OpenCV
target_include_directories(test PRIVATE
    ${PDAL_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS})

# Add PCL definitions and compile options
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})
