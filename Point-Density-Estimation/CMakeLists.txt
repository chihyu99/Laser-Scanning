cmake_minimum_required(VERSION 3.6)
project(WritingTutorial)

# Find PDAL
find_package(PDAL 2.0.0 REQUIRED CONFIG)

# Find OpenCV
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(density_map_generator generatePDM.cpp)

# Link PDAL libraries
target_link_libraries(density_map_generator PRIVATE ${PDAL_LIBRARIES})

# Link OpenCV libraries
target_link_libraries(density_map_generator PRIVATE ${OpenCV_LIBS})

# Include directories for PDAL and OpenCV
target_include_directories(density_map_generator PRIVATE
    ${PDAL_INCLUDE_DIRS}
    ${PDAL_INCLUDE_DIRS}/pdal
    ${OpenCV_INCLUDE_DIRS})
