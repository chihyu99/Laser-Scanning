# Point Cloud Density Map Generator

## Description
This project provides a C++ implementation to read point cloud data from LAS files and generate a density map. The density map is visualized using a colormap generated with OpenCV.

## Features
- Reads LAS files to process point cloud data.
- Calculates the density of points in each specified cell of the point cloud.
- Generates a density map visualized as a colored image.

## Dependencies
- [PDAL](https://pdal.io/): Point Data Abstraction Library for processing point cloud data.
- [OpenCV](https://opencv.org/): Open Source Computer Vision Library for image processing and visualization.

## Build and Run
Ensure you have PDAL and OpenCV installed on your system.

### Building the Project
Using CMake:

```bash
mkdir build
cd build
cmake ..
make
```

### Running the Application
Execute the binary created after building. 
```bash
./density_map_generator
```
Follow the prompts to input the path to the LAS file and the desired cell size for density calculation.
