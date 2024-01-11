#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <float.h>
#include <exception>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/PointView.hpp>
#include <pdal/Options.hpp>
#include <opencv2/opencv.hpp> 

using namespace std;
using namespace cv;

vector<vector<int>> generateDensityMap(){
    // Read LAS file
    string filePath;
    cout << "Path of the file to be processed: ";
    cin >> filePath;

    cout << "\nStart reading " << filePath << " ...\n";

    if (!filesystem::exists(filePath)) throw std::runtime_error("File does not exist.");

    pdal::PointTable table;

    pdal::StageFactory factory;
    pdal::Stage* reader = factory.createStage("readers.las");

    pdal::Options options;
    options.add("filename", filePath);
    reader->setOptions(options);

    reader->prepare(table);
    pdal::PointViewSet pointViewSet = reader->execute(table);

    size_t totalPoints = 0;
    double min_x = DBL_MAX, max_x = DBL_MIN, min_y = DBL_MAX, max_y = DBL_MIN;
    for (const auto& pointView : pointViewSet) {
        totalPoints += pointView->size();
        for (pdal::PointId idx = 0; idx < pointView->size(); ++idx) { 
            double x = pointView->getFieldAs<double>(pdal::Dimension::Id::X, idx);
            double y = pointView->getFieldAs<double>(pdal::Dimension::Id::Y, idx);
            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            min_y = std::min(min_y, y);
            max_y = std::max(max_y, y);
        }
    }
    cout << "Total number of points: " << totalPoints << endl;
    cout << "min_x = " << min_x << ", max_x = " << max_x << ", min_y = " << min_y << ", max_y = " << max_y << endl;

    // Compute offsets and number of cells
    float cellSize;
    cout << "\nWhat is the cell size (m)? ";
    cin >> cellSize;    

    int numXCells = std::ceil((max_x - min_x) / cellSize), numYCells = std::ceil((max_y - min_y) / cellSize);
    cout << "\nTotal number of cells: (" << numXCells << ", " << numYCells << ")\n";

    vector<vector<int>> hist(numYCells, std::vector<int>(numXCells));
    for (const auto& pointView : pointViewSet) {
        for (pdal::PointId idx = 0; idx < pointView->size(); ++idx) { 
            double x = pointView->getFieldAs<double>(pdal::Dimension::Id::X, idx);
            double y = pointView->getFieldAs<double>(pdal::Dimension::Id::Y, idx);
            int cellX = static_cast<int>(floor((x - min_x) / cellSize));
            int cellY = static_cast<int>(floor((y - min_y) / cellSize));
            hist[cellY][cellX]++;
        }
    }

    // Generate the point density map with the function LPD = n/r^2
    for (int i = 0; i < numYCells; ++i) {
        for (int j = 0; j < numXCells; ++j) {
            hist[i][j] /= pow(cellSize, 2);
        }
    }

    return hist;
}

void plotColormap(const vector<vector<int>>& data) {
    int rows = data.size();
    int cols = data.empty() ? 0 : data[0].size();

    // Create a Mat object with single channel (grayscale)
    cv::Mat mat(rows, cols, CV_32FC1);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat.at<float>(i, j) = data[i][j];
        }
    }

    // Normalize the values for proper colormap representation
    cv::normalize(mat, mat, 0, 255, cv::NORM_MINMAX);
    mat.convertTo(mat, CV_8UC1);

    cv::Mat colorMat;
    cv::applyColorMap(mat, colorMat, cv::COLORMAP_JET);

    // Display the image
    // cv::imshow("Colormap", colorMat);
    // cv::waitKey(0); 

    cv::imwrite("point_density_map.png", colorMat);

    cout << "\nGenerate point_density_map.png successfully!\n";
}

int main() {

    vector<vector<int>> hist = generateDensityMap();
    plotColormap(hist);

    return 0;
}
