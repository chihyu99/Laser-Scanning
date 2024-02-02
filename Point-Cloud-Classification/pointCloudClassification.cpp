#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <float.h>
#include <exception>
#include <chrono>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/PointView.hpp>
#include <pdal/Options.hpp>
#include <pdal/io/LasReader.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <opencv2/opencv.hpp> 
#include <Eigen/Dense>
#include <thread>

pcl::PointCloud<pcl::PointXYZ>::Ptr readLAS(){

    // Read LAS file
    std::string filePath;
    std::cout << "Path of the file to be processed: ";
    std::cin >> filePath;

    std::cout << "\nStart reading " << filePath << " ...\n\n";

    if (!std::filesystem::exists(filePath)) throw std::runtime_error("Error: File " + filePath + " does not exist.");

    pdal::PointTable table;
    pdal::StageFactory factory;
    pdal::Stage* reader = factory.createStage("readers.las");
    pdal::Options options;
    options.add("filename", filePath);
    reader->setOptions(options);
    reader->prepare(table);
    pdal::PointViewSet pointViewSet = reader->execute(table);

    if (pointViewSet.empty()) throw std::runtime_error("Error: No points were found in the file.");

    const pdal::PointViewPtr pointView = *pointViewSet.begin();

    // Create a PCL point cloud and reserve space
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->reserve(pointView->size());

    // Extract XYZ coordinates
    for (pdal::PointId id = 0; id < pointView->size(); ++id) {
        cloud->push_back(pcl::PointXYZ(
            pointView->getFieldAs<float>(pdal::Dimension::Id::X, id),
            pointView->getFieldAs<float>(pdal::Dimension::Id::Y, id),
            pointView->getFieldAs<float>(pdal::Dimension::Id::Z, id)
        ));
    }

    // Display the size of the point cloud
    std::cout << "point_size: (" << pointView->size() << ", " << 3 << ")\n";

    return cloud;
}


void processSet(int setIdx, pcl::KdTreeFLANN<pcl::PointXYZ> kdtree, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int startIdx, int endIdx, int K, std::vector<int>& classId) {   
    std::cout << "=== Set " << setIdx << " ===" << std::endl;
    std::cout << "Points: " << startIdx << " to " << endIdx << "\n\n";

    for (int pointIdx = startIdx; pointIdx < endIdx; ++pointIdx) {

        pcl::PointXYZ searchPoint = cloud->points[pointIdx];

        std::vector<int> pointIdxNKNSearch(K);
        std::vector<float> pointNKNSquaredDistance(K);

        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        std::vector<Eigen::Vector3d> neighborPts;

        if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
            // pointIdxNKNSearch now contains the indices of the k nearest neighbors
            // pointNKNSquaredDistance has their squared distances to searchPoint
            for (int neighboridx : pointIdxNKNSearch) {
                Eigen::Vector3d neighborPt(cloud->points[neighboridx].x, cloud->points[neighboridx].y, cloud->points[neighboridx].z);
                centroid += neighborPt;
                neighborPts.push_back(neighborPt);
            }
            centroid /= K;

            // Shift points and compute covariance matrix
            Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
            for (auto& pt : neighborPts) {
                pt -= centroid;
                covariance += pt * pt.transpose();
            }
            covariance /= K;

            // Eigenvalue decomposition
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(covariance);
            Eigen::Vector3d eigenvalues = eig.eigenvalues().real();

            // Dimensionality-based feature classification
            double l1 = eigenvalues[0], l2 = eigenvalues[1], l3 = eigenvalues[2];
            Eigen::Vector3d temp(3);
            temp[0] = (std::sqrt(l1) - std::sqrt(l2)) / std::sqrt(l1);
            temp[1] = (std::sqrt(l2) - std::sqrt(l3)) / std::sqrt(l1);
            temp[2] = std::sqrt(l3) / std::sqrt(l1);

            Eigen::Index maxIndex;
            double maxValue = temp.maxCoeff(&maxIndex);
            classId[pointIdx] = maxIndex;
        }
    }
}

int main() {

    ////////// Read LAS File //////////
    auto start = std::chrono::high_resolution_clock::now();
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = readLAS();
    int numPoints = cloud->points.size();
    
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Read LAS File - Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    ///////////////////////////////////

    // Number of nearest neighbors
    std::cout << "\nNumber of nearest neighbors: ";
    std::cin >> K;

    // Separate the points into 20 + 1 sets
    int numPointsInOneSet = numPoints/20;
    std::cout << "\nnumPointsInOneSet = " << numPointsInOneSet << std::endl;
    // int numPointsRemaining = numPoints - numPointsInOneSet*20;
    // std::cout << "\nnumPointsInOneSet = " << numPointsInOneSet << ", numPointsRemaining = " << numPointsRemaining << std::endl;


    ////////// Build KD Tree //////////
    start = std::chrono::high_resolution_clock::now();

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << "Build KD Tree - Elapsed time: " << elapsed.count() << " seconds" << "\n\n";
    ///////////////////////////////////


    int numThreads = std::thread::hardware_concurrency(); 
    std::vector<std::thread> threads;
    std::vector<int> classId(numPoints);

    ////////// Process all sets //////////
    start = std::chrono::high_resolution_clock::now();

    for (int setIdx = 0; setIdx < 20; ++setIdx) {
        int startIdx = numPointsInOneSet * setIdx;
        int endIdx = (setIdx == 19) ? numPoints : startIdx + numPointsInOneSet;
        
        // Launch a thread for each set
        threads.push_back(std::thread(processSet, setIdx, kdtree, cloud, startIdx, endIdx, K, std::ref(classId)));
        
        if (threads.size() >= numThreads || setIdx == 19) {
            // Join the threads
            for (auto& th : threads) {
                th.join();
            }
            threads.clear();
        }
    }

    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << "Process one set - Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    ///////////////////////////////////

    return 0;
}
