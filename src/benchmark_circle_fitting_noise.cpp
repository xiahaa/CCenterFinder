#include <cstdlib>
#include <iostream>
#include <vector>
#include <list>
#include <iterator>
#include <filesystem>
#include <random>
#include <cstdint>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <fstream>

#include <pcl/memory.h>
#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>

#include "opencv2/opencv.hpp"

#include "Fit3DCircle.hpp"

int find_circle_pcl(pcl::PointCloud<pcl::PointXYZ> &cloud, Eigen::Vector3d &center, double &radius, double tolerance = 0.01, bool use_ransac = true, int max_iterations = 1000)
{
    // use pcl circle3d to find the circle
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_CIRCLE3D);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(tolerance);

    if (!use_ransac)
    {
        seg.setMaxIterations(1);
    }
    else
    {
        seg.setMaxIterations(max_iterations);
    }

    seg.setInputCloud(cloud.makeShared());
    seg.segment(*inliers, *coefficients);

    // assign result
    center << coefficients->values[0], coefficients->values[1], coefficients->values[2];
    radius = coefficients->values[3];

    return 0;
}

int find_circle_cga(pcl::PointCloud<pcl::PointXYZ> &cloud, Eigen::Vector3d &center, double &radius, double tolerance = 0.01)
{
    std::vector<cv::Point3d> circle_points;
    for (const auto& point : cloud.points)
    {
        circle_points.push_back(cv::Point3d(point.x, point.y, point.z));
    }
    // do circle fitting, find the center and radius, remove radius that is not correct.
    Eigen::Matrix<double, 3, 1> fit_center;
    double fit_radius;

    ConformalFit3DCicle::Fit(circle_points, fit_center, fit_radius);

    // assign to center and radius
    center = fit_center;
    radius = fit_radius;

    return 0;
}

int read_pts(std::string filename, std::vector<cv::Point3d> &pts)
{
    // read from file
    std::ifstream infile;
    infile.open(filename);
    if (!infile.is_open())
    {
        std::cerr << "Cannot open file: " << "/mnt/d/data/IROS/workspace/pcn.txt" << std::endl;
        return -1;
    }
    // read the file, file is composed of n lines, each line is x,y,z
    std::string line;

    pts.clear();
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        std::vector<double> data;
        std::string val;
        // data are separated by comma parse it
        while (std::getline(iss, val, ' '))
        {
            data.push_back(std::stod(val));
        }
        if (data.size() != 3)
        {
            std::cerr << "Invalid data: " << line << std::endl;
            continue;
        }
        cv::Point3d p;
        p.x = data[0];
        p.y = data[1];
        p.z = data[2];
        pts.push_back(p);
    }

    return 0;
}
int main(int argc, const char * argv[]) {

    std::string base_folder = "/mnt/d/data/IROS/data/3d_experiment/";

    int num_experiments=500;
    std::vector<double> noise_levels = {1e-4,1e-3,1e-2,1e-1,1};

    std::map<double, std::vector<double>> error_center_pcl_map;
    std::map<double, std::vector<double>> error_center_cga_map;

    // make a tqdm like bar
    for (auto noise : noise_levels)
    {
        std::cout << "Noise: " << noise << std::endl;
        std::vector<double> error_center_pcl_list;
        std::vector<double> error_center_cga_list;
        for (int i = 0; i < num_experiments; i++)
        {
            if (i % 100 == 0)
            {
                std::cout << "Experiment: " << i << std::endl;
            }
            std::ostringstream filename;
            filename << base_folder << "pcn_" << std::fixed << std::setprecision(6) << noise << "_" << std::setw(4) << std::setfill('0') << i << ".txt";

            std::vector<cv::Point3d> circle_points;

            read_pts(filename.str(), circle_points);

            pcl::PointCloud<pcl::PointXYZ> cloud;
            for (const auto& point : circle_points)
            {
                pcl::PointXYZ p;
                p.x = point.x;
                p.y = point.y;
                p.z = point.z;
                cloud.push_back(p);
            }
            Eigen::Vector3d center_pcl;
            double radius_pcl;
            find_circle_pcl(cloud, center_pcl, radius_pcl, noise, true);

            Eigen::Matrix<double, 3, 1> fit_center;
            double fit_radius;

            // ConformalFit3DCicle::Fit(circle_points, fit_center, fit_radius);
            find_circle_cga(cloud, fit_center, fit_radius, noise);

            // std::cout << "center pcl: " << center_pcl.transpose() << " radius pcl: " << radius_pcl << std::endl;

            // std::cout << "center cga: " << fit_center.transpose() << " radius cga: " << fit_radius << std::endl;

            //# write out also the result to a file, line by line
            std::ostringstream result_filename;
            result_filename << base_folder << "centers_" << std::fixed << std::setprecision(6) << noise << "_" << std::setw(4) << std::setfill('0') << i << ".txt";

            std::vector<cv::Point3d> centers;
            read_pts(result_filename.str(), centers);
            // std::cout << " Center GT: " << centers[0].x << " " << centers[0].y << " " << centers[0].z << std::endl;
            // break;

            // compute error of centers
            Eigen::Vector3d error_center_pcl = center_pcl - Eigen::Vector3d(centers[0].x, centers[0].y, centers[0].z);
            Eigen::Vector3d error_center_cga = fit_center - Eigen::Vector3d(centers[0].x, centers[0].y, centers[0].z);

            error_center_pcl_list.push_back(error_center_pcl.norm());
            error_center_cga_list.push_back(error_center_cga.norm());
        }
        // break;

        // compute mean error
        double mean_error_center_pcl = std::accumulate(error_center_pcl_list.begin(), error_center_pcl_list.end(), 0.0) / error_center_pcl_list.size();
        double mean_error_center_cga = std::accumulate(error_center_cga_list.begin(), error_center_cga_list.end(), 0.0) / error_center_cga_list.size();

        std::cout << "Mean error center pcl: " << mean_error_center_pcl << std::endl;
        std::cout << "Mean error center cga: " << mean_error_center_cga << std::endl;

        error_center_pcl_map[noise] = error_center_pcl_list;
        error_center_cga_map[noise] = error_center_cga_list;
    }
    return 0;
}
