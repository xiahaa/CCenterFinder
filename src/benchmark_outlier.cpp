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
#include "RobustFit3DCircle.hpp"
#include "rtl/RTL.hpp"


class Point3
{
public:
    Point3() : x_(0), y_(0), z_(0) { }

    Point3(double _x, double _y, double _z) : x_(_x), y_(_y), z_(_z) { }

    friend std::ostream& operator<<(std::ostream& out, const Point3& p) { return out << p.x_ << ", " << p.y_ << ", " << p.z_; }

    double x_, y_, z_;
};

class Circle3D
{
public:
    Circle3D(){ }

    Circle3D(double cx, double cy, double cz, double radius, double nx, double ny, double nz){
        center_ << cx, cy, cz;
        radius_ = radius;
        normal_ << nx, ny, nz;
     }

    friend std::ostream& operator<<(std::ostream& out, const Circle3D& c) { return out << c.center_.transpose() << ", " << c.radius_ << ", " << c.normal_.transpose(); }

    double radius_;
    Eigen::Matrix<double, 3, 1> center_;
    Eigen::Matrix<double, 3, 1> normal_;
};

class Circle3DEstimator : public RTL::Estimator<Circle3D, Point3, std::vector<Point3> >
{
public:
    // Calculate the mean of data at the sample indices
    virtual Circle3D ComputeModel(const std::vector<Point3>& data, const std::set<int>& samples)
    {
        std::vector<cv::Point3d> pcd;

        for (auto itr = samples.begin(); itr != samples.end(); itr++)
        {
            // std::cout << "Sample: " << *itr << std::endl;
            const Point3& p = data[*itr];
            pcd.push_back(cv::Point3d(p.x_, p.y_, p.z_));
        }

        Eigen::Matrix<double, 3, 1> fit_center;
        double fit_radius;
        Eigen::Matrix<double, 3, 1> fit_normal;
        // Use robust CGA fit with normalization for better stability
        robust_cga::RobustFit3DCircle::Fit(pcd, fit_center, fit_radius, &fit_normal);

        // std::cout << "Fit center: " << fit_center.transpose() << ";" << fit_radius << ";" << fit_normal.transpose() << std::endl;

        Circle3D circle;
        circle.center_ = fit_center;
        circle.radius_ = fit_radius;
        circle.normal_ = fit_normal;
        return circle;
    }

    // Calculate error between the mean and given datum
    virtual double ComputeError(const Circle3D& circle, const Point3& point)
    {
        Eigen::Vector3d pt(point.x_, point.y_, point.z_);
        Eigen::Vector3d d = circle.center_ - pt;
        Eigen::Vector3d vecC = circle.normal_;
        if (vecC.norm() == 0)
            return std::numeric_limits<double>::max();

        // Calculate the plane equation
        double k = -vecC.dot(circle.center_);
        Eigen::Vector4d plane_eq(vecC[0], vecC[1], vecC[2], k);

        // Distance from a point to the circle's plane
        double dist_pt_plane = (plane_eq[0] * pt[0] + plane_eq[1] * pt[1] + plane_eq[2] * pt[2] + plane_eq[3]) / vecC.norm();

        // Distance from a point to the circle hull if it is infinite along its axis (perpendicular distance to the plane)
        Eigen::Vector3d dist_pt_inf_circle_vec = vecC.cross(d);
        double dist_pt_inf_circle = dist_pt_inf_circle_vec.norm() - circle.radius_;

        // std::cout << "Dist pt inf circle: " << dist_pt_inf_circle << std::endl;
        // std::cout << "Dist pt plane: " << dist_pt_plane << std::endl;

         // The distance from a point to a circle will be the hypotenuse
         double dist_pt = std::sqrt(std::pow(dist_pt_inf_circle, 2) + std::pow(dist_pt_plane, 2));

        return dist_pt;
    }
};



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

    if (use_ransac)
    {
        seg.setMaxIterations(max_iterations);
    }
    else
    {
        seg.setMaxIterations(1);
    }

    seg.setInputCloud(cloud.makeShared());
    seg.segment(*inliers, *coefficients);

    // assign result
    center << coefficients->values[0], coefficients->values[1], coefficients->values[2];
    radius = coefficients->values[3];

    return 0;
}

int find_circle_cga(pcl::PointCloud<pcl::PointXYZ> &cloud, Eigen::Vector3d &center, double &radius, double tolerance = 0.01, int max_iterations = 1000)
{
    std::vector<Point3> circle_points;
    for (const auto& point : cloud.points)
    {
        circle_points.push_back(Point3(point.x, point.y, point.z));
    }
    Circle3D model;
    Circle3DEstimator estimator;
    RTL::RANSAC<Circle3D, Point3, std::vector<Point3> > ransac(&estimator);

    ransac.SetParamIteration(max_iterations);
    ransac.SetParamThreshold(tolerance);

    double loss = ransac.FindBest(model, circle_points, circle_points.size(), 5);
    std::vector<int> inliers = ransac.FindInliers(model, circle_points, circle_points.size());

    center = model.center_;
    radius = model.radius_;

    // print the inliers
    // std::cout << "Inliers: " << inliers.size() << std::endl;
    // for (auto &idx : inliers)
    // {
        // std::cout << "Inlier: " << idx << " " << circle_points[idx].x_ << ", " << circle_points[idx].y_ << ", " << circle_points[idx].z_ << std::endl;
    // }

    // Refit on inliers using robust CGA for better accuracy
    std::vector<cv::Point3d> inlier_points;
    inlier_points.reserve(inliers.size());
    for (auto &idx : inliers)
    {
        inlier_points.emplace_back(circle_points[idx].x_, circle_points[idx].y_, circle_points[idx].z_);
    }
    Eigen::Matrix<double, 3, 1> fit_center;
    double fit_radius;
    if (inlier_points.size() >= 5)
    {
        robust_cga::RobustFit3DCircle::Fit(inlier_points, fit_center, fit_radius);
        if (fit_radius > 0 && fit_center.allFinite())
        {
            center = fit_center;
            radius = fit_radius;
        }
    }

    return 0;
}

int read_pts(std::string filename, std::vector<cv::Point3d> &pts)
{
    // read from file
    std::ifstream infile;
    // std:: cout << "Reading file: " << filename << std::endl;
    infile.open(filename);
    if (!infile.is_open())
    {
        std::cerr << "Cannot open file: " << filename << std::endl;
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

    std::string base_folder = "/mnt/d/data/IROS/data/3d_experiment_outlier/";

    int num_experiments=100;
    double noise = 1e-1;
    int span = 360;
    std::vector<double> outlier_prob = {0.1, 0.2, 0.3, 0.4, 0.5};

    std::map<double, std::vector<double>> error_center_pcl_map;
    std::map<double, std::vector<double>> error_center_cga_map;

    // ensure output directory exists
    const std::string out_dir = "results";
    try {
        std::filesystem::create_directories(out_dir);
    } catch (const std::exception &e) {
        std::cerr << "Failed to create results directory: " << e.what() << std::endl;
    }

    std::random_device SeedDevice;
    std::mt19937 RNG = std::mt19937(SeedDevice());
    std::uniform_int_distribution<int> UniDist(10, 20); // [Incl, Incl]
    int Perturb = 0.01;
    std::normal_distribution<double> PerturbDist(0, Perturb);

    // make a tqdm like bar
    for (auto prob : outlier_prob)
    {
        std::cout << "outlier_prob: " << prob << std::endl;
        std::vector<double> error_center_pcl_list;
        std::vector<double> error_center_cga_list;

        // open per-method result files for this outlier probability
        std::ostringstream prob_str;
        prob_str << std::fixed << std::setprecision(2) << prob;
        const std::string prob_token = prob_str.str();

        std::ostringstream pcl_path;
        pcl_path << out_dir << "/outlier_prob_" << prob_token << "_pcl_results.txt";
        std::ofstream pcl_out(pcl_path.str());
        pcl_out << "center_error,radius_error,success" << std::endl;

        std::ostringstream cga_path;
        cga_path << out_dir << "/outlier_prob_" << prob_token << "_cga_results.txt";
        std::ofstream cga_out(cga_path.str());
        cga_out << "center_error,radius_error,success" << std::endl;
        for (int i = 0; i < num_experiments; i++)
        {
            if (i % 100 == 0)
            {
                std::cout << "Experiment: " << i << std::endl;
            }
            std::ostringstream filename;
            filename << base_folder << "pcn_" << std::setw(3) << std::setfill('0') << span << "_" << std::fixed << std::setprecision(6) << noise << "_" << std::setw(4) << std::setfill('0') << i << ".txt";

            // std::cout << filename.str() << std::endl;

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

            int num_outliers = prob * cloud.size();

            // std::cout << "cloud.size(): " << cloud.size() << std::endl;
            // std::cout << "num_outliers: " << num_outliers << std::endl;

            for (int j = 0; j < num_outliers; ++j)
            {
                cv::Point3d outlier_pt(UniDist(RNG) + PerturbDist(RNG),
                                       UniDist(RNG) + PerturbDist(RNG),
                                       UniDist(RNG) + PerturbDist(RNG));
                cloud.push_back(pcl::PointXYZ(outlier_pt.x, outlier_pt.y, outlier_pt.z));
            }

            // shuffle the cloud
            // std::shuffle(cloud.begin(), cloud.end(), RNG);

            Eigen::Vector3d center_pcl;
            double radius_pcl;
            find_circle_pcl(cloud, center_pcl, radius_pcl, 0.1, true, 1000);

            Eigen::Matrix<double, 3, 1> fit_center;
            double fit_radius;

            // ConformalFit3DCicle::Fit(circle_points, fit_center, fit_radius);
            find_circle_cga(cloud, fit_center, fit_radius, 0.2, 1000);

            // std::cout << "center pcl: " << center_pcl.transpose() << " radius pcl: " << radius_pcl << std::endl;

            // std::cout << "center cga: " << fit_center.transpose() << " radius cga: " << fit_radius << std::endl;

            //# write out also the result to a file, line by line
            std::ostringstream result_filename;
            result_filename << base_folder << "centers_" << std::setw(3) << std::setfill('0') << span << "_" << std::fixed << std::setprecision(6) << noise << "_" << std::setw(4) << std::setfill('0') << i << ".txt";
            std::vector<cv::Point3d> centers;
            read_pts(result_filename.str(), centers);
            // std::cout << " Center GT: " << centers[0].x << " " << centers[0].y << " " << centers[0].z << std::endl;

            // compute error of centers
            Eigen::Vector3d gt_center(centers[0].x, centers[0].y, centers[0].z);
            Eigen::Vector3d error_center_pcl = center_pcl - gt_center;
            Eigen::Vector3d error_center_cga = fit_center - gt_center;

            // estimate ground-truth radius from original (pre-outlier) points
            double gt_radius = 0.0;
            if (!circle_points.empty()) {
                double acc = 0.0;
                for (const auto &pt : circle_points) {
                    Eigen::Vector3d v(pt.x, pt.y, pt.z);
                    acc += (v - gt_center).norm();
                }
                gt_radius = acc / static_cast<double>(circle_points.size());
            }

            double radius_error_pcl = std::abs(radius_pcl - gt_radius);
            double radius_error_cga = std::abs(fit_radius - gt_radius);

            // record
            error_center_pcl_list.push_back(error_center_pcl.norm());
            error_center_cga_list.push_back(error_center_cga.norm());

            // success flag: always 1 for now (both methods returned a result)
            pcl_out << error_center_pcl.norm() << "," << radius_error_pcl << ",1" << std::endl;
            cga_out << error_center_cga.norm() << "," << radius_error_cga << ",1" << std::endl;
        }

        // compute mean error
        double mean_error_center_pcl = std::accumulate(error_center_pcl_list.begin(), error_center_pcl_list.end(), 0.0) / error_center_pcl_list.size();
        double mean_error_center_cga = std::accumulate(error_center_cga_list.begin(), error_center_cga_list.end(), 0.0) / error_center_cga_list.size();

        std::cout << "Mean error center pcl: " << mean_error_center_pcl << std::endl;
        std::cout << "Mean error center cga: " << mean_error_center_cga << std::endl;

        error_center_pcl_map[noise] = error_center_pcl_list;
        error_center_cga_map[noise] = error_center_cga_list;

        // close per-method files
        pcl_out.close();
        cga_out.close();

        // append to summary CSV
        std::ostringstream summary_path;
        summary_path << out_dir << "/outlier_summary.csv";

        const bool summary_exists = std::filesystem::exists(summary_path.str());
        std::ofstream summary_out(summary_path.str(), std::ios::app);
        if (!summary_exists) {
            summary_out << "prob,method,mean_center_error" << std::endl;
        }
        summary_out << prob_token << ",PCL," << mean_error_center_pcl << std::endl;
        summary_out << prob_token << ",CGA," << mean_error_center_cga << std::endl;
        summary_out.close();

    }
    return 0;
}
