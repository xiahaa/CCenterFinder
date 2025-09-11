#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <iomanip>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <filesystem>

#include "Fit3DCircle.hpp"

// Function to generate points on a circle with noise
std::vector<cv::Point3d> generate_circle_points(
    const Eigen::Vector3d& center,
    double radius,
    const Eigen::Vector3d& normal,
    int num_points = 100,
    double noise_std = 0.1,
    int seed = 42
) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise_dist(0.0, noise_std);

    // Create orthonormal basis for the circle plane
    Eigen::Vector3d n = normal.normalized();
    Eigen::Vector3d u, v;

    // Find a vector not parallel to n
    if (std::abs(n.z()) < 0.9) {
        u = n.cross(Eigen::Vector3d(0, 0, 1));
    } else {
        u = n.cross(Eigen::Vector3d(1, 0, 0));
    }
    u.normalize();
    v = n.cross(u);
    v.normalize();

    std::vector<cv::Point3d> points;
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);

    for (int i = 0; i < num_points; ++i) {
        double t = angle_dist(rng);
        Eigen::Vector3d point = center + radius * (std::cos(t) * u + std::sin(t) * v);

        // Add noise
        point.x() += noise_dist(rng);
        point.y() += noise_dist(rng);
        point.z() += noise_dist(rng);

        points.push_back(cv::Point3d(point.x(), point.y(), point.z()));
    }

    return points;
}

// Function to generate limited arc points (non-uniform distribution)
std::vector<cv::Point3d> generate_limited_arc_points(
    const Eigen::Vector3d& center,
    double radius,
    const Eigen::Vector3d& normal,
    int num_points = 100,
    double arc_deg = 70.0,
    double noise_std = 0.1,
    int seed = 42
) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise_dist(0.0, noise_std);
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

    // Create orthonormal basis
    Eigen::Vector3d n = normal.normalized();
    Eigen::Vector3d u, v;

    if (std::abs(n.z()) < 0.9) {
        u = n.cross(Eigen::Vector3d(0, 0, 1));
    } else {
        u = n.cross(Eigen::Vector3d(1, 0, 0));
    }
    u.normalize();
    v = n.cross(u);
    v.normalize();

    std::vector<cv::Point3d> points;
    double arc_rad = arc_deg * M_PI / 180.0;

    for (int i = 0; i < num_points; ++i) {
        // Bias density to one end of arc
        double u_rand = uniform_dist(rng);
        double t = std::pow(u_rand, 2.0) * arc_rad - 0.2 * arc_rad;

        Eigen::Vector3d point = center + radius * (std::cos(t) * u + std::sin(t) * v);

        // Add noise
        point.x() += noise_dist(rng);
        point.y() += noise_dist(rng);
        point.z() += noise_dist(rng);

        points.push_back(cv::Point3d(point.x(), point.y(), point.z()));
    }

    return points;
}

// Function to generate sparse non-uniform points
std::vector<cv::Point3d> generate_sparse_points(
    const Eigen::Vector3d& center,
    double radius,
    const Eigen::Vector3d& normal,
    int num_points = 12,
    double noise_std = 0.1,
    int seed = 42
) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise_dist(0.0, noise_std);

    // Create orthonormal basis
    Eigen::Vector3d n = normal.normalized();
    Eigen::Vector3d u, v;

    if (std::abs(n.z()) < 0.9) {
        u = n.cross(Eigen::Vector3d(0, 0, 1));
    } else {
        u = n.cross(Eigen::Vector3d(1, 0, 0));
    }
    u.normalize();
    v = n.cross(u);
    v.normalize();

    std::vector<cv::Point3d> points;

    // Generate 2-3 angular clusters
    int num_clusters = std::min(3, num_points / 4);
    std::uniform_real_distribution<double> cluster_center_dist(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<double> cluster_width_dist(M_PI / 30.0, M_PI / 9.0);

    for (int cluster = 0; cluster < num_clusters; ++cluster) {
        double cluster_center = cluster_center_dist(rng);
        double cluster_width = cluster_width_dist(rng);
        int points_in_cluster = num_points / num_clusters;

        if (cluster == num_clusters - 1) {
            points_in_cluster = num_points - points.size(); // Remaining points
        }

        for (int i = 0; i < points_in_cluster; ++i) {
            double t = cluster_center + std::normal_distribution<double>(0.0, cluster_width)(rng);

            Eigen::Vector3d point = center + radius * (std::cos(t) * u + std::sin(t) * v);

            // Add noise
            point.x() += noise_dist(rng);
            point.y() += noise_dist(rng);
            point.z() += noise_dist(rng);

            points.push_back(cv::Point3d(point.x(), point.y(), point.z()));
        }
    }

    return points;
}

// Function to generate two-side symmetric arcs with non-uniform intervals
std::vector<cv::Point3d> generate_symmetric_nonuniform_points(
    const Eigen::Vector3d& center,
    double radius,
    const Eigen::Vector3d& normal,
    int num_points = 20,
    double arc_deg = 200.0,
    double noise_std = 0.1,
    int seed = 42
) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise_dist(0.0, noise_std);

    // Orthonormal basis
    Eigen::Vector3d n = normal.normalized();
    Eigen::Vector3d u, v;
    if (std::abs(n.z()) < 0.9) {
        u = n.cross(Eigen::Vector3d(0, 0, 1));
    } else {
        u = n.cross(Eigen::Vector3d(1, 0, 0));
    }
    u.normalize();
    v = n.cross(u);
    v.normalize();

    // Helper: build symmetric non-uniform arc around 0 with extent arc_rad
    auto build_symmetric_arc = [&](double center_angle, double arc_rad, int count) {
        std::vector<double> angles;
        if (count <= 1) {
            angles.push_back(center_angle);
            return angles;
        }

        int half = count / 2;
        // generate random positive intervals for half arc
        std::vector<double> intervals(half, 0.0);
        std::uniform_real_distribution<double> uni(0.8, 1.2);
        double sum = 0.0;
        for (int i = 0; i < half; ++i) { intervals[i] = uni(rng); sum += intervals[i]; }
        for (int i = 0; i < half; ++i) { intervals[i] = intervals[i] / sum * (arc_rad / 2.0); }

        // cumulative to get offsets
        std::vector<double> t_half; t_half.reserve(half+1);
        double acc = 0.0; t_half.push_back(0.0);
        for (int i = 0; i < half; ++i) { acc += intervals[i]; t_half.push_back(acc); }

        // mirror to get full set
        // left side (negative), then right side (positive) excluding duplicate center
        for (int i = static_cast<int>(t_half.size()) - 1; i >= 0; --i) {
            angles.push_back(center_angle - t_half[i]);
        }
        for (size_t i = 1; i < t_half.size(); ++i) {
            angles.push_back(center_angle + t_half[i]);
        }

        // if odd count, ensure exact count by trimming last
        if (static_cast<int>(angles.size()) > count) {
            angles.resize(count);
        }
        return angles;
    };

    double arc_rad = arc_deg * M_PI / 180.0;
    // Build symmetric arc around 0
    auto angles = build_symmetric_arc(0.0, arc_rad, num_points);

    // Map to [0, 2pi)
    for (auto &ang : angles) {
        ang = std::fmod(ang, 2.0 * M_PI);
        if (ang < 0) ang += 2.0 * M_PI;
    }

    // Create points
    std::vector<cv::Point3d> points;
    points.reserve(angles.size());
    for (double t : angles) {
        Eigen::Vector3d point = center + radius * (std::cos(t) * u + std::sin(t) * v);
        point.x() += noise_dist(rng);
        point.y() += noise_dist(rng);
        point.z() += noise_dist(rng);
        points.emplace_back(point.x(), point.y(), point.z());
    }

    return points;
}

// PCL circle fitting function
bool fit_circle_pcl(const std::vector<cv::Point3d>& points,
                   Eigen::Vector3d& center,
                   double& radius,
                   double distance_threshold = 0.1) {
    // Convert to PCL point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    for (const auto& point : points) {
        pcl::PointXYZ pcl_point;
        pcl_point.x = static_cast<float>(point.x);
        pcl_point.y = static_cast<float>(point.y);
        pcl_point.z = static_cast<float>(point.z);
        cloud->points.push_back(pcl_point);
    }

    if (cloud->points.size() < 3) {
        return false;
    }

    // Set up SAC segmentation for circle fitting
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_CIRCLE3D);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(distance_threshold);
    seg.setMaxIterations(1000);
    seg.setProbability(0.99);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() < 3) {
        return false;
    }

    // Extract results
    if (coefficients->values.size() >= 4) {
        center << coefficients->values[0],
                 coefficients->values[1],
                 coefficients->values[2];
        radius = coefficients->values[3];
        return true;
    }

    return false;
}

// CGA circle fitting function
bool fit_circle_cga(const std::vector<cv::Point3d>& points,
                   Eigen::Vector3d& center,
                   double& radius) {
    try {
        Eigen::Matrix<double, 3, 1> fit_center;
        double fit_radius;

        // Use the robust fitting method
        int result = ConformalFit3DCicle::Fit(points, fit_center, fit_radius);

        if (result == 0 && fit_radius > 0 && fit_center.allFinite()) {
            center = fit_center;
            radius = fit_radius;
            return true;
        } else {
            // Fallback to standard fitting
            result = ConformalFit3DCicle::Fit(points, fit_center, fit_radius);
            if (result == 0 && fit_radius > 0 && fit_center.allFinite()) {
                center = fit_center;
                radius = fit_radius;
                return true;
            }
        }

        return false;
    } catch (const std::exception& e) {
        std::cerr << "CGA fitting exception: " << e.what() << std::endl;
        return false;
    } catch (...) {
        return false;
    }
}

// Function to run Monte Carlo experiments
void run_monte_carlo_experiment(
    const std::string& scenario_name,
    std::function<std::vector<cv::Point3d>(const Eigen::Vector3d&, double, const Eigen::Vector3d&, int)> point_generator,
    int num_experiments = 1000,
    int seed = 42,
    const std::string& output_dir = "results"
) {
    std::cout << "Running Monte Carlo experiments for: " << scenario_name << std::endl;

    // Create output directory
    std::filesystem::create_directories(output_dir);

    // Open output files
    std::ofstream pcl_file(output_dir + "/" + scenario_name + "_pcl_results.txt");
    std::ofstream cga_file(output_dir + "/" + scenario_name + "_cga_results.txt");

    // Write headers
    pcl_file << "center_error radius_error success\n";
    cga_file << "center_error radius_error success\n";

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> center_dist(-2.0, 2.0);
    std::uniform_real_distribution<double> radius_dist(1.0, 5.0);
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    int pcl_successes = 0, cga_successes = 0;

    for (int exp = 0; exp < num_experiments; ++exp) {
        if (exp % 100 == 0) {
            std::cout << "Experiment " << exp << "/" << num_experiments << std::endl;
        }

        // Generate random ground truth
        Eigen::Vector3d true_center(center_dist(rng), center_dist(rng), center_dist(rng));
        double true_radius = radius_dist(rng);
        Eigen::Vector3d true_normal(normal_dist(rng), normal_dist(rng), normal_dist(rng));
        true_normal.normalize();

        // Generate points
        std::vector<cv::Point3d> points = point_generator(true_center, true_radius, true_normal, rng());

        // Test PCL method
        Eigen::Vector3d pcl_center;
        double pcl_radius;
        bool pcl_success = fit_circle_pcl(points, pcl_center, pcl_radius);

        if (pcl_success) {
            double center_error = (pcl_center - true_center).norm();
            double radius_error = std::abs(pcl_radius - true_radius);
            pcl_file << center_error << " " << radius_error << " 1\n";
            pcl_successes++;
        } else {
            pcl_file << "nan nan 0\n";
        }

        // Test CGA method
        Eigen::Vector3d cga_center;
        double cga_radius;
        bool cga_success = fit_circle_cga(points, cga_center, cga_radius);

        if (cga_success) {
            double center_error = (cga_center - true_center).norm();
            double radius_error = std::abs(cga_radius - true_radius);
            cga_file << center_error << " " << radius_error << " 1\n";
            cga_successes++;
        } else {
            cga_file << "nan nan 0\n";
        }
    }

    std::cout << "Results for " << scenario_name << ":" << std::endl;
    std::cout << "  PCL: " << pcl_successes << "/" << num_experiments << " successes" << std::endl;
    std::cout << "  CGA: " << cga_successes << "/" << num_experiments << " successes" << std::endl;

    pcl_file.close();
    cga_file.close();
}

int main(int argc, char* argv[]) {
    int num_experiments = 1000;
    std::string output_dir = "results";

    if (argc > 1) {
        num_experiments = std::atoi(argv[1]);
    }
    if (argc > 2) {
        output_dir = argv[2];
    }

    std::cout << "Monte Carlo Benchmark for 3D Circle Fitting" << std::endl;
    std::cout << "Number of experiments: " << num_experiments << std::endl;
    std::cout << "Output directory: " << output_dir << std::endl;
    std::cout << "===========================================" << std::endl;

    // Scenario 1: Isotropic 3D noise
    auto scenario1 = [](const Eigen::Vector3d& center, double radius, const Eigen::Vector3d& normal, int seed) {
        return generate_circle_points(center, radius, normal, 100, 0.2, seed);
    };
    run_monte_carlo_experiment("isotropic_noise", scenario1, num_experiments, 42, output_dir);

    // Scenario 2: Limited arc
    auto scenario2 = [](const Eigen::Vector3d& center, double radius, const Eigen::Vector3d& normal, int seed) {
        return generate_limited_arc_points(center, radius, normal, 100, 70.0, 0.2, seed);
    };
    run_monte_carlo_experiment("limited_arc", scenario2, num_experiments, 42, output_dir);

    // Scenario 3: Sparse points
    auto scenario3 = [](const Eigen::Vector3d& center, double radius, const Eigen::Vector3d& normal, int seed) {
        return generate_sparse_points(center, radius, normal, 12, 0.2, seed);
    };
    run_monte_carlo_experiment("sparse_points", scenario3, num_experiments, 42, output_dir);

    // Scenario 4: Symmetric distribution with non-uniform intervals
    auto scenario4 = [](const Eigen::Vector3d& center, double radius, const Eigen::Vector3d& normal, int seed) {
        return generate_symmetric_nonuniform_points(center, radius, normal, 20, 200.0, 0.2, seed);
    };
    run_monte_carlo_experiment("symmetric_distribution", scenario4, num_experiments, 42, output_dir);

    std::cout << "Monte Carlo experiments completed!" << std::endl;
    std::cout << "Results saved to: " << output_dir << std::endl;

    return 0;
}
