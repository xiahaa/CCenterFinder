//
//  simple_test.cpp
//  fit3dcicle
//
//  Created by xiao.hu on 2023/8/9.
//  Copyright © 2023 xiao.hu. All rights reserved.
//
//  Unified test program for 3D circle fitting with multiple modes:
//  - Generated: Create synthetic circle data
//  - Given: Use provided test data
//  - RANSAC: Use RANSAC with noise and outliers for robust fitting
//

#include <iostream>
#include "Fit3DCircle.hpp"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include "timer.hpp"
#include <cmath>
#include <random>
#include <cstdint>
#include "rtl/RTL.hpp"
#include "RobustFit3DCircle.hpp"

/**
 * @brief Simple 3D point class for RANSAC operations
 *
 * This class represents a 3D point with x, y, z coordinates.
 * Used as the data type for RANSAC circle fitting.
 */
class Point3
{
public:
    // Default constructor - initializes to origin
    Point3() : x_(0), y_(0), z_(0) { }

    // Constructor with explicit coordinates
    Point3(double _x, double _y, double _z) : x_(_x), y_(_y), z_(_z) { }

    // Stream output operator for debugging
    friend std::ostream& operator<<(std::ostream& out, const Point3& p) {
        return out << p.x_ << ", " << p.y_ << ", " << p.z_;
    }

    // 3D coordinates
    double x_, y_, z_;
};

/**
 * @brief 3D circle model class for RANSAC fitting
 *
 * Represents a 3D circle with center, radius, and normal vector.
 * The normal vector defines the plane in which the circle lies.
 */
class Circle3D
{
public:
    // Default constructor
    Circle3D(){ }

    // Constructor with explicit parameters
    Circle3D(double cx, double cy, double cz, double radius, double nx, double ny, double nz){
        center_ << cx, cy, cz;
        radius_ = radius;
        normal_ << nx, ny, nz;
     }

    // Stream output operator for debugging
    friend std::ostream& operator<<(std::ostream& out, const Circle3D& c) {
        return out << c.center_.transpose() << ", " << c.radius_ << ", " << c.normal_.transpose();
    }

    // Circle parameters
    double radius_;                                    // Circle radius
    Eigen::Matrix<double, 3, 1> center_;              // Circle center point
    Eigen::Matrix<double, 3, 1> normal_;              // Normal vector of the circle's plane
};

/**
 * @brief RANSAC estimator for 3D circle fitting
 *
 * Implements the RTL::Estimator interface for fitting 3D circles using RANSAC.
 * Uses robust conformal geometric algebra (CGA) fitting for model computation.
 */
class Circle3DEstimator : public RTL::Estimator<Circle3D, Point3, std::vector<Point3> >
{
public:
    /**
     * @brief Compute a circle model from a set of sample points
     *
     * @param data The full dataset
     * @param samples Indices of points to use for model fitting
     * @return Circle3D The fitted circle model
     */
    virtual Circle3D ComputeModel(const std::vector<Point3>& data, const std::set<int>& samples)
    {
        // Convert sample points to cv::Point3d format for CGA fitting
        std::vector<cv::Point3d> pcd;
        for (auto itr = samples.begin(); itr != samples.end(); itr++)
        {
            const Point3& p = data[*itr];
            pcd.push_back(cv::Point3d(p.x_, p.y_, p.z_));
        }

        // Fit circle using robust CGA method
        Eigen::Matrix<double, 3, 1> fit_center;
        double fit_radius;
        Eigen::Matrix<double, 3, 1> fit_normal;
        robust_cga::RobustFit3DCircle::Fit(pcd, fit_center, fit_radius, &fit_normal);

        // Create and return the circle model
        Circle3D circle;
        circle.center_ = fit_center;
        circle.radius_ = fit_radius;
        circle.normal_ = fit_normal;
        return circle;
    }

    /**
     * @brief Compute the error between a point and a circle model
     *
     * Calculates the 3D distance from a point to a circle, considering both
     * the distance to the circle's plane and the distance to the circle's edge.
     *
     * @param circle The circle model
     * @param point The point to evaluate
     * @return double The error distance
     */
    virtual double ComputeError(const Circle3D& circle, const Point3& point)
    {
        Eigen::Vector3d pt(point.x_, point.y_, point.z_);
        Eigen::Vector3d d = circle.center_ - pt;
        Eigen::Vector3d vecC = circle.normal_;

        // Handle degenerate case (zero normal vector)
        if (vecC.norm() == 0)
            return std::numeric_limits<double>::max();

        // Calculate the plane equation: ax + by + cz + d = 0
        double k = -vecC.dot(circle.center_);
        Eigen::Vector4d plane_eq(vecC[0], vecC[1], vecC[2], k);

        // Distance from point to the circle's plane
        double dist_pt_plane = (plane_eq[0] * pt[0] + plane_eq[1] * pt[1] +
                               plane_eq[2] * pt[2] + plane_eq[3]) / vecC.norm();

        // Distance from point to the circle's edge (projected onto the plane)
        Eigen::Vector3d dist_pt_inf_circle_vec = vecC.cross(d);
        double dist_pt_inf_circle = dist_pt_inf_circle_vec.norm() - circle.radius_;

        // Total distance from point to circle (hypotenuse of the two distances)
        double dist_pt = std::sqrt(std::pow(dist_pt_inf_circle, 2) + std::pow(dist_pt_plane, 2));

        return dist_pt;
    }
};

/**
 * @brief Generate 3D circle points using spherical coordinates
 *
 * Creates a set of 3D points that lie on a circle defined by:
 * - Center point c
 * - Radius r
 * - Orientation defined by spherical angles theta (azimuth) and phi (zenith)
 *
 * @tparam T Numeric type for calculations
 * @tparam Derived Eigen vector type for center
 * @tparam Point3 Point type for output
 * @param tlist List of parameter values (angles) for circle generation
 * @param c Center of the circle
 * @param r Radius of the circle
 * @param theta Azimuth angle (rotation around z-axis)
 * @param phi Zenith angle (inclination from z-axis)
 * @param circle_points Output vector to store generated points
 */
template <typename T, typename Derived, typename Point3>
void GenerateCircleByAngles(std::vector<T> tlist, Derived c, T r, T theta, T phi, std::vector<Point3> &circle_points)
{
    // Calculate normal vector of the circle's plane using spherical coordinates
    Eigen::Matrix<typename Derived::Scalar, 3, 1> n;
    n << std::cos(phi) * std::sin(theta), std::sin(phi)*std::sin(theta), std::cos(theta);

    // Calculate first tangent vector (perpendicular to normal and z-axis)
    Eigen::Matrix<typename Derived::Scalar, 3, 1> u;
    u << -std::sin(phi), std::cos(phi), 0;

    // Generate circle points using parametric equation
    for (auto t : tlist)
    {
        Point3 pt;
        Eigen::Matrix<typename Derived::Scalar, 3, 1> res;

        // Parametric circle equation: P(t) = center + r*cos(t)*u + r*sin(t)*(n×u)
        res = r*std::cos(t)*u + r*std::sin(t)*n.cross(u) + c;

        // Convert to point format
        pt.x = res(0), pt.y = res(1), pt.z = res(2);
        circle_points.push_back(pt);
    }
    return;
}
// Predefined test data - array of 3D points (x,y,z coordinates)
// This data represents points that should lie approximately on a 3D circle
double test_data[] = {
    1.6832706697750046,0.9751299488013927,4.121819521070186,
    1.843071406095262,0.7714591223004006,4.094673424895004,
    1.816443669906431,0.8744312527813478,3.9992711793853513,
    1.882498796938751,0.6720245510990221,3.9548845779908017,
    1.9124038269665051,0.6756148848505107,3.9984407538242723,
    1.852426825069269,0.7493801553605415,3.7813420143435614,
    1.9318126078707376,0.6321608438616902,3.8205186895300356,
    1.8989291761130698,0.8031515806840624,3.7328057885844066,
    1.87444779522176,0.7134817149264335,3.6456948387326547,
    1.8959929528784527,0.5665439382368892,3.7069186767995004,
    2.074777172437536,0.7528660959347276,3.6627283603717173,
    2.278200739366805,0.7353306141520216,3.5359040559856028,
    2.3177394917543532,0.6717856523054778,3.348976235316639,
    2.305122649239423,0.660809869147072,3.5320512352574833,
    2.328644215104718,0.6025670134983305,3.4754577768540944,
    2.3646131618508766,0.7052421861829917,3.526691759725171,
    2.4496403918187495,0.7008108833530603,3.3834635993081896,
    2.4625428969754033,0.6060341798153763,3.2991988241820773,
    2.459598055881362,0.6380424767963573,3.2339487308629566,
    2.4333692205158166,0.7350057113155481,3.34907161842683,
    2.6612830592852337,0.7699069605299078,3.1779879352448983,
    2.5652844635545202,0.5888959835347078,3.200551020585818,
    2.796291785613804,0.7139923452222791,3.103028381424681,
    2.6824101650250403,0.6399130479387766,3.014390962551298,
    2.8078804876655994,0.6776192360873912,3.0450302226391,
    2.8778230000129574,0.5584791570435007,3.1233622613121654,
    2.7808259799123554,0.7116047058933415,2.9365123289132224,
    2.967485255370753,0.7940495616727585,2.7734102021046003,
    2.9795129365494977,0.7693198288225658,2.7478675886566126,
    2.921432650864718,0.742617776348045,2.6544893991711938,
    3.015849194779801,0.859691782563724,2.7429563712144804,
    3.249059450098446,0.7375809815211556,2.812714967461209,
    3.1229394278256404,0.7405916751236974,2.816171122280753,
    3.219887684335651,0.9191430876821542,2.6694336855446776,
    3.3234114066894183,0.9537012634044723,2.8807533621707444,
    3.164346472520131,0.8825146818484788,2.8032202555873904,
    3.5336650087317096,1.0012780816490583,2.5555530405503313,
    3.407079181315902,0.7526002680975172,2.720176919108199,
    3.5784043807882338,1.09452320739042,2.7381482444391207,
    3.4284506698079418,0.899237780060474,2.666486444364594,
    3.55568499492551,1.0088243522438902,2.4928477784369107,
    3.51579514334479,0.8391467295548509,2.6763350438528133,
    3.6371637885387544,0.9509122388736055,2.4458316419891184,
    3.7117506115847663,1.355455781085893,2.4360537601755823,
    3.568600234870548,1.2186646971059631,2.4884524030723427,
    3.599411049371955,1.1541658848585827,2.4930315886754286,
    3.772557672628144,1.4012966263285285,2.42282262251528,
    4.016701934662973,1.2942828488698044,2.5244328369386295,
    3.8652954670150637,1.367569882866456,2.412422382448171,
    3.8302022668509803,1.240424288488636,2.485432417198807,
    3.8343047142850453,1.348980469472031,2.3200750301472928,
    4.08508416334152,1.229503576685295,2.423488948520432,
    3.901828367352792,1.4666341094820898,2.143298392129167,
    4.392281642937492,1.5849067925396854,2.305274094067744,
    4.203665655267543,1.6117558324231003,2.1568316552713296,
    4.174328084910846,1.5157240917306751,2.066272886664373,
    4.222410645052283,1.6475693293037594,2.221028098513643,
    4.12177200040485,1.8013112198498322,2.0476373467942515,
    4.188322743677865,1.8884112026333881,2.2820534956069993,
    4.347932977657649,1.6843595854756568,2.096634240230431,
    4.424667893702422,1.8635918489918526,2.4491848294425504,
    4.400175984243966,1.8952183498212494,2.291597418249493,
    4.433836083296191,1.8004272166261182,2.178445983324643,
    4.572516583128506,1.8925074903727979,2.1888342668455367,
    4.45342249217274,2.146007074493802,2.188705529278964,
    4.593015594427465,2.0502199623856727,2.201317075661221,
    4.4904901287500785,1.9393120905237866,2.2853143629397445,
    4.48585129078437,2.188277010955246,2.39854604991796,
    4.6370581841570715,2.206291014739608,2.3663210826669347,
    4.4872581646291065,2.2031179538773453,2.206867448077289,
    4.525609389882044,2.36790635658834,2.3395836253543827,
    4.778876433459592,2.468573056961086,2.284612229109273,
    4.76878049730126,2.4372755215116912,2.120985949905867,
    4.643132973203124,2.5093090308842836,2.314532362543428,
    4.5658226931142565,2.6302909275584887,2.196888500933942,
    4.703806113860315,2.51173152937348,2.4319101992695993,
    5.039329054953187,2.7991182804479116,2.2800047168313844,
    4.703126724423914,2.7266278805270017,2.348431602078394,
    4.855046788451264,2.9038829100647674,2.320587181850809,
    4.818469949847069,2.7473394040166816,2.471952083911469,
    4.725013315004643,2.866847137789248,2.13530126877799,
    4.84907888090141,3.0723508439272047,2.3020663781560864,
    4.848062197186836,3.0822709450411137,2.345329794494057,
    4.759637148205683,3.127518123997316,2.679341890471018,
    5.035430122684494,3.1329523482633803,2.426217341000672,
    5.034492205905964,2.977363539078546,2.4219136460326096,
    4.945411029907279,3.0145034120955625,2.362180552029566,
    4.9885919195043,3.330232918146335,2.45644769390192,
    4.937669767865428,3.525405198513244,2.4532177695853674,
    4.88486399287941,3.407820675625762,2.419526383984406,
    5.229903963787904,3.4640570003453055,2.566646693105073,
    5.153975009051558,3.7345233135703064,2.5478530269319277,
    4.815612338210983,3.4514655046099074,2.501894769186868,
    5.171228460778208,3.4021244851695926,2.542885167937023,
    4.951821635204731,3.6703226988810074,2.5370021669712304,
    4.927876479084568,3.8735292440343003,2.633319791925958,
    4.982084159326887,3.729396204701583,2.6725810271587327,
    4.894529735887229,3.91279536669174,2.6782561393074564,
    4.831560930720577,3.8450123897266977,2.6281700194971513,
    5.009755789421601,3.6679954318559833,2.9565565082123086};

/**
 * @brief Test modes for the circle fitting program
 */
enum class TestMode {
    GENERATED,  // Generate synthetic circle data
    GIVEN,      // Use provided test_data
    RANSAC      // Use RANSAC with noise and outliers
};

/**
 * @brief Parse command line mode string to TestMode enum
 *
 * @param mode_str String from command line argument
 * @return TestMode Corresponding enum value
 */
TestMode parseMode(const std::string& mode_str) {
    if (mode_str == "generated" || mode_str == "gen" || mode_str == "g") {
        return TestMode::GENERATED;
    } else if (mode_str == "given" || mode_str == "test" || mode_str == "data" || mode_str == "t") {
        return TestMode::GIVEN;
    } else if (mode_str == "ransac" || mode_str == "r") {
        return TestMode::RANSAC;
    } else {
        std::cerr << "Unknown mode: " << mode_str << std::endl;
        std::cerr << "Usage: " << std::endl;
        std::cerr << "  generated/gen/g - Generate synthetic circle data" << std::endl;
        std::cerr << "  given/test/data/t - Use provided test_data" << std::endl;
        std::cerr << "  ransac/r - Use RANSAC with noise and outliers" << std::endl;
        exit(1);
    }
}

/**
 * @brief Print usage information for the program
 *
 * @param program_name Name of the executable (argv[0])
 */
void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [mode] [options]" << std::endl;
    std::cout << "Modes:" << std::endl;
    std::cout << "  generated/gen/g - Generate synthetic circle data (default)" << std::endl;
    std::cout << "  given/test/data/t - Use provided test_data" << std::endl;
    std::cout << "  ransac/r - Use RANSAC with noise and outliers" << std::endl;
    std::cout << std::endl;
    std::cout << "RANSAC options:" << std::endl;
    std::cout << "  --threshold <value>  - RANSAC threshold (default: 0.1)" << std::endl;
    std::cout << "  --iterations <value> - RANSAC iterations (default: 300)" << std::endl;
    std::cout << "  --noise <value>      - Noise level (default: 0.01)" << std::endl;
    std::cout << "  --outliers <value>   - Number of outliers (default: 30)" << std::endl;
}

/**
 * @brief Main function for 3D circle fitting test program
 *
 * Supports three modes:
 * 1. GENERATED: Create synthetic circle data and fit it
 * 2. GIVEN: Use predefined test data and fit it
 * 3. RANSAC: Create noisy data with outliers and use RANSAC for robust fitting
 *
 * @param argc Number of command line arguments
 * @param argv Command line arguments
 * @return int Exit status
 */
int main(int argc, const char * argv[])
{
    // Parse command line arguments
    TestMode mode = TestMode::GENERATED;  // Default mode
    double ransac_threshold = 0.1;        // RANSAC distance threshold
    int ransac_iterations = 300;          // Number of RANSAC iterations
    double noise_level = 0.01;            // Gaussian noise standard deviation
    int num_outliers = 30;                // Number of outlier points to add

    if (argc > 1) {
        std::string arg1(argv[1]);
        if (arg1 == "--help" || arg1 == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        mode = parseMode(arg1);
    }

    // Parse additional RANSAC options
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) break;

        std::string option(argv[i]);
        std::string value(argv[i + 1]);

        if (option == "--threshold") {
            ransac_threshold = std::stod(value);
        } else if (option == "--iterations") {
            ransac_iterations = std::stoi(value);
        } else if (option == "--noise") {
            noise_level = std::stod(value);
        } else if (option == "--outliers") {
            num_outliers = std::stoi(value);
        }
    }

    std::vector<cv::Point3d> circle_points;

    if (mode == TestMode::GENERATED) {
        /*-------------------------------------------------------------------------------
         MODE 1: Generate synthetic circle data
         Creates a perfect 3D circle and fits it to test the basic fitting algorithm
        -------------------------------------------------------------------------------*/
        double r = 2.5;               // Circle radius
        double c_array[] = {3,3,4};   // Circle center coordinates
        Eigen::Map<Eigen::VectorXd, 0> c(c_array, 3);
        double theta = 45.0/180*M_PI;     // Azimuth angle (rotation around z-axis)
        double phi   = 0.0/180*M_PI;    // Zenith angle (inclination from z-axis)

        // Generate parameter values for circle points
        std::vector<double> tlist;
        double ang = 0;
        const double step = 2*M_PI / 150.0;  // 150 points around the circle
        while (ang <= 2*M_PI)
        {
            tlist.push_back(ang);
            ang+=step;
        }

        // Generate the circle points
        GenerateCircleByAngles<double, Eigen::VectorXd, cv::Point3d>(tlist, c, r, theta, phi, circle_points);
        std::cout << "[INFO] Using generated circle data (" << circle_points.size() << " points)" << std::endl;
        std::cout << "real_center: " << c << std::endl;
        std::cout << "real_radius: " << r << std::endl;
    } else if (mode == TestMode::GIVEN) {
        /*-------------------------------------------------------------------------------
         MODE 2: Use predefined test data
         Loads the hardcoded test_data array and fits a circle to it
        -------------------------------------------------------------------------------*/
        circle_points.clear();
        size_t num_points = sizeof(test_data)/sizeof(double)/3;  // Each point has 3 coordinates

        // Convert test_data array to cv::Point3d format
        for (size_t i = 0; i < num_points; i++)
        {
            circle_points.push_back(cv::Point3d(test_data[(i)*3],test_data[(i)*3+1],test_data[(i)*3+2]));
        }
        std::cout << "[INFO] Using provided test_data (" << circle_points.size() << " points)" << std::endl;
    } else if (mode == TestMode::RANSAC) {
        /*-------------------------------------------------------------------------------
         MODE 3: RANSAC with noise and outliers
         Creates a circle, adds noise and outliers, then uses RANSAC for robust fitting
        -------------------------------------------------------------------------------*/
        double r = 2.5;               // Circle radius
        double c_array[] = {3,3,4};   // Circle center
        Eigen::Map<Eigen::VectorXd, 0> c(c_array, 3);
        double theta = 45.0/180*M_PI;     // Azimuth angle
        double phi   = -30.0/180*M_PI;    // Zenith angle

        // Generate fewer points for RANSAC (computational efficiency)
        std::vector<double> tlist;
        double ang = 0;
        const double step = 2*M_PI / 100.0;  // 100 points around the circle
        while (ang <= 2*M_PI)
        {
            tlist.push_back(ang);
            ang+=step;
        }

        // Generate the base circle points
        GenerateCircleByAngles<double, Eigen::VectorXd, cv::Point3d>(tlist, c, r, theta, phi, circle_points);

        // Setup random number generators for noise and outliers
        std::random_device SeedDevice;
        std::mt19937 RNG = std::mt19937(SeedDevice());
        std::uniform_int_distribution<int> UniDist(5, 10);  // Range for outlier coordinates
        std::normal_distribution<double> PerturbDist(0, noise_level);  // Gaussian noise

        // Convert to Point3 format and add noise to inlier points
        std::vector<Point3> ransac_points;
        for (int i = 0; i < circle_points.size(); ++i)
        {
            cv::Point3d pt(circle_points[i].x + PerturbDist(RNG),
                           circle_points[i].y + PerturbDist(RNG),
                           circle_points[i].z + PerturbDist(RNG));
            ransac_points.push_back(Point3(pt.x, pt.y, pt.z));
        }

        // Add random outlier points
        for (int i = 0; i < num_outliers; ++i)
        {
            cv::Point3d pt(UniDist(RNG) + PerturbDist(RNG),
                           UniDist(RNG) + PerturbDist(RNG),
                           UniDist(RNG) + PerturbDist(RNG));
            ransac_points.push_back(Point3(pt.x, pt.y, pt.z));
        }

        std::cout << "[INFO] Using RANSAC mode with " << circle_points.size() << " inliers + "
                  << num_outliers << " outliers (total: " << ransac_points.size() << " points)" << std::endl;
        std::cout << "[INFO] RANSAC parameters: threshold=" << ransac_threshold
                  << ", iterations=" << ransac_iterations << ", noise=" << noise_level << std::endl;

        // Setup and run RANSAC
        Timer timer;
        Circle3D model;
        Circle3DEstimator estimator;
        RTL::RANSAC<Circle3D, Point3, std::vector<Point3> > ransac(&estimator);

        ransac.SetParamIteration(ransac_iterations);
        ransac.SetParamThreshold(ransac_threshold);

        // Run RANSAC algorithm
        timer.tic();
        double loss = ransac.FindBest(model, ransac_points, ransac_points.size(), 5);
        timer.toc(std::string("RANSAC Estimate: "));

        // Find inliers from the best model
        std::vector<int> inliers = ransac.FindInliers(model, ransac_points, ransac_points.size());

        if (inliers.size() > 0)
        {
            // Convert inliers back to cv::Point3d format for final fitting
            circle_points.clear();
            for (auto &idx : inliers)
            {
                const Point3& pt = ransac_points[idx];
                circle_points.emplace_back(pt.x_, pt.y_, pt.z_);
            }
            std::cout << "[INFO] RANSAC found " << circle_points.size() << " inliers" << std::endl;
        } else {
            std::cout << "[WARNING] RANSAC found no inliers, using original data" << std::endl;
        }
        std::cout << "real_center: " << c << std::endl;
        std::cout << "real_radius: " << r << std::endl;
    }

    // Optional: Print all points for debugging
    // for (auto pt : circle_points)
    //     std::cout << pt.x << "," << pt.y << "," << pt.z << std::endl;

    /*-------------------------------------------------------------------------------
     Final circle fitting using Conformal Geometric Algebra (CGA)
     This is the core fitting algorithm that works on the final point set
    -------------------------------------------------------------------------------*/
    Eigen::Matrix<double, 3, 1> fit_center;  // Fitted circle center
    double fit_radius;                        // Fitted circle radius

    // Perform the actual circle fitting
    Timer fit_timer;
    fit_timer.tic();
    ConformalFit3DCicle::Fit(circle_points, fit_center, fit_radius);
    fit_timer.toc(std::string("Circle Fitting: "));

    // Output results
    std::cout << "fit_center: " << fit_center << std::endl;
    std::cout << "fit_radius: " << fit_radius << std::endl;

    return 0;
}
