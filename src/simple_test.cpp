//
//  main.cpp
//  fit3dcicle
//
//  Created by xiao.hu on 2023/8/9.
//  Copyright © 2023 xiao.hu. All rights reserved.
//

#include <iostream>
#include "Fit3DCircle.hpp"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include "timer.hpp"


template <typename T, typename Derived, typename Point3>
void GenerateCircleByAngles(std::vector<T> tlist, Derived c, T r, T theta, T phi, std::vector<Point3> &circle_points)
{
    Eigen::Matrix<typename Derived::Scalar, 3, 1> n;
    Eigen::Matrix<typename Derived::Scalar, 3, 1> u;
    n << std::cos(phi) * std::sin(theta), std::sin(phi)*std::sin(theta), std::cos(theta);
    u << -std::sin(phi), std::cos(phi), 0;


    for (auto t : tlist)
    {
        Point3 pt;
        Eigen::Matrix<typename Derived::Scalar, 3, 1> res;
        res = r*std::cos(t)*u + r*std::sin(t)*n.cross(u) + c;

        pt.x = res(0), pt.y = res(1), pt.z = res(2);
        circle_points.push_back(pt);
    }
    return;
}
#if 0
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

int main(int argc, const char * argv[])
{
    // Configuration: set to true to generate synthetic data, false to use given test_data
    bool use_generated_data = true;
    if (argc > 1) {
        std::string arg1(argv[1]);
        if (arg1 == "test" || arg1 == "given" || arg1 == "data") {
            use_generated_data = false;
        }
    }

    std::vector<cv::Point3d> circle_points;

    if (use_generated_data) {
        /*-------------------------------------------------------------------------------
         Generating circle
        -------------------------------------------------------------------------------*/
        double r = 2.5;               // Radius
        double c_array[] = {3,3,4};
        Eigen::Map<Eigen::VectorXd, 0> c(c_array, 3);// Center
        double theta = 45.0/180*M_PI;     // Azimuth
        double phi   = -30.0/180*M_PI;    // Zenith

        std::vector<double> tlist;
        double ang = 0;
        const double step = 2*M_PI / 150.0;
        while (ang <= 2*M_PI)
        {
            tlist.push_back(ang);
            ang+=step;
        }

        GenerateCircleByAngles<double, Eigen::VectorXd, cv::Point3d>(tlist, c, r, theta, phi, circle_points);
        std::cout << "[INFO] Using generated circle data (" << circle_points.size() << " points)" << std::endl;
    } else {
        // Use given test_data
        circle_points.clear();
        size_t num_points = sizeof(test_data)/sizeof(double)/3;
        for (size_t i = 0; i < num_points; i++)
        {
            circle_points.push_back(cv::Point3d(test_data[(i)*3],test_data[(i)*3+1],test_data[(i)*3+2]));
        }
        std::cout << "[INFO] Using provided test_data (" << circle_points.size() << " points)" << std::endl;
    }

    // Optionally print points
    // for (auto pt : circle_points)
    //     std::cout << pt.x << "," << pt.y << "," << pt.z << std::endl;

    Eigen::Matrix<double, 3, 1> fit_center;
    double fit_radius;

    ConformalFit3DCicle::Fit(circle_points, fit_center, fit_radius);

    std::cout << "fit_center: " << fit_center << std::endl;
    std::cout << "fit_radius: " << fit_radius << std::endl;

    return 0;
}
#endif

#if 0
#include <iostream>
//#include <opencv2/opencv.hpp>
#include <cmath>
#include <random>
#include <cstdint>

#include "GRANSAC.hpp"
#include "Circle3DModel.hpp"

int main(int argc, char * argv[])
{
    /*-------------------------------------------------------------------------------
     Generating circle
     -------------------------------------------------------------------------------*/
    double r = 2.5;               // Radius
    double c_array[] = {3,3,4};
    Eigen::Map<Eigen::VectorXd, 0> c(c_array, 3);// Center
    double theta = 45.0/180*M_PI;     // Azimuth
    double phi   = -30.0/180*M_PI;    // Zenith

    //    Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(100,0.0,2*M_PI).transpose();
    std::vector<double> tlist;
    double ang = 0;
    const double step = 2*M_PI / 100;
    while (ang <= 2*M_PI)
    {
        tlist.push_back(ang);
        ang+=step;
    }
    std::vector<cv::Point3d> circle_points;
    GenerateCircleByAngles<double, Eigen::VectorXd, cv::Point3d>(tlist, c, r, theta, phi, circle_points);



    std::random_device SeedDevice;
    std::mt19937 RNG = std::mt19937(SeedDevice());
    std::uniform_int_distribution<int> UniDist(5, 10); // [Incl, Incl]
    int Perturb = 0.01;
    std::normal_distribution<GRANSAC::VPFloat> PerturbDist(0, Perturb);

    std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> CandPoints;
    for (int i = 0; i < circle_points.size(); ++i)
    {
        cv::Point3d pt(circle_points[i].x+PerturbDist(RNG),
                       circle_points[i].y+PerturbDist(RNG),
                       circle_points[i].z+PerturbDist(RNG));
        std::shared_ptr<GRANSAC::AbstractParameter> CandPt = std::make_shared<Point3D>(pt.x, pt.y, pt.z);
        CandPoints.push_back(CandPt);
    }

    for (int i = 0; i < 30; ++i)
    {
        cv::Point3d pt(UniDist(RNG) + PerturbDist(RNG),
                       UniDist(RNG) + PerturbDist(RNG),
                       UniDist(RNG) + PerturbDist(RNG));
        std::shared_ptr<GRANSAC::AbstractParameter> CandPt = std::make_shared<Point3D>(pt.x, pt.y, pt.z);
        CandPoints.push_back(CandPt);
    }

    Timer timer;
    GRANSAC::RANSAC<Circle3DModel, 5> Estimator;
    Estimator.Initialize(0.1, 300); // Threshold, iterations
    timer.tic();
    Estimator.Estimate(CandPoints);
    timer.toc(std::string("Estimate: "));

    auto BestInliers = Estimator.GetBestInliers();
    if (BestInliers.size() > 0)
    {
        circle_points.clear();
        for (auto& Inlier : BestInliers)
        {
            auto RPt = std::dynamic_pointer_cast<Point3D>(Inlier);
            cv::Point3d Pt(RPt->x, RPt->y, RPt->z);
            circle_points.emplace_back(Pt);
        }
        Eigen::Matrix<double, 3, 1> fit_center;
        double fit_radius;

        ConformalFit3DCicle::Fit(circle_points, fit_center, fit_radius);

        std::cout << "fit_center: " << fit_center << std::endl;
        std::cout << "fit_radius: " << fit_radius << std::endl;
    }

    return 0;
}
#endif
