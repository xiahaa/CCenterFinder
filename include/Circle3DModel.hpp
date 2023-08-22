//
//  Circle3DModel.hpp
//  fit3dcicle
//
//  Created by xiao.hu on 2023/8/21.
//  Copyright © 2023 xiao.hu. All rights reserved.
//

#ifndef Circle3DModel_h
#define Circle3DModel_h

#include "AbstractModel.hpp"
#include <Eigen/Dense>
#include <Eigen/Core>
#include "Fit3DCircle.hpp"

class Point3D
: public GRANSAC::AbstractParameter
{
public:
    Point3D(GRANSAC::VPFloat x_, GRANSAC::VPFloat y_, GRANSAC::VPFloat z_):x(x_),y(y_),z(z_){}
    GRANSAC::VPFloat x;
    GRANSAC::VPFloat y;
    GRANSAC::VPFloat z;
};

class Circle3DModel
: public GRANSAC::AbstractModel<5>
{
protected:
    // Parametric form
    Eigen::Matrix<GRANSAC::VPFloat, 3, 1> center_;
    GRANSAC::VPFloat radius_;
    
    virtual GRANSAC::VPFloat ComputeDistanceMeasure(std::shared_ptr<GRANSAC::AbstractParameter> Param) override
    {
        auto ExtPoint3D = std::dynamic_pointer_cast<Point3D>(Param);
        if (ExtPoint3D == nullptr)
            throw std::runtime_error("Circle3DModel::ComputeDistanceMeasure() - Passed parameter are not of type Point3D.");
        
        // Return distance
        Eigen::Matrix<GRANSAC::VPFloat, 3, 1> pt;
        pt << ExtPoint3D->x, ExtPoint3D->y, ExtPoint3D->z;
        auto d = center_ - pt;
        GRANSAC::VPFloat dist = fabs(d.norm() - radius_);
        
        return dist;
    };
    
public:
    Circle3DModel(const std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> &InputParams)
    {
        Initialize(InputParams);
    };
    
    virtual void Initialize(const std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> &InputParams) override
    {
        if (InputParams.size() != 5)
            throw std::runtime_error("Circle3DModel - Number of input parameters does not match minimum number required for this model.");
        
        // Check for AbstractParamter types
        std::vector<Point3D> pcd;
        for (int i = 0; i < 5; i++)
        {
            auto point = std::dynamic_pointer_cast<Point3D>(InputParams[i]);
            if (point == nullptr)
                throw std::runtime_error("Circle3DModel - InputParams type mismatch. It is not a Point3D.");
            pcd.emplace_back(*point);
        }
        
        std::copy(InputParams.begin(), InputParams.end(), m_MinModelParams.begin());
        
        Eigen::Matrix<double, 3, 1> fit_center;
        double fit_radius;
        
        ConformalFit3DCicle::Fit(pcd, fit_center, fit_radius);
        center_ = fit_center;
        radius_ = fit_radius;
    };
    
    virtual std::pair<GRANSAC::VPFloat, std::vector<std::shared_ptr<GRANSAC::AbstractParameter>>> Evaluate(const std::vector<std::shared_ptr<GRANSAC::AbstractParameter>>& EvaluateParams, GRANSAC::VPFloat Threshold)
    {
        std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> Inliers;
        int nTotalParams = EvaluateParams.size();
        int nInliers = 0;
        
        for (auto& Param : EvaluateParams)
        {
            if (ComputeDistanceMeasure(Param) < Threshold)
            {
                Inliers.push_back(Param);
                nInliers++;
            }
        }
        
        GRANSAC::VPFloat InlierFraction = GRANSAC::VPFloat(nInliers) / GRANSAC::VPFloat(nTotalParams); // This is the inlier fraction
        
        return std::make_pair(InlierFraction, Inliers);
    };
};

#endif /* Circle3DModel_h */
