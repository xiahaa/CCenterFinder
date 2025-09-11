//
//  fit3dcicle.hpp
//  fit3dcicle
//
//  Created by xiao.hu on 2023/8/9.
//  Copyright © 2023 xiao.hu. All rights reserved.
//

#ifndef fit3dcicle_hpp
#define fit3dcicle_hpp

#include <stdio.h>
#include <stdlib.h>
#include <glog/logging.h>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <Eigen/Core>

class ConformalFit3DCicle{
private:
    ConformalFit3DCicle() = default;

public:
    template <typename Derived>
    static int Skew(const Eigen::Matrix<typename Derived::Scalar, 3, 1> &x, Eigen::MatrixBase<Derived> &xhat)
    {
        xhat << 0,   -x[2],  x[1],
                x[2],    0, -x[0],
               -x[1], x[0],     0;
        return 0;
    }

    template <typename Derived>
    static int OuterProduct(const Eigen::MatrixBase<Derived> &y,
                            const Eigen::MatrixBase<Derived> &x,
                            Eigen::Matrix<typename Derived::Scalar, 10, 1> &val)
    {
        typedef typename Derived::Scalar Scalar_t;
        Eigen::Matrix<Scalar_t,3,3> yhat;
        Skew(y.head(3), yhat);
        Eigen::Matrix<Scalar_t,3,3> y0_eye = y[3] * Eigen::Matrix<Scalar_t,3,3>::Identity();
        Eigen::Matrix<Scalar_t,3,3> yinf_eye = y[4] * Eigen::Matrix<Scalar_t, 3, 3>::Identity();
        Eigen::Matrix<Scalar_t, 10, 5> A;
        A.setZero();
        A.template block<3, 3>(0, 0) = yhat;
        A.template block<3, 3>(3, 0) = y0_eye;
        A.template block<3, 1>(3, 3) = -y.head(3);
        A.template block<3, 3>(6, 0) = -yinf_eye;
        A.template block<3, 1>(6, 4) = y.head(3);
        A(9,3) = -y(4);
        A(9,4) = y(3);
        val = A*x;
        return 0;
    }


    template <typename T, typename Derived>
    static int EstablishPmat(const std::vector<T> &pcd, Eigen::MatrixBase<Derived> &Pmat)
    {
        typedef typename Derived::Scalar Scalar_t;
        std::size_t n = pcd.size();
        Eigen::Matrix<Scalar_t, 5, 5> DDt;
        DDt.setZero();
        for (std::size_t i = 0; i < pcd.size(); i++)
        {
            Eigen::Matrix<Scalar_t,3,1> v;
            v << pcd[i].x, pcd[i].y, pcd[i].z;
            Scalar_t vnorm = v.norm();
            Scalar_t vnorm2 = vnorm*vnorm;
            Scalar_t vnorm4 = vnorm2*vnorm2;

            DDt.template block<3,3>(0, 0) = DDt.template block<3,3>(0, 0) + v * v.transpose();
            DDt.template block<3,1>(0, 3) = DDt.template block<3,1>(0, 3) -0.5 * vnorm2 * v;
            DDt.template block<3,1>(0, 4) = DDt.template block<3,1>(0, 4) -v;

            DDt.template block<1,3>(3, 0) = DDt.template block<1,3>(3, 0) + v.transpose();

            DDt(3, 3) = DDt(3, 3) - 0.5 * vnorm2;
            DDt(3, 4) = DDt(3, 4) - 1;

            DDt.template block<1, 3>(4, 0) = DDt.template block<1,3>(4, 0) + 0.5 * vnorm2 * v.transpose();
            DDt(4, 3) = DDt(4, 3) -0.25 * vnorm4;
            DDt(4, 4) = DDt(4, 4) -0.5 * vnorm2;
        }
        Pmat = DDt / n;
        return 0;
    }

    template <typename Derived>
    static int ExtractGeometricParameters(const Eigen::Matrix<typename Derived::Scalar, 10, 1> &e,
                                          Eigen::MatrixBase<Derived> &c,
                                          typename Derived::Scalar &radius)
    {
        typedef typename Derived::Scalar Scalar_t;
        auto ei = e.head(3);
        auto eoi = e.template segment<3>(3);//e.middleRows(3,3);
        auto einfi = e.template segment<3>(6);//e.middleRows(6,3);
        auto eoinf = -e[9];
        auto alpha = eoi.norm();
        auto n1 = -eoi/alpha;
        auto n = -eoi;
        auto B0 = eoinf;
        auto B1 = ei[0], B2 = ei[1], B3 = ei[2];

        Eigen::Matrix<Scalar_t, 3, 3> A;
        A.row(0) << B0,-B3,B2;
        A.row(1) << B3, B0,-B1;
        A.row(2) << -B2,B1,B0;
        c = A*n/(n.norm() * n.norm());
        radius = c.norm()*c.norm()-2*n1.dot(einfi)/alpha-2*(c.dot(n1))*(c.dot(n1));
        if(radius>0)
            radius = std::sqrt(radius);
        else
            radius = 0;
        return 0;
    }

    template <typename T, typename Derived>
    static int Fit(const std::vector<T> &ps, Eigen::MatrixBase<Derived> &center, typename Derived::Scalar &radius)
    {
        typedef typename Derived::Scalar Scalar_t;
        Eigen::Matrix<Scalar_t, 5, 5> M;
        EstablishPmat<T>(ps, M);
        Eigen::EigenSolver<Eigen::Matrix<Scalar_t, 5, 5>> eigSolver(M);
        auto evals = eigSolver.eigenvalues();
        std::vector<int> index({0,1,2,3,4});
        std::sort(index.begin(), index.end(), [evals](const int i, const int j){
            return evals[i].real() < evals[j].real();
        });
        std::size_t i = 0;
        for (i = 0; i < index.size(); i++)
        {
            if (evals[index[i]].real()>0)
                break;
        }
        auto index1 = index[i];
        auto index2 = index[i+1];
        Eigen::Matrix<Scalar_t, 5, 1> sol2 = eigSolver.eigenvectors().col(index2).real();
        Eigen::Matrix<Scalar_t, 5, 1> sol1 = eigSolver.eigenvectors().col(index1).real();
        Eigen::Matrix<Scalar_t, 10, 1> sol_final;

        OuterProduct(sol2, sol1, sol_final);
        ExtractGeometricParameters(sol_final, center, radius);
        return 0;
    }

    // todo, add ransac

};

#endif /* fit3dcicle_hpp */
