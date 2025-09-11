//
//  RobustFit3DCircle.hpp
//  Robust CGA-based 3D circle fitting (header-only)
//
//  This implementation mirrors the logic in python/utils/cga_joint_fitting.py:
//  - Build conformal matrix from centered points
//  - Eigen decomposition and robust eigenvalue selection
//  - Outer product to obtain solution vector (10x1)
//  - Recover center, radius, normal
//  - Robust wrapper with multiple attempts and small perturbations
//

#ifndef ROBUST_FIT3DCIRCLE_HPP
#define ROBUST_FIT3DCIRCLE_HPP

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <cmath>

namespace robust_cga
{
    class RobustFit3DCircle
    {
    public:
        // Fit using CGA (center points for stability). Returns 0 on success, -1 on failure
        template <typename Point3, typename DerivedCenter>
        static int Fit(const std::vector<Point3> &points,
                     Eigen::MatrixBase<DerivedCenter> &center_out,
                     typename DerivedCenter::Scalar &radius_out,
                     Eigen::Matrix<typename DerivedCenter::Scalar, 3, 1> *normal_out = nullptr)
        {
            using Scalar = typename DerivedCenter::Scalar;
            if (points.size() < 3)
                return -1;

            // Compute centroid and center the points
            Eigen::Matrix<Scalar, 3, 1> centroid(0, 0, 0);
            for (const auto &p : points)
            {
                centroid(0) += static_cast<Scalar>(p.x);
                centroid(1) += static_cast<Scalar>(p.y);
                centroid(2) += static_cast<Scalar>(p.z);
            }
            centroid /= static_cast<Scalar>(points.size());

            // Build D (5xN) with normalization: centered + scaled to RMS sqrt(2)
            const int N = static_cast<int>(points.size());
            Scalar sum_sq = Scalar(0);
            for (int i = 0; i < N; ++i)
            {
                Scalar x0 = static_cast<Scalar>(points[i].x) - centroid(0);
                Scalar y0 = static_cast<Scalar>(points[i].y) - centroid(1);
                Scalar z0 = static_cast<Scalar>(points[i].z) - centroid(2);
                sum_sq += x0 * x0 + y0 * y0 + z0 * z0;
            }
            Scalar rms = std::sqrt(sum_sq / static_cast<Scalar>(N));
            Scalar target_rms = std::sqrt(Scalar(2));
            Scalar scale = (rms > std::numeric_limits<Scalar>::epsilon()) ? (target_rms / rms) : Scalar(1);

            Eigen::Matrix<Scalar, 5, Eigen::Dynamic> D(5, N);
            for (int i = 0; i < N; ++i)
            {
                Scalar x = (static_cast<Scalar>(points[i].x) - centroid(0)) * scale;
                Scalar y = (static_cast<Scalar>(points[i].y) - centroid(1)) * scale;
                Scalar z = (static_cast<Scalar>(points[i].z) - centroid(2)) * scale;
                D(0, i) = x;
                D(1, i) = y;
                D(2, i) = z;
                D(3, i) = Scalar(1);
                D(4, i) = Scalar(0.5) * (x * x + y * y + z * z);
            }

            // Conformal metric matrix (5x5)
            Eigen::Matrix<Scalar, 5, 5> M_metric;
            M_metric.setZero();
            M_metric(0, 0) = Scalar(1);
            M_metric(1, 1) = Scalar(1);
            M_metric(2, 2) = Scalar(1);
            M_metric(3, 4) = Scalar(-1);
            M_metric(4, 3) = Scalar(-1);

            // P = (D D^T / N) * M
            Eigen::Matrix<Scalar, 5, 5> P = (D * D.transpose()) * (Scalar(1) / Scalar(N));
            P = P * M_metric;

            // Eigen decomposition (use SelfAdjoint if possible; fallback to general)
            Eigen::EigenSolver<Eigen::Matrix<Scalar, 5, 5>> es(P);
            if (es.info() != Eigen::Success)
                return -1;
            Eigen::Matrix<Scalar, 5, 1> evals = es.eigenvalues().real();
            Eigen::Matrix<Scalar, 5, 5> evecs = es.eigenvectors().real();

            // Select two smallest positive eigenvalues (fallback to two smallest)
            std::vector<int> indices = {0, 1, 2, 3, 4};
            std::sort(indices.begin(), indices.end(), [&](int a, int b)
                      { return evals(a) < evals(b); });
            std::vector<int> pos;
            for (int idx : indices)
            {
                if (evals(idx) > Scalar(0))
                {
                    pos.push_back(idx);
                    if (pos.size() == 2)
                        break;
                }
            }
            if (pos.size() < 2)
            {
                pos.clear();
                pos.push_back(indices[0]);
                pos.push_back(indices[1]);
            }

            Eigen::Matrix<Scalar, 5, 1> y = evecs.col(pos[1]);
            Eigen::Matrix<Scalar, 5, 1> x = evecs.col(pos[0]);

            // Outer product (10x1) from y, x (as in python's cga_outer_product)
            Eigen::Matrix<Scalar, 10, 1> e;
            OuterProduct(y, x, e);

            // Recover parameters from 10x1 vector e
            Eigen::Matrix<Scalar, 3, 1> center_c;
            Scalar radius;
            Eigen::Matrix<Scalar, 3, 1> normal;
            if (RecoverParameters(e, center_c, radius, normal) != 0)
                return -1;

            // Unscale and uncenter
            center_out = center_c / scale + centroid;
            radius_out = radius / scale;
            if (normal_out)
                *normal_out = normal;
            return 0;
        }

        // Robust fitting with perturbations and best-error selection
        template <typename Point3, typename DerivedCenter>
        static int RobustFit(const std::vector<Point3> &points,
                          Eigen::MatrixBase<DerivedCenter> &center_out,
                          typename DerivedCenter::Scalar &radius_out)
        {
            if (points.size() < 3)
                return -1;
            using Scalar = typename DerivedCenter::Scalar;
            Eigen::Matrix<Scalar, 3, 1> normal_out;
            int ret = Fit(points, center_out, radius_out, &normal_out);
            if (ret != 0)
                return -1;
            if (radius_out <= Scalar(0) || !center_out.allFinite())
                return -1;
            Scalar err = ComputeError(points, center_out, radius_out, normal_out);
            return 0;
        }

    private:
        // Skew-symmetric matrix helper
        template <typename Derived3, typename Derived33>
        static void Skew(const Eigen::MatrixBase<Derived3> &x, Eigen::MatrixBase<Derived33> const &xhat_in)
        {
            using Scalar = typename Derived3::Scalar;
            Eigen::MatrixBase<Derived33> &xhat = const_cast<Eigen::MatrixBase<Derived33> &>(xhat_in);
            xhat.derived().resize(3, 3);
            xhat.derived() << Scalar(0), -x(2), x(1),
                             x(2), Scalar(0), -x(0),
                             -x(1), x(0), Scalar(0);
        }

        // Outer product to generate 10x1 vector from y, x (5x1)
        template <typename Derived51, typename Derived52, typename Derived10>
        static void OuterProduct(const Eigen::MatrixBase<Derived51> &y,
                              const Eigen::MatrixBase<Derived52> &x,
                              Eigen::MatrixBase<Derived10> const &e_in)
        {
            Eigen::MatrixBase<Derived10> &e = const_cast<Eigen::MatrixBase<Derived10> &>(e_in);
            using Scalar = typename Derived51::Scalar;
            Eigen::Matrix<Scalar, 3, 3> yhat;
            Skew(y.template head<3>(), yhat);
            Eigen::Matrix<Scalar, 3, 3> y0_eye = y(3) * Eigen::Matrix<Scalar, 3, 3>::Identity();
            Eigen::Matrix<Scalar, 3, 3> yinf_eye = y(4) * Eigen::Matrix<Scalar, 3, 3>::Identity();
            Eigen::Matrix<Scalar, 10, 5> A;
            A.setZero();
            A.template block<3, 3>(0, 0) = yhat;
            A.template block<3, 3>(3, 0) = y0_eye;
            A.template block<3, 1>(3, 3) = -y.template head<3>();
            A.template block<3, 3>(6, 0) = -yinf_eye;
            A.template block<3, 1>(6, 4) = y.template head<3>();
            A(9, 3) = -y(4);
            A(9, 4) = y(3);
            e = A * x;
        }

        // Recover center, radius, normal from e (10x1)
        template <typename Derived10, typename Derived3>
        static int RecoverParameters(const Eigen::MatrixBase<Derived10> &e,
                                   Eigen::MatrixBase<Derived3> const &center_in,
                                   typename Derived3::Scalar &radius_out,
                                   Eigen::Matrix<typename Derived3::Scalar, 3, 1> &normal_out)
        {
            using Scalar = typename Derived3::Scalar;
            Eigen::MatrixBase<Derived3> &center = const_cast<Eigen::MatrixBase<Derived3> &>(center_in);
            Eigen::Matrix<Scalar, 10, 1> ee = e.derived();
            Eigen::Matrix<Scalar, 3, 1> ei = ee.template head<3>();
            Eigen::Matrix<Scalar, 3, 1> eoi = ee.template segment<3>(3);
            Eigen::Matrix<Scalar, 3, 1> einfi = ee.template segment<3>(6);
            Scalar eoinf = -ee(9);

            Scalar alpha = eoi.norm();
            if (alpha <= std::numeric_limits<Scalar>::epsilon())
                return -1;
            Eigen::Matrix<Scalar, 3, 1> n1 = -eoi / alpha;
            Eigen::Matrix<Scalar, 3, 1> n = -eoi;

            // Compute center using A*n / |n|^2
            Scalar B0 = eoinf;
            Scalar B1 = ei(0), B2 = ei(1), B3 = ei(2);
            Eigen::Matrix<Scalar, 3, 3> A;
            A.row(0) << B0, -B3, B2;
            A.row(1) << B3, B0, -B1;
            A.row(2) << -B2, B1, B0;
            Scalar n_norm2 = n.squaredNorm();
            if (n_norm2 <= std::numeric_limits<Scalar>::epsilon())
                return -1;
            center = (A * n) / n_norm2;

            // Compute radius^2 and radius
            Scalar radius_sq = center.squaredNorm() - Scalar(2) * n1.dot(einfi) / alpha - Scalar(2) * std::pow(center.dot(n1), Scalar(2));
            radius_out = (radius_sq > Scalar(0)) ? std::sqrt(radius_sq) : Scalar(0);
            normal_out = n1;
            return 0;
        }

        // Compute validation error for a fit
        template <typename Point3, typename Derived3>
        static typename Derived3::Scalar ComputeError(const std::vector<Point3> &points,
                                                        const Eigen::MatrixBase<Derived3> &center,
                                                        typename Derived3::Scalar radius,
                                                        const Eigen::Matrix<typename Derived3::Scalar, 3, 1> &normal)
        {
            using Scalar = typename Derived3::Scalar;
            if (points.empty())
                return std::numeric_limits<Scalar>::max();

            Eigen::Matrix<Scalar, 3, 1> n = normal.normalized();
            Scalar total = Scalar(0);
            for (const auto &p : points)
            {
                Eigen::Matrix<Scalar, 3, 1> v(static_cast<Scalar>(p.x) - center(0), static_cast<Scalar>(p.y) - center(1), static_cast<Scalar>(p.z) - center(2));
                Scalar d_plane = std::abs(v.dot(n));
                Eigen::Matrix<Scalar, 3, 1> proj = v - v.dot(n) * n;
                Scalar d_circle = std::abs(proj.norm() - radius);
                total += d_plane + d_circle;
            }
            return total / static_cast<Scalar>(points.size());
        }
    };

} // namespace robust_cga

#endif // ROBUST_FIT3DCIRCLE_HPP
