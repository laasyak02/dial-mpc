#pragma once

#include "unitree-go2-env.h"
#include <omp.h>

#include <Eigen/Dense>
#include <unsupported/Eigen/Splines>

#include <random>
#include <string>
#include <vector>
#include <iostream>
#include <tuple>
#include <algorithm>
#include <stdexcept>

namespace dial
{
  using VectorXd = Eigen::VectorXd;
  using MatrixXd = Eigen::MatrixXd;

  /*
   * Corrected, detailed translation of the Python DIAL-MPC code into C++/Eigen.
   *
   * Key points addressed:
   *   1) rollout_us returns the *entire time-sequence* of rewards,
   *      exactly as the Python code does.
   *   2) reverse_once computes logp0 = (mean(rews_sample) - rew_Ybar_i) / std(rews_sample) / temp_sample,
   *      with standard deviation across time steps for each sample.
   *   3) We incorporate the "sigma_control * (traj_diffuse_factor^i)" approach
   *      during each diffusion iteration, matching the Python's 'scan'.
   *   4) We demonstrate an initial "reverse" call before the main rollout,
   *      which the Python code sometimes uses (commented out in the script).
   *   5) The environment dimension and controls are fully generic.
   *
   * This code should yield *identical math* to the Python version.
   * Only environment details are stubbed out (replace with your real environment).
   */
  
  struct JaxRNG {
    std::mt19937_64 rng;

    // Constructor initializes with a seed
    explicit JaxRNG(uint64_t seed) : rng(seed) {}

    // Split function: creates two new independent RNGs
    std::tuple<JaxRNG, JaxRNG> split() {
        std::uniform_int_distribution<uint64_t> dist;
        uint64_t new_seed1 = dist(rng);
        uint64_t new_seed2 = dist(rng);
        return {JaxRNG(new_seed1), JaxRNG(new_seed2)};
    }
  };

  //////////////////////////////////////////////////////////////
  // Configuration structure, mirroring Python's DialConfig
  //////////////////////////////////////////////////////////////
  struct DialConfig
  {
    int seed = 0;
    int Hsample = 16; // horizon sample
    int Hnode = 4;    // number of node points
    // int Nsample = 1024;                    // number of samples at each diffusion iteration
    int Ndiffuse = 2;                    // how many times we run "reverse_once" each planning step
    int Ndiffuse_init = 10;              // used at the first iteration
    double temp_sample = 0.05;           // temperature
    int n_steps = 400;                   // number of rollout steps
    double ctrl_dt = 0.0025;             // control dt
    double horizon_diffuse_factor = 0.9; // multiplies sigma_control
    double traj_diffuse_factor = 0.5;    // factor^i in each "reverse_once" iteration
  };

  // reverse_once: replicate the Python method exactly
  struct ReverseInfo
  {
    // We store arrays that replicate "info" in Python
    VectorXd rews;            // shape (Nsample+1,) of average rewards for each sample
    MatrixXd qbar;            // placeholders
    MatrixXd qdbar;           // placeholders
    MatrixXd xbar;            // placeholders
    VectorXd new_noise_scale; // new sigma
  };

  /**
   * @brief Compute the piecewise cubic Hermite interpolation (natural spline)
   *        of the given data at specified query times.
   *
   * @param[in]  states      An N x M matrix, where row i is the state at time knotTimes(i).
   * @param[in]  knotTimes   A length-N vector of strictly increasing knot times.
   * @param[in]  queryTimes  A length-Q vector of times at which to interpolate.
   * @return     A Q x M matrix of interpolated values.  Row q corresponds
   *             to queryTimes(q), and column m corresponds to dimension m.
   *
   * The algorithm:
   *  1. For each of the M columns, solve for the "natural spline" second derivatives via a
   *     simple tridiagonal O(N) pass (the classical cubic-spline approach).
   *  2. From these second derivatives, recover the knot *first derivatives*,
   *     which fully determine the Hermite form on each interval.
   *  3. For each query time, identify the appropriate interval [t_i, t_{i+1}] and
   *     evaluate the cubic Hermite polynomial using the standard Hermite basis.
   */
  MatrixXd piecewiseCubicHermiteInterpolate1(
      const MatrixXd &states,
      const VectorXd &knotTimes,
      const VectorXd &queryTimes)
  {
    using namespace Eigen;

    // Basic checks
    const int N = static_cast<int>(knotTimes.size()); // number of knot points
    const int M = static_cast<int>(states.cols());    // dimension of the states
    if (states.rows() != N)
    {
      throw std::runtime_error("states.rows() must match knotTimes.size()");
    }
    if (N < 2)
    {
      throw std::runtime_error("Need at least 2 knot points for cubic spline");
    }

    const int Q = static_cast<int>(queryTimes.size()); // number of query points
    MatrixXd result(Q, M);
    if (Q == 0)
    {
      return result; // empty
    }

    std::vector<double> h(N - 1);
    for (int i = 0; i < N - 1; ++i)
    {
      double dt = knotTimes(i + 1) - knotTimes(i);
      if (dt <= 0.0)
      {
        throw std::runtime_error("knotTimes must be strictly increasing");
      }
      h[i] = dt;
    }

    MatrixXd secondDerivs(N, M);
    secondDerivs.setZero();

    // We'll do each dimension's second-derivative solution in turn:
    for (int mIdx = 0; mIdx < M; ++mIdx)
    {
      // We'll build 'alpha', 'l', 'mu', 'z' (classic notations) for dimension mIdx
      VectorXd alpha = VectorXd::Zero(N);

      // Compute alpha for i=1..N-2
      // alpha_i = 3 * [(y_{i+1}-y_i)/h_i - (y_i - y_{i-1})/h_{i-1}]
      for (int i = 1; i < N - 1; ++i)
      {
        double y_im1 = states(i - 1, mIdx);
        double y_i = states(i, mIdx);
        double y_ip1 = states(i + 1, mIdx);
        alpha(i) = 3.0 * ((y_ip1 - y_i) / h[i] - (y_i - y_im1) / h[i - 1]);
      }
      // Natural boundary => alpha(0)=0, alpha(N-1)=0

      // l, mu, z for the forward pass
      VectorXd l = VectorXd::Zero(N);
      VectorXd mu = VectorXd::Zero(N);
      VectorXd z = VectorXd::Zero(N);

      l(0) = 1.0; // natural boundary
      mu(0) = 0.0;
      z(0) = 0.0;
      for (int i = 1; i < N - 1; ++i)
      {
        l(i) = 2.0 * (knotTimes(i + 1) - knotTimes(i - 1)) - h[i - 1] * mu(i - 1);
        mu(i) = h[i] / l(i);
        z(i) = (alpha(i) - h[i - 1] * z(i - 1)) / l(i);
      }
      // boundary
      l(N - 1) = 1.0;
      z(N - 1) = 0.0;

      secondDerivs(N - 1, mIdx) = 0.0;
      for (int i = N - 2; i >= 0; --i)
      {
        secondDerivs(i, mIdx) = z(i) - mu(i) * secondDerivs(i + 1, mIdx);
      }
    }

    MatrixXd firstDerivs(N, M);
    for (int mIdx = 0; mIdx < M; ++mIdx)
    {
      for (int i = 0; i < N - 1; ++i)
      {
        double y_i = states(i, mIdx);
        double y_ip1 = states(i + 1, mIdx);
        double M_i = secondDerivs(i, mIdx);
        double M_ip1 = secondDerivs(i + 1, mIdx);
        double Hi = h[i];

        double Bi = (y_ip1 - y_i) / Hi - (Hi / 6.0) * (2.0 * M_i + M_ip1);
        // That is the slope that the spline takes *leaving* point i
        firstDerivs(i, mIdx) = Bi;
      }
      {
        int i = N - 1;
        double y_im1 = states(i - 1, mIdx);
        double y_i = states(i, mIdx);
        double M_im1 = secondDerivs(i - 1, mIdx);
        double M_i = secondDerivs(i, mIdx);
        double Hi = h[N - 2];
        double BiLast = (y_i - y_im1) / Hi - (Hi / 6.0) * (2.0 * M_im1 + M_i);
        firstDerivs(i, mIdx) = BiLast;
      }
    }

    int intervalIndex = 0; // we will move forward through the knot intervals
    for (int q = 0; q < Q; ++q)
    {
      double tq = queryTimes(q);

      // Advance intervalIndex as needed so that:
      //   knotTimes(intervalIndex) <= tq < knotTimes(intervalIndex+1)
      while (intervalIndex < N - 2 && tq > knotTimes(intervalIndex + 1))
      {
        intervalIndex++;
      }

      // Clamp or assume in-range
      if (intervalIndex >= N - 1)
      {
        intervalIndex = N - 2; // handle boundary
      }

      double t0 = knotTimes(intervalIndex);
      double t1 = knotTimes(intervalIndex + 1);
      double hInt = t1 - t0;
      double u = (tq - t0) / hInt; // in [0,1]

      // Precompute Hermite basis polynomials
      double u2 = u * u;
      double u3 = u2 * u;
      double H00 = 2.0 * u3 - 3.0 * u2 + 1.0;
      double H10 = u3 - 2.0 * u2 + u;
      double H01 = -2.0 * u3 + 3.0 * u2;
      double H11 = u3 - u2;

      for (int mIdx = 0; mIdx < M; ++mIdx)
      {
        double p_i = states(intervalIndex, mIdx);
        double p_ip1 = states(intervalIndex + 1, mIdx);
        double m_i = firstDerivs(intervalIndex, mIdx);
        double m_ip1 = firstDerivs(intervalIndex + 1, mIdx);

        // S(t) = p_i*H00 + (h*m_i)*H10 + p_{i+1}*H01 + (h*m_{i+1})*H11
        double val = p_i * H00 + (hInt * m_i) * H10 + p_ip1 * H01 + (hInt * m_ip1) * H11;

        result(q, mIdx) = val;
      }
    }

    return result;
  }
  /*
  class InterpolatedUnivariateSpline {
    private:
        int k;                      // Spline order
        Eigen::VectorXd x;          // x coordinates
        Eigen::VectorXd y;          // y coordinates
        Eigen::VectorXd coefficients; // Computed coefficients
        std::string endpoints;      // Endpoint condition
    
    public:
        InterpolatedUnivariateSpline(const Eigen::VectorXd& x_input, 
                                     const Eigen::VectorXd& y_input,
                                     int k_input = 3,
                                     const std::string& endpoints_input = "not-a-knot") 
            : k(k_input), x(x_input), y(y_input), endpoints(endpoints_input) {
            
            // Verify inputs
            if (k != 1 && k != 2 && k != 3) {
                throw std::runtime_error("Order k must be in {1, 2, 3}.");
            }
            
            if (x.size() != y.size()) {
                throw std::runtime_error("Input arrays must be the same length.");
            }
            
            int n_data = x.size();
            
            // Difference vectors
            Eigen::VectorXd h = Eigen::VectorXd::Zero(n_data - 1);
            Eigen::VectorXd p = Eigen::VectorXd::Zero(n_data - 1);
            
            for (int i = 0; i < n_data - 1; i++) {
                h(i) = x(i + 1) - x(i);
                p(i) = y(i + 1) - y(i);
            }
            
            // Build the linear system of equations depending on k
            if (k == 1) {
                if (n_data <= 1) {
                    throw std::runtime_error("Not enough input points for linear spline.");
                }
                
                // For linear splines, coefficients are just the slopes
                coefficients = Eigen::VectorXd::Zero(n_data - 1);
                for (int i = 0; i < n_data - 1; i++) {
                    coefficients(i) = p(i) / h(i);
                }
            }
            else if (k == 2) {
                if (n_data <= 2) {
                    throw std::runtime_error("Not enough input points for quadratic spline.");
                }
                
                if (endpoints != "not-a-knot") {
                    throw std::runtime_error("Only 'not-a-knot' endpoint condition is supported for quadratic splines.");
                }
                
                // Knots are in between data points
                Eigen::VectorXd knots = Eigen::VectorXd::Zero(n_data - 1);
                for (int i = 0; i < n_data - 1; i++) {
                    knots(i) = (x(i + 1) + x(i)) / 2.0;
                }
                
                // Add artificial knots at the ends
                double first_knot = x(0) - (x(1) - x(0)) / 2.0;
                double last_knot = x(n_data - 1) + (x(n_data - 1) - x(n_data - 2)) / 2.0;
                
                // Create complete knot sequence
                Eigen::VectorXd full_knots(n_data + 1);
                full_knots(0) = first_knot;
                for (int i = 0; i < n_data - 1; i++) {
                    full_knots(i + 1) = knots(i);
                }
                full_knots(n_data) = last_knot;
                
                // Compute interval lengths for these new knots
                Eigen::VectorXd h_knots = Eigen::VectorXd::Zero(n_data);
                for (int i = 0; i < n_data; i++) {
                    h_knots(i) = full_knots(i + 1) - full_knots(i);
                }
                
                // Position of data points inside the intervals
                Eigen::VectorXd dt = Eigen::VectorXd::Zero(n_data);
                for (int i = 0; i < n_data; i++) {
                    dt(i) = x(i) - full_knots(i);
                }
                
                // Build the system matrix A
                int n = full_knots.size();
                Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
                
                // Diagonal elements
                A(0, 0) = 1.0;
                A(n - 1, n - 1) = 1.0;
                
                for (int i = 1; i < n - 1; i++) {
                    A(i, i) = 2.0 * dt(i) - (dt(i) * dt(i)) / h_knots(i) - 
                              (dt(i - 1) * dt(i - 1)) / h_knots(i - 1) + h_knots(i - 1);
                }
                
                // Upper diagonal 1
                A(0, 1) = -(1.0 + h_knots(0) / h_knots(1));
                for (int i = 1; i < n - 1; i++) {
                    A(i, i + 1) = (dt(i) * dt(i)) / h_knots(i);
                }
                
                // Upper diagonal 2
                A(0, 2) = h_knots(0) / h_knots(1);
                
                // Lower diagonal 1
                for (int i = 1; i < n - 1; i++) {
                    A(i, i - 1) = h_knots(i - 1) - 2.0 * dt(i - 1) + (dt(i - 1) * dt(i - 1)) / h_knots(i - 1);
                }
                A(n - 1, n - 2) = -(1.0 + h_knots(n - 2) / h_knots(n - 3));
                
                // Lower diagonal 2
                A(n - 1, n - 3) = h_knots(n - 2) / h_knots(n - 3);
                
                // RHS vector
                Eigen::VectorXd s = Eigen::VectorXd::Zero(n);
                for (int i = 1; i < n - 1; i++) {
                    s(i) = 2.0 * p(i - 1);
                }
                
                // Solve the system
                coefficients = A.fullPivLu().solve(s);
            }
            else if (k == 3) {
                if (n_data <= 3) {
                    throw std::runtime_error("Not enough input points for cubic spline.");
                }
                
                if (endpoints != "natural" && endpoints != "not-a-knot") {
                    endpoints = "natural";
                }
                
                // Construct the tri-diagonal matrix A
                Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n_data, n_data);
                
                // Special values for first and last equations
                double A00, A01, A02, ANN, AN1, AN2;
                
                if (endpoints == "natural") {
                    A00 = 1.0;
                    A01 = 0.0;
                    A02 = 0.0;
                    ANN = 1.0;
                    AN1 = -1.0;
                    AN2 = 0.0;
                } else { // not-a-knot
                    A00 = h(1);
                    A01 = -(h(0) + h(1));
                    A02 = h(0);
                    ANN = h(n_data - 3);
                    AN1 = -(h(n_data - 3) + h(n_data - 2));
                    AN2 = h(n_data - 2);
                }
                
                // Diagonal of A
                A(0, 0) = A00;
                A(n_data - 1, n_data - 1) = ANN;
                
                for (int i = 1; i < n_data - 1; i++) {
                    A(i, i) = 2.0 * (h(i - 1) + h(i));
                }
                
                // Upper diagonal 1
                A(0, 1) = A01;
                for (int i = 1; i < n_data - 1; i++) {
                    A(i, i + 1) = h(i);
                }
                
                // Upper diagonal 2
                A(0, 2) = A02;
                
                // Lower diagonal 1
                for (int i = 1; i < n_data - 1; i++) {
                    A(i, i - 1) = h(i - 1);
                }
                A(n_data - 1, n_data - 2) = AN1;
                
                // Lower diagonal 2
                A(n_data - 1, n_data - 3) = AN2;
                
                // RHS vector s
                Eigen::VectorXd s = Eigen::VectorXd::Zero(n_data);
                
                for (int i = 1; i < n_data - 1; i++) {
                    s(i) = 3.0 * (p(i) / h(i) - p(i - 1) / h(i - 1));
                }
                
                // Solve the system for coefficients (second derivatives at the knots)
                coefficients = A.fullPivLu().solve(s);
            }
        }
        
        double operator()(double x_eval) const {
            if (k == 1) {
                auto result = computeCoeffs1(x_eval);
                double t = std::get<0>(result);
                double a = std::get<1>(result);
                double b = std::get<2>(result);
                return a + b * t;
            }
            if (k == 2) {
                auto result = computeCoeffs2(x_eval);
                double t = std::get<0>(result);
                double a = std::get<1>(result);
                double b = std::get<2>(result);
                double c = std::get<3>(result);
                return a + b * t + c * t * t;
            }
            if (k == 3) {
                auto result = computeCoeffs3(x_eval);
                double t = std::get<0>(result);
                double a = std::get<1>(result);
                double b = std::get<2>(result);
                double c = std::get<3>(result);
                double d = std::get<4>(result);
                return a + b * t + c * t * t + d * t * t * t;
            }
            return 0.0;
        }
        
        // Evaluate spline at multiple points
        Eigen::VectorXd evaluate(const Eigen::VectorXd& x_eval) const {
            Eigen::VectorXd result(x_eval.size());
            for (int i = 0; i < x_eval.size(); i++) {
                result(i) = (*this)(x_eval(i));
            }
            return result;
        }
        
        // Derivative evaluation
        double derivative(double x_eval, int n = 1) const {
            if (n < 0 || n > k) {
                throw std::runtime_error("Invalid derivative order.");
            }
            
            if (n == 0) {
                return (*this)(x_eval);
            }
            
            if (k == 1) {
                auto coef = computeCoeffs1(x_eval);
                double b = std::get<2>(coef);
                return b;
            }
            else if (k == 2) {
                auto coef = computeCoeffs2(x_eval);
                double t = std::get<0>(coef);
                double b = std::get<2>(coef);
                double c = std::get<3>(coef);
                
                if (n == 1) {
                    return b + 2.0 * c * t;
                }
                else if (n == 2) {
                    return 2.0 * c;
                }
            }
            else if (k == 3) {
                auto coef = computeCoeffs3(x_eval);
                double t = std::get<0>(coef);
                double b = std::get<2>(coef);
                double c = std::get<3>(coef);
                double d = std::get<4>(coef);
                
                if (n == 1) {
                    return b + 2.0 * c * t + 3.0 * d * t * t;
                }
                else if (n == 2) {
                    return 2.0 * c + 6.0 * d * t;
                }
                else if (n == 3) {
                    return 6.0 * d;
                }
            }
            
            return 0.0;
        }
        
        // Antiderivative calculation (similar to the Python version)
        double antiderivative(double x_eval) const {
            if (k == 1) {
                // Retrieve parameters
                Eigen::VectorXd knots = x;
                
                // Determine the interval that x lies in
                int ind = std::upper_bound(knots.data(), knots.data() + knots.size(), x_eval) - knots.data() - 1;
                ind = std::max(0, std::min(ind, static_cast<int>(knots.size()) - 2));
                double t = x_eval - knots(ind);
                
                double a = y(ind);
                double b = coefficients(ind);
                double h = knots(ind + 1) - knots(ind);
                
                // Create cumulative sum vector
                std::vector<double> cst(knots.size());
                cst[0] = 0.0;
                for (int i = 1; i < knots.size(); i++) {
                    double prev_h = knots(i) - knots(i - 1);
                    cst[i] = cst[i - 1] + y(i - 1) * prev_h + coefficients(i - 1) * prev_h * prev_h / 2.0;
                }
                
                return cst[ind] + a * t + b * t * t / 2.0;
            }
            else if (k == 2) {
                // For quadratic, we redefine the knots
                int n_data = x.size();
                
                Eigen::VectorXd mid_knots(n_data - 1);
                for (int i = 0; i < n_data - 1; i++) {
                    mid_knots(i) = (x(i + 1) + x(i)) / 2.0;
                }
                
                double first_knot = x(0) - (x(1) - x(0)) / 2.0;
                double last_knot = x(n_data - 1) + (x(n_data - 1) - x(n_data - 2)) / 2.0;
                
                Eigen::VectorXd knots(n_data + 1);
                knots(0) = first_knot;
                for (int i = 0; i < n_data - 1; i++) {
                    knots(i + 1) = mid_knots(i);
                }
                knots(n_data) = last_knot;
                
                // Determine the interval that x lies in
                int ind = std::upper_bound(knots.data(), knots.data() + knots.size(), x_eval) - knots.data() - 1;
                ind = std::max(0, std::min(ind, static_cast<int>(knots.size()) - 2));
                double t = x_eval - knots(ind);
                
                Eigen::VectorXd h = Eigen::VectorXd::Zero(knots.size() - 1);
                for (int i = 0; i < knots.size() - 1; i++) {
                    h(i) = knots(i + 1) - knots(i);
                }
                
                Eigen::VectorXd dt = Eigen::VectorXd::Zero(n_data);
                for (int i = 0; i < n_data; i++) {
                    dt(i) = x(i) - knots(i);
                }
                
                Eigen::VectorXd b = coefficients.head(n_data);
                Eigen::VectorXd b1 = coefficients.tail(n_data);
                
                Eigen::VectorXd a = Eigen::VectorXd::Zero(n_data);
                Eigen::VectorXd c = Eigen::VectorXd::Zero(n_data);
                
                for (int i = 0; i < n_data; i++) {
                    a(i) = y(i) - b(i) * dt(i) - (b1(i) - b(i)) * dt(i) * dt(i) / (2.0 * h(i));
                    c(i) = (b1(i) - b(i)) / (2.0 * h(i));
                }
                
                // Create cumulative sum vector
                std::vector<double> cst(knots.size());
                cst[0] = 0.0;
                for (int i = 1; i < knots.size(); i++) {
                    double prev_h = h(i - 1);
                    cst[i] = cst[i - 1] + a(i - 1) * prev_h + b(i - 1) * prev_h * prev_h / 2.0 + 
                             c(i - 1) * prev_h * prev_h * prev_h / 3.0;
                }
                
                return cst[ind] + a(ind) * t + b(ind) * t * t / 2.0 + c(ind) * t * t * t / 3.0;
            }
            else if (k == 3) {
                Eigen::VectorXd knots = x;
                
                // Determine the interval that x lies in
                int ind = std::upper_bound(knots.data(), knots.data() + knots.size(), x_eval) - knots.data() - 1;
                ind = std::max(0, std::min(ind, static_cast<int>(knots.size()) - 2));
                double t = x_eval - knots(ind);
                
                Eigen::VectorXd h = Eigen::VectorXd::Zero(knots.size() - 1);
                for (int i = 0; i < knots.size() - 1; i++) {
                    h(i) = knots(i + 1) - knots(i);
                }
                
                double c = coefficients(ind);
                double c1 = coefficients(ind + 1);
                double a = y(ind);
                double a1 = y(ind + 1);
                double b = (a1 - a) / h(ind) - (2.0 * c + c1) * h(ind) / 3.0;
                double d = (c1 - c) / (3.0 * h(ind));
                
                // Create cumulative sum vector
                std::vector<double> cst(knots.size());
                cst[0] = 0.0;
                for (int i = 1; i < knots.size(); i++) {
                    int j = i - 1;
                    double prev_h = h(j);
                    double prev_c = coefficients(j);
                    double prev_c1 = coefficients(j + 1);
                    double prev_a = y(j);
                    double prev_a1 = y(j + 1);
                    double prev_b = (prev_a1 - prev_a) / prev_h - (2.0 * prev_c + prev_c1) * prev_h / 3.0;
                    double prev_d = (prev_c1 - prev_c) / (3.0 * prev_h);
                    
                    cst[i] = cst[i - 1] + prev_a * prev_h + prev_b * prev_h * prev_h / 2.0 + 
                             prev_c * prev_h * prev_h * prev_h / 3.0 + prev_d * prev_h * prev_h * prev_h * prev_h / 4.0;
                }
                
                return cst[ind] + a * t + b * t * t / 2.0 + c * t * t * t / 3.0 + d * t * t * t * t / 4.0;
            }
            
            return 0.0;
        }
        
        // Definite integral calculation
        double integral(double a, double b) const {
            // Swap integration bounds if needed
            double sign = 1.0;
            if (b < a) {
                std::swap(a, b);
                sign = -1.0;
            }
            
            return sign * (antiderivative(b) - antiderivative(a));
        }
        
    private:
        // Helper functions to compute coefficients
        std::tuple<double, double, double> computeCoeffs1(double x_eval) const {
            // Determine the interval that x lies in
            Eigen::VectorXd knots = x;
            int ind = std::upper_bound(knots.data(), knots.data() + knots.size(), x_eval) - knots.data() - 1;
            ind = std::max(0, std::min(ind, static_cast<int>(knots.size()) - 2));
            double t = x_eval - knots(ind);
            double a = y(ind);
            double b = coefficients(ind);
            
            return std::make_tuple(t, a, b);
        }
        
        std::tuple<double, double, double, double> computeCoeffs2(double x_eval) const {
            // For quadratic, we redefine the knots
            int n_data = x.size();
            
            Eigen::VectorXd mid_knots(n_data - 1);
            for (int i = 0; i < n_data - 1; i++) {
                mid_knots(i) = (x(i + 1) + x(i)) / 2.0;
            }
            
            double first_knot = x(0) - (x(1) - x(0)) / 2.0;
            double last_knot = x(n_data - 1) + (x(n_data - 1) - x(n_data - 2)) / 2.0;
            
            Eigen::VectorXd knots(n_data + 1);
            knots(0) = first_knot;
            for (int i = 0; i < n_data - 1; i++) {
                knots(i + 1) = mid_knots(i);
            }
            knots(n_data) = last_knot;
            
            // Determine the interval that x lies in
            int ind = std::upper_bound(knots.data(), knots.data() + knots.size(), x_eval) - knots.data() - 1;
            ind = std::max(0, std::min(ind, static_cast<int>(knots.size()) - 2));
            double t = x_eval - knots(ind);
            double h = knots(ind + 1) - knots(ind);
            
            // Compute dt for the data point
            double dt = x(ind) - knots(ind);
            
            double b = coefficients(ind);
            double b1 = coefficients(ind + 1);
            double a = y(ind) - b * dt - (b1 - b) * dt * dt / (2.0 * h);
            double c = (b1 - b) / (2.0 * h);
            
            return std::make_tuple(t, a, b, c);
        }
        
        std::tuple<double, double, double, double, double> computeCoeffs3(double x_eval) const {
            // Determine the interval that x lies in
            Eigen::VectorXd knots = x;
            int ind = std::upper_bound(knots.data(), knots.data() + knots.size(), x_eval) - knots.data() - 1;
            ind = std::max(0, std::min(ind, static_cast<int>(knots.size()) - 2));
            double t = x_eval - knots(ind);
            double h = knots(ind + 1) - knots(ind);
            
            double c = coefficients(ind);
            double c1 = coefficients(ind + 1);
            double a = y(ind);
            double a1 = y(ind + 1);
            double b = (a1 - a) / h - (2.0 * c + c1) * h / 3.0;
            double d = (c1 - c) / (3.0 * h);
            
            return std::make_tuple(t, a, b, c, d);
        }
    };*/
    class InterpolatedUnivariateSpline {
      private:
          int k;                    // Spline order
          Eigen::VectorXd x;        // Knot points
          Eigen::VectorXd y;        // Values at knot points
          Eigen::VectorXd coefficients; // Spline coefficients
          std::string endpoints;    // Endpoint condition for cubic splines
      
          // Helper functions for coefficient computation
          std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> computeCoeffs1(const Eigen::VectorXd& xs) const;
          std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> computeCoeffs2(const Eigen::VectorXd& xs) const;
          std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> computeCoeffs3(const Eigen::VectorXd& xs) const;
      
      public:
          InterpolatedUnivariateSpline(const Eigen::VectorXd& x, const Eigen::VectorXd& y, 
                                      int k = 3, const std::string& endpoints = "not-a-knot");
      
          // Evaluate the spline at point(s) x
          Eigen::VectorXd operator()(const Eigen::VectorXd& xs) const;
          double operator()(double x) const;
      
          // Compute the nth derivative at point(s) x
          Eigen::VectorXd derivative(const Eigen::VectorXd& xs, int n = 1) const;
          double derivative(double x, int n = 1) const;
      
          // Compute the antiderivative at point(s) x
          Eigen::VectorXd antiderivative(const Eigen::VectorXd& xs) const;
          
          // Compute the definite integral over [a, b]
          double integral(double a, double b) const;
      };
      
      InterpolatedUnivariateSpline::InterpolatedUnivariateSpline(
          const Eigen::VectorXd& x, const Eigen::VectorXd& y, int k, const std::string& endpoints)
          : k(k), x(x), y(y), endpoints(endpoints) {
          
          // Verify inputs
          if (k < 1 || k > 3) {
              throw std::runtime_error("Order k must be in {1, 2, 3}.");
          }
          if (x.size() != y.size()) {
              throw std::runtime_error("Input arrays must be the same length.");
          }
          int n_data = x.size();
      
          // Difference vectors
          Eigen::VectorXd h = Eigen::VectorXd::Zero(n_data - 1);  // x[i+1] - x[i]
          Eigen::VectorXd p = Eigen::VectorXd::Zero(n_data - 1);  // y[i+1] - y[i]
          
          for (int i = 0; i < n_data - 1; i++) {
              h(i) = x(i + 1) - x(i);
              p(i) = y(i + 1) - y(i);
          }
      
          // Build the linear system of equations depending on k
          if (k == 1) {
              if (n_data <= 1) {
                  throw std::runtime_error("Not enough input points for linear spline.");
              }
              // No matrix necessary for k=1
              coefficients = p.array() / h.array();
          }
          else if (k == 2) {
              if (n_data <= 2) {
                  throw std::runtime_error("Not enough input points for quadratic spline.");
              }
              if (endpoints != "not-a-knot") {
                  std::cerr << "Warning: endpoints not recognized for k=2. Using not-a-knot." << std::endl;
              }
      
              // The knots are actually in between data points
              Eigen::VectorXd knots(n_data - 1);
              for (int i = 0; i < n_data - 1; i++) {
                  knots(i) = (x(i + 1) + x(i)) / 2.0;
              }
              
              // We add 2 artificial knots before and after
              Eigen::VectorXd extended_knots(n_data + 1);
              extended_knots(0) = x(0) - (x(1) - x(0)) / 2.0;
              extended_knots.segment(1, n_data - 1) = knots;
              extended_knots(n_data) = x(n_data - 1) + (x(n_data - 1) - x(n_data - 2)) / 2.0;
              
              int n = extended_knots.size();
              
              // Compute interval lengths for these new knots
              Eigen::VectorXd h_knots(n - 1);
              for (int i = 0; i < n - 1; i++) {
                  h_knots(i) = extended_knots(i + 1) - extended_knots(i);
              }
              
              // Position of data point inside the interval
              Eigen::VectorXd dt(n_data);
              for (int i = 0; i < n_data; i++) {
                  dt(i) = x(i) - extended_knots(i);
              }
              
              // Build the system matrix
              Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
              
              // Main diagonal
              A(0, 0) = 1.0;
              A(n - 1, n - 1) = 1.0;
              for (int i = 1; i < n - 1; i++) {
                  A(i, i) = 2 * dt(i) - std::pow(dt(i), 2) / h_knots(i) - std::pow(dt(i - 1), 2) / h_knots(i - 1) + h_knots(i - 1);
              }
              
              // Upper diagonals
              A(0, 1) = -(1 + h_knots(0) / h_knots(1));
              for (int i = 1; i < n - 1; i++) {
                  A(i, i + 1) = std::pow(dt(i), 2) / h_knots(i);
              }
              A(0, 2) = h_knots(0) / h_knots(1);
              
              // Lower diagonals
              for (int i = 1; i < n - 1; i++) {
                  A(i, i - 1) = h_knots(i - 1) - 2 * dt(i - 1) + std::pow(dt(i - 1), 2) / h_knots(i - 1);
              }
              A(n - 1, n - 2) = -(1 + h_knots(n - 2) / h_knots(n - 3));
              A(n - 1, n - 3) = h_knots(n - 2) / h_knots(n - 3);
              
              // RHS vector
              Eigen::VectorXd s = Eigen::VectorXd::Zero(n);
              s.segment(1, n - 2) = 2 * p;
              
              // Solve the system
              coefficients = A.colPivHouseholderQr().solve(s);
          }
          else if (k == 3) {
              if (n_data <= 3) {
                  throw std::runtime_error("Not enough input points for cubic spline.");
              }
              std::string local_endpoints = endpoints;
              if (endpoints != "natural" && endpoints != "not-a-knot") {
                  std::cerr << "Warning: endpoints not recognized for k=3. Using natural." << std::endl;
                  local_endpoints = "natural";
              }
              
              // Build the tridiagonal system
              Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n_data, n_data);
              
              // Special values for the first and last equations
              double A00 = local_endpoints == "natural" ? 1.0 : h(1);
              double A01 = local_endpoints == "natural" ? 0.0 : -(h(0) + h(1));
              double A02 = local_endpoints == "natural" ? 0.0 : h(0);
              double ANN = local_endpoints == "natural" ? 1.0 : h(n_data - 3);
              double AN1 = local_endpoints == "natural" ? -1.0 : -(h(n_data - 3) + h(n_data - 2));
              double AN2 = local_endpoints == "natural" ? 0.0 : h(n_data - 2);
              
              // Main diagonal
              A(0, 0) = A00;
              A(n_data - 1, n_data - 1) = ANN;
              for (int i = 1; i < n_data - 1; i++) {
                  A(i, i) = 2 * (h(i - 1) + h(i));
              }
              
              // Upper diagonals
              A(0, 1) = A01;
              A(0, 2) = A02;
              for (int i = 1; i < n_data - 1; i++) {
                  A(i, i + 1) = h(i);
              }
              
              // Lower diagonals
              for (int i = 1; i < n_data - 1; i++) {
                  A(i, i - 1) = h(i - 1);
              }
              A(n_data - 1, n_data - 2) = AN1;
              A(n_data - 1, n_data - 3) = AN2;
              
              // RHS vector
              Eigen::VectorXd s = Eigen::VectorXd::Zero(n_data);
              for (int i = 1; i < n_data - 1; i++) {
                  s(i) = 3 * (p(i) / h(i) - p(i - 1) / h(i - 1));
              }
              
              // Solve the system
              coefficients = A.colPivHouseholderQr().solve(s);
          }
      }
      
      std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> 
      InterpolatedUnivariateSpline::computeCoeffs1(const Eigen::VectorXd& xs) const {
          int n_data = x.size();
          int n_xs = xs.size();
          
          // Determine the interval that x lies in
          Eigen::VectorXi ind(n_xs);
          for (int i = 0; i < n_xs; i++) {
              double xi = xs(i);
              
              // Binary search to find the right interval
              int low = 0, high = n_data - 1;
              while (low < high - 1) {
                  int mid = (low + high) / 2;
                  if (x(mid) <= xi) low = mid;
                  else high = mid;
              }
              ind(i) = low;
          }
          
          // Clip to valid range
          for (int i = 0; i < n_xs; i++) {
              ind(i) = std::max(0, std::min(ind(i), n_data - 2));
          }
          
          // Calculate t
          Eigen::VectorXd t(n_xs);
          for (int i = 0; i < n_xs; i++) {
              t(i) = xs(i) - x(ind(i));
          }
          
          // Get a and b coefficients
          Eigen::VectorXd a(n_xs), b(n_xs);
          for (int i = 0; i < n_xs; i++) {
              a(i) = y(ind(i));
              b(i) = coefficients(ind(i));
          }
          
          return {t, a, b};
      }
      
      std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> 
      InterpolatedUnivariateSpline::computeCoeffs2(const Eigen::VectorXd& xs) const {
          int n_data = x.size();
          int n_xs = xs.size();
          
          // Create knots (midpoints between data points)
          Eigen::VectorXd knots(n_data - 1);
          for (int i = 0; i < n_data - 1; i++) {
              knots(i) = (x(i + 1) + x(i)) / 2.0;
          }
          
          // Add artificial knots
          Eigen::VectorXd extended_knots(n_data + 1);
          extended_knots(0) = x(0) - (x(1) - x(0)) / 2.0;
          extended_knots.segment(1, n_data - 1) = knots;
          extended_knots(n_data) = x(n_data - 1) + (x(n_data - 1) - x(n_data - 2)) / 2.0;
          
          // Determine the interval that xs lies in
          Eigen::VectorXi ind(n_xs);
          for (int i = 0; i < n_xs; i++) {
              double xi = xs(i);
              
              // Binary search to find the right interval
              int low = 0, high = extended_knots.size() - 1;
              while (low < high - 1) {
                  int mid = (low + high) / 2;
                  if (extended_knots(mid) <= xi) low = mid;
                  else high = mid;
              }
              ind(i) = low;
          }
          
          // Clip to valid range
          for (int i = 0; i < n_xs; i++) {
              ind(i) = std::max(0, std::min(ind(i), static_cast<int>(extended_knots.size() - 2)));
          }
          
          // Calculate t
          Eigen::VectorXd t(n_xs);
          for (int i = 0; i < n_xs; i++) {
              t(i) = xs(i) - extended_knots(ind(i));
          }
          
          // Calculate h (interval widths)
          Eigen::VectorXd h(extended_knots.size() - 1);
          for (int i = 0; i < extended_knots.size() - 1; i++) {
              h(i) = extended_knots(i + 1) - extended_knots(i);
          }
          
          // Calculate dt (position of x within intervals)
          Eigen::VectorXd dt(n_data);
          for (int i = 0; i < n_data; i++) {
              int idx = std::upper_bound(extended_knots.data(), extended_knots.data() + extended_knots.size(), x(i)) - extended_knots.data() - 1;
              idx = std::max(0, std::min(idx, static_cast<int>(extended_knots.size()) - 2));
              dt(i) = x(i) - extended_knots(idx);
          }
          
          // Get coefficients
          Eigen::VectorXd a(n_xs), b(n_xs), c(n_xs);
          for (int i = 0; i < n_xs; i++) {
              int idx = ind(i);
              // Find which data point corresponds to this interval
              int data_idx = -1;
              for (int j = 0; j < n_data; j++) {
                  if (x(j) >= extended_knots(idx) && x(j) < extended_knots(idx + 1)) {
                      data_idx = j;
                      break;
                  }
              }
              if (data_idx == -1) {
                  if (xs(i) <= x(0)) data_idx = 0;
                  else data_idx = n_data - 1;
              }
              
              // Calculate coefficients
              double b_val = coefficients(idx);
              double b1_val = idx + 1 < coefficients.size() ? coefficients(idx + 1) : b_val;
              double dt_val = dt(data_idx);
              double h_val = h(idx);
              
              b(i) = b_val;
              c(i) = (b1_val - b_val) / (2 * h_val);
              a(i) = y(data_idx) - b_val * dt_val - (b1_val - b_val) * dt_val * dt_val / (2 * h_val);
          }
          
          return {t, a, b, c};
      }
      
      std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> 
      InterpolatedUnivariateSpline::computeCoeffs3(const Eigen::VectorXd& xs) const {
          int n_data = x.size();
          int n_xs = xs.size();
          
          // Determine the interval that x lies in
          Eigen::VectorXi ind(n_xs);
          for (int i = 0; i < n_xs; i++) {
              double xi = xs(i);
              
              // Binary search to find the right interval
              int low = 0, high = n_data - 1;
              while (low < high - 1) {
                  int mid = (low + high) / 2;
                  if (x(mid) <= xi) low = mid;
                  else high = mid;
              }
              ind(i) = low;
          }
          
          // Clip to valid range
          for (int i = 0; i < n_xs; i++) {
              ind(i) = std::max(0, std::min(ind(i), n_data - 2));
          }
          
          // Calculate t
          Eigen::VectorXd t(n_xs);
          for (int i = 0; i < n_xs; i++) {
              t(i) = xs(i) - x(ind(i));
          }
          
          // Calculate h (interval widths)
          Eigen::VectorXd h(n_data - 1);
          for (int i = 0; i < n_data - 1; i++) {
              h(i) = x(i + 1) - x(i);
          }
          
          // Get coefficients
          Eigen::VectorXd a(n_xs), b(n_xs), c(n_xs), d(n_xs);
          for (int i = 0; i < n_xs; i++) {
              int idx = ind(i);
              double c_val = coefficients(idx);
              double c1_val = coefficients(idx + 1);
              double a_val = y(idx);
              double a1_val = y(idx + 1);
              double h_val = h(idx);
              
              a(i) = a_val;
              b(i) = (a1_val - a_val) / h_val - (2 * c_val + c1_val) * h_val / 3.0;
              c(i) = c_val;
              d(i) = (c1_val - c_val) / (3 * h_val);
          }
          
          return {t, a, b, c, d};
      }
      
      Eigen::VectorXd InterpolatedUnivariateSpline::operator()(const Eigen::VectorXd& xs) const {
          int n_xs = xs.size();
          Eigen::VectorXd result(n_xs);
          
          if (k == 1) {
              auto [t, a, b] = computeCoeffs1(xs);
              for (int i = 0; i < n_xs; i++) {
                  result(i) = a(i) + b(i) * t(i);
              }
          }
          else if (k == 2) {
              auto [t, a, b, c] = computeCoeffs2(xs);
              for (int i = 0; i < n_xs; i++) {
                  result(i) = a(i) + b(i) * t(i) + c(i) * t(i) * t(i);
              }
          }
          else if (k == 3) {
              auto [t, a, b, c, d] = computeCoeffs3(xs);
              for (int i = 0; i < n_xs; i++) {
                  result(i) = a(i) + b(i) * t(i) + c(i) * t(i) * t(i) + d(i) * t(i) * t(i) * t(i);
              }
          }
          
          return result;
      }
      
      double InterpolatedUnivariateSpline::operator()(double x) const {
          Eigen::VectorXd xs(1);
          xs(0) = x;
          return operator()(xs)(0);
      }
      
      Eigen::VectorXd InterpolatedUnivariateSpline::derivative(const Eigen::VectorXd& xs, int n) const {
          if (n < 0 || n > k) {
              throw std::runtime_error("Invalid derivative order n.");
          }
          
          int n_xs = xs.size();
          Eigen::VectorXd result(n_xs);
          
          if (n == 0) {
              return operator()(xs);
          }
          
          if (k == 1) {
              auto [t, a, b] = computeCoeffs1(xs);
              // Only first derivative is non-zero
              if (n == 1) {
                  result = b;
              } 
              else {
                  result.setZero();
              }
          }
          else if (k == 2) {
              auto [t, a, b, c] = computeCoeffs2(xs);
              if (n == 1) {
                  for (int i = 0; i < n_xs; i++) {
                      result(i) = b(i) + 2 * c(i) * t(i);
                  }
              }
              else if (n == 2) {
                  result = 2 * c;
              }
              else {
                  result.setZero();
              }
          }
          else if (k == 3) {
              auto [t, a, b, c, d] = computeCoeffs3(xs);
              if (n == 1) {
                  for (int i = 0; i < n_xs; i++) {
                      result(i) = b(i) + 2 * c(i) * t(i) + 3 * d(i) * t(i) * t(i);
                  }
              }
              else if (n == 2) {
                  for (int i = 0; i < n_xs; i++) {
                      result(i) = 2 * c(i) + 6 * d(i) * t(i);
                  }
              }
              else if (n == 3) {
                  result = 6 * d;
              }
              else {
                  result.setZero();
              }
          }
          
          return result;
      }
      
      double InterpolatedUnivariateSpline::derivative(double x, int n) const {
          Eigen::VectorXd xs(1);
          xs(0) = x;
          return derivative(xs, n)(0);
      }
      
      Eigen::VectorXd InterpolatedUnivariateSpline::antiderivative(const Eigen::VectorXd& xs) const {
          int n_xs = xs.size();
          Eigen::VectorXd result(n_xs);
          
          if (k == 1) {
              auto [t, a, b] = computeCoeffs1(xs);
              
              // Calculate h (interval widths)
              int n_data = x.size();
              Eigen::VectorXd h(n_data - 1);
              for (int i = 0; i < n_data - 1; i++) {
                  h(i) = x(i + 1) - x(i);
              }
              
              // Calculate constants for each interval
              Eigen::VectorXd cst(n_data);
              cst(0) = 0.0;
              for (int i = 1; i < n_data; i++) {
                  int idx = i - 1;
                  cst(i) = cst(idx) + a(0) * h(idx) + coefficients(idx) * h(idx) * h(idx) / 2.0;
              }
              
              // Calculate result
              for (int i = 0; i < n_xs; i++) {
                  int idx = std::upper_bound(x.data(), x.data() + n_data, xs(i)) - x.data() - 1;
                  idx = std::max(0, std::min(idx, n_data - 2));
                  
                  double a_val = y(idx);
                  double b_val = coefficients(idx);
                  double t_val = t(i);
                  
                  result(i) = cst(idx) + a_val * t_val + b_val * t_val * t_val / 2.0;
              }
          }
          else if (k == 2) {
              auto [t, a, b, c] = computeCoeffs2(xs);
              
              // Calculate knots
              int n_data = x.size();
              Eigen::VectorXd knots(n_data - 1);
              for (int i = 0; i < n_data - 1; i++) {
                  knots(i) = (x(i + 1) + x(i)) / 2.0;
              }
              
              // Add artificial knots
              Eigen::VectorXd extended_knots(n_data + 1);
              extended_knots(0) = x(0) - (x(1) - x(0)) / 2.0;
              extended_knots.segment(1, n_data - 1) = knots;
              extended_knots(n_data) = x(n_data - 1) + (x(n_data - 1) - x(n_data - 2)) / 2.0;
              
              // Calculate h (interval widths)
              Eigen::VectorXd h(extended_knots.size() - 1);
              for (int i = 0; i < extended_knots.size() - 1; i++) {
                  h(i) = extended_knots(i + 1) - extended_knots(i);
              }
              
              // Calculate constants for each interval
              Eigen::VectorXd cst(extended_knots.size());
              cst(0) = 0.0;
              
              for (int i = 1; i < extended_knots.size(); i++) {
                  int idx = i - 1;
                  double a_val = a(idx);
                  double b_val = b(idx);
                  double c_val = c(idx);
                  double h_val = h(idx);
                  
                  cst(i) = cst(idx) + a_val * h_val + b_val * h_val * h_val / 2.0 + c_val * h_val * h_val * h_val / 3.0;
              }
              
              // Calculate result
              for (int i = 0; i < n_xs; i++) {
                  int idx = std::upper_bound(extended_knots.data(), 
                                             extended_knots.data() + extended_knots.size(), 
                                             xs(i)) - extended_knots.data() - 1;
                  idx = std::max(0, std::min(idx, static_cast<int>(extended_knots.size()) - 2));
                  
                  double t_val = t(i);
                  double a_val = a(i);
                  double b_val = b(i);
                  double c_val = c(i);
                  
                  result(i) = cst(idx) + a_val * t_val + b_val * t_val * t_val / 2.0 + c_val * t_val * t_val * t_val / 3.0;
              }
          }
          else if (k == 3) {
              auto [t, a, b, c, d] = computeCoeffs3(xs);
              
              // Calculate h (interval widths)
              int n_data = x.size();
              Eigen::VectorXd h(n_data - 1);
              for (int i = 0; i < n_data - 1; i++) {
                  h(i) = x(i + 1) - x(i);
              }
              
              // Calculate constants for each interval
              Eigen::VectorXd cst(n_data);
              cst(0) = 0.0;
              
              for (int i = 1; i < n_data; i++) {
                  int idx = i - 1;
                  double a_val = y(idx);
                  double c_val = coefficients(idx);
                  double c1_val = coefficients(i);
                  double h_val = h(idx);
                  
                  double b_val = (y(i) - a_val) / h_val - (2 * c_val + c1_val) * h_val / 3.0;
                  double d_val = (c1_val - c_val) / (3 * h_val);
                  
                  cst(i) = cst(idx) + a_val * h_val + b_val * h_val * h_val / 2.0 + 
                          c_val * h_val * h_val * h_val / 3.0 + d_val * h_val * h_val * h_val * h_val / 4.0;
              }
              
              // Calculate result
              for (int i = 0; i < n_xs; i++) {
                  int idx = std::upper_bound(x.data(), x.data() + n_data, xs(i)) - x.data() - 1;
                  idx = std::max(0, std::min(idx, n_data - 2));
                  
                  double t_val = t(i);
                  double a_val = a(i);
                  double b_val = b(i);
                  double c_val = c(i);
                  double d_val = d(i);
                  
                  result(i) = cst(idx) + a_val * t_val + b_val * t_val * t_val / 2.0 + 
                             c_val * t_val * t_val * t_val / 3.0 + d_val * t_val * t_val * t_val * t_val / 4.0;
              }
          }
          
          return result;
      }
      
      double InterpolatedUnivariateSpline::integral(double a, double b) const {
          // Swap integration bounds if needed
          double sign = 1.0;
          if (b < a) {
              std::swap(a, b);
              sign = -1.0;
          }
          
          Eigen::VectorXd xs(2);
          xs << a, b;
          
          Eigen::VectorXd antideriv = antiderivative(xs);
    
          // The function should return a single value (the definite integral),
          double result = sign * (antideriv(1) - antideriv(0));
          
          return result;
      }

    // The function to match the piecewiseCubicHermiteInterpolate signature but using our new implementation
    Eigen::MatrixXd piecewiseCubicHermiteInterpolate(
        const Eigen::MatrixXd &states,
        const Eigen::VectorXd &knotTimes,
        const Eigen::VectorXd &queryTimes) {
        
        int N = knotTimes.size();
        int M = states.cols();
        int Q = queryTimes.size();
        
        // Basic checks
        if (states.rows() != N) {
            throw std::runtime_error("states.rows() must match knotTimes.size()");
        }
        if (N < 2) {
            throw std::runtime_error("Need at least 2 knot points for interpolation");
        }
        
        Eigen::MatrixXd result(Q, M);
        
        // Create a spline for each column (dimension) of the states matrix
        for (int mIdx = 0; mIdx < M; mIdx++) {
            // Extract the column
            Eigen::VectorXd column_values(N);
            for (int i = 0; i < N; i++) {
                column_values(i) = states(i, mIdx);
            }
            
            // Create the spline for this dimension
            // InterpolatedUnivariateSpline spline(knotTimes, column_values, 3, "natural");
            InterpolatedUnivariateSpline spline(knotTimes, column_values, 2, "not-a-knot");
            
            // Evaluate at query times
            for (int q = 0; q < Q; q++) {
                result(q, mIdx) = spline(queryTimes(q));
            }
        }
        
        return result;
    }

  MatrixXd piecewiseLinearInterpolate(
      const MatrixXd &vals,
      const VectorXd &knotTimes,
      const VectorXd &queryTimes)
  {
    using namespace Eigen;

    const int N = static_cast<int>(knotTimes.size());
    const int M = static_cast<int>(vals.cols());
    if (vals.rows() != N)
    {
      throw std::runtime_error("vals.rows() must match knotTimes.size()");
    }
    if (N < 2)
    {
      throw std::runtime_error("Need at least 2 knot points for linear interpolation");
    }

    const int Q = static_cast<int>(queryTimes.size());
    MatrixXd result(Q, M);
    if (Q == 0)
      return result;

    int intervalIndex = 0;
    for (int q = 0; q < Q; ++q)
    {
      double tq = queryTimes(q);
      while (intervalIndex < N - 2 && tq > knotTimes(intervalIndex + 1))
      {
        intervalIndex++;
      }
      if (intervalIndex >= N - 1)
      {
        intervalIndex = N - 2;
      }

      double t0 = knotTimes(intervalIndex);
      double t1 = knotTimes(intervalIndex + 1);
      double u = (tq - t0) / (t1 - t0);

      for (int mIdx = 0; mIdx < M; ++mIdx)
      {
        double p0 = vals(intervalIndex, mIdx);
        double p1 = vals(intervalIndex + 1, mIdx);
        result(q, mIdx) = p0 + (p1 - p0) * u;
      }
    }
    return result;
  }

  inline MatrixXd node2u(const MatrixXd &nodes,
                         const VectorXd &step_nodes,
                         const VectorXd &step_us)
  {
    // nodes has shape (Hnode+1, nu)
    // return piecewiseLinearInterpolate(nodes, step_nodes, step_us);
    return piecewiseCubicHermiteInterpolate(nodes, step_nodes, step_us);
  }

  inline MatrixXd u2node(const MatrixXd &us,
                         const VectorXd &step_us,
                         const VectorXd &step_nodes)
  {
    // us has shape (Hsample+1, nu)
    // return piecewiseLinearInterpolate(us, step_us, step_nodes);
    return piecewiseCubicHermiteInterpolate(us, step_us, step_nodes);
  }

  // Computes: mu_0tm1 = sum_n weights[n] * Y0s[n]  (with Y0s[n] of shape (Hnode+1, nu))
  inline std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
  softmax_update(const Eigen::VectorXd &weights,
                 const std::vector<Eigen::MatrixXd> &Y0s,
                 const Eigen::VectorXd &sigma,
                 const Eigen::MatrixXd &mu_0t)
  {
    // Check that the number of weights matches the number of candidate matrices.
    if (weights.size() != static_cast<Eigen::Index>(Y0s.size()))
    {
      throw std::invalid_argument("Size of weights must equal the number of candidate matrices in Y0s.");
    }

    // Assume that Y0s is nonempty.
    if (Y0s.empty())
    {
      throw std::invalid_argument("Y0s must contain at least one candidate matrix.");
    }

    // Get the expected dimensions from the first candidate.
    const Eigen::Index Hnode_plus_one = Y0s[0].rows();
    const Eigen::Index nu = Y0s[0].cols();

    // Optionally, one could check that all Y0s[i] have the same dimensions.
    for (size_t i = 0; i < Y0s.size(); ++i)
    {
      if (Y0s[i].rows() != Hnode_plus_one || Y0s[i].cols() != nu)
      {
        throw std::invalid_argument("All candidate matrices in Y0s must have the same dimensions (Hnode+1 x nu).");
      }
    }

    // Create an output matrix with the same dimensions (Hnode+1, nu) and initialize it to zero.
    Eigen::MatrixXd mu_0tm1 = Eigen::MatrixXd::Zero(Hnode_plus_one, nu);

    // Compute the weighted sum: mu_0tm1 = sum_{n} weights[n] * Y0s[n]
    for (Eigen::Index n = 0; n < weights.size(); ++n)
    {
      mu_0tm1 += weights(n) * Y0s[n];
    }

    // The sigma is returned unchanged.
    return std::make_tuple(mu_0tm1, sigma);
  }

  //////////////////////////////////////////////////////////////
  // MBDPI Class
  //////////////////////////////////////////////////////////////
  template <int NUMSAMPLES_>
  class MBDPI
  {
  public:
    static const int NUMSAMPLES = NUMSAMPLES_;

    MBDPI(const DialConfig &args, go2env::UnitreeGo2Env<NUMSAMPLES + 1> &env)
        : args_(args), env_(env), nu_(env.action_size())
    {
      // 1) Precompute sigmas_ for i in [0..Ndiffuse-1]
      double sigma0 = 1e-2, sigma1 = 1.0;
      double A = sigma0;
      double B = std::log(sigma1 / sigma0) / args_.Ndiffuse;
      sigmas_ = VectorXd::Zero(args_.Ndiffuse);
      for (int i = 0; i < args_.Ndiffuse; i++)
      {
        sigmas_(i) = A * std::exp(B * i);
      }

      // 2) sigma_control_ = horizon_diffuse_factor^( [Hnode..0] ) (in Python, reversed)
      sigma_control_ = VectorXd::Zero(args_.Hnode + 1);
      for (int i = 0; i <= args_.Hnode; i++)
      {
        // reversed exponent, i.e. sigma_control[0] = horizon_diffuse_factor^Hnode
        int exponent = args_.Hnode - i;
        sigma_control_(i) = std::pow(args_.horizon_diffuse_factor, exponent);
      }

      // 3) Create step_us_, step_nodes_
      // std::cout<<args_.ctrl_dt<<std::endl;
      // std::cout<<args_.Hsample<<std::endl;
      // std::cout<<args_.Hnode<<std::endl;
      double tmax = args_.ctrl_dt * args_.Hsample;
      step_us_ = VectorXd::Zero(args_.Hsample + 1);
      step_nodes_ = VectorXd::Zero(args_.Hnode + 1);
      for (int i = 0; i <= args_.Hsample; i++)
      {
        step_us_(i) = (double)i / (double)args_.Hsample * tmax;
      }
      for (int i = 0; i <= args_.Hnode; i++)
      {
        step_nodes_(i) = (double)i / (double)args_.Hnode * tmax;
      }
    }

    // Roll out the full control trajectories for each candidate.
    // all_us: vector of matrices, each of shape (Hsample+1, nu)
    // Returns: vector of reward vectors (each of length Hsample+1)
    std::vector<VectorXd> rollout_us_batch(const go2env::EnvState &state, const std::vector<MatrixXd> &all_us)
    {
      // Loop over all candidates (should be NUMSAMPLES+1 in total)
      const size_t num_samples = all_us.size();
      std::vector<Eigen::VectorXd> rews_batch(num_samples);

      // auto start_time = std::chrono::high_resolution_clock::now();
// Parallelize the loop over candidates.
// #pragma omp parallel for schedule(dynamic) num_threads(16)
      for (int i = 0; i < static_cast<int>(num_samples); ++i)
      {
        // Run the simulation for the i-th candidate.
        std::vector<go2env::EnvState> traj_sample = env_.stepTrajectory(i, state, all_us[i]);
        if (i == 0 || i == 1 || i  == static_cast<int>(num_samples)-2 || i == static_cast<int>(num_samples)-1)
        {
            std::cout << i << std::endl;
            std::cout << "state qpos: " << std::endl;
            for (int k = 0; k<traj_sample.size(); k=k+1){
                std::cout << traj_sample[k].pipeline_state.qpos.transpose() << std::endl;
            }
            std::cout << "state qvel: " << std::endl;
            for (int k = 0; k<traj_sample.size(); k=k+1){
                std::cout << traj_sample[k].pipeline_state.qvel.transpose() << std::endl;
            }
            std::cout << "state x (pos): " << std::endl;
            for (int k = 0; k<traj_sample.size(); k=k+1){
                std::cout << traj_sample[k].pipeline_state.x.pos_.transpose() << std::endl;
            }
        }
        // Allocate and initialize the reward vector.
        rews_batch[i] = Eigen::VectorXd::Zero(args_.Hsample + 1);

        // Fill in the reward trajectory.
        for (size_t t = 0; t < traj_sample.size(); ++t)
        {
          rews_batch[i](t) = traj_sample[t].reward;
        }
        if (i == 0 || i == 1 || i  == static_cast<int>(num_samples)-2 || i == static_cast<int>(num_samples)-1)
        {
            std::cout << "Rews: " << rews_batch[i] << std::endl;
            std::cin.get();
        }
      }
      // std::cout << "Average rollout time per sample: " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time).count() / (double)num_samples << "us" << std::endl;
      return rews_batch;
    }

    // reverse_once: one step of reverse diffusion.
    //   - state: current environment state
    //   - rng: random number generator (passed by reference)
    //   - Ybar_i: current nominal control nodes (shape: Hnode+1 x nu)
    //   - noise_scale: vector (length Hnode+1)
    // Returns: tuple (Ybar, info), where Ybar is updated nodes.
    std::tuple<MatrixXd, ReverseInfo, JaxRNG>
    reverse_once(const go2env::EnvState &state,
                 JaxRNG &rng,
                 const MatrixXd &Ybar_i,
                 const VectorXd &noise_scale)
    {
      std::cout << "---------- INSIDE REVERSE ONCE ----------" <<std::endl;
      std::cout << "Ybar_i being sent inside: " << Ybar_i << std::endl;
      std::cout << "Noise scale being sent inside: " << noise_scale << std::endl;
      auto [new_rng, Y0s_rng] = rng.split();
      rng = new_rng;
      // 1) Sample from q_i
      // Generate NUMSAMPLES candidates (each is a matrix of shape (Hnode+1, nu))
      std::normal_distribution<double> dist(0.0, 1.0);
      std::vector<MatrixXd> Y0s;
      Y0s.reserve(NUMSAMPLES);

      // auto start_time_sampling = std::chrono::high_resolution_clock::now();
      for (int s = 0; s < NUMSAMPLES; s++)
      {
        // std::mt19937_64 *rng_tmp = new std::mt19937_64(std::chrono::system_clock::now().time_since_epoch().count() + s);
        MatrixXd eps = MatrixXd::Zero(args_.Hnode + 1, nu_);
        for (int i = 0; i <= args_.Hnode; i++)
        {
          for (int j = 0; j < nu_; j++)
          {
            // double z = dist(Y0s_rng.rng);
            double z = 0.5;
            // std::cout<<"z: "<<z<<std::endl;
            eps(i, j) = z * noise_scale(i);
          }
        }
        MatrixXd candidate = Ybar_i + eps;
        // Fix the first node (control) to remain unchanged.16
        candidate.row(0) = Ybar_i.row(0);
        Y0s.push_back(candidate);

        // delete rng_tmp;
      }
      // std::cout << "Sampling time in reverse_once: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time_sampling).count() << "ms" << std::endl;

      // Append Ybar_i as the last candidate so that total candidates = NUMSAMPLES+1.
      std::vector<MatrixXd> all_Y0s = Y0s;
      all_Y0s.push_back(Ybar_i);
      std::cout << "1st Y0 before clipping" << all_Y0s[0]<<std::endl;
      std::cout << "2nd Y0 before clipping" << all_Y0s[1]<<std::endl;
      std::cout << "Last 2nd Y0 before clipping" << all_Y0s[all_Y0s.size()-2]<<std::endl;
      std::cout << "Last Y0 before clipping" << all_Y0s[all_Y0s.size()-1]<<std::endl;

      // std::cout << "Candidate node trajectories dimensions: " << all_Y0s.size()
      //           << " x " << all_Y0s[0].rows() << " x " << all_Y0s[0].cols() << std::endl;
      // std::cout << "Expected dimensions: " << (NUMSAMPLES + 1) << " x " << (args_.Hnode + 1) << " x " << nu_ << std::endl;
      // Expected: (NUMSAMPLES+1) x (Hnode+1) x (nu)

      // 2) Clip each candidate to the range [-1, 1].
      // auto start_time_clip = std::chrono::high_resolution_clock::now();
      for (MatrixXd &mat : all_Y0s)
      {
        for (int r = 0; r < mat.rows(); r++)
        {
          for (int c = 0; c < mat.cols(); c++)
          {
            if (mat(r, c) < -1.0)
              mat(r, c) = -1.0;
            if (mat(r, c) > 1.0)
              mat(r, c) = 1.0;
          }
        }
      }

      std::cout << "1st Y0 after clipping" << all_Y0s[0]<<std::endl;
      std::cout << "2nd Y0 after clipping" << all_Y0s[1]<<std::endl;
      std::cout << "Last 2nd Y0 after clipping" << all_Y0s[all_Y0s.size()-2]<<std::endl;
      std::cout << "Last Y0 after clipping" << all_Y0s[all_Y0s.size()-1]<<std::endl;

      // std::cout << "Clipping time in reverse_once: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time_clip).count() << "ms" << std::endl;

      // 3) Convert each candidate from node-space to a full control trajectory.
      // Each candidate: (Hnode+1, nu) -> (Hsample+1, nu)
      // auto start_time_node2u = std::chrono::high_resolution_clock::now();
      std::vector<MatrixXd> batch_us;
      batch_us.reserve(all_Y0s.size());
      for (size_t i = 0; i < all_Y0s.size(); i++)
      {
        Eigen::MatrixXd u_traj = node2u(all_Y0s[i], step_nodes_, step_us_);
        batch_us.push_back(u_traj);
      }

      std::cout << "1st u: " << batch_us[0]<<std::endl;
      std::cout << "2nd u: " << batch_us[1]<<std::endl;
      std::cout << "Last 2nd u: " << batch_us[batch_us.size()-2]<<std::endl;
      std::cout << "Last u: " << batch_us[batch_us.size()-1]<<std::endl;

      std::cout << "state qpos: " << state.pipeline_state.qpos << std::endl;
      std::cout << "state qvel: " << state.pipeline_state.qvel << std::endl;
      std::cout << "state x (pos, quat): " << state.pipeline_state.x.pos_<< ", " << state.pipeline_state.x.quat_ << std::endl;
      std::cout << "state xd (lin, ang): " << state.pipeline_state.xd.lin_ << ", " << state.pipeline_state.xd.ang_ << std::endl;
      std::cout << "state reward: " << state.reward << std::endl;
      // std::cout << "Node2u time in reverse_once: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time_node2u).count() << "ms" << std::endl;

      // std::cout << "Batch control trajectories dimensions: " << batch_us.size()
      //           << " x " << batch_us[0].rows() << " x " << batch_us[0].cols() << std::endl;
      // std::cout << "Expected dimensions: " << (NUMSAMPLES + 1) << " x " << (args_.Hsample + 1)
      //           << " x " << nu_ << std::endl;
      // Expected: (NUMSAMPLES+1) x (Hsample+1) x nu

      // 4) Roll out each candidate trajectory.
      // auto start_time_rollout_us = std::chrono::high_resolution_clock::now();
      std::vector<VectorXd> rews_batch = rollout_us_batch(state, batch_us);

      std::cout << "Batch_us size: " << batch_us.size() << ", " << batch_us[0].rows() << ", " << batch_us[0].cols() << std::endl;

      // Print elements
    //   for (size_t i = 0; i < rews_batch.size(); ++i) {
    //     std::cout << "Vector " << i << " (size: " << rews_batch[i].size() << "): ";
    //     std::cout << rews_batch[i].transpose() << std::endl;
    //   }
    //   std::cout << "Size of rews_batch: " << rews_batch.size() << std::endl;
      std::cout << NUMSAMPLES << ", " << args_.Hnode + 1 << ", " << nu_ <<std::endl;
      // std::cout << "Rollout time in reverse_once: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time_rollout_us).count() << "ms" << std::endl;

      // 5) Compute the average reward for the nominal candidate (the last one).
      Eigen::VectorXd rews_Ybar_i = rews_batch.back(); // shape: (Hsample+1)
      double rew_Ybar_i = rews_Ybar_i.mean();

      // 6) For each candidate, compute the mean reward and its standard deviation (over time).
      int Nall = static_cast<int>(rews_batch.size()); // should be NUMSAMPLES+1
      Eigen::VectorXd meanRews(Nall), stdRews(Nall);
      for (int s = 0; s < Nall; s++)
      {
        double m = rews_batch[s].mean();
        meanRews(s) = m;
        double sum_sq = 0.0;
        int T = rews_batch[s].size();
        for (int t = 0; t < T; t++)
        {
          double diff = rews_batch[s](t) - m;
          sum_sq += diff * diff;
        }
        double var = sum_sq / T;
        double stdev = (var > 1e-14) ? std::sqrt(var) : 1e-7;
        stdRews(s) = stdev;
      }
      
      // 7) Compute log probabilities.
      Eigen::VectorXd logp0(Nall);
      for (int s = 0; s < Nall; s++)
      {
        logp0(s) = (meanRews(s) - rew_Ybar_i) / (stdRews(s) * args_.temp_sample);
      }

      // 8) Compute softmax weights.
      double max_val = logp0.maxCoeff();
      Eigen::VectorXd exps = (logp0.array() - max_val).exp();
      double sum_exps = exps.sum();
      Eigen::VectorXd weights = exps / sum_exps; // length = NUMSAMPLES+1

      // 9) Update Ybar using the softmax_update function.
      std::tuple<Eigen::MatrixXd, Eigen::VectorXd> res_softmax = softmax_update(weights, all_Y0s, noise_scale, Ybar_i);
      Eigen::MatrixXd Ybar = std::get<0>(res_softmax);
      Eigen::VectorXd new_sigma = std::get<1>(res_softmax);

      // 10) (Placeholders for qbar, qdbar, xbar)
      Eigen::MatrixXd qbar = Eigen::MatrixXd::Zero(args_.Hnode + 1, 1);
      Eigen::MatrixXd qdbar = Eigen::MatrixXd::Zero(args_.Hnode + 1, 1);
      Eigen::MatrixXd xbar = Eigen::MatrixXd::Zero(args_.Hnode + 1, 1);

      // Fill ReverseInfo
      ReverseInfo info;
      info.rews = meanRews; // vector of length (NUMSAMPLES+1)
      info.qbar = qbar;
      info.qdbar = qdbar;
      info.xbar = xbar;
      info.new_noise_scale = new_sigma;

      return std::make_tuple(Ybar, info, rng);
    }

    // reverse: iteratively apply reverse_once from i = Ndiffuse-1 down to 1.
    // MatrixXd reverse(const go2env::EnvState &state,
    //                  const MatrixXd &YN,
    //                  std::mt19937_64 &rng)
    // {
    //   MatrixXd Yi = YN;
    //   for (int i = args_.Ndiffuse - 1; i >= 1; i--)
    //   {
    //     VectorXd scale = VectorXd::Constant(args_.Hnode + 1, sigmas_(i));
    //     std::tuple<MatrixXd, ReverseInfo> res_reverse = reverse_once(state, rng, Yi, scale);
    //     MatrixXd newY = std::get<0>(res_reverse);
    //     ReverseInfo info = std::get<1>(res_reverse);

    //     Yi = newY;
    //   }
    //   return Yi;
    // }

    // shift: Convert node parameters to full control trajectory, roll by one time step,
    // set the final control to zero, then convert back to node parametrization.
    Eigen::MatrixXd shift(const Eigen::MatrixXd &Y)
    {
      // std::cout<<"sn, su: "<<step_nodes_<< " "<<step_us_<<std::endl;
      Eigen::MatrixXd u = node2u(Y, step_nodes_, step_us_);
      Eigen::MatrixXd u_shifted = u;
      for (int i = 0; i < u.rows() - 1; i++)
      {
        u_shifted.row(i) = u.row(i + 1);
      }
      u_shifted.row(u.rows() - 1).setZero();
      Eigen::MatrixXd Ynew = u2node(u_shifted, step_us_, step_nodes_);
      return Ynew;
    }

  public:
    DialConfig args_;
    go2env::UnitreeGo2Env<NUMSAMPLES + 1> &env_;
    int nu_;

    Eigen::VectorXd sigmas_;        // length: Ndiffuse
    Eigen::VectorXd sigma_control_; // length: Hnode+1
    Eigen::VectorXd step_us_;       // length: Hsample+1
    Eigen::VectorXd step_nodes_;    // length: Hnode+1
  };

} // namespace dial