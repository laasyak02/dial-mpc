#pragma once

#include "unitree-go2-env.h"

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
    int n_steps = 200;                   // number of rollout steps
    double ctrl_dt = 0.0025;               // control dt
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
  MatrixXd piecewiseCubicHermiteInterpolate(
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
      std::vector<VectorXd> rews_batch;
      rews_batch.reserve(all_us.size()); // rews_batch should be Nsample+1 x Hsample+1
      for (int i = 0; i < all_us.size(); ++i)
      {
        rews_batch.push_back(VectorXd::Zero(args_.Hsample + 1));
        std::vector<go2env::EnvState> traj_sample_i = env_.stepTrajectory(i, state, all_us[i]);
        for (size_t t = 0; t < traj_sample_i.size(); ++t) // data locality sucks but whatever
        {
          rews_batch[i](t) = traj_sample_i[t].reward;
        }
      }
      return rews_batch;
    }

    // reverse_once: one step of reverse diffusion.
    //   - state: current environment state
    //   - rng: random number generator (passed by reference)
    //   - Ybar_i: current nominal control nodes (shape: Hnode+1 x nu)
    //   - noise_scale: vector (length Hnode+1)
    // Returns: tuple (Ybar, info), where Ybar is updated nodes.
    std::tuple<MatrixXd, ReverseInfo>
    reverse_once(const go2env::EnvState &state,
                 std::mt19937_64 &rng,
                 const MatrixXd &Ybar_i,
                 const VectorXd &noise_scale)
    {
      // 1) Sample from q_i
      // Generate NUMSAMPLES candidates (each is a matrix of shape (Hnode+1, nu))
      std::normal_distribution<double> dist(0.0, 1.0);
      std::vector<MatrixXd> Y0s;
      Y0s.reserve(NUMSAMPLES);

      for (int s = 0; s < NUMSAMPLES; s++)
      {
        std::mt19937_64 *rng_tmp = new std::mt19937_64(std::chrono::system_clock::now().time_since_epoch().count() + s);
        MatrixXd eps = MatrixXd::Zero(args_.Hnode + 1, nu_);
        for (int i = 0; i <= args_.Hnode; i++)
        {
          for (int j = 0; j < nu_; j++)
          {
            double z = dist(*rng_tmp);
            eps(i, j) = z * noise_scale(i);
          }
        }
        MatrixXd candidate = Ybar_i + eps;
        // Fix the first node (control) to remain unchanged.
        candidate.row(0) = Ybar_i.row(0);
        Y0s.push_back(candidate);

        delete rng_tmp;
      }

      // Append Ybar_i as the last candidate so that total candidates = NUMSAMPLES+1.
      std::vector<MatrixXd> all_Y0s = Y0s;
      all_Y0s.push_back(Ybar_i);

      // std::cout << "Candidate node trajectories dimensions: " << all_Y0s.size()
      //           << " x " << all_Y0s[0].rows() << " x " << all_Y0s[0].cols() << std::endl;
      // std::cout << "Expected dimensions: " << (NUMSAMPLES + 1) << " x " << (args_.Hnode + 1) << " x " << nu_ << std::endl;
      // Expected: (NUMSAMPLES+1) x (Hnode+1) x (nu)

      // 2) Clip each candidate to the range [-1, 1].
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

      // 3) Convert each candidate from node-space to a full control trajectory.
      // Each candidate: (Hnode+1, nu) -> (Hsample+1, nu)
      std::vector<MatrixXd> batch_us;
      batch_us.reserve(all_Y0s.size());
      for (size_t i = 0; i < all_Y0s.size(); i++)
      {
        Eigen::MatrixXd u_traj = node2u(all_Y0s[i], step_nodes_, step_us_);
        batch_us.push_back(u_traj);
      }

      // std::cout << "Batch control trajectories dimensions: " << batch_us.size()
      //           << " x " << batch_us[0].rows() << " x " << batch_us[0].cols() << std::endl;
      // std::cout << "Expected dimensions: " << (NUMSAMPLES + 1) << " x " << (args_.Hsample + 1)
      //           << " x " << nu_ << std::endl;
      // Expected: (NUMSAMPLES+1) x (Hsample+1) x nu

      // 4) Roll out each candidate trajectory.
      std::vector<VectorXd> rews_batch = rollout_us_batch(state, batch_us);

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

      return std::make_tuple(Ybar, info);
    }

    // reverse: iteratively apply reverse_once from i = Ndiffuse-1 down to 1.
    MatrixXd reverse(const go2env::EnvState &state,
                     const MatrixXd &YN,
                     std::mt19937_64 &rng)
    {
      MatrixXd Yi = YN;
      for (int i = args_.Ndiffuse - 1; i >= 1; i--)
      {
        VectorXd scale = VectorXd::Constant(args_.Hnode + 1, sigmas_(i));
        std::tuple<MatrixXd, ReverseInfo> res_reverse = reverse_once(state, rng, Yi, scale);
        MatrixXd newY = std::get<0>(res_reverse);
        ReverseInfo info = std::get<1>(res_reverse);

        Yi = newY;
      }
      return Yi;
    }

    // shift: Convert node parameters to full control trajectory, roll by one time step,
    // set the final control to zero, then convert back to node parametrization.
    Eigen::MatrixXd shift(const Eigen::MatrixXd &Y)
    {
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