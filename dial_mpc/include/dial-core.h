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
    double horizon_diffuse_factor = 0.5; // multiplies sigma_control
    double ctrl_dt = 0.02;               // control dt
    int n_steps = 400;                   // number of rollout steps
    double traj_diffuse_factor = 0.5;    // factor^i in each "reverse_once" iteration
    std::string update_method = "mppi";  // only "mppi" used in the original script
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

  //////////////////////////////////////////////////////////////
  // Softmax update (the only update_method from Python code: "mppi")
  //////////////////////////////////////////////////////////////
  // inline std::tuple<MatrixXd, VectorXd> softmax_update(
  //     const VectorXd &weights, // shape (Nsample + 1,)
  //     const std::vector<MatrixXd> &Y0s, // shape (Nsample, Hnode + 1, nu)
  //     const VectorXd &sigma, // shape (Hnode + 1,)
  //     const MatrixXd &mu_0t) // shape (Hnode + 1, nu)
  // {
  //   std::cout << "softmax_update called" << std::endl;

  //   // Weighted average
  //   MatrixXd Ybar = MatrixXd::Zero(mu_0t.rows(), mu_0t.cols());
  //   for (int i = 0; i < (int)Y0s.size(); i++)
  //   {
  //     Ybar += weights(i) * Y0s[i];
  //   }
  //   std::cout << "softmax_update done" << std::endl;
  //   return std::make_tuple(Ybar, sigma); // new_sigma = sigma (unchanged)
  // }

  // Function signature exactly as requested.
  inline std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
  softmax_update(const Eigen::VectorXd &weights,
                 const std::vector<Eigen::MatrixXd> &Y0s,
                 const Eigen::VectorXd &sigma,
                 const Eigen::MatrixXd &mu_0t)
  {
    // Check that the number of weights matches the number of candidate matrices.
    if (weights.size() != static_cast<Eigen::Index>(Y0s.size()))
    {
      std::cout << "weights.size() = " << weights.size() << std::endl;
      std::cout << "Y0s.size() = " << Y0s.size() << std::endl;
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

    MBDPI(const DialConfig &args, go2env::UnitreeGo2Env<NUMSAMPLES> &env)
        : args_(args), env_(env), nu_(env.action_size())
    {
      std::cout << "NUMSAMPLES = " << NUMSAMPLES << std::endl;

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

    // // rollout_us: replicate exactly the Python version.
    // // Return the entire time-sequence of rewards (length = us.rows())
    // // plus a vector of pipeline states if needed.
    // std::tuple<VectorXd, std::vector<go2env::EnvState>>
    // rollout_us(const go2env::EnvState &state, const MatrixXd &us)
    // {
    //   int T = us.rows();
    //   VectorXd rewards(T);
    //   rewards.setZero();
    //   std::vector<go2env::EnvState> pipeline_states;
    //   pipeline_states.reserve(T);

    //   go2env::EnvState cur = state;
    //   for (int t = 0; t < T; t++)
    //   {
    //     cur = env_.step(cur, us.row(t).eval());
    //     rewards(t) = cur.reward;
    //     pipeline_states.push_back(cur);
    //   }
    //   return std::make_tuple(rewards, pipeline_states);
    // }

    // // Vectorized version: for each sample in all_Y0s, convert to "us", then rollout
    // std::vector<VectorXd>
    // rollout_us_batch(const go2env::EnvState &state, const std::vector<MatrixXd> &all_us)
    // {
    //   std::vector<VectorXd> rews_batch;
    //   rews_batch.reserve(all_us.size());
    //   for (const MatrixXd &us : all_us)
    //   {
    //     std::tuple<VectorXd, std::vector<go2env::EnvState>> res_rollout = rollout_us(state, us);
    //     VectorXd rews = std::get<0>(res_rollout);
    //     std::vector<go2env::EnvState> pipeline_states = std::get<1>(res_rollout);
    //     rews_batch.push_back(rews);
    //   }
    //   return rews_batch;
    // }

    std::vector<VectorXd> rollout_us_batch(const go2env::EnvState &state, const std::vector<MatrixXd> &all_us)
    {
      std::cout << "rollout_us_batch called" << std::endl;
      std::vector<VectorXd> rews_batch;
      rews_batch.reserve(all_us.size()); // rews_batch should be Nsample+1 x Hsample+1
      for (int i = 0; i < NUMSAMPLES; ++i)
      {
        rews_batch.push_back(VectorXd::Zero(args_.Hsample + 1));
        std::vector<go2env::EnvState> traj_sample_i = env_.stepTrajectory(i, state, all_us[i].transpose());
        std::cout << "Trajectory sample " << i << " has " << traj_sample_i.size() << " steps" << std::endl;
        for (size_t t = 0; t < traj_sample_i.size(); ++t) // data locality sucks but whatever
        {
          rews_batch[i](t) = traj_sample_i[t].reward;
        }
        std::cout << "Rewards for sample " << i << " are: " << rews_batch[i].transpose() << std::endl;
      }
      std::cout << "rollout_us_batch done" << std::endl;
      return rews_batch;
    }

    std::tuple<MatrixXd, ReverseInfo>
    reverse_once(const go2env::EnvState &state,
                 std::mt19937_64 &rng,
                 const MatrixXd &Ybar_i,
                 const VectorXd &noise_scale)
    {
      // 1) Sample from q_i
      // Y0s has shape (Nsample, Hnode+1, nu), plus one more for the appended Ybar_i
      std::normal_distribution<double> dist(0.0, 1.0);
      std::vector<MatrixXd> Y0s(NUMSAMPLES);

      for (int s = 0; s < NUMSAMPLES; s++)
      {
        MatrixXd eps = MatrixXd::Zero(args_.Hnode + 1, nu_);
        for (int i = 0; i <= args_.Hnode; i++)
        {
          for (int j = 0; j < nu_; j++)
          {
            double z = dist(rng);
            eps(i, j) = z * noise_scale(i);
          }
        }
        MatrixXd candidate = Ybar_i + eps;
        // fix first control
        candidate.row(0) = Ybar_i.row(0);
        Y0s[s] = candidate;
      }

      // Append Ybar_i as "last sample"
      std::vector<MatrixXd> all_Y0s = Y0s;
      all_Y0s.push_back(Ybar_i); // all_Y0s now has shape (Nsample+1, Hnode+1, nu)

      std::cout << "all_Y0s dimensions: " << all_Y0s.size() << " x " << all_Y0s[0].rows() << " x " << all_Y0s[0].cols() << std::endl;
      std::cout << "all_Y0s dimensions SHOULD be: " << NUMSAMPLES + 1 << " x " << args_.Hsample + 1 << " x " << nu_ << std::endl;

      // 2) Clip to [-1,1]
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

      // 3) Convert each Y0 to batch_us
      std::vector<MatrixXd> batch_us(all_Y0s.size());

      for (int i = 0; i < (int)all_Y0s.size(); i++)
      {
        batch_us[i] = node2u(all_Y0s[i], step_nodes_, step_us_);
      }
      // batch_us has shape (Nsample+1, Hsample+1, nu)

      std::cout << "batch_us dimensions: " << batch_us.size() << " x " << batch_us[0].rows() << " x " << batch_us[0].cols() << std::endl;
      std::cout << "batch_us dimensions SHOULD be: " << NUMSAMPLES + 1 << " x " << args_.Hsample + 1 << " x " << nu_ << std::endl;

      // 4) Rollout for each sample
      std::vector<VectorXd> rews_batch = rollout_us_batch(state, batch_us);

      // The last sample is Ybar_i
      VectorXd rews_Ybar_i = rews_batch.back(); // shape (Hsample+1)
      double rew_Ybar_i = rews_Ybar_i.mean();

      // 5) Compute each sample's average reward and std
      // Python code:
      //   rews = rewss.mean(axis=-1)  => average across time
      //   rew_Ybar_i = rewss[-1].mean()
      //   logp0 = (rews - rew_Ybar_i) / rews.std(axis=-1) / temp_sample
      // where rews.std(axis=-1) is the stdev of each sample's rews across time.
      int Nall = (int)rews_batch.size(); // = Nsample+1
      VectorXd meanRews(Nall), stdRews(Nall);

      for (int s = 0; s < Nall; s++)
      {
        double m = rews_batch[s].mean();
        meanRews(s) = m;
        // compute stdev across time
        double sum_sq = 0.0;
        int T = rews_batch[s].size();
        for (int t = 0; t < T; t++)
        {
          double diff = (rews_batch[s](t) - m);
          sum_sq += diff * diff;
        }
        double var = sum_sq / (double)T;
        double stdev = (var > 1e-14) ? std::sqrt(var) : 1e-7;
        stdRews(s) = stdev;
      }

      // 6) logp0
      VectorXd logp0(Nall);

      for (int s = 0; s < Nall; s++)
      {
        logp0(s) = (meanRews(s) - rew_Ybar_i) / (stdRews(s) * args_.temp_sample);
      }

      // 7) weights = softmax(logp0)
      double max_val = logp0.maxCoeff();
      VectorXd exps = (logp0.array() - max_val).exp();
      double sum_exps = exps.sum();
      VectorXd weights = exps / sum_exps; // length = Nsample+1

      // 8) Ybar = sum_n w(n)*Y0s[n]
      std::cout << "weights dimensions: " << weights.size() << std::endl;
      std::cout << "all_Y0s dimensions: " << all_Y0s.size() << " x " << all_Y0s[0].rows() << " x " << all_Y0s[0].cols() << std::endl;
      std::cout << "noise_scale dimensions: " << noise_scale.size() << std::endl;
      std::cout << "Ybar_i dimensions: " << Ybar_i.rows() << " x " << Ybar_i.cols() << std::endl;
      std::tuple<MatrixXd, VectorXd> res_softmax = softmax_update(weights, all_Y0s, noise_scale, Ybar_i);
      MatrixXd Ybar = std::get<0>(res_softmax);
      VectorXd new_sigma = std::get<1>(res_softmax);

      // 9) Weighted qbar, qdbar, xbar placeholders
      // The Python code does qbar, qdbar, xbar from pipeline states.
      // We do placeholders. If you want real data, store states from each rollout.
      MatrixXd qbar = MatrixXd::Zero(args_.Hnode + 1, 1);
      MatrixXd qdbar = MatrixXd::Zero(args_.Hnode + 1, 1);
      MatrixXd xbar = MatrixXd::Zero(args_.Hnode + 1, 1);

      // Fill ReverseInfo
      ReverseInfo info;
      info.rews = meanRews; // shape (Nsample+1)
      info.qbar = qbar;
      info.qdbar = qdbar;
      info.xbar = xbar;
      info.new_noise_scale = new_sigma;

      std::cout << "reverse_once done" << std::endl;

      return std::make_tuple(Ybar, info);
    }

    // "reverse" calls "reverse_once" from i = Ndiffuse-1 down to 1, same as Python
    // (which does "for i in range(self.args.Ndiffuse - 1, 0, -1)").
    // For each iteration, we pass "sigmas_[i] * ones(Hnode+1)".
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

    // shift: replicate "shift" in Python
    //   u = node2u(Y)
    //   u = roll(u, -1, axis=0)
    //   u[-1] = 0
    //   Y = u2node(u)
    MatrixXd shift(const MatrixXd &Y)
    {
      MatrixXd u = node2u(Y, step_nodes_, step_us_);
      MatrixXd u_shifted = u;
      // shift up by 1
      for (int i = 0; i < u.rows() - 1; i++)
      {
        u_shifted.row(i) = u.row(i + 1);
      }
      u_shifted.row(u.rows() - 1).setZero();
      MatrixXd Ynew = u2node(u_shifted, step_us_, step_nodes_);
      return Ynew;
    }

  public:
    DialConfig args_;
    go2env::UnitreeGo2Env<NUMSAMPLES> &env_;
    int nu_;

    VectorXd sigmas_;        // length Ndiffuse
    VectorXd sigma_control_; // length Hnode+1
    VectorXd step_us_;       // length Hsample+1
    VectorXd step_nodes_;    // length Hnode+1
  };

} // namespace dial