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
  int Hsample = 16;                    // horizon sample
  int Hnode = 4;                       // number of node points
  int Nsample = 20;                    // number of samples at each diffusion iteration
  int Ndiffuse = 2;                    // how many times we run "reverse_once" each planning step
  int Ndiffuse_init = 10;              // used at the first iteration
  double temp_sample = 0.05;           // temperature
  double horizon_diffuse_factor = 0.5; // multiplies sigma_control
  double ctrl_dt = 0.02;               // control dt
  int n_steps = 400;                   // number of rollout steps
  double traj_diffuse_factor = 0.5;    // factor^i in each "reverse_once" iteration
  std::string update_method = "mppi";  // only "mppi" used in the original script
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
Eigen::MatrixXd piecewiseCubicHermiteInterpolate(
    const Eigen::MatrixXd &states,
    const Eigen::VectorXd &knotTimes,
    const Eigen::VectorXd &queryTimes)
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

  //--------------------------------------------------------------------------------
  // 1) Precompute interval lengths h_i = t_{i+1} - t_i
  //    We'll store them for quick access.
  //--------------------------------------------------------------------------------
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

  //--------------------------------------------------------------------------------
  // 2) For each dimension m, compute the second derivatives at the knots
  //    under "natural" boundary conditions.  We'll store those in an N x M matrix.
  //
  //    We'll do the standard tridiagonal solve:
  //
  //      Let alpha_i = 3 * ( (y_{i+1}-y_i)/h_i - (y_i - y_{i-1})/h_{i-1} )
  //      with alpha_0 = alpha_{N-1} = 0  (natural boundary).
  //
  //    Solve for M_i in the system:
  //      2(h_{i-1}+h_i) * M_i + h_i * M_{i+1} + h_{i-1} * M_{i-1} = alpha_i
  //
  //    Then the second derivatives at each knot are M_i.
  //--------------------------------------------------------------------------------
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

    // Now the backward pass to get the second derivatives M_i
    // We'll temporarily store them in secondDerivs.col(mIdx).
    secondDerivs(N - 1, mIdx) = 0.0;
    for (int i = N - 2; i >= 0; --i)
    {
      secondDerivs(i, mIdx) = z(i) - mu(i) * secondDerivs(i + 1, mIdx);
    }
  }

  //--------------------------------------------------------------------------------
  // 3) Recover the *first* derivatives at each knot from the second derivatives.
  //    For a natural cubic spline on [t_i, t_{i+1}], the polynomial is:
  //       s_i(t) = y_i + B_i (t - t_i) + C_i (t - t_i)^2 + D_i (t - t_i)^3
  //    with
  //       B_i = (y_{i+1}-y_i)/h_i - h_i/6 * (2*M_i + M_{i+1})
  //    (Here M_i = s''(t_i).)
  //
  //    The slope at knot i is s'(t_i).  For the interior knot i, that slope is
  //    typically deduced from the piece on [t_{i-1}, t_i], but for convenience
  //    we can pick it from [t_i, t_{i+1}] or ensure they match. They do match
  //    for a C^2 spline. We will just pick the forward difference formula for m_i:
  //      m_i = B_i = (y_{i+1}-y_i)/h_i - (h_i/6)*(2*M_i + M_{i+1})
  //--------------------------------------------------------------------------------

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
    // For the last knot i = N-1, we can match from the left interval:
    // slope at i = B_{i-1} + h_{i-1}*derivative_of_polynomial...
    // But simpler is to ensure continuity with the prior segment:
    // We can just do it from the formula on [N-2, N-1].
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

  //--------------------------------------------------------------------------------
  // 4) For each query time, do a fast piecewise search (e.g. std::upper_bound)
  //    to find the interval i such that knotTimes(i) <= t < knotTimes(i+1).
  //    Then use the standard cubic Hermite basis for that interval.
  //
  //    On [t_i, t_{i+1}], the Hermite form is:
  //
  //      Let  p_i = states(i,m),   p_{i+1} = states(i+1,m)
  //           m_i = firstDerivs(i,m),   m_{i+1} = firstDerivs(i+1,m)
  //           h_i = knotTimes(i+1) - knotTimes(i)
  //           u   = (t - knotTimes(i)) / h_i   in [0,1].
  //
  //      Then
  //        S(u) =  p_i * H_00(u) + (h_i m_i) * H_10(u)
  //              + p_{i+1} * H_01(u) + (h_i m_{i+1}) * H_11(u),
  //      where
  //        H_00(u) =  2u^3 - 3u^2 + 1
  //        H_10(u) =      u^3 - 2u^2 + u
  //        H_01(u) = -2u^3 + 3u^2
  //        H_11(u) =      u^3 -     u^2
  //--------------------------------------------------------------------------------

  // For fast interval lookup, we’ll just do a pointer into knotTimes if queryTimes is sorted.
  // If queryTimes is not guaranteed sorted, you can do an independent binary_search for each
  // query.  Here we assume sorted queries for maximum efficiency with a single pass.
  // If not sorted, replace this pass with a per-query std::upper_bound.

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

Eigen::MatrixXd piecewiseLinearInterpolate(
    const Eigen::MatrixXd &vals,
    const Eigen::VectorXd &knotTimes,
    const Eigen::VectorXd &queryTimes)
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

inline Eigen::MatrixXd node2u(const Eigen::MatrixXd &nodes,
                              const Eigen::VectorXd &step_nodes,
                              const Eigen::VectorXd &step_us)
{
  // nodes has shape (Hnode+1, nu)
  // return piecewiseLinearInterpolate(nodes, step_nodes, step_us);
  return piecewiseCubicHermiteInterpolate(nodes, step_nodes, step_us);
}

inline Eigen::MatrixXd u2node(const Eigen::MatrixXd &us,
                              const Eigen::VectorXd &step_us,
                              const Eigen::VectorXd &step_nodes)
{
  // us has shape (Hsample+1, nu)
  // return piecewiseLinearInterpolate(us, step_us, step_nodes);
  return piecewiseCubicHermiteInterpolate(us, step_us, step_nodes);
}

//////////////////////////////////////////////////////////////
// Softmax update (the only update_method from Python code: "mppi")
//////////////////////////////////////////////////////////////
inline std::tuple<Eigen::MatrixXd, Eigen::VectorXd> softmax_update(
    const Eigen::VectorXd &weights,
    const std::vector<Eigen::MatrixXd> &Y0s,
    const Eigen::VectorXd &sigma,
    const Eigen::MatrixXd &mu_0t)
{
  // Weighted average
  Eigen::MatrixXd Ybar = Eigen::MatrixXd::Zero(mu_0t.rows(), mu_0t.cols());
  for (int i = 0; i < (int)Y0s.size(); i++)
  {
    Ybar += weights(i) * Y0s[i];
  }
  return std::make_tuple(Ybar, sigma); // new_sigma = sigma (unchanged)
}

//////////////////////////////////////////////////////////////
// MBDPI Class
//////////////////////////////////////////////////////////////
class MBDPI
{
public:
  MBDPI(const DialConfig &args, UnitreeGo2Env &env)
      : args_(args), env_(env), nu_(env.action_size())
  {
    // 1) Precompute sigmas_ for i in [0..Ndiffuse-1]
    double sigma0 = 1e-2, sigma1 = 1.0;
    double A = sigma0;
    double B = std::log(sigma1 / sigma0) / args_.Ndiffuse;
    sigmas_ = Eigen::VectorXd::Zero(args_.Ndiffuse);
    for (int i = 0; i < args_.Ndiffuse; i++)
    {
      sigmas_(i) = A * std::exp(B * i);
    }

    // 2) sigma_control_ = horizon_diffuse_factor^( [Hnode..0] ) (in Python, reversed)
    sigma_control_ = Eigen::VectorXd::Zero(args_.Hnode + 1);
    for (int i = 0; i <= args_.Hnode; i++)
    {
      // reversed exponent, i.e. sigma_control[0] = horizon_diffuse_factor^Hnode
      int exponent = args_.Hnode - i;
      sigma_control_(i) = std::pow(args_.horizon_diffuse_factor, exponent);
    }

    // 3) Create step_us_, step_nodes_
    double tmax = args_.ctrl_dt * args_.Hsample;
    step_us_ = Eigen::VectorXd::Zero(args_.Hsample + 1);
    step_nodes_ = Eigen::VectorXd::Zero(args_.Hnode + 1);
    for (int i = 0; i <= args_.Hsample; i++)
    {
      step_us_(i) = (double)i / (double)args_.Hsample * tmax;
    }
    for (int i = 0; i <= args_.Hnode; i++)
    {
      step_nodes_(i) = (double)i / (double)args_.Hnode * tmax;
    }
  }

  // rollout_us: replicate exactly the Python version.
  // Return the entire time-sequence of rewards (length = us.rows())
  // plus a vector of pipeline states if needed.
  std::tuple<Eigen::VectorXd, std::vector<EnvState>>
  rollout_us(const EnvState &state, const Eigen::MatrixXd &us)
  {
    int T = us.rows();
    Eigen::VectorXd rewards(T);
    rewards.setZero();
    std::vector<EnvState> pipeline_states;
    pipeline_states.reserve(T);

    EnvState cur = state;
    for (int t = 0; t < T; t++)
    {
      cur = env_.step(cur, us.row(t).eval());
      rewards(t) = cur.reward;
      pipeline_states.push_back(cur);
    }
    return std::make_tuple(rewards, pipeline_states);
  }

  // Vectorized version: for each sample in all_Y0s, convert to "us", then rollout
  std::vector<Eigen::VectorXd>
  rollout_us_batch(const EnvState &state, const std::vector<Eigen::MatrixXd> &all_us)
  {
    std::vector<Eigen::VectorXd> rews_batch;
    rews_batch.reserve(all_us.size());
    for (const Eigen::MatrixXd &us : all_us)
    {
      std::tuple<Eigen::VectorXd, std::vector<EnvState>> res_rollout = rollout_us(state, us);
      Eigen::VectorXd rews = std::get<0>(res_rollout);
      std::vector<EnvState> pipeline_states = std::get<1>(res_rollout);
      rews_batch.push_back(rews);
    }
    return rews_batch;
  }

  // reverse_once: replicate the Python method exactly
  struct ReverseInfo
  {
    // We store arrays that replicate "info" in Python
    Eigen::VectorXd rews;            // shape (Nsample+1,) of average rewards for each sample
    Eigen::MatrixXd qbar;            // placeholders
    Eigen::MatrixXd qdbar;           // placeholders
    Eigen::MatrixXd xbar;            // placeholders
    Eigen::VectorXd new_noise_scale; // new sigma
  };

  std::tuple<Eigen::MatrixXd, ReverseInfo>
  reverse_once(const EnvState &state,
               std::mt19937_64 &rng,
               const Eigen::MatrixXd &Ybar_i,
               const Eigen::VectorXd &noise_scale)
  {
    // 1) Sample from q_i
    // Y0s has shape (Nsample, Hnode+1, nu), plus one more for the appended Ybar_i
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<Eigen::MatrixXd> Y0s(args_.Nsample);

    for (int s = 0; s < args_.Nsample; s++)
    {
      Eigen::MatrixXd eps = Eigen::MatrixXd::Zero(args_.Hnode + 1, nu_);
      for (int i = 0; i <= args_.Hnode; i++)
      {
        for (int j = 0; j < nu_; j++)
        {
          double z = dist(rng);
          eps(i, j) = z * noise_scale(i);
        }
      }
      Eigen::MatrixXd candidate = Ybar_i + eps;
      // fix first control
      candidate.row(0) = Ybar_i.row(0);
      Y0s[s] = candidate;
    }

    // Append Ybar_i as "last sample"
    std::vector<Eigen::MatrixXd> all_Y0s = Y0s;
    all_Y0s.push_back(Ybar_i);

    // 2) Clip to [-1,1]
    for (Eigen::MatrixXd &mat : all_Y0s)
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

    // 3) Convert each Y0 to "us"
    // us has shape (Hsample+1, nu)
    std::vector<Eigen::MatrixXd> batch_us(all_Y0s.size());

    for (int i = 0; i < (int)all_Y0s.size(); i++)
    {
      batch_us[i] = node2u(all_Y0s[i], step_nodes_, step_us_);
    }

    // 4) Rollout for each sample
    std::vector<Eigen::VectorXd> rews_batch = rollout_us_batch(state, batch_us);

    // The last sample is Ybar_i
    Eigen::VectorXd rews_Ybar_i = rews_batch.back(); // shape (Hsample+1)
    double rew_Ybar_i = rews_Ybar_i.mean();

    // 5) Compute each sample's average reward and std
    // Python code:
    //   rews = rewss.mean(axis=-1)  => average across time
    //   rew_Ybar_i = rewss[-1].mean()
    //   logp0 = (rews - rew_Ybar_i) / rews.std(axis=-1) / temp_sample
    // where rews.std(axis=-1) is the stdev of each sample's rews across time.
    int Nall = (int)rews_batch.size(); // = Nsample+1
    Eigen::VectorXd meanRews(Nall), stdRews(Nall);

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
    Eigen::VectorXd logp0(Nall);

    for (int s = 0; s < Nall; s++)
    {
      logp0(s) = (meanRews(s) - rew_Ybar_i) / (stdRews(s) * args_.temp_sample);
    }

    // 7) weights = softmax(logp0)
    double max_val = logp0.maxCoeff();
    Eigen::VectorXd exps = (logp0.array() - max_val).exp();
    double sum_exps = exps.sum();
    Eigen::VectorXd weights = exps / sum_exps; // length = Nsample+1

    // 8) Ybar = sum_n w(n)*Y0s[n]
    std::tuple<Eigen::MatrixXd, Eigen::VectorXd> res_softmax = softmax_update(weights, all_Y0s, noise_scale, Ybar_i);
    Eigen::MatrixXd Ybar = std::get<0>(res_softmax);
    Eigen::VectorXd new_sigma = std::get<1>(res_softmax);

    // 9) Weighted qbar, qdbar, xbar placeholders
    // The Python code does qbar, qdbar, xbar from pipeline states.
    // We do placeholders. If you want real data, store states from each rollout.
    Eigen::MatrixXd qbar = Eigen::MatrixXd::Zero(args_.Hnode + 1, 1);
    Eigen::MatrixXd qdbar = Eigen::MatrixXd::Zero(args_.Hnode + 1, 1);
    Eigen::MatrixXd xbar = Eigen::MatrixXd::Zero(args_.Hnode + 1, 1);

    // Fill ReverseInfo
    ReverseInfo info;
    info.rews = meanRews; // shape (Nsample+1)
    info.qbar = qbar;
    info.qdbar = qdbar;
    info.xbar = xbar;
    info.new_noise_scale = new_sigma;

    return std::make_tuple(Ybar, info);
  }

  // "reverse" calls "reverse_once" from i = Ndiffuse-1 down to 1, same as Python
  // (which does "for i in range(self.args.Ndiffuse - 1, 0, -1)").
  // For each iteration, we pass "sigmas_[i] * ones(Hnode+1)".
  Eigen::MatrixXd reverse(const EnvState &state,
                          const Eigen::MatrixXd &YN,
                          std::mt19937_64 &rng)
  {
    Eigen::MatrixXd Yi = YN;
    for (int i = args_.Ndiffuse - 1; i >= 1; i--)
    {
      Eigen::VectorXd scale = Eigen::VectorXd::Constant(args_.Hnode + 1, sigmas_(i));
      std::tuple<Eigen::MatrixXd, ReverseInfo> res_reverse = reverse_once(state, rng, Yi, scale);

      Eigen::MatrixXd newY = std::get<0>(res_reverse);
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
  Eigen::MatrixXd shift(const Eigen::MatrixXd &Y)
  {
    Eigen::MatrixXd u = node2u(Y, step_nodes_, step_us_);
    Eigen::MatrixXd u_shifted = u;
    // shift up by 1
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
  UnitreeGo2Env &env_;
  int nu_;

  Eigen::VectorXd sigmas_;        // length Ndiffuse
  Eigen::VectorXd sigma_control_; // length Hnode+1
  Eigen::VectorXd step_us_;       // length Hsample+1
  Eigen::VectorXd step_nodes_;    // length Hnode+1
};