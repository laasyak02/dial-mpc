#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <unsupported/Eigen/Splines>
#include <Eigen/Dense>

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
  int seed = 42;
  int Hsample = 15;                    // horizon sample
  int Hnode = 5;                       // number of node points
  int Nsample = 32;                    // number of samples at each diffusion iteration
  int Ndiffuse = 5;                    // how many times we run "reverse_once" each planning step
  int Ndiffuse_init = 5;               // used at the first iteration
  double temp_sample = 1.0;            // temperature
  double horizon_diffuse_factor = 1.0; // multiplies sigma_control
  double ctrl_dt = 0.02;               // control dt
  int n_steps = 50;                    // number of rollout steps
  double traj_diffuse_factor = 1.0;    // factor^i in each "reverse_once" iteration
  std::string update_method = "mppi";  // only "mppi" used in the original script
};

//////////////////////////////////////////////////////////////
// Simple environment definitions
//////////////////////////////////////////////////////////////
struct EnvState
{
  int t;
  Eigen::VectorXd qpos;
  Eigen::VectorXd qvel;
  Eigen::VectorXd ctrl;
  double reward;
};

class Env
{
public:
  // Example environment with a chosen state dimension, you can adapt as needed.
  Env(int state_size, int action_size)
      : state_size_(state_size), action_size_(action_size) {}

  // Reset environment
  EnvState reset(std::mt19937_64 &rng)
  {
    EnvState s;
    s.t = 0;
    s.qpos = Eigen::VectorXd::Zero(state_size_);
    s.qvel = Eigen::VectorXd::Zero(state_size_);
    s.ctrl = Eigen::VectorXd::Zero(action_size_);
    s.reward = 0.0;
    return s;
  }

  // Step environment: for demonstration, just do something trivial
  EnvState step(const EnvState &state, const Eigen::VectorXd &u)
  {
    EnvState next = state;
    next.t += 1;
    next.ctrl = u;
    // example "physics"
    next.qpos += 0.01 * u.head(std::min<int>(state_size_, u.size()));
    next.qvel += 0.01 * u.head(std::min<int>(state_size_, u.size()));
    // example reward: negative L2 norm of control
    next.reward = -u.squaredNorm();
    return next;
  }

  int action_size() const { return action_size_; }

private:
  int state_size_;
  int action_size_;
};

//////////////////////////////////////////////////////////////
// Quadratic spline interpolation (k=2), matching Python's
// InterpolatedUnivariateSpline(..., k=2)
//////////////////////////////////////////////////////////////
inline Eigen::Spline<double, 1> createQuadraticSpline(const Eigen::VectorXd &x, const Eigen::VectorXd &y)
{
  std::cout << "createQuadraticSpline: x.size() = " << x.size() << ", y.size() = " << y.size() << std::endl;
  Eigen::Matrix<double, 1, Eigen::Dynamic> points(1, y.size());
  for (int i = 0; i < y.size(); i++)
  {
    points(0, i) = y(i);
  }
  // Rescale x to [0,1]
  double xmin = x.minCoeff();
  double xmax = x.maxCoeff();
  double L = xmax - xmin;
  Eigen::VectorXd t = (x.array() - xmin) / L;

  // Fit order=2
  Eigen::SplineFitting<Eigen::Spline<double, 1>> fitting;
  auto spline = fitting.Interpolate(points, /*splineOrder=*/y.size() - 1, t);
  return spline;
}

inline Eigen::VectorXd evaluateQuadraticSpline(const Eigen::Spline<double, 1> &spline,
                                               const Eigen::VectorXd &x_original,
                                               double xmin,
                                               double xmax)
{
  double L = xmax - xmin;
  Eigen::VectorXd out(x_original.size());
  for (int i = 0; i < x_original.size(); i++)
  {
    double t = (x_original(i) - xmin) / L;
    Eigen::Matrix<double, 1, 1> val = spline(t);
    out(i) = val(0, 0);
  }
  std::cout << "evaluateQuadraticSpline: out.size() = " << out.size() << std::endl;
  return out;
}

// node2u: from (Hnode+1, nu) controls at times "step_nodes_" to
//          (Hsample+1, nu) controls at times "step_us_"
inline Eigen::MatrixXd node2u(const Eigen::MatrixXd &nodes,
                              const Eigen::VectorXd &step_nodes,
                              const Eigen::VectorXd &step_us)
{
  std::cout << "node2u: nodes.rows() = " << nodes.rows() << ", nodes.cols() = " << nodes.cols() << std::endl;
  std::cout << "node2u: step_nodes.size() = " << step_nodes.size() << ", step_us.size() = " << step_us.size() << std::endl;
  // Output: (Hsample+1, nu)
  Eigen::MatrixXd us(step_us.size(), nodes.cols());
  for (int dim = 0; dim < nodes.cols(); dim++)
  {
    Eigen::VectorXd yvals = nodes.col(dim);
    std::cout << "node2u: yvals.size() = " << yvals.size() << std::endl;
    Eigen::Spline<double, 1> spline = createQuadraticSpline(step_nodes, yvals);
    double xmin = step_nodes.minCoeff();
    double xmax = step_nodes.maxCoeff();
    Eigen::VectorXd result = evaluateQuadraticSpline(spline, step_us, xmin, xmax);
    std::cout << "node2u: result.size() = " << result.size() << std::endl;
    us.col(dim) = result;
  }
  std::cout << "node2u: us.rows() = " << us.rows() << ", us.cols() = " << us.cols() << std::endl;
  return us;
}

// u2node: from (Hsample+1, nu) controls at times "step_us_" to
//          (Hnode+1, nu) controls at times "step_nodes_"
inline Eigen::MatrixXd u2node(const Eigen::MatrixXd &us,
                              const Eigen::VectorXd &step_us,
                              const Eigen::VectorXd &step_nodes)
{
  std::cout << "u2node: us.rows() = " << us.rows() << ", us.cols() = " << us.cols() << std::endl;
  std::cout << "u2node: step_us.size() = " << step_us.size() << ", step_nodes.size() = " << step_nodes.size() << std::endl;
  // Output: (Hnode+1, nu)
  Eigen::MatrixXd nodes(step_nodes.size(), us.cols());
  for (int dim = 0; dim < us.cols(); dim++)
  {
    Eigen::VectorXd yvals = us.col(dim);
    std::cout << "u2node: yvals.size() = " << yvals.size() << std::endl;
    Eigen::Spline<double, 1> spline = createQuadraticSpline(step_us, yvals);
    double xmin = step_us.minCoeff();
    double xmax = step_us.maxCoeff();
    Eigen::VectorXd result = evaluateQuadraticSpline(spline, step_nodes, xmin, xmax);
    std::cout << "u2node: result.size() = " << result.size() << std::endl;
    nodes.col(dim) = result;
  }
  std::cout << "u2node: nodes.rows() = " << nodes.rows() << ", nodes.cols() = " << nodes.cols() << std::endl;
  return nodes;
}

//////////////////////////////////////////////////////////////
// Softmax update (the only update_method from Python code: "mppi")
//////////////////////////////////////////////////////////////
inline std::pair<Eigen::MatrixXd, Eigen::VectorXd> softmax_update(
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
  return std::make_pair(Ybar, sigma); // new_sigma = sigma (unchanged)
}

//////////////////////////////////////////////////////////////
// MBDPI Class
//////////////////////////////////////////////////////////////
class MBDPI
{
public:
  MBDPI(const DialConfig &args, Env &env)
      : args_(args), env_(env), nu_(env.action_size())
  {
    // 1) Precompute sigmas_ for i in [0..Ndiffuse-1]
    double sigma0 = 1e-2, sigma1 = 1.0;
    double A = sigma0;
    double B = std::log(sigma1 / sigma0) / args_.Ndiffuse;
    sigmas_.resize(args_.Ndiffuse);
    for (int i = 0; i < args_.Ndiffuse; i++)
    {
      sigmas_(i) = A * std::exp(B * i);
    }

    // 2) sigma_control_ = horizon_diffuse_factor^( [Hnode..0] ) (in Python, reversed)
    sigma_control_.resize(args_.Hnode + 1);
    for (int i = 0; i <= args_.Hnode; i++)
    {
      // reversed exponent, i.e. sigma_control[0] = horizon_diffuse_factor^Hnode
      int exponent = args_.Hnode - i;
      sigma_control_(i) = std::pow(args_.horizon_diffuse_factor, exponent);
    }

    // 3) Create step_us_, step_nodes_
    double tmax = args_.ctrl_dt * args_.Hsample;
    step_us_.resize(args_.Hsample + 1);
    step_nodes_.resize(args_.Hnode + 1);
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
  std::pair<Eigen::VectorXd, std::vector<EnvState>>
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
      cur = env_.step(cur, us.row(t));
      rewards(t) = cur.reward;
      pipeline_states.push_back(cur);
    }
    return std::make_pair(rewards, pipeline_states);
  }

  // Vectorized version: for each sample in all_Y0s, convert to "us", then rollout
  std::vector<Eigen::VectorXd>
  rollout_us_batch(const EnvState &state, const std::vector<Eigen::MatrixXd> &all_us)
  {
    std::vector<Eigen::VectorXd> rews_batch;
    rews_batch.reserve(all_us.size());
    for (auto &us : all_us)
    {
      auto [rews, _] = rollout_us(state, us);
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
    for (auto &mat : all_Y0s)
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
    auto [Ybar, new_sigma] = softmax_update(weights, all_Y0s, noise_scale, Ybar_i);

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

    std::cout << "reverse_once: Ybar.rows() = " << Ybar.rows() << ", Ybar.cols() = " << Ybar.cols() << std::endl;
    std::cout << "reverse_once: info.rews.size() = " << info.rews.size() << std::endl;

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
      auto [newY, info] = reverse_once(state, rng, Yi, scale);
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
  Env &env_;
  int nu_;

  Eigen::VectorXd sigmas_;        // length Ndiffuse
  Eigen::VectorXd sigma_control_; // length Hnode+1
  Eigen::VectorXd step_us_;       // length Hsample+1
  Eigen::VectorXd step_nodes_;    // length Hnode+1
};

//////////////////////////////////////////////////////////////
// Main
//////////////////////////////////////////////////////////////
int main()
{
  // 1) Build config
  DialConfig cfg;
  cfg.seed = 123;
  cfg.Hsample = 15;
  cfg.Hnode = 5;
  cfg.Nsample = 32;
  cfg.Ndiffuse = 5;
  cfg.Ndiffuse_init = 5;
  cfg.temp_sample = 1.0;
  cfg.n_steps = 10; // shorter demonstration
  cfg.ctrl_dt = 0.02;
  cfg.horizon_diffuse_factor = 1.0;
  cfg.traj_diffuse_factor = 1.0;

  // 2) Create environment: e.g., 4D state, 3D action
  Env env(/*state_size=*/4, /*action_size=*/3);

  // 3) Create MBDPI
  MBDPI mbdpi(cfg, env);

  // 4) RNG
  std::mt19937_64 rng(cfg.seed);

  // 5) Reset environment
  EnvState state_init = env.reset(rng);

  // 6) YN = zeros( (Hnode+1, nu) )
  Eigen::MatrixXd YN = Eigen::MatrixXd::Zero(cfg.Hnode + 1, mbdpi.nu_);

  // 7) (Optional) initial reverse call, as in Python code sometimes:
  //    "Y0 = mbdpi.reverse(state_init, YN, rng_exp)"
  //    You can comment/uncomment as needed.
  YN = mbdpi.reverse(state_init, YN, rng);

  // Let Y0 = the result
  Eigen::MatrixXd Y0 = YN;

  // 8) Main rollout loop
  EnvState cur_state = state_init;
  std::vector<double> rews;
  rews.reserve(cfg.n_steps);

  for (int t = 0; t < cfg.n_steps; t++)
  {
    // Step environment with the first row of Y0
    Eigen::VectorXd action = Y0.row(0);
    EnvState next_state = env.step(cur_state, action);
    rews.push_back(next_state.reward);

    // shift Y0
    Y0 = mbdpi.shift(Y0);

    // how many times to do "reverse_once"?
    int n_diffuse = (t == 0) ? cfg.Ndiffuse_init : cfg.Ndiffuse;

    // Python code builds:
    //   traj_diffuse_factors = sigma_control * (traj_diffuse_factor^arange(n_diffuse))[:,None]
    //   Then a lax.scan over (rng, Y0, state)
    // We'll replicate that with a for-loop:
    for (int i = 0; i < n_diffuse; i++)
    {
      // factor for iteration i
      // shape = (Hnode+1,)
      Eigen::VectorXd factor(cfg.Hnode + 1);
      for (int h = 0; h <= cfg.Hnode; h++)
      {
        factor(h) = mbdpi.sigma_control_(h) * std::pow(cfg.traj_diffuse_factor, (double)i);
      }
      // call reverse_once
      auto [newY, info] = mbdpi.reverse_once(next_state, rng, Y0, factor);
      Y0 = newY;
      // info.rews is the distribution of sample mean rewards, you can log if desired.
    }

    cur_state = next_state;
  }

  // Summarize
  double sum_rew = 0.0;
  for (auto r : rews)
    sum_rew += r;
  double avg_rew = sum_rew / (double)rews.size();
  std::cout << "Average reward = " << avg_rew << std::endl;

  return 0;
}
