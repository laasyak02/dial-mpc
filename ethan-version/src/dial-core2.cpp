/*****************************************************************************
 * Minimal C++/Eigen MBDPI Implementation
 * (Mirroring the key ideas from the provided Python script)
 *****************************************************************************/
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/Splines> // For spline interpolation

/*****************************************************************************
 * Toy Environment
 * ---------------
 * A simple environment to illustrate "rollout" logic. In practice, replace
 * this with your real environment logic (e.g., a physics simulation).
 *****************************************************************************/

struct EnvState {
  // Example continuous state (position, velocity in 1D)
  double x;
  double v;
  // Reward from the last step
  double reward;
};

class ToyEnv {
public:
  ToyEnv() : dt_(0.01) {}
  // Step the environment with a given control action "u"
  // Returns updated EnvState
  EnvState step(const EnvState &state, double u) const {
    EnvState next = state;

    // Very simple 1D dynamics: x' = x + v*dt, v' = v + u*dt
    next.x += next.v * dt_;
    next.v += u * dt_;

    // A simple reward: negative distance squared to x=0.0 goal
    next.reward = -(next.x * next.x);

    return next;
  }

  // Reset environment
  EnvState reset(unsigned int seed = 0) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    EnvState s;
    s.x = dist(gen);
    s.v = dist(gen);
    s.reward = 0.0;
    return s;
  }

  double dt() const { return dt_; }

private:
  double dt_; // time step
};

/*****************************************************************************
 * Config
 * ------
 * Key hyperparameters from the Python version.
 *****************************************************************************/
struct DialConfig {
  // horizon for control "nodes"
  int Hnode; 
  // horizon for "samples" (finer steps)
  int Hsample;  
  // number of sample rollouts for each reverse pass
  int Nsample;  
  // number of reverse passes
  int Ndiffuse; 
  // special number of diffuse steps on the first iteration
  int Ndiffuse_init; 
  // temperature for softmax weighting
  double temp_sample;
  // horizon diffuse factor 
  double horizon_diffuse_factor; 
  // factor used to diffuse over multiple reverse passes
  double traj_diffuse_factor;
  // total steps in the main loop
  int n_steps;
  // random seed
  int seed;

  DialConfig() {
    // Some default example values (tune as needed)
    Hnode = 5;
    Hsample = 20;
    Nsample = 64;
    Ndiffuse = 5;
    Ndiffuse_init = 10;
    temp_sample = 1.0;
    horizon_diffuse_factor = 1.0; 
    traj_diffuse_factor = 1.0; 
    n_steps = 15;
    seed = 42;
  }
};

/*****************************************************************************
 * Spline Utilities (node2u / u2node)
 * ----------------------------------
 * We replicate the logic of "InterpolatedUnivariateSpline" with k=2 in Python
 * using Eigen's SplineFitting. For a multi-dimensional control, you can either
 * do it dimension-by-dimension or with a higher dimensional Spline type.
 *****************************************************************************/

// Build a spline (k=2 means quadratic) that interpolates the points
// times[i] -> values[i] in 1D. Returns an Eigen::Spline<double, 1>.
static Eigen::Spline<double, 1> fitQuadraticSpline1D(const Eigen::VectorXd &times,
                                                     const Eigen::VectorXd &values) {
  // We want to fit a 1D spline: Spline<double, 1>
  // "points" should have shape = (1, #points). Each col is a sample in 1D.
  Eigen::MatrixXd points(1, times.size());
  for (int i = 0; i < times.size(); i++) {
    points(0, i) = values(i);
  }

  // We fit on "times" as the domain
  return Eigen::SplineFitting<Eigen::Spline<double, 1>>::Interpolate(
      points,  // 1D values
      2,       // spline order k=2
      times);
}

// Evaluate the 1D spline at given queryTime
static double evalSpline1D(const Eigen::Spline<double, 1> &spline,
                           double queryTime) {
  Eigen::Matrix<double, 1, 1> val = spline(queryTime);
  return val(0, 0);
}

/*****************************************************************************
 * MBDPI Class
 * -----------
 * Holds the control representation (nodes -> finer steps), sampling logic,
 * and reverse pass. This is analogous to the Python "MBDPI" class.
 *****************************************************************************/
class MBDPI {
public:
  MBDPI(const DialConfig &args, const ToyEnv &env)
      : args_(args), env_(env), nu_(1) { // For demonstration, action_size = 1

    // Precompute the sigmas for diffusion
    double sigma0 = 1e-2;
    double sigma1 = 1.0;
    double A = sigma0;
    double B = std::log(sigma1 / sigma0) / (double)args_.Ndiffuse;
    sigmas_.resize(args_.Ndiffuse);
    for (int i = 0; i < args_.Ndiffuse; i++) {
      sigmas_[i] = A * std::exp(B * i);
    }

    // sigma_control = horizon_diffuse_factor^(0..Hnode in reverse)
    sigmaControl_.resize(args_.Hnode + 1);
    for (int i = 0; i <= args_.Hnode; i++) {
      // reversed exponent: Hnode-i
      sigmaControl_[i] = std::pow(args_.horizon_diffuse_factor, (args_.Hnode - i));
    }

    // create time vectors for nodes and for fine steps
    ctrl_dt_ = 0.02;
    stepUs_.resize(args_.Hsample + 1);
    stepNodes_.resize(args_.Hnode + 1);
    for (int i = 0; i <= args_.Hsample; i++) {
      stepUs_[i] = ctrl_dt_ * i;
    }
    for (int i = 0; i <= args_.Hnode; i++) {
      stepNodes_[i] = ctrl_dt_ * (double)args_.Hsample * i / (double)args_.Hnode;
    }
  }

  // Convert a node vector of length (Hnode+1) into a control sequence of
  // length (Hsample+1) by spline interpolation. For nu_=1 here.
  Eigen::VectorXd node2u(const Eigen::VectorXd &nodes) const {
    // Fit a spline times=stepNodes_, values=nodes, then evaluate at stepUs_
    Eigen::VectorXd times = Eigen::Map<const Eigen::VectorXd>(stepNodes_.data(),
                                                             stepNodes_.size());
    // "nodes" is same length as times
    Eigen::VectorXd sVals = nodes;

    Eigen::Spline<double, 1> spline = fitQuadraticSpline1D(times, sVals);

    Eigen::VectorXd out(stepUs_.size());
    for (int i = 0; i < (int)stepUs_.size(); i++) {
      out[i] = evalSpline1D(spline, stepUs_[i]);
    }
    return out;
  }

  // Convert a control sequence of length (Hsample+1) back to a node vector
  // of length (Hnode+1) by spline interpolation. For nu_=1 here.
  Eigen::VectorXd u2node(const Eigen::VectorXd &us) const {
    Eigen::VectorXd times = Eigen::Map<const Eigen::VectorXd>(stepUs_.data(),
                                                             stepUs_.size());
    Eigen::Spline<double, 1> spline = fitQuadraticSpline1D(times, us);

    Eigen::VectorXd out(stepNodes_.size());
    for (int i = 0; i < (int)stepNodes_.size(); i++) {
      out[i] = evalSpline1D(spline, stepNodes_[i]);
    }
    return out;
  }

  // Roll out the environment with a sequence of controls "us".
  // Returns the mean reward across all steps (or you could accumulate).
  double rollout_us(const EnvState &initState, const Eigen::VectorXd &us) const {
    EnvState s = initState;
    double sumReward = 0.0;
    for (int i = 0; i < us.size(); i++) {
      s = env_.step(s, us[i]);
      sumReward += s.reward;
    }
    // We can return average or total. In the Python code, it looks like
    // they used the mean of the last dimension, but we’ll do average of all.
    return sumReward / (double)us.size();
  }

  // The main "reverse_once" step from the Python code:
  // 1. Sample candidate node sequences from a Normal around Ybar_i
  // 2. Roll them out, compute rewards
  // 3. Softmax weighting
  // 4. Update Ybar
  Eigen::VectorXd reverse_once(const EnvState &state,
                               std::mt19937 &gen,
                               const Eigen::VectorXd &Ybar_i,
                               double noiseScale) const {
    // We will sample (Nsample) node sequences around Ybar_i
    // dimension of each node seq is (Hnode+1)
    std::normal_distribution<double> dist(0.0, noiseScale);

    // Build matrix Y0s of shape Nsample x (Hnode+1)
    // plus 1 extra to evaluate Ybar_i itself
    int nNodes = args_.Hnode + 1;
    Eigen::MatrixXd Y0s(args_.Nsample + 1, nNodes);
    for (int i = 0; i < args_.Nsample; i++) {
      for (int j = 0; j < nNodes; j++) {
        double eps = dist(gen);
        double val = Ybar_i(j) + eps;
        // We can clamp to [-1,1] as in Python clip
        if (val < -1.0) val = -1.0;
        if (val >  1.0) val =  1.0;
        // first node forced to Ybar_i(0) (like the Python code)
        if (j == 0) val = Ybar_i(0);
        Y0s(i, j) = val;
      }
    }
    // The last row is exactly Ybar_i
    Y0s.row(args_.Nsample) = Ybar_i.transpose();

    // For each row in Y0s, convert to control sequence -> rollout
    // store the average reward
    Eigen::VectorXd rewards(args_.Nsample + 1);
    for (int i = 0; i < args_.Nsample + 1; i++) {
      Eigen::VectorXd us = node2u(Y0s.row(i));
      double r = rollout_us(state, us);
      rewards(i) = r;
    }

    // Softmax weighting
    // Normalizing with the standard deviation of rewards, as in code:
    //   (rews - rew_Ybar_i).std() => but let's approximate with a sample std
    // The Python snippet: 
    //   rew_Ybar_i = rewss[-1].mean() --> "Ybar" was last, 
    //   logp0 = (rews - rew_Ybar_i) / rews.std() / temp_sample
    // We'll do a simpler approach that is consistent in spirit
    double rewYbar = rewards(args_.Nsample); // last is Ybar’s reward
    double meanR = rewards.mean();
    double varR = (rewards.array() - meanR).square().sum() / rewards.size();
    double stdR = std::sqrt(varR + 1e-8);

    Eigen::VectorXd logw(args_.Nsample + 1);
    for (int i = 0; i < args_.Nsample + 1; i++) {
      double scaled = (rewards(i) - rewYbar) / (stdR * args_.temp_sample);
      logw(i) = scaled;
    }
    // numerical stable softmax
    double maxLogw = logw.maxCoeff();
    Eigen::VectorXd wExp = (logw.array() - maxLogw).exp();
    double wExpSum = wExp.sum();
    Eigen::VectorXd weights = wExp / wExpSum; // shape (Nsample+1)

    // Weighted update
    // Ybar = sum_i weights_i * Y0s.row(i)
    // result is 1 x nNodes
    Eigen::VectorXd newYbar = Eigen::VectorXd::Zero(nNodes);
    for (int i = 0; i < args_.Nsample + 1; i++) {
      newYbar += weights(i) * Y0s.row(i).transpose();
    }
    return newYbar;
  }

  // Shifts the node vector (roll by -1). The first step in Python is:
  //   us = node2u(Y)
  //   us <- roll us by -1, set last to 0
  //   Y <- u2node(us)
  Eigen::VectorXd shift(const Eigen::VectorXd &Y) const {
    // Convert node->u, roll, u->node
    Eigen::VectorXd us = node2u(Y);

    // shift by -1
    Eigen::VectorXd usShifted = us;
    for (int i = 0; i < us.size() - 1; i++) {
      usShifted[i] = us[i + 1];
    }
    usShifted[us.size() - 1] = 0.0; // set last to zero

    // back to node form
    Eigen::VectorXd Yout = u2node(usShifted);
    return Yout;
  }

  // Accessors
  int actionSize() const { return nu_; }
  const std::vector<double> &sigmas() const { return sigmas_; }
  const std::vector<double> &sigmaControl() const { return sigmaControl_; }
  const DialConfig &args() const { return args_; }

private:
  DialConfig args_;
  ToyEnv env_;
  int nu_;

  std::vector<double> sigmas_;        // diffusion scales
  std::vector<double> sigmaControl_;  // horizon-based diffusion scale

  std::vector<double> stepUs_;        // time points for fine control (Hsample+1)
  std::vector<double> stepNodes_;     // time points for nodes (Hnode+1)
  double ctrl_dt_;
};

/*****************************************************************************
 * MAIN
 * ----
 * Demonstration of the iterative procedure:
 *   1) Initialize environment & MBDPI
 *   2) For each rollout step:
 *       - Step environment with the first node's action
 *       - Shift the node vector
 *       - Perform a number of "reverse_once" calls to refine the plan
 *****************************************************************************/

int main() {
  // --------------------------------------------------------------------------
  // 1. Setup
  // --------------------------------------------------------------------------
  DialConfig config;
  // You may set config fields here if you want non-default:
  // config.Hnode = 5; config.Hsample = 20; ...

  ToyEnv env;
  MBDPI mbdpi(config, env);

  // Initialize random generator
  std::mt19937 gen(config.seed);

  // Initial environment state
  EnvState state = env.reset(config.seed);

  // Initialize the node vector (Y0) to all zeros
  // shape: (Hnode+1) x 1 (but we store as VectorXd)
  Eigen::VectorXd Y = Eigen::VectorXd::Zero(config.Hnode + 1);

  // --------------------------------------------------------------------------
  // 2. Rollout Loop
  //    For each iteration, we:
  //    - apply action = Y(0) in environment
  //    - shift Y
  //    - do Ndiffuse (or Ndiffuse_init) reverse passes
  // --------------------------------------------------------------------------
  double totalReward = 0.0;
  for (int step = 0; step < config.n_steps; step++) {
    // 2a) Step environment with the first node's action
    double action = Y(0);
    state = env.step(state, action);
    totalReward += state.reward;

    // 2b) Shift Y
    Y = mbdpi.shift(Y);

    // 2c) Number of diffuse passes
    int nDiff = (step == 0) ? config.Ndiffuse_init : config.Ndiffuse;

    // We can also incorporate the "sigmaControl" factor that the Python code
    // uses for each diffusion iteration. For simplicity, here we just do
    // sigma = sigmas_[i]*someFactor. You can replicate the loop with a scan
    // if desired. We'll do a simpler approach: all steps use the same scale
    // or a decaying scale.
    for (int i = 0; i < nDiff; i++) {
      // e.g. multiply by traj_diffuse_factor^(i)
      double factor = std::pow(config.traj_diffuse_factor, i);
      double noiseScale = mbdpi.sigmas()[std::min(i, (int)mbdpi.sigmas().size() - 1)]
                          * factor;
      Y = mbdpi.reverse_once(state, gen, Y, noiseScale);
    }

    std::cout << "[Step " << step << "] reward=" << state.reward << std::endl;
  }

  std::cout << "Total reward over " << config.n_steps << " steps = "
            << totalReward << std::endl;

  return 0;
}
