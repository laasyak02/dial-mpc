#include "dial-core.h"

//////////////////////////////////////////////////////////////
// Main
//////////////////////////////////////////////////////////////
int main()
{
  DialConfig cfg;
  cfg.seed = 0;
  cfg.Hsample = 25;
  cfg.Hnode = 5;
  cfg.Nsample = 2048;
  cfg.Ndiffuse = 2;
  cfg.Ndiffuse_init = 10;
  cfg.temp_sample = 0.05;
  cfg.n_steps = 400;
  cfg.ctrl_dt = 0.02;
  cfg.horizon_diffuse_factor = 0.9;
  cfg.traj_diffuse_factor = 0.5;

  UnitreeGo2EnvConfig go2_config;
  go2_config.kp = 30.0;
  go2_config.kd = 0.65;
  go2_config.action_scale = 1.0;
  go2_config.default_vx = 0.8;
  go2_config.default_vy = 0.0;
  go2_config.default_vyaw = 0.0;
  go2_config.ramp_up_time = 1.0;
  go2_config.gait = "stand";
  go2_config.timestep = 0.0025;
  go2_config.randomize_tasks = false;
  go2_config.leg_control = "torque";

  std::string model_path = "/home/quant/dial_mpc_ws/src/dial-mpc/models/unitree_go2/mjx_scene_force.xml";

  // 2) Create environment
  UnitreeGo2Env env(go2_config, model_path);

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
      std::tuple<Eigen::MatrixXd, MBDPI::ReverseInfo> res_reverse = mbdpi.reverse_once(next_state, rng, Y0, factor);
      Eigen::MatrixXd newY = std::get<0>(res_reverse);
      MBDPI::ReverseInfo info = std::get<1>(res_reverse);
      Y0 = newY;
      // info.rews is the distribution of sample mean rewards, you can log if desired.
    }

    cur_state = next_state;
  }

  // Summarize
  double sum_rew = 0.0;
  for (double r : rews)
    sum_rew += r;
  double avg_rew = sum_rew / (double)rews.size();
  std::cout << "Average reward = " << avg_rew << std::endl;

  return 0;
}
