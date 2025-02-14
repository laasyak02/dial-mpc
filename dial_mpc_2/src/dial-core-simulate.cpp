#include "dial-core.h"
#include "mujoco-sim.h"

struct MultiBodyState
{
    Eigen::VectorXd qpos;
    Eigen::VectorXd qvel;
};

void loop(const mjModel *m, mjData *d);

template <int NUMSAMPLES>
void ComputeControlTrajectory(const dial::DialConfig &config, go2env::UnitreeGo2Env<NUMSAMPLES + 1> &env);

Eigen::VectorXd act2joint(const Eigen::VectorXd &act, const mjModel *m, const mjData *d);

Eigen::VectorXd act2tau(const Eigen::VectorXd &act, const mjModel *m, const mjData *d);

MujocoEnvironment mjEnv(loop);

dial::DialConfig cfg;
go2env::UnitreeGo2EnvConfig go2_config;

std::vector<Eigen::VectorXd> all_us;
std::vector<go2env::EnvState> all_states;

Eigen::Matrix<double, 12, 2> joint_range;
Eigen::Matrix<double, 12, 2> joint_torque_range;

bool controller_started = false;
double mj_start_time = 0.0;
double curr_time = 0.0;

static const int NUMBER_OF_SAMPLES = 2048;

//////////////////////////////////////////////////////////////
// Main
//////////////////////////////////////////////////////////////
int main()
{
    cfg.seed = 0;
    cfg.Hsample = 16;
    cfg.Hnode = 4;
    // cfg.Nsample = 2048; // added as a template parameter instead
    cfg.Ndiffuse = 2;
    cfg.Ndiffuse_init = 10;
    cfg.temp_sample = 0.05;
    cfg.n_steps = 200;
    cfg.ctrl_dt = 0.02;
    cfg.horizon_diffuse_factor = 0.9;
    cfg.traj_diffuse_factor = 0.5;

    go2_config.kp = 30.0;
    go2_config.kd = 0.0;
    go2_config.action_scale = 1.0;
    go2_config.default_vx = 0.0;
    go2_config.default_vy = 0.0;
    go2_config.default_vyaw = 0.0;
    go2_config.ramp_up_time = 1.0;
    go2_config.gait = "stand";
    go2_config.timestep = 0.02;

    const std::string model_path = "/home/laasya/dial-mpc-python/dial_mpc/models/unitree_go2/mjx_scene_force.xml";
    // const std::string model_path = "/home/quant/dial_mpc_ws/src/dial-mpc/models/unitree_go2/mjx_scene_force.xml";
    
    go2env::UnitreeGo2Env<NUMBER_OF_SAMPLES + 1> env(go2_config, model_path);

    joint_range = env.joint_range();
    joint_torque_range = env.joint_torque_range();

    auto start_time = std::chrono::high_resolution_clock::now();
    ComputeControlTrajectory<NUMBER_OF_SAMPLES>(cfg, env);
    std::cout << "ComputeControlTrajectory time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

    mjEnv.Initialize(model_path);
    controller_started = true;

    mjModel *m = mjEnv.GetModel();
    mjData *d = mjEnv.GetData();

    m->opt.timestep = go2_config.timestep;

    int home_id = mj_name2id(m, mjOBJ_KEY, "home");

    for (size_t i = 0; i <m->nq; i++)
    {
        d->qpos[i] =m->key_qpos[home_id * m->nq + i];
    }
    for (size_t i = 7; i < m->nq; i++)
    {
        d->qpos[i] = m->key_qpos[home_id * m->nq + i];
    }

    mj_start_time = d->time;

    mjEnv.Loop();

    mjEnv.Exit();

    return 0;
}

template <int NUMSAMPLES>
void ComputeControlTrajectory(const dial::DialConfig &config, go2env::UnitreeGo2Env<NUMSAMPLES + 1> &env)
{
    // Create MBDPI
    dial::MBDPI<NUMSAMPLES> mbdpi(config, env);

    std::mt19937_64 rng(config.seed);

    // Reset environment
    go2env::EnvState state_init = env.reset(rng);

    Eigen::MatrixXd YN = Eigen::MatrixXd::Zero(config.Hnode + 1, mbdpi.nu_); // (Hnode+1, nu)
    // YN = mbdpi.reverse(state_init, YN, rng);

    Eigen::MatrixXd Y0 = YN;

    //  Main rollout loop
    std::vector<double> rews;
    rews.reserve(config.n_steps);

    all_us.reserve(config.n_steps);
    all_states.reserve(config.n_steps);

    go2env::EnvState cur_state = state_init;
    all_states.push_back(cur_state);

    auto start_time = std::chrono::high_resolution_clock::now();

    double avg_t_env_step = 0.0;
    double avg_t_shift = 0.0;
    double avg_t_reverse_once = 0.0;
    for (int t = 0; t < config.n_steps; t++)
    {
        // Step environment with the first row of Y0
        Eigen::VectorXd action = Y0.row(0);

        auto start_time_env_step = std::chrono::high_resolution_clock::now();
        go2env::EnvState next_state = env.step(cur_state, action);
        avg_t_env_step += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time_env_step).count();

        /*
        std::cout<<"cur_state: "<<std::endl;
        // std::cout << "randomize_target: " << (cur_state.info.randomize_target ? "true" : "false") << std::endl;
        std::cout<<"Target Position: "<< cur_state.info.pos_tar[0]<<","<< cur_state.info.pos_tar[1]<<","<< cur_state.info.pos_tar[2]<<std::endl;
        // std::cout<<"Current Position: "<<cur_state.data->qpos[0]<<","<<cur_state.data->qpos[1]<<","<<cur_state.data->qpos[2]<<std::endl;
        std::cout<<"Target Velocity: "<< cur_state.info.vel_tar[0]<<","<< cur_state.info.vel_tar[1]<<","<< cur_state.info.vel_tar[2]<<std::endl;
        // std::cout<<"Current Velocity: "<<cur_state.data->qvel[0]<<","<<cur_state.data->qvel[1]<<","<<cur_state.data->qvel[2]<<std::endl;
        std::cout << "Target Angular Velocity: " << cur_state.info.ang_vel_tar.transpose() << std::endl;
        std::cout << "Target Yaw: " << cur_state.info.yaw_tar << std::endl;
        std::cout<<"Target Foot Heights: "<< cur_state.info.z_feet_tar[0]<<","<< cur_state.info.z_feet_tar[1]<<","<< cur_state.info.z_feet_tar[2]<<","<< cur_state.info.z_feet_tar[3]<<std::endl;
        std::cout<<"Current Foot Heights: "<< cur_state.info.z_feet[0]<<","<< cur_state.info.z_feet[1]<<","<< cur_state.info.z_feet[2]<<","<< cur_state.info.z_feet[3]<<std::endl;
        std::cout<<"Last Contact: "<< cur_state.info.last_contact[0]<<","<< cur_state.info.last_contact[1]<<","<< cur_state.info.last_contact[2]<<","<< cur_state.info.last_contact[3]<<std::endl;
        std::cout << "Feet Air Time: ";
        for (const auto& val : cur_state.info.feet_air_time) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
        
        std::cout << std::endl;
        std::cout<<"Action: \n"<<action<<std::endl;
        std::cout << std::endl;

        std::cout<<"next_state: "<<std::endl;
        // std::cout << "randomize_target: " << (next_state.info.randomize_target ? "true" : "false") << std::endl;
        std::cout<<"Target Position: "<< next_state.info.pos_tar[0]<<","<< next_state.info.pos_tar[1]<<","<< next_state.info.pos_tar[2]<<std::endl;
        // std::cout<<"Current Position: "<<next_state.data->qpos[0]<<","<<next_state.data->qpos[1]<<","<<next_state.data->qpos[2]<<std::endl;
        std::cout<<"Target Velocity: "<< next_state.info.vel_tar[0]<<","<< next_state.info.vel_tar[1]<<","<< next_state.info.vel_tar[2]<<std::endl;
        // std::cout<<"Current Velocity: "<<next_state.data->qvel[0]<<","<<next_state.data->qvel[1]<<","<<next_state.data->qvel[2]<<std::endl;
        std::cout << "Target Angular Velocity: " << next_state.info.ang_vel_tar.transpose() << std::endl;
        std::cout << "Target Yaw: " << next_state.info.yaw_tar << std::endl;
        std::cout<<"Target Foot Heights: "<< next_state.info.z_feet_tar[0]<<","<< next_state.info.z_feet_tar[1]<<","<< next_state.info.z_feet_tar[2]<<","<< next_state.info.z_feet_tar[3]<<std::endl;
        std::cout<<"Current Foot Heights: "<< next_state.info.z_feet[0]<<","<< next_state.info.z_feet[1]<<","<< next_state.info.z_feet[2]<<","<< next_state.info.z_feet[3]<<std::endl;
        std::cout<<"Last Contact: "<< next_state.info.last_contact[0]<<","<< next_state.info.last_contact[1]<<","<< next_state.info.last_contact[2]<<","<< next_state.info.last_contact[3]<<std::endl;
        // std::cout << "Feet Air Time: " << next_state.info.feet_air_time.transpose() << std::endl;
        std::cout << "Feet Air Time: ";
        for (const auto& val : next_state.info.feet_air_time) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        std::cin.get();
        */
        
        rews.push_back(next_state.reward);
        all_us.push_back(action);
        all_states.push_back(next_state);

        // shift Y0
        auto start_time_shift = std::chrono::high_resolution_clock::now();
        Y0 = mbdpi.shift(Y0);
        avg_t_shift += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time_shift).count();

        // how many times to do "reverse_once"?
        int n_diffuse = (t == 0) ? config.Ndiffuse_init : config.Ndiffuse;

        // std::cout<<"Initial Y0 going in: \n" << Y0 << std::endl;

        // Python code builds:
        //   traj_diffuse_factors = sigma_control * (traj_diffuse_factor^arange(n_diffuse))[:,None]
        //   Then a lax.scan over (rng, Y0, state)
        // We'll replicate that with a for-loop:
        for (int i = 0; i < n_diffuse; i++)
        {
            // factor for iteration i
            // shape = (Hnode+1,)
            Eigen::VectorXd factor(config.Hnode + 1);
            for (int h = 0; h <= config.Hnode; h++)
            {
                factor(h) = mbdpi.sigma_control_(h) * std::pow(config.traj_diffuse_factor, (double)i);
            }
            // call reverse_once

            auto start_time_reverse_once = std::chrono::high_resolution_clock::now();
            std::tuple<Eigen::MatrixXd, dial::ReverseInfo> res_reverse = mbdpi.reverse_once(next_state, rng, Y0, factor);
            avg_t_reverse_once += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time_reverse_once).count();

            Eigen::MatrixXd newY = std::get<0>(res_reverse);
            dial::ReverseInfo info = std::get<1>(res_reverse);
            Y0 = newY;
            // info.rews is the distribution of sample mean rewards, you can log if desired.
            /*
            std::cout << "Factor: \n" << factor << std::endl;
            std::cout << "Y0: \n" << Y0 << std::endl;
            std::cout << "Info: \n";
            std::cout << "Size of rewards" << info.rews.size() <<std::endl;
            std::cout << "rewards: " << info.rews << std::endl;
            std::cout << "new_noise_scale: " << info.new_noise_scale<< std::endl;
            std::cout << "qbar: " << info.qbar << std::endl;
            std::cout << "qdbar: " << info.qdbar<< std::endl;
            std::cout << "xbar: " << info.xbar << std::endl;
            std::cin.get();
            */
        }
        std::cout << "Step " << t << " done" << std::endl;
        cur_state = next_state;
    }
    std::cout << "Average time per environment step: " << avg_t_env_step / (double)config.n_steps << "us" << std::endl;
    std::cout << "Average time per shift: " << avg_t_shift / (double)config.n_steps << "us" << std::endl;
    std::cout << "Average time per reverse_once: " << avg_t_reverse_once / (double)config.n_steps << "us" << std::endl;
    std::cout << "Average time per mpc step: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() / (double)config.n_steps << "ms" << std::endl;

    // Summarize
    double sum_rew = 0.0;
    for (double r : rews)
        sum_rew += r;
    double avg_rew = sum_rew / (double)rews.size();
    std::cout << "Average reward = " << avg_rew << std::endl;

    // Reset environment so that the mujoco model and data are what they were
    env.reset(rng);
}

void loop(const mjModel *m, mjData *d)
{
    if (!controller_started)
    {
        return;
    }

    // Use a PD controller and feedforward to track the all_us

    // Get the current state
    MultiBodyState mbs;
    mbs.qpos = Eigen::VectorXd::Map(d->qpos, m->nq);
    mbs.qvel = Eigen::VectorXd::Map(d->qvel, m->nv);

    // The interval between control steps
    double ctrl_dt = cfg.ctrl_dt;

    // Get the time
    double time_elapsed = d->time - mj_start_time;

    // Get the control action based on the time elapsed
    int idx = (int)(time_elapsed / ctrl_dt);
    if (idx >= all_us.size())
    {
        // Done
        mjEnv.Exit();
        return;
    }

    Eigen::VectorXd action = all_us[idx];

    // Convert action to torque
    Eigen::VectorXd tau = act2tau(action, m, d);

    std::cout << "tau = " << tau.transpose() << std::endl;

    // Set the control
    for (int i = 0; i < m->nu; i++)
    {
        d->ctrl[i] = tau[i];
    }
}

Eigen::VectorXd act2joint(const Eigen::VectorXd &act, const mjModel *m, const mjData *d)
{
    // 1) act_normalized = (act * action_scale + 1.0)/2.0
    // 2) joint_targets = joint_range_[:,0] + ...
    // 3) clip to physical_joint_range_

    Eigen::VectorXd result(m->nu);
    for (int i = 0; i < m->nu; i++)
    {
        double a = act[i] * go2_config.action_scale + 1.0;
        double act_normalized = a * 0.5; // => range [0..1], if act in [-1..1]
        // scale to joint_range
        double low = joint_range(i, 0);
        double high = joint_range(i, 1);
        double jt = low + act_normalized * (high - low);

        // clip to physical_joint_range_ (same in our case)
        double p_low = joint_range(i, 0);
        double p_high = joint_range(i, 1);
        jt = go2env::clamp(jt, p_low, p_high);

        result[i] = jt;
    }
    return result;
}

Eigen::VectorXd act2tau(const Eigen::VectorXd &act, const mjModel *m, const mjData *d)
{
    Eigen::VectorXd joint_target = act2joint(act, m, d);

    Eigen::VectorXd q(m->nq - 7);
    Eigen::VectorXd qd(m->nv - 6);
    for (int i = 0; i < m->nq - 7; i++)
    {
        q[i] = d->qpos[7 + i];
    }
    for (int i = 0; i < m->nv - 6; i++)
    {
        qd[i] = d->qvel[6 + i];
    }

    // PD
    Eigen::VectorXd q_err = joint_target - q;
    Eigen::VectorXd tau(m->nu);
    for (int i = 0; i < m->nu; i++)
    {
        double val = go2_config.kp * q_err[i] - go2_config.kd * qd[i];
        // clip to joint_torque_range_
        double tmin = joint_torque_range(i, 0);
        double tmax = joint_torque_range(i, 1);
        tau[i] = go2env::clamp(val, tmin, tmax);
    }
    return tau;
}