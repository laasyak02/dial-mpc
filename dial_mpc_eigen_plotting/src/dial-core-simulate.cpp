#include "dial-core.h"
#include "mujoco-sim.h"
#include "gnuplot-iostream.h"

Gnuplot gp;

struct MultiBodyState
{
    Eigen::VectorXd qpos;
    Eigen::VectorXd qvel;
};

void loop(const mjModel *m, mjData *d);

void ComputeControlTrajectory(const DialConfig &config, UnitreeGo2Env &env);

Eigen::VectorXd act2joint(const Eigen::VectorXd &act, const mjModel *m, const mjData *d);

Eigen::VectorXd act2tau(const Eigen::VectorXd &act, const mjModel *m, const mjData *d);

MujocoEnvironment mjEnv(loop);

DialConfig cfg;
UnitreeGo2EnvConfig go2_config;

std::vector<Eigen::VectorXd> all_us;
std::vector<MultiBodyState> all_xs;

Eigen::Matrix<double, 12, 2> joint_range;
Eigen::Matrix<double, 12, 2> joint_torque_range;

bool controller_started = false;
double mj_start_time = 0.0;
double curr_time = 0.0;

//////////////////////////////////////////////////////////////
// Main
//////////////////////////////////////////////////////////////
int main()
{
    cfg.seed = 0;
    cfg.Hsample = 16;
    cfg.Hnode = 4;
    cfg.Nsample = 2048;
    cfg.Ndiffuse = 2;
    cfg.Ndiffuse_init = 10;
    cfg.temp_sample = 0.05;
    cfg.n_steps = 100;
    cfg.ctrl_dt = 0.02;
    cfg.horizon_diffuse_factor = 0.9;
    cfg.traj_diffuse_factor = 0.5;

    go2_config.kp = 30.0;
    go2_config.kd = 1.0;
    go2_config.action_scale = 1.0;
    go2_config.default_vx = 0.8;
    go2_config.default_vy = 0.0;
    go2_config.default_vyaw = 0.0;
    go2_config.ramp_up_time = 1.0;
    go2_config.gait = "stand";
    go2_config.timestep = 0.0025;
    go2_config.randomize_tasks = false;
    go2_config.leg_control = "torque";

    std::string model_path = "/home/laasya/dial-mpc-python/dial_mpc/models/unitree_go2/mjx_scene_force.xml";

    UnitreeGo2Env env(go2_config, model_path);

    joint_range = env.joint_range();
    joint_torque_range = env.joint_torque_range();

    ComputeControlTrajectory(cfg, env);

    mjEnv.Initialize(model_path);
    controller_started = true;
    mj_start_time = mjEnv.GetTime();
    mjEnv.Finalize();

    mjEnv.Loop();

    mjEnv.Exit();

    return 0;
}

void ComputeControlTrajectory(const DialConfig &config, UnitreeGo2Env &env)
{
    // Create MBDPI
    MBDPI mbdpi(config, env);

    std::mt19937_64 rng(config.seed);

    Eigen::MatrixXd state_vals;
    state_vals.resize(config.n_steps+1, 37);

    // Reset environment
    EnvState state_init = env.reset(rng);

    for (size_t i = 0; i < 19; ++i){
    	state_vals(0,i) = state_init.data->qpos[i];
    }
    for (size_t i = 19; i < 37; ++i){
    	state_vals(0,i) = state_init.data->qvel[i-19];
    }

    // YN = zeros( (Hnode+1, nu) )
    Eigen::MatrixXd YN = Eigen::MatrixXd::Zero(config.Hnode + 1, mbdpi.nu_);

    //    (Optional) initial reverse call, as in Python code sometimes:
    //    "Y0 = mbdpi.reverse(state_init, YN, rng_exp)"
    //    You can comment/uncomment as needed.
    YN = mbdpi.reverse(state_init, YN, rng);

    // Let Y0 = the result
    Eigen::MatrixXd Y0 = YN;

    //  Main rollout loop
    std::vector<double> rews;
    rews.reserve(config.n_steps);

    // std::vector<Eigen::VectorXd> all_us;
    all_us.reserve(config.n_steps);

    std::vector<MultiBodyState> all_xs;
    all_xs.reserve(config.n_steps);

    EnvState cur_state = state_init;

    MultiBodyState mbs;
    mbs.qpos = Eigen::VectorXd::Map(cur_state.data->qpos, cur_state.model->nq);
    mbs.qvel = Eigen::VectorXd::Map(cur_state.data->qvel, cur_state.model->nv);
    all_xs.push_back(mbs);

    for (int t = 0; t < config.n_steps; t++)
    {
        // Step environment with the first row of Y0
        Eigen::VectorXd action = Y0.row(0);
        std::cout<<"Step: "<< t <<std::endl;

        EnvState next_state = env.step(cur_state, action);

        // std::cout<<"Afer calling step once "<<std::endl;
        std::cout<<"cur_state: "<<std::endl;
        std::cout << "randomize_target: " << (cur_state.info.randomize_target ? "true" : "false") << std::endl;
        std::cout<<"Target Position: "<< cur_state.info.pos_tar[0]<<","<< cur_state.info.pos_tar[1]<<","<< cur_state.info.pos_tar[2]<<std::endl;
        std::cout<<"Current Position: "<<cur_state.data->qpos[0]<<","<<cur_state.data->qpos[1]<<","<<cur_state.data->qpos[2]<<std::endl;
        std::cout<<"Target Velocity: "<< cur_state.info.vel_tar[0]<<","<< cur_state.info.vel_tar[1]<<","<< cur_state.info.vel_tar[2]<<std::endl;
        std::cout<<"Current Velocity: "<<cur_state.data->qvel[0]<<","<<cur_state.data->qvel[1]<<","<<cur_state.data->qvel[2]<<std::endl;
        std::cout << "Target Angular Velocity: " << cur_state.info.ang_vel_tar.transpose() << std::endl;
        std::cout << "Target Yaw: " << cur_state.info.yaw_tar << std::endl;
        // std::cout<<"Target Foot Heights: "<< cur_state.info.z_feet_tar[0]<<","<< cur_state.info.z_feet_tar[1]<<","<< cur_state.info.z_feet_tar[2]<<","<< cur_state.info.z_feet_tar[3]<<std::endl;
        // std::cout<<"Current Foot Heights: "<< cur_state.info.z_feet[0]<<","<< cur_state.info.z_feet[1]<<","<< cur_state.info.z_feet[2]<<","<< cur_state.info.z_feet[3]<<std::endl;
        // std::cout<<"Last Contact: "<< cur_state.info.last_contact[0]<<","<< cur_state.info.last_contact[1]<<","<< cur_state.info.last_contact[2]<<","<< cur_state.info.last_contact[3]<<std::endl;
        // std::cout << "Feet Air Time: " << cur_state.info.feet_air_time.transpose() << std::endl;
        
        std::cout << std::endl;
        std::cout<<"Action: \n"<<action<<std::endl;
        std::cout << std::endl;

        std::cout<<"next_state: "<<std::endl;
        std::cout << "randomize_target: " << (next_state.info.randomize_target ? "true" : "false") << std::endl;
        std::cout<<"Target Position: "<< next_state.info.pos_tar[0]<<","<< next_state.info.pos_tar[1]<<","<< next_state.info.pos_tar[2]<<std::endl;
        std::cout<<"Current Position: "<<next_state.data->qpos[0]<<","<<next_state.data->qpos[1]<<","<<next_state.data->qpos[2]<<std::endl;
        std::cout<<"Target Velocity: "<< next_state.info.vel_tar[0]<<","<< next_state.info.vel_tar[1]<<","<< next_state.info.vel_tar[2]<<std::endl;
        std::cout<<"Current Velocity: "<<next_state.data->qvel[0]<<","<<next_state.data->qvel[1]<<","<<next_state.data->qvel[2]<<std::endl;
        std::cout << "Target Angular Velocity: " << next_state.info.ang_vel_tar.transpose() << std::endl;
        std::cout << "Target Yaw: " << next_state.info.yaw_tar << std::endl;
        // std::cout<<"Target Foot Heights: "<< next_state.info.z_feet_tar[0]<<","<< next_state.info.z_feet_tar[1]<<","<< next_state.info.z_feet_tar[2]<<","<< next_state.info.z_feet_tar[3]<<std::endl;
        // std::cout<<"Current Foot Heights: "<< next_state.info.z_feet[0]<<","<< next_state.info.z_feet[1]<<","<< next_state.info.z_feet[2]<<","<< next_state.info.z_feet[3]<<std::endl;
        // std::cout<<"Last Contact: "<< next_state.info.last_contact[0]<<","<< next_state.info.last_contact[1]<<","<< next_state.info.last_contact[2]<<","<< next_state.info.last_contact[3]<<std::endl;
        // std::cout << "Feet Air Time: " << next_state.info.feet_air_time.transpose() << std::endl;

        // std::cin.get();

        rews.push_back(next_state.reward);
        mbs.qpos = Eigen::VectorXd::Map(next_state.data->qpos, next_state.model->nq);
        mbs.qvel = Eigen::VectorXd::Map(next_state.data->qvel, next_state.model->nv);
        all_xs.push_back(mbs);
        all_us.push_back(action);

        // shift Y0
        Y0 = mbdpi.shift(Y0);

        // how many times to do "reverse_once"?
        int n_diffuse = (t == 0) ? config.Ndiffuse_init : config.Ndiffuse;

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
            std::tuple<Eigen::MatrixXd, MBDPI::ReverseInfo> res_reverse = mbdpi.reverse_once(next_state, rng, Y0, factor);
            Eigen::MatrixXd newY = std::get<0>(res_reverse);
            MBDPI::ReverseInfo info = std::get<1>(res_reverse);
            Y0 = newY;
            // info.rews is the distribution of sample mean rewards, you can log if desired.
        }

        cur_state = next_state;

        for (size_t i = 0; i < 19; ++i){
    	    state_vals(t+1,i) = cur_state.data->qpos[i];
        }
        for (size_t i = 19; i < 37; ++i){
    	    state_vals(t+1,i) = cur_state.data->qvel[i-19];
        }
    	
    }

    std::vector<std::vector<double>> pos_data;
    for (size_t i = 0; i < 19; ++i) {
    	Eigen::VectorXd column = state_vals.col(i);

        // Convert Eigen::VectorXd to std::vector
        std::vector<double> col_vector(column.data(), column.data() + column.size());
    	pos_data.push_back(col_vector);
	//plt::plot(pos_data, {{"label", "Position " + std::to_string(i)}});
    }
    
    std::vector<std::vector<double>> vel_data;
    for (size_t i = 19; i < 37; ++i) {
    	Eigen::VectorXd column = state_vals.col(i);

        // Convert Eigen::VectorXd to std::vector
        std::vector<double> col_vector(column.data(), column.data() + column.size());
    	vel_data.push_back(col_vector);
	//plt::plot(pos_data, {{"label", "Position " + std::to_string(i)}});
    }

    gp << "set title 'Graph 1: Base Position (x,y,z)'\n";
    gp << "set term wxt 0\n";
    gp << "plot";
    
    for (size_t i = 0; i < 3; ++i) {
        gp << " '-' with lines title 'Position " << i << "'";
        if (i < 3 - 1) gp << ",";
    }
    gp << std::endl;

    for (size_t i = 0; i < 3; ++i) {
    	gp.send(pos_data[i]);
    }

    gp << "set title 'Graph 5: Base Orientation '\n";
    gp << "set term wxt 4\n";
    gp << "plot";
    
    for (size_t i = 3; i < 7; ++i) {
        gp << " '-' with lines title 'Orientation " << i-3 << "'";
        if (i < 7 - 1) gp << ",";
    }
    gp << std::endl;

    for (size_t i = 3; i < 7; ++i) {
    	gp.send(pos_data[i]);
    }

    gp << "set title 'Graph 2: Joints Position'\n";
    gp << "set term wxt 1\n";
    gp << "plot";
    
    for (size_t i = 7; i < pos_data.size(); ++i) {
        gp << " '-' with lines title 'Joint " << i-6 << "'";
        if (i < pos_data.size() - 1) gp << ",";
    }
    gp << std::endl;
    
    for (size_t i = 7; i < pos_data.size(); ++i) {
    	gp.send(pos_data[i]);
    }
    
    gp << "set title 'Graph 3: Base Velocity'\n";
    gp << "set term wxt 2\n";
    gp << "plot";
    
    for (size_t i = 0; i < 3; ++i) {
        gp << " '-' with lines title 'Velocity " << i << "'";
        if (i < 3 - 1) gp << ",";
    }
    gp << std::endl;
    
    for (size_t i = 0; i < 3; ++i) {
    	gp.send(vel_data[i]);
    }

    gp << "set title 'Graph 6: Base Angular Velocity'\n";
    gp << "set term wxt 5\n";
    gp << "plot";
    
    for (size_t i = 3; i < 6; ++i) {
        gp << " '-' with lines title 'Angular Velocity " << i-3 << "'";
        if (i < 6 - 1) gp << ",";
    }
    gp << std::endl;
    
    for (size_t i = 3; i < 6; ++i) {
    	gp.send(vel_data[i]);
    }
    
    gp << "set title 'Graph 4: Joints Velocity'\n";
    gp << "set term wxt 3\n";
    gp << "plot";
    
    for (size_t i = 6; i < vel_data.size(); ++i) {
        gp << " '-' with lines title 'Joint " << i-5 << "'";
        if (i < vel_data.size() - 1) gp << ",";
    }
    gp << std::endl;
    
    for (size_t i = 6; i < vel_data.size(); ++i) {
    	gp.send(vel_data[i]);
    }

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
        jt = clamp(jt, p_low, p_high);

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
        tau[i] = clamp(val, tmin, tmax);
    }
    return tau;
}