#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include <Eigen/Dense>
#include <random>
#include <string>
#include <vector>
#include <map>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <cmath>
#include <cassert>

//-----------------------------------------------------
// Configuration structure similar to UnitreeGo2EnvConfig
//-----------------------------------------------------
struct UnitreeGo2EnvConfig
{
    double kp = 30.0;
    double kd = 0.0;
    double action_scale = 1.0; // newly added to match 'act2joint' scale

    double default_vx = 1.0;
    double default_vy = 0.0;
    double default_vyaw = 0.0;
    double ramp_up_time = 2.0;
    std::string gait = "trot";
    double timestep = 0.0025;

    // If true, the environment will randomize targets
    bool randomize_tasks = false;

    // Leg control mode: "position" or "torque"
    std::string leg_control = "torque";
};

//-----------------------------------------------------
// A structure to store environment state info
//-----------------------------------------------------
struct StateInfo
{
    Eigen::VectorXd pos_tar;     // (3,)
    Eigen::VectorXd vel_tar;     // (3,)
    Eigen::VectorXd ang_vel_tar; // (3,)
    double yaw_tar;
    int step;
    Eigen::VectorXd z_feet;     // (4,) foot heights
    Eigen::VectorXd z_feet_tar; // (4,) desired foot heights
    bool randomize_target;
    Eigen::VectorXi last_contact;  // (4,) boolean in 0/1
    Eigen::VectorXd feet_air_time; // (4,) how long each foot is in air
};

//-----------------------------------------------------
// A structure to hold the entire env step state
//-----------------------------------------------------
struct EnvState
{
    mjModel *model = nullptr;
    mjData *data = nullptr;

    Eigen::VectorXd obs; // observation
    double reward;       // reward
    bool done;           // termination flag
    StateInfo info;
};

//-----------------------------------------------------
// Helper: clamp
//-----------------------------------------------------
inline double clamp(double x, double low, double high)
{
    return (x < low) ? low : ((x > high) ? high : x);
}

//-----------------------------------------------------
// Helper: replicate the modulo-then-shift logic
//-----------------------------------------------------
inline double angleMod(double x, double modulus)
{
    // replicate (x % modulus) in a floating sense
    double y = std::fmod(x, modulus);
    if (y < 0)
    {
        y += modulus;
    }
    return y;
}

//-----------------------------------------------------
// Helper for getFootStep: step_height(t, footphase, duty_ratio)
// exactly replicates the Python logic
//-----------------------------------------------------
double step_height(double t, double footphase, double duty_ratio)
{
    // angle = (t + pi - footphase) % (2 pi) - pi
    double raw = angleMod(t + M_PI - footphase, 2.0 * M_PI);
    double angle = raw - M_PI;

    // angle = jnp.where(duty_ratio < 1, angle * 0.5 / (1 - duty_ratio), angle)
    // i.e., if (duty_ratio < 1) => angle *= 0.5 / (1 - duty_ratio)
    if (duty_ratio < 1.0)
    {
        angle *= (0.5 / (1.0 - duty_ratio));
    }

    // clipped_angle = clip(angle, -pi/2, pi/2)
    double clipped_angle = clamp(angle, -M_PI / 2.0, M_PI / 2.0);

    // value = jnp.where(duty_ratio < 1, cos(clipped_angle), 0)
    double value = 0.0;
    if (duty_ratio < 1.0)
    {
        value = std::cos(clipped_angle);
    }

    // final_value = jnp.where(abs(value) >= 1e-6, abs(value), 0.0)
    double final_value = 0.0;
    if (std::fabs(value) >= 1e-6)
    {
        final_value = std::fabs(value);
    }
    return final_value;
}

//-----------------------------------------------------
// get_foot_step(duty_ratio, cadence, amplitude, phases, time)
// Exactly replicates the Python version
//-----------------------------------------------------
Eigen::VectorXd getFootStep(double duty_ratio, double cadence, double amplitude,
                            const Eigen::VectorXd &phases, double time)
{
    // phases is assumed to be length 4 (or N for N legs).
    // h_steps = amplitude * vmap(step_height) over each foot-phase
    // but we do a for loop in C++.
    int n = phases.size();
    Eigen::VectorXd h_steps(n);
    // t in Python = time * 2*pi*cadence + pi
    double T = time * 2.0 * M_PI * cadence + M_PI;
    for (int i = 0; i < n; ++i)
    {
        double footphase = 2.0 * M_PI * phases[i];
        double val = step_height(T, footphase, duty_ratio);
        h_steps[i] = amplitude * val;
    }
    return h_steps;
}

//-----------------------------------------------------
// Convert a quaternion to a 3x3 rotation matrix
//-----------------------------------------------------
Eigen::Matrix3d quatToMat(const double *quat)
{
    // MuJoCo quaternion layout: w, x, y, z
    Eigen::Quaterniond q(quat[0], quat[1], quat[2], quat[3]);
    return q.toRotationMatrix();
}

//-----------------------------------------------------
// Convert quaternion to yaw angle (Z euler)
//-----------------------------------------------------
double quatToYaw(const double *quat)
{
    Eigen::Quaterniond q(quat[0], quat[1], quat[2], quat[3]);
    // eulerAngles(2,1,0) returns (Z, Y, X)
    Eigen::Vector3d euler = q.toRotationMatrix().eulerAngles(2, 1, 0);
    double yaw = euler[0];
    return yaw;
}

//-----------------------------------------------------
// Rotate a global velocity into body frame: v_body = R^T * v_global
//-----------------------------------------------------
Eigen::Vector3d globalToBodyVelocity(const Eigen::Vector3d &vel_global,
                                     const double *body_quat)
{
    Eigen::Matrix3d R = quatToMat(body_quat);
    return R.transpose() * vel_global;
}

//-----------------------------------------------------
// Pipeline initialization
//-----------------------------------------------------
void pipeline_init(mjModel *m, mjData *d,
                   const Eigen::VectorXd &init_q,
                   const Eigen::VectorXd &init_dq)
{
    // set qpos
    for (int i = 0; i < init_q.size(); ++i)
    {
        d->qpos[i] = init_q[i];
    }
    // set qvel
    for (int i = 0; i < init_dq.size(); ++i)
    {
        d->qvel[i] = init_dq[i];
    }
    mj_forward(m, d);
}

//-----------------------------------------------------
// Pipeline step
//-----------------------------------------------------
void pipeline_step(mjModel *m, mjData *d, const Eigen::VectorXd &ctrl)
{
    for (int i = 0; i < ctrl.size() && i < m->nu; ++i)
    {
        d->ctrl[i] = ctrl[i];
    }
    mj_step(m, d);
}

//-----------------------------------------------------
// The environment class
//-----------------------------------------------------
class UnitreeGo2Env
{
public:
    UnitreeGo2Env(const UnitreeGo2EnvConfig &config, const std::string &model_path)
        : config_(config)
    {
        char error[1000] = "Could not load XML model";
        m_ = mj_loadXML(model_path.c_str(), nullptr, error, 1000);
        if (!m_)
        {
            std::cerr << "Load model error: " << error << std::endl;
            std::exit(1);
        }
        d_ = mj_makeData(m_);

        // set global timestep
        m_->opt.timestep = config_.timestep;

        // Identify "base" body index
        torso_idx_ = mj_name2id(m_, mjOBJ_BODY, "base");
        if (torso_idx_ < 0)
        {
            std::cerr << "[Warning] body 'base' not found in model." << std::endl;
            torso_idx_ = 0; // fallback
        }

        // Identify feet site indices
        std::vector<std::string> feet_site_names = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"};
        for (auto &site_name : feet_site_names)
        {
            int sid = mj_name2id(m_, mjOBJ_SITE, site_name.c_str());
            if (sid < 0)
            {
                std::cerr << "[Warning] Site not found: " << site_name << std::endl;
            }
            feet_site_id_.push_back(sid);
        }

        // Attempt to find keyframe "home"
        int home_id = mj_name2id(m_, mjOBJ_KEY, "home");
        init_q_ = Eigen::VectorXd::Zero(m_->nq);
        default_pose_ = Eigen::VectorXd::Zero(m_->nq - 7);
        if (home_id < 0)
        {
            std::cerr << "Keyframe 'home' not found; defaulting qpos=0" << std::endl;
        }
        else
        {
            for (int i = 0; i < m_->nq; i++)
            {
                init_q_[i] = m_->key_qpos[home_id * m_->nq + i];
            }
            for (int i = 7; i < m_->nq; i++)
            {
                default_pose_[i - 7] = m_->key_qpos[home_id * m_->nq + i];
            }
        }

        // Example joint_range_ from Python snippet (12x2).
        joint_range_ = Eigen::MatrixXd(12, 2);
        joint_range_ << -0.5, 0.5,
            0.4, 1.4,
            -2.3, -0.85,
            -0.5, 0.5,
            0.4, 1.4,
            -2.3, -0.85,
            -0.5, 0.5,
            0.4, 1.4,
            -2.3, -1.3,
            -0.5, 0.5,
            0.4, 1.4,
            -2.3, -1.3;

        // We do NOT know the actual physical_joint_range_ from your code;
        // as an example, set them same as joint_range_:
        physical_joint_range_ = joint_range_;

        // We also define a placeholder for joint_torque_range_ (12x2),
        // e.g. min torque = -50, max torque = 50 for each joint:
        joint_torque_range_ = Eigen::MatrixXd(12, 2);
        joint_torque_range_.col(0).setConstant(-50.0);
        joint_torque_range_.col(1).setConstant(50.0);

        foot_radius_ = 0.0175;

        // Populate the known gait phases/params:
        setupGaitTables();

        if (!kGaitPhases_.count(config_.gait))
        {
            std::cerr << "Gait not recognized: " << config_.gait << std::endl;
            gait_ = "trot";
        }
        else
        {
            gait_ = config_.gait;
        }
    }

    ~UnitreeGo2Env()
    {
        if (d_)
            mj_deleteData(d_);
        if (m_)
            mj_deleteModel(m_);
    }

    // -----------------------------------
    // reset the environment
    // -----------------------------------
    EnvState reset(std::mt19937 &rng_engine)
    {
        mj_resetData(m_, d_);
        Eigen::VectorXd zero_dq(m_->nv);
        zero_dq.setZero();
        pipeline_init(m_, d_, init_q_, zero_dq);

        EnvState s;
        s.model = m_;
        s.data = d_;

        // Fill StateInfo
        s.info.pos_tar = Eigen::Vector3d(0.282, 0.0, 0.3);
        s.info.vel_tar = Eigen::Vector3d(0.0, 0.0, 0.0);
        s.info.ang_vel_tar = Eigen::Vector3d(0.0, 0.0, 0.0);
        s.info.yaw_tar = 0.0;
        s.info.step = 0;
        s.info.z_feet = Eigen::VectorXd::Zero(4);
        s.info.z_feet_tar = Eigen::VectorXd::Zero(4);
        s.info.randomize_target = config_.randomize_tasks;
        s.info.last_contact = Eigen::VectorXi::Zero(4);
        s.info.feet_air_time = Eigen::VectorXd::Zero(4);

        Eigen::VectorXd ctrl_init = Eigen::VectorXd::Zero(m_->nu);
        s.obs = get_obs(s.info, ctrl_init);
        s.reward = 0.0;
        s.done = false;

        return s;
    }

    // -----------------------------------
    // step the environment
    // -----------------------------------
    EnvState step(EnvState state, const Eigen::VectorXd &action, std::mt19937 &rng_engine)
    {
        // 1) Convert action to position or torque
        Eigen::VectorXd ctrl;
        if (config_.leg_control == "position")
        {
            ctrl = act2joint(action);
        }
        else
        { // torque
            ctrl = act2tau(action);
        }

        // 2) pipeline_step
        pipeline_step(m_, d_, ctrl);

        // 3) gather new obs
        Eigen::VectorXd obs = get_obs(state.info, ctrl);

        // Possibly randomize velocity commands every 500 steps
        if (state.info.randomize_target && (state.info.step % 500 == 0))
        {
            auto cmds = sample_command(rng_engine);
            state.info.vel_tar = cmds.first;
            state.info.ang_vel_tar = cmds.second;
        }
        else
        {
            // Ramp up to default velocities
            double scaled = std::min(config_.default_vx * (state.info.step * dt()) / config_.ramp_up_time,
                                     config_.default_vx);
            state.info.vel_tar[0] = scaled;
            scaled = std::min(config_.default_vy * (state.info.step * dt()) / config_.ramp_up_time,
                              config_.default_vy);
            state.info.vel_tar[1] = scaled;
            double yaw_scaled = std::min(config_.default_vyaw * (state.info.step * dt()) / config_.ramp_up_time,
                                         config_.default_vyaw);
            state.info.ang_vel_tar[2] = yaw_scaled;
        }

        // 4) compute reward
        double reward = computeReward(state.info, ctrl);

        // 5) check done
        bool done = checkTermination();

        // 6) update state info (z_feet, contact, etc.)
        state.info.step += 1;
        // site z positions:
        Eigen::Vector4d z_feet;
        for (int i = 0; i < 4; i++)
        {
            z_feet[i] = d_->site_xpos[feet_site_id_[i] * 3 + 2];
        }
        state.info.z_feet = z_feet;
        // needed for debugging or logging

        // feet contact + last_contact
        Eigen::Vector4i contact;
        for (int i = 0; i < 4; i++)
        {
            double foot_z = d_->site_xpos[feet_site_id_[i] * 3 + 2] - foot_radius_;
            contact[i] = (foot_z < 1e-3) ? 1 : 0;
        }
        Eigen::Vector4i contact_filt_mm;
        for (int i = 0; i < 4; i++)
        {
            contact_filt_mm[i] = (contact[i] || state.info.last_contact[i]) ? 1 : 0;
        }
        // feet_air_time
        for (int i = 0; i < 4; i++)
        {
            if (contact_filt_mm[i] == 0)
            {
                state.info.feet_air_time[i] += dt();
            }
            else
            {
                state.info.feet_air_time[i] = 0.0;
            }
        }
        state.info.last_contact = contact;

        // also store the updated obs, reward, done
        EnvState newState = state;
        newState.obs = obs;
        newState.reward = reward;
        newState.done = done;
        return newState;
    }

    mjModel *model() { return m_; }
    mjData *data() { return d_; }

    // --------------------------------------------------
    // dt
    // --------------------------------------------------
    double dt() const { return config_.timestep; }

private:
    UnitreeGo2EnvConfig config_;

    mjModel *m_ = nullptr;
    mjData *d_ = nullptr;

    int torso_idx_;
    std::vector<int> feet_site_id_;

    // From the Python code
    Eigen::Matrix<double, 12, 2> joint_range_;
    Eigen::Matrix<double, 12, 2> physical_joint_range_;
    Eigen::Matrix<double, 12, 2> joint_torque_range_;

    Eigen::VectorXd init_q_;
    Eigen::VectorXd default_pose_;
    double foot_radius_;

    std::string gait_;

    // Gait phase table
    std::map<std::string, Eigen::Vector4d> kGaitPhases_;
    // Gait param table: (duty_ratio, cadence, amplitude)
    std::map<std::string, Eigen::Vector3d> kGaitParams_;

    // --------------------------------------------------
    // Implementation of the exact Python get_foot_step
    // logic via the function above
    // --------------------------------------------------
    // Called inside computeReward to get foot reference
    // height vs. actual foot height.
    // --------------------------------------------------
    Eigen::VectorXd computeFootStep(const StateInfo &info)
    {
        // from the Python snippet:
        // duty_ratio, cadence, amplitude = self._gait_params[self._gait]
        Eigen::Vector3d gp = kGaitParams_[gait_];
        double duty_ratio = gp[0];
        double cadence = gp[1];
        double amplitude = gp[2];

        // phases = self._gait_phase[self._gait]
        Eigen::Vector4d phases = kGaitPhases_[gait_];

        double time_sec = info.step * dt();
        // call the exact function
        Eigen::VectorXd foot_target = getFootStep(duty_ratio, cadence, amplitude, phases, time_sec);
        return foot_target; // length=4
    }

    // --------------------------------------------------
    // act2joint: replicate python
    // --------------------------------------------------
    Eigen::VectorXd act2joint(const Eigen::VectorXd &act)
    {
        // 1) act_normalized = (act * action_scale + 1.0)/2.0
        // 2) joint_targets = joint_range_[:,0] + ...
        // 3) clip to physical_joint_range_

        int N = joint_range_.rows(); // e.g. 12
        Eigen::VectorXd result(N);
        for (int i = 0; i < N; i++)
        {
            double a = act[i] * config_.action_scale + 1.0;
            double act_normalized = a * 0.5; // => range [0..1], if act in [-1..1]
            // scale to joint_range
            double low = joint_range_(i, 0);
            double high = joint_range_(i, 1);
            double jt = low + act_normalized * (high - low);

            // clip to physical_joint_range_
            double p_low = physical_joint_range_(i, 0);
            double p_high = physical_joint_range_(i, 1);
            jt = clamp(jt, p_low, p_high);

            result[i] = jt;
        }
        return result;
    }

    // --------------------------------------------------
    // act2tau: replicate python
    // --------------------------------------------------
    Eigen::VectorXd act2tau(const Eigen::VectorXd &act)
    {
        // from python:
        // 1) joint_target = act2joint(act)
        // 2) q = pipeline_state.qpos[7:7+N]
        // 3) qd = pipeline_state.qvel[6:6+N]
        // 4) q_err = joint_target - q
        // 5) tau = kp * q_err - kd * qd
        // 6) clip(tau, torque_range)
        const int N = joint_range_.rows(); // e.g. 12
        Eigen::VectorXd joint_target = act2joint(act);

        // gather q, qd from the MuJoCo data
        // qpos has size m_->nq, but we only want the joint portion (skip 7 root dofs).
        // qvel has size m_->nv, skip 6 root dofs.
        Eigen::VectorXd q(N);
        Eigen::VectorXd qd(N);
        for (int i = 0; i < N; i++)
        {
            q[i] = d_->qpos[7 + i];
            qd[i] = d_->qvel[6 + i];
        }
        // PD
        Eigen::VectorXd q_err = joint_target - q;
        Eigen::VectorXd tau(N);
        for (int i = 0; i < N; i++)
        {
            double val = config_.kp * q_err[i] - config_.kd * qd[i];
            // clip to joint_torque_range_
            double tmin = joint_torque_range_(i, 0);
            double tmax = joint_torque_range_(i, 1);
            tau[i] = clamp(val, tmin, tmax);
        }
        return tau;
    }

    // --------------------------------------------------
    // Build observation vector, just as in python
    // --------------------------------------------------
    Eigen::VectorXd get_obs(const StateInfo &info,
                            const Eigen::VectorXd &ctrl)
    {
        // Collect velocities from d_->cvel
        // MuJoCo cvel per body: [angular(3), linear(3)]
        // Here: torso_idx_ * 6 => offset
        double w_x = d_->cvel[torso_idx_ * 6 + 0];
        double w_y = d_->cvel[torso_idx_ * 6 + 1];
        double w_z = d_->cvel[torso_idx_ * 6 + 2];
        double v_x = d_->cvel[torso_idx_ * 6 + 3];
        double v_y = d_->cvel[torso_idx_ * 6 + 4];
        double v_z = d_->cvel[torso_idx_ * 6 + 5];

        Eigen::Vector3d w_global(w_x, w_y, w_z);
        Eigen::Vector3d v_global(v_x, v_y, v_z);

        const double *body_quat = &d_->xquat[torso_idx_ * 4];
        Eigen::Vector3d vb = globalToBodyVelocity(v_global, body_quat);
        Eigen::Vector3d ab = globalToBodyVelocity(w_global, body_quat);

        // qpos, qvel
        Eigen::VectorXd qpos(m_->nq);
        for (int i = 0; i < m_->nq; i++)
        {
            qpos[i] = d_->qpos[i];
        }
        Eigen::VectorXd qvel(m_->nv);
        for (int i = 0; i < m_->nv; i++)
        {
            qvel[i] = d_->qvel[i];
        }

        // match python ordering:
        // [vel_tar(3), ang_vel_tar(3), ctrl(m_->nu), qpos(m_->nq),
        //  vb(3), ab(3), qvel[6..end]]
        int nu = m_->nu;
        int tail_v = m_->nv - 6; // length of qvel we keep
        int obs_dim = 3 + 3 + nu + m_->nq + 3 + 3 + tail_v;

        Eigen::VectorXd obs(obs_dim);
        int idx = 0;
        // vel_tar
        obs.segment<3>(idx) = info.vel_tar;
        idx += 3;
        // ang_vel_tar
        obs.segment<3>(idx) = info.ang_vel_tar;
        idx += 3;
        // ctrl
        obs.segment(idx, nu) = ctrl;
        idx += nu;
        // qpos
        obs.segment(idx, m_->nq) = qpos;
        idx += m_->nq;
        // vb
        obs.segment<3>(idx) = vb;
        idx += 3;
        // ab
        obs.segment<3>(idx) = ab;
        idx += 3;
        // qvel[6..]
        obs.segment(idx, tail_v) = qvel.segment(6, tail_v);

        return obs;
    }

    // --------------------------------------------------
    // Reward calculation, matching the python snippet
    // --------------------------------------------------
    double computeReward(StateInfo &info, const Eigen::VectorXd &ctrl)
    {
        // 1) get actual foot z
        Eigen::Vector4d z_feet;
        for (int i = 0; i < 4; i++)
        {
            z_feet[i] = d_->site_xpos[feet_site_id_[i] * 3 + 2];
        }
        // get foot step reference
        Eigen::VectorXd z_feet_tar = computeFootStep(info);

        // reward_gaits = - sum(((z_feet_tar - z_feet)/0.05)^2)
        double reward_gaits = 0.0;
        for (int i = 0; i < 4; i++)
        {
            double diff = (z_feet_tar[i] - z_feet[i]) / 0.05;
            reward_gaits -= diff * diff;
        }

        // torso pos, orientation
        Eigen::Vector3d torso_pos(d_->xpos[torso_idx_ * 3 + 0],
                                  d_->xpos[torso_idx_ * 3 + 1],
                                  d_->xpos[torso_idx_ * 3 + 2]);
        Eigen::Matrix3d R = quatToMat(&d_->xquat[torso_idx_ * 4]);

        // reward_upright: measure alignment with global up
        Eigen::Vector3d up_global(0, 0, 1);
        Eigen::Vector3d up_body = R * up_global;
        double reward_upright = -(up_body - up_global).squaredNorm();

        // yaw orientation reward
        double yaw_tar = info.yaw_tar + info.ang_vel_tar[2] * dt() * info.step;
        double yaw = quatToYaw(&d_->xquat[torso_idx_ * 4]);
        double d_yaw = yaw - yaw_tar;
        double wrapped = atan2(std::sin(d_yaw), std::cos(d_yaw));
        double reward_yaw = -(wrapped * wrapped);

        // velocity
        Eigen::Vector3d v_global(d_->cvel[torso_idx_ * 6 + 3],
                                 d_->cvel[torso_idx_ * 6 + 4],
                                 d_->cvel[torso_idx_ * 6 + 5]);
        Eigen::Vector3d w_global(d_->cvel[torso_idx_ * 6 + 0],
                                 d_->cvel[torso_idx_ * 6 + 1],
                                 d_->cvel[torso_idx_ * 6 + 2]);
        Eigen::Vector3d vb = globalToBodyVelocity(v_global, &d_->xquat[torso_idx_ * 4]);
        Eigen::Vector3d ab = globalToBodyVelocity(w_global, &d_->xquat[torso_idx_ * 4]);
        double reward_vel = -(vb.head<2>() - info.vel_tar.head<2>()).squaredNorm();
        double reward_ang_vel = -std::pow(ab[2] - info.ang_vel_tar[2], 2.0);

        // height reward
        // pos_tar z is info.pos_tar[2], but also the snippet does something like
        // pos_tar = info.pos_tar + ...
        // We'll just replicate the final line from the snippet:
        double reward_height = -std::pow((torso_pos[2] - info.pos_tar[2]), 2);

        // sum up with python weights
        double reward = 0.1 * reward_gaits + 0.5 * reward_upright + 0.3 * reward_yaw + 1.0 * reward_vel + 1.0 * reward_ang_vel + 1.0 * reward_height;

        return reward;
    }

    // --------------------------------------------------
    // termination check
    // --------------------------------------------------
    bool checkTermination()
    {
        // dot(up_body, up_global) < 0 => inverted
        Eigen::Matrix3d R = quatToMat(&d_->xquat[torso_idx_ * 4]);
        Eigen::Vector3d up_body = R * Eigen::Vector3d(0, 0, 1);
        if (up_body.dot(Eigen::Vector3d(0, 0, 1)) < 0.0)
        {
            return true;
        }

        // check joint angles out of range
        const int N = joint_range_.rows();
        for (int i = 0; i < N; i++)
        {
            double angle = d_->qpos[7 + i];
            double lower = joint_range_(i, 0);
            double upper = joint_range_(i, 1);
            if (angle < lower || angle > upper)
            {
                return true;
            }
        }

        // check if torso fell below 0.18
        double z_torso = d_->xpos[torso_idx_ * 3 + 2];
        if (z_torso < 0.18)
        {
            return true;
        }
        return false;
    }

    // --------------------------------------------------
    // sample_command: replicate python random
    // --------------------------------------------------
    std::pair<Eigen::Vector3d, Eigen::Vector3d> sample_command(std::mt19937 &rng)
    {
        std::uniform_real_distribution<double> dist_lin_x(-1.5, 1.5);
        std::uniform_real_distribution<double> dist_lin_y(-0.5, 0.5);
        std::uniform_real_distribution<double> dist_yaw(-1.5, 1.5);

        double lx = dist_lin_x(rng);
        double ly = dist_lin_y(rng);
        double yw = dist_yaw(rng);

        Eigen::Vector3d lin(lx, ly, 0.0);
        Eigen::Vector3d ang(0.0, 0.0, yw);
        return {lin, ang};
    }

    // --------------------------------------------------
    // fill in the gait tables (phases, params)
    // --------------------------------------------------
    void setupGaitTables()
    {
        kGaitPhases_["stand"] = (Eigen::Vector4d() << 0.0, 0.0, 0.0, 0.0).finished();
        kGaitPhases_["walk"] = (Eigen::Vector4d() << 0.0, 0.5, 0.75, 0.25).finished();
        kGaitPhases_["trot"] = (Eigen::Vector4d() << 0.0, 0.5, 0.5, 0.0).finished();
        kGaitPhases_["canter"] = (Eigen::Vector4d() << 0.0, 0.33, 0.33, 0.66).finished();
        kGaitPhases_["gallop"] = (Eigen::Vector4d() << 0.0, 0.05, 0.4, 0.35).finished();

        // (duty_ratio, cadence, amplitude)
        kGaitParams_["stand"] = (Eigen::Vector3d() << 1.0, 1.0, 0.0).finished();
        kGaitParams_["walk"] = (Eigen::Vector3d() << 0.75, 1.0, 0.08).finished();
        kGaitParams_["trot"] = (Eigen::Vector3d() << 0.45, 2.0, 0.08).finished();
        kGaitParams_["canter"] = (Eigen::Vector3d() << 0.4, 4.0, 0.06).finished();
        kGaitParams_["gallop"] = (Eigen::Vector3d() << 0.3, 3.5, 0.10).finished();
    }
};

// MuJoCo data structures
mjModel *m = NULL; // MuJoCo model
mjData *d = NULL;  // MuJoCo data
mjvCamera cam;     // abstract camera
mjvOption opt;     // visualization options
mjvScene scn;      // abstract scene
mjrContext con;    // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;

// keyboard callback
void keyboard(GLFWwindow *window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE)
    {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
}

// mouse button callback
void mouse_button(GLFWwindow *window, int button, int act, int mods)
{
    // update button state
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}

// mouse move callback
void mouse_move(GLFWwindow *window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if (!button_left && !button_middle && !button_right)
    {
        return;
    }

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if (button_right)
    {
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    }
    else if (button_left)
    {
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    }
    else
    {
        action = mjMOUSE_ZOOM;
    }

    // move camera
    mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}

// scroll callback
void scroll(GLFWwindow *window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}

// main function
int main(int argc, const char **argv)
{
    // Create config
    UnitreeGo2EnvConfig config;
    config.gait = "trot";
    config.randomize_tasks = false;
    config.leg_control = "torque";
    config.action_scale = 1.0; // from the snippet
    config.timestep = 0.0025;

    std::string model_path = "/home/quant/dial_mpc_ws/src/dial-mpc/models/unitree_go2/mjx_scene_force.xml";

    // Create environment
    UnitreeGo2Env env(config, model_path);

    // RNG
    std::random_device rd;
    std::mt19937 rng_engine(rd());

    // reset
    EnvState state = env.reset(rng_engine);

    m = env.model();
    d = env.data();

    // init GLFW
    if (!glfwInit())
    {
        mju_error("Could not initialize GLFW");
    }

    // create window, make OpenGL context current, request v-sync
    GLFWwindow *window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    int i = 0;
    // run main loop, target real-time simulation and 60 fps rendering
    while (!glfwWindowShouldClose(window))
    {
        // advance interactive simulation for 1/60 sec
        //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
        //  this loop will finish on time for the next frame to be rendered at 60 fps.
        //  Otherwise add a cpu timer and exit this loop when it is time to render.
        mjtNum simstart = d->time;
        // example: random action (12-dim)
        Eigen::VectorXd action(12);
        action.setZero(); // or random
        while (d->time - simstart < 1.0 / 60.0)
        {
            state = env.step(state, action, rng_engine);
        }

        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();

        if (state.done)
        {
            std::cout << "Terminated at step " << i
                      << " reward = " << state.reward << std::endl;
            break;
        }

        ++i;
    }

    // free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // // free MuJoCo model and data
    // mj_deleteData(d);
    // mj_deleteModel(m);

    // terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
    glfwTerminate();
#endif

    return 1;
}