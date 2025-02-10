#pragma once

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include <Eigen/Dense>

#include <random>
#include <time.h>
#include <thread>

#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <algorithm>

/* Thread-safe function that returns a random number between min and max (inclusive).
This function takes ~142% the time that calling rand() would take. For this extra
cost you get a better uniform distribution and thread-safety. */
double doubleRand(const double &min, const double &max)
{
    static thread_local std::mt19937 *generator = nullptr;
    if (!generator)
        generator = new std::mt19937(std::chrono::system_clock::now().time_since_epoch().count() + std::hash<std::thread::id>()(std::this_thread::get_id()));
    std::uniform_real_distribution<double> distribution(min, max);
    return distribution(*generator);
}

//-----------------------------------------------------
// Configuration structure similar to UnitreeGo2EnvConfig
//-----------------------------------------------------
struct UnitreeGo2EnvConfig
{
    double kp = 30.0;
    double kd = 1.0;
    double action_scale = 1.0; // newly added to match 'act2joint' scale

    double default_vx = 0.0;
    double default_vy = 0.0;
    double default_vyaw = 0.0;
    double ramp_up_time = 1.0;
    std::string gait = "stand";
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
    std::mt19937_64 rng;
    Eigen::VectorXd pos_tar;     // (3,)
    Eigen::VectorXd vel_tar;     // (3,)
    Eigen::VectorXd ang_vel_tar; // (3,)
    double yaw_tar;
    int step;
    bool randomize_target;
};

//-----------------------------------------------------
// A structure to hold the entire env step state
//-----------------------------------------------------
struct EnvState
{
    mjModel *model = nullptr;
    mjData *data = nullptr;

    double reward; // reward
    bool done;     // termination flag
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
// Convert quaternion to yaw angle (Z euler)
//-----------------------------------------------------
double quatToYaw(Eigen::Vector4d quat)
{
    Eigen::Quaterniond q(quat[0], quat[1], quat[2], quat[3]);
    // eulerAngles(2,1,0) returns (Z, Y, X)
    Eigen::Vector3d euler = q.toRotationMatrix().eulerAngles(2, 1, 0);
    double yaw = euler[0];
    return yaw;
}

Eigen::Vector4d quat_inv(const Eigen::Vector4d &q)
{
    Eigen::Vector4d q_inv = q;
    q_inv.segment(1, 3) *= -1;
    return q_inv;
}

Eigen::Vector3d rotate(const Eigen::Vector3d &v, const Eigen::Vector4d &q)
{
    double s = q[0];
    Eigen::Vector3d u = q.segment(1, 3);
    Eigen::Vector3d r = 2 * (u.dot(v) * u) + (s * s - u.dot(u)) * v;
    r = r.eval() + 2 * s * u.cross(v);
    return r;
}

Eigen::Vector3d inv_rotate(const Eigen::Vector3d &v, const Eigen::Vector4d &q)
{
    Eigen::Vector4d q_inv = quat_inv(q);
    return rotate(v, q_inv);
}

//-----------------------------------------------------
// Rotate vel_global by inverse of body_quat
//-----------------------------------------------------
Eigen::Vector3d globalToBodyVelocity(const Eigen::Vector3d &vel_global,
                                     const Eigen::Vector4d &body_quat)
{
    return inv_rotate(vel_global, body_quat);
}

//-----------------------------------------------------
// Rotate vel_local by body_quat
//-----------------------------------------------------
Eigen::Vector3d localToGlobalVelocity(const Eigen::Vector3d &vel_local,
                                      const Eigen::Vector4d &body_quat)
{
    return rotate(vel_local, body_quat);
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
        std::cout << "Torso index: " << torso_idx_ << std::endl;

        // Identify feet site indices
        std::vector<std::string> feet_site_names = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"};
        for (auto &site_name : feet_site_names)
        {
            int sid = mj_name2id(m_, mjOBJ_SITE, site_name.c_str());
            if (sid < 0)
            {
                std::cerr << "[Warning] Site not found: " << site_name << std::endl;
            }
            std::cout << "Site id for " << site_name << ": " << sid << std::endl;
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
        std::cout << "Initial config: " << init_q_.transpose() << std::endl;
        std::cout << "Default pose: " << default_pose_.transpose() << std::endl;

        // joint_range_ = Eigen::MatrixXd(12, 2);
        // joint_range_ << -0.5, 0.5,
        //     0.4, 1.4,
        //     -2.3, -0.85,
        //     -0.5, 0.5,
        //     0.4, 1.4,
        //     -2.3, -0.85,
        //     -0.5, 0.5,
        //     0.4, 1.4,
        //     -2.3, -1.3,
        //     -0.5, 0.5,
        //     0.4, 1.4,
        //     -2.3, -1.3;

        // get joint limits from mujoco instead:
        joint_range_ = Eigen::MatrixXd(12, 2);
        for (int i = 0; i < 12; i++)
        {
            joint_range_(i, 0) = m_->jnt_range[i * 2];
            joint_range_(i, 1) = m_->jnt_range[i * 2 + 1];
        }

        std::cout << "Joint range: " << joint_range_ << std::endl;

        physical_joint_range_ = joint_range_;

        // joint_torque_range_ = Eigen::MatrixXd(12, 2);
        // joint_torque_range_.col(0).setConstant(-500.0);
        // joint_torque_range_.col(1).setConstant(500.0);

        // get torque limits from mujoco instead:
        joint_torque_range_ = Eigen::MatrixXd(12, 2);
        for (int i = 0; i < 12; i++)
        {
            // if lb and ub = 0, then set to inf
            if (std::fabs(m_->actuator_ctrlrange[i * 2]) < 1e-6 && std::fabs(m_->actuator_ctrlrange[i * 2 + 1]) < 1e-6)
            {
                joint_torque_range_(i, 0) = -std::numeric_limits<double>::infinity();
                joint_torque_range_(i, 1) = std::numeric_limits<double>::infinity();
            }
            else
            {
                joint_torque_range_(i, 0) = m_->actuator_ctrlrange[i * 2];
                joint_torque_range_(i, 1) = m_->actuator_ctrlrange[i * 2 + 1];
            }
        }

        std::cout << "Torque range: " << joint_torque_range_ << std::endl;

        action_size_ = m_->nu; // 12
        std::cout << "Action size: " << action_size_ << std::endl;

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
        // if (d_)
        //     mj_deleteData(d_);
        // if (m_)
        //     mj_deleteModel(m_);
    }

    // -----------------------------------
    // reset the environment
    // -----------------------------------
    EnvState reset(std::mt19937_64 &rng)
    {
        mj_resetData(m_, d_);
        Eigen::VectorXd zero_dq(m_->nv);
        zero_dq.setZero();
        pipeline_init(m_, d_, init_q_, zero_dq);

        EnvState s;
        s.model = m_;
        s.data = d_;

        // Fill StateInfo
        s.info.rng = rng;
        s.info.pos_tar = Eigen::Vector3d(0.282, 0.0, 0.3);
        s.info.vel_tar = Eigen::Vector3d(0.0, 0.0, 0.0);
        s.info.ang_vel_tar = Eigen::Vector3d(0.0, 0.0, 0.0);
        s.info.yaw_tar = 0.0;
        s.info.step = 0;
        s.info.randomize_target = config_.randomize_tasks;

        Eigen::VectorXd ctrl_init = Eigen::VectorXd::Zero(m_->nu);
        s.reward = 0.0;
        s.done = false;

        return s;
    }

    // -----------------------------------
    // step the environment
    // -----------------------------------
    EnvState step(EnvState state, const Eigen::VectorXd &action)
    {
        std::mt19937_64 rng = state.info.rng;
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

        // 3) compute observation
        // omitted

        // Possibly randomize velocity commands every 500 steps
        if (state.info.randomize_target && (state.info.step % 500 == 0))
        {
            auto cmds = sample_command(rng);
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
        /*
        // site z positions:
        std::cout << "Size of feet_site_id_: " << feet_site_id_.size() << std::endl;
        std::cout << "feet_site_id_ values: ";
        
        for (int id : feet_site_id_)
        {
            std::cout << id << " ";
        }
        std::cout << std::endl;
        */
        Eigen::Vector4d z_feet;
        for (int i = 0; i < 4; i++)
        {
            // std::cout << "z_feet " << i << " is value (from site_xpos)" << d_->site_xpos[feet_site_id_[i] * 3 + 2] <<std::endl;
            z_feet[i] = d_->site_xpos[feet_site_id_[i] * 3 + 2];
        }
        // get foot step reference
        Eigen::VectorXd z_feet_tar = computeFootStep(state.info);
        // std::cout << std::endl << "z_feet_tar:" << std::endl << z_feet_tar << std::endl;

        // reward_gaits = - sum(((z_feet_tar - z_feet)/0.05)^2)
        double reward_gaits = 0.0;
        for (int i = 0; i < 4; i++)
        {
            double diff = (z_feet_tar[i] - z_feet[i]) / 0.05;
            reward_gaits -= diff * diff;
        }

        // reward_upright: measure alignment with global up
        Eigen::Vector3d up_global(0, 0, 1);
        Eigen::Vector4d torso_quat;
        for (int i = 0; i < 4; i++)
        {
            torso_quat[i] = d_->xquat[torso_idx_ * 4 + i];
        }
        // Eigen::Matrix3d R = quatToMat(&d_->xquat[torso_idx_ * 4]); // torso orientation
        // Eigen::Vector3d up_body = R * up_global;
        Eigen::Vector3d up_body = rotate(up_global, torso_quat);
        double reward_upright = -(up_body - up_global).squaredNorm();

        // yaw orientation reward
        double yaw_tar = state.info.yaw_tar + state.info.ang_vel_tar[2] * dt() * state.info.step;
        double yaw = quatToYaw(torso_quat);
        double d_yaw = yaw - yaw_tar;
        double wrapped = atan2(std::sin(d_yaw), std::cos(d_yaw));
        double reward_yaw = -(wrapped * wrapped);

        // velocity reward

        // we need to replicate this:
        // x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
        // cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
        // offset = data.xpos[1:, :] - data.subtree_com[sys.body_rootid[1:]]
        // offset = Transform.create(pos=offset)
        // xd = offset.vmap().do(cvel)

        // vb = global_to_body_velocity(
        //     xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1]
        // )
        // ab = global_to_body_velocity(
        //     xd.ang[self._torso_idx - 1] * jnp.pi / 180.0, x.rot[self._torso_idx - 1]
        // )
        // reward_vel = -jnp.sum((vb[:2] - state.info["vel_tar"][:2]) ** 2)
        // reward_ang_vel = -jnp.sum((ab[2] - state.info["ang_vel_tar"][2]) ** 2)

        // this is wrong! does not use the transformed velocities:
        // Eigen::Vector3d v_global(d_->cvel[torso_idx_ * 6 + 3],
        //                          d_->cvel[torso_idx_ * 6 + 4],
        //                          d_->cvel[torso_idx_ * 6 + 5]);
        // Eigen::Vector3d w_global(d_->cvel[torso_idx_ * 6 + 0],
        //                          d_->cvel[torso_idx_ * 6 + 1],
        //                          d_->cvel[torso_idx_ * 6 + 2]);
        // Eigen::Vector3d vb = globalToBodyVelocity(v_global, torso_quat);
        // Eigen::Vector3d ab = globalToBodyVelocity(w_global, torso_quat);
        // double reward_vel = -(vb.head<2>() - state.info.vel_tar.head<2>()).squaredNorm();
        // double reward_ang_vel = -std::pow(ab[2] - state.info.ang_vel_tar[2], 2.0);

        // this is right:
        Eigen::Vector3d offset;
        offset << d_->xpos[torso_idx_ * 3 + 0] - d_->subtree_com[m_->body_rootid[torso_idx_] * 3 + 0],
            d_->xpos[torso_idx_ * 3 + 1] - d_->subtree_com[m_->body_rootid[torso_idx_] * 3 + 1],
            d_->xpos[torso_idx_ * 3 + 2] - d_->subtree_com[m_->body_rootid[torso_idx_] * 3 + 2];

        Eigen::Vector4d identity_quat(1.0, 0.0, 0.0, 0.0);

        Eigen::Vector3d cvel_ang;
        cvel_ang << d_->cvel[torso_idx_ * 6 + 0],
            d_->cvel[torso_idx_ * 6 + 1],
            d_->cvel[torso_idx_ * 6 + 2];

        Eigen::Vector3d cvel_lin;
        cvel_lin << d_->cvel[torso_idx_ * 6 + 3],
            d_->cvel[torso_idx_ * 6 + 4],
            d_->cvel[torso_idx_ * 6 + 5];

        Eigen::Vector4d rot_inv = quat_inv(identity_quat);
        Eigen::Vector3d vel_ang = rotate(cvel_ang, rot_inv);
        Eigen::Vector3d vel_lin = rotate(cvel_lin - offset.cross(cvel_ang), rot_inv);

        Eigen::Vector3d vb = globalToBodyVelocity(vel_lin, torso_quat);
        Eigen::Vector3d ab = globalToBodyVelocity(vel_ang, torso_quat);

        double reward_vel = -(vb.head<2>() - state.info.vel_tar.head<2>()).squaredNorm();
        double reward_ang_vel = -std::pow(ab[2] - state.info.ang_vel_tar[2], 2.0);

        // height reward
        double reward_height = -std::pow((d_->xpos[torso_idx_ * 3 + 2] - state.info.pos_tar[2]), 2);

        // sum up with weights
        double reward = 0.1 * reward_gaits + 0.5 * reward_upright + 0.3 * reward_yaw + 1.0 * reward_vel + 1.0 * reward_ang_vel + 1.0 * reward_height;
        // double reward = 0.0 * reward_gaits + 5.0 * reward_upright + 1.0 * reward_height;

        // 5) check done
        bool done = false;

        // dot(up_body, up_global) < 0 => inverted
        if (up_body.dot(up_global) < 0.0)
        {
            done = true;
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
                done = true;
            }
        }

        // check if torso fell below 0.18
        double z_torso = d_->xpos[torso_idx_ * 3 + 2];
        if (z_torso < 0.18)
        {
            done = true;
        }

        state.info.rng = rng;

        // 6) update state info (z_feet, contact, etc.)
        state.info.step += 1;

        // also store the updated reward, done
        EnvState newState = state;
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

    int action_size() const { return action_size_; }

    Eigen::Matrix<double, 12, 2> joint_range() const { return joint_range_; }

    Eigen::Matrix<double, 12, 2> joint_torque_range() const { return joint_torque_range_; }

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

    int action_size_;

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
    // sample_command: replicate python random
    // --------------------------------------------------
    std::pair<Eigen::Vector3d, Eigen::Vector3d> sample_command(std::mt19937_64 &rng)
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