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

// go2 has nq = 19, nv = 18, nu = 12
namespace go2env
{
    using VectorXd = Eigen::VectorXd;

    using Vector19d = Eigen::Matrix<double, 19, 1>; // for qpos
    using Vector18d = Eigen::Matrix<double, 18, 1>; // for qvel

    using Vector12d = Eigen::Matrix<double, 12, 1>; // for joints, torques
    using Matrix12Xd = Eigen::Matrix<double, 12, Eigen::Dynamic>;
    using Matrix12Bounds = Eigen::Matrix<double, 12, 2>; // for joint limits
    using Vector3d = Eigen::Vector3d;
    using Vector4d = Eigen::Vector4d;

    //-----------------------------------------------------
    // Convert quaternion to yaw angle (Z euler)
    //-----------------------------------------------------
    static double quat_to_yaw(Vector4d quat)
    {
        Eigen::Quaterniond q(quat[0], quat[1], quat[2], quat[3]);
        // eulerAngles(2,1,0) returns (Z, Y, X)
        Vector3d euler = q.toRotationMatrix().eulerAngles(2, 1, 0);
        double yaw = euler[0];
        return yaw;
    }

    //-----------------------------------------------------
    // Compute the quaternion inverse
    //-----------------------------------------------------
    static Vector4d quat_inv(const Vector4d &q)
    {
        Vector4d q_inv = q;
        q_inv.segment(1, 3) *= -1;
        return q_inv;
    }

    //-----------------------------------------------------
    // Rotate a vector by a quaternion
    //-----------------------------------------------------
    static Vector3d rotate(const Vector3d &v, const Vector4d &q)
    {
        double s = q[0];
        Vector3d u = q.segment(1, 3);
        Vector3d r = 2 * (u.dot(v) * u) + (s * s - u.dot(u)) * v;
        r = r.eval() + 2 * s * u.cross(v);
        return r;
    }

    //-----------------------------------------------------
    // Rotate a vector by the inverse of a quaternion
    //-----------------------------------------------------
    static Vector3d inv_rotate(const Vector3d &v, const Vector4d &q)
    {
        Vector4d q_inv = quat_inv(q);
        return rotate(v, q_inv);
    }

    //-----------------------------------------------------
    // Rotate vel_global by inverse of body_quat
    //-----------------------------------------------------
    static Vector3d global_to_body_velocity(const Vector3d &vel_global,
                                            const Vector4d &body_quat)
    {
        return inv_rotate(vel_global, body_quat);
    }

    //-----------------------------------------------------
    // Rotate vel_local by body_quat
    //-----------------------------------------------------
    static Vector3d local_to_global_velocity(const Vector3d &vel_local,
                                             const Vector4d &body_quat)
    {
        return rotate(vel_local, body_quat);
    }

    class Transform;

    class Motion
    {
    public:
        Motion()
        {
            lin_ = Vector3d::Zero();
            ang_ = Vector3d::Zero();
        }

        Motion(const Vector3d &lin, const Vector3d &ang)
        {
            lin_ = lin;
            ang_ = ang;
        }

        Vector3d lin_;
        Vector3d ang_;

    }; // class Motion

    class Transform
    {
    public:
        Transform()
        {
            pos_ = Vector3d::Zero();
            quat_ = Vector4d(1.0, 0.0, 0.0, 0.0);
        }

        Transform(const Vector3d &pos, const Vector4d &quat)
        {
            pos_ = pos;
            quat_ = quat;
        }

        Transform(const Vector3d &pos)
        {
            pos_ = pos;
            quat_ = Vector4d(1.0, 0.0, 0.0, 0.0);
        }

        Motion apply(const Motion &m)
        {
            Vector4d quat_inv = quat_inv(quat_);
            Vector3d ang = rotate(m.ang_, quat_inv);
            Vector3d lin = rotate(m.lin_ - pos_.cross(m.ang_), quat_inv);
            return Motion(lin, ang);
        }

        Motion apply_inv(const Motion &m)
        {
            Vector3d ang = rotate(m.ang_, quat_);
            Vector3d lin = rotate(m.lin_, quat_) + pos_.cross(ang);
            return Motion(lin, ang);
        }

        Vector3d pos_;
        Vector4d quat_;

    }; // class Transform

    //-----------------------------------------------------
    // Configuration structure similar to UnitreeGo2EnvConfig
    //-----------------------------------------------------
    struct UnitreeGo2EnvConfig
    {
        UnitreeGo2EnvConfig(double kp, double kd, double action_scale,
                            double default_vx, double default_vy, double default_vyaw,
                            double ramp_up_time, std::string gait, double timestep)
            : kp(kp), kd(kd), action_scale(action_scale),
              default_vx(default_vx), default_vy(default_vy), default_vyaw(default_vyaw),
              ramp_up_time(ramp_up_time), gait(gait), timestep(timestep)
        {
        }

        UnitreeGo2EnvConfig()
        {
        }

        double kp = 30.0;
        double kd = 1.0;
        double action_scale = 1.0;

        double default_vx = 0.0;
        double default_vy = 0.0;
        double default_vyaw = 0.0;
        double ramp_up_time = 1.0;
        std::string gait = "stand";
        double timestep = 0.0025; // dt of the underlying simulator step
    };

    //-----------------------------------------------------
    // A structure to store environment state info
    //-----------------------------------------------------
    struct StateInfo
    {
        std::mt19937_64 rng;
        Vector3d pos_tar;
        Vector3d vel_tar;
        Vector3d ang_vel_tar;
        double yaw_tar;
        size_t step;

        Vector4d z_feet;
        Vector4d z_feet_tar;
        std::vector<bool> last_contact;
        std::vector<double> feet_air_time;
    };

    //-----------------------------------------------------
    // A structure to hold the pipeline state
    //-----------------------------------------------------
    struct PipelineState
    {
        Vector19d qpos;
        Vector18d qvel;
        Transform x;
        Motion xd;
    };

    //-----------------------------------------------------
    // A structure to hold the entire env step state
    //-----------------------------------------------------
    struct EnvState
    {
        PipelineState pipeline_state;
        double reward;
        bool done;
        StateInfo info;
    };

    //-----------------------------------------------------
    // Random number generator
    //-----------------------------------------------------
    static double double_rand(size_t id, double min, double max)
    {
        static std::mt19937_64 *generators[200];
        if (!generators[id])
        {
            generators[id] = new std::mt19937_64(std::chrono::system_clock::now().time_since_epoch().count() + id);
        }
        std::uniform_real_distribution<double> distribution(min, max);
        return distribution(*generators[id]);
    }

    //-----------------------------------------------------
    // Clamp
    //-----------------------------------------------------
    static inline double clamp(double x, double low, double high)
    {
        return (x < low) ? low : ((x > high) ? high : x);
    }

    //-----------------------------------------------------
    // Angle modulus
    //-----------------------------------------------------
    static inline double angle_mod(double x, double modulus)
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
    static double step_height(double t, double footphase, double duty_ratio)
    {
        double raw = angle_mod(t + M_PI - footphase, 2.0 * M_PI);
        double angle = raw - M_PI;

        if (duty_ratio < 1.0)
        {
            angle *= (0.5 / (1.0 - duty_ratio));
        }

        double clipped_angle = clamp(angle, -M_PI / 2.0, M_PI / 2.0);

        double value = 0.0;
        if (duty_ratio < 1.0)
        {
            value = std::cos(clipped_angle);
        }

        double final_value = 0.0;
        if (std::fabs(value) >= 1e-6)
        {
            final_value = std::fabs(value);
        }
        return final_value;
    }

    //-----------------------------------------------------
    // Compute footstep heights
    //-----------------------------------------------------
    static Vector4d get_foot_step(double duty_ratio, double cadence, double amplitude,
                                  const Vector4d &phases, double time)
    {
        size_t n = phases.size();
        Vector4d h_steps(n);
        double T = time * 2.0 * M_PI * cadence + M_PI;
        for (size_t i = 0; i < n; ++i)
        {
            double footphase = 2.0 * M_PI * phases[i];
            double val = step_height(T, footphase, duty_ratio);
            h_steps[i] = amplitude * val;
        }
        return h_steps;
    }

    //-----------------------------------------------------
    // Helper to copy Eigen vectors to Mujoco
    //-----------------------------------------------------
    template <typename Derived>
    static void copy_eigen_to_mujoco(double *dst, const Eigen::MatrixBase<Derived> &src, size_t size)
    {
        static_assert(Derived::ColsAtCompileTime == 1, "Only column (double) vectors are supported");
        // we don't check if the scalars are double, but oh well who cares
        for (size_t i = 0; i < size; ++i)
        {
            dst[i] = src[i];
        }
    }

    template <typename Derived>
    static void copy_mujoco_to_eigen(const double *src, Eigen::MatrixBase<Derived> &dst, size_t size)
    {
        static_assert(Derived::ColsAtCompileTime == 1, "Only column (double) vectors are supported");
        for (size_t i = 0; i < size; ++i)
        {
            dst[i] = src[i];
        }
    }

    //-----------------------------------------------------
    // The environment class
    //-----------------------------------------------------
    template <int BATCH_SIZE_>
    class UnitreeGo2Env
    {
    public:
        static const int BATCH_SIZE = BATCH_SIZE_;

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

            d_main_ = mj_makeData(m_);
            for (int i = 0; i < BATCH_SIZE; i++)
            {
                d_[i] = mj_makeData(m_);
            }

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
            init_q_ = Vector19d::Zero(m_->nq);
            default_pose_ = Vector12d::Zero(m_->nq - 7);
            if (home_id < 0)
            {
                std::cerr << "Keyframe 'home' not found; defaulting qpos=0" << std::endl;
            }
            else
            {
                for (size_t i = 0; i < m_->nq; i++)
                {
                    init_q_[i] = m_->key_qpos[home_id * m_->nq + i];
                }
                for (size_t i = 7; i < m_->nq; i++)
                {
                    default_pose_[i - 7] = m_->key_qpos[home_id * m_->nq + i];
                }
            }
            std::cout << "Initial config: " << init_q_.transpose() << std::endl;
            std::cout << "Default pose: " << default_pose_.transpose() << std::endl;

            // get joint limits from mujoco instead:
            joint_range_ = Matrix12Bounds::Zero();
            for (size_t i = 0; i < 12; i++)
            {
                joint_range_(i, 0) = m_->jnt_range[i * 2];
                joint_range_(i, 1) = m_->jnt_range[i * 2 + 1];
            }

            std::cout << "Joint range: " << joint_range_ << std::endl;

            physical_joint_range_ = joint_range_;

            // get torque limits from mujoco instead:
            joint_torque_range_ = Matrix12Bounds::Zero();
            for (size_t i = 0; i < 12; i++)
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
                gait_ = "stand";
            }
            else
            {
                gait_ = config_.gait;
            }
        }

        ~UnitreeGo2Env()
        {
            mj_deleteData(d_main_);
            for (int i = 0; i < BATCH_SIZE; i++)
            {
                mj_deleteData(d_[i]);
            }
            mj_deleteModel(m_);
        }

        // steps a particular mjdata for a sequence of actions given the initial state, returns the sequence of environment states
        template <typename Derived>
        std::vector<EnvState> stepTrajectory(size_t data_index, const EnvState &state_init, const Eigen::MatrixBase<Derived> &actions)
        {
            assert(actions.cols() == 12 && "Actions must be Nx12");
            std::vector<EnvState> states;
            states.reserve(actions.rows());
            EnvState state = state_init;
            copy_eigen_to_mujoco(d_[data_index]->qpos, state_init.pipeline_state.qpos, 19);
            copy_eigen_to_mujoco(d_[data_index]->qvel, state_init.pipeline_state.qvel, 18);
            for (size_t i = 0; i < actions.rows(); i++)
            {
                Vector12d transposed_action = actions.row(i).transpose();
                state = step(data_index, state, transposed_action);
                states.push_back(state);
            }
            return states;
        }

        template <typename Derived>
        EnvState step(const EnvState &state_init, const Eigen::MatrixBase<Derived> &action)
        {
            return step(d_main_, state_init, action);
        }

        template <typename Derived>
        EnvState step(size_t data_index, const EnvState &state_init, const Eigen::MatrixBase<Derived> &action)
        {
            return step(d_[data_index], state_init, action);
        }

        template <typename Derived>
        EnvState step(mjData *d, const EnvState &state_init, const Eigen::MatrixBase<Derived> &action)
        {
            assert(action.rows() == 12 && "Action must be 12x1");

            // Vector19d qpos_init = state_init.pipeline_state.qpos;
            // Vector18d qvel_init = state_init.pipeline_state.qvel;

            Vector19d qpos_init = Vector19d::Zero();
            Vector18d qvel_init = Vector18d::Zero();
            copy_mujoco_to_eigen(d->qpos, qpos_init, 19);
            copy_mujoco_to_eigen(d->qvel, qvel_init, 18);

            Vector12d ctrl = act2tau(qpos_init.tail<12>(), qvel_init.tail<12>(), action);

            pipelineStep(d, qpos_init, qvel_init, ctrl);

            PipelineState pipeline_state = createTransformedState(d);
            EnvState new_state = state_init;

            double scaled = std::min(config_.default_vx * (new_state.info.step * dt()) / config_.ramp_up_time, config_.default_vx);
            new_state.info.vel_tar[0] = scaled;
            scaled = std::min(config_.default_vy * (new_state.info.step * dt()) / config_.ramp_up_time, config_.default_vy);
            new_state.info.vel_tar[1] = scaled;
            double yaw_scaled = std::min(config_.default_vyaw * (new_state.info.step * dt()) / config_.ramp_up_time, config_.default_vyaw);
            new_state.info.ang_vel_tar[2] = yaw_scaled;

            Vector4d z_feet;
            for (size_t i = 0; i < 4; i++)
            {
                z_feet[i] = d->site_xpos[feet_site_id_[i] * 3 + 2];
            }
            Vector4d z_feet_tar = computeFootStep(new_state.info);

            double reward_gaits = 0.0;
            for (size_t i = 0; i < 4; i++)
            {
                double diff = (z_feet_tar[i] - z_feet[i]) / 0.05;
                reward_gaits -= diff * diff;
            }

            Vector3d up_global(0, 0, 1);
            Vector3d up_body = rotate(up_global, pipeline_state.x.quat_);
            double reward_upright = -(up_body - up_global).squaredNorm();

            double yaw_tar = new_state.info.yaw_tar + new_state.info.ang_vel_tar[2] * dt() * new_state.info.step;
            double yaw = quat_to_yaw(pipeline_state.x.quat_);
            double yaw_error = yaw - yaw_tar;
            double wrapped = atan2(std::sin(yaw_error), std::cos(yaw_error));
            double reward_yaw = -(wrapped * wrapped);

            Vector3d vb = global_to_body_velocity(pipeline_state.xd.lin_, pipeline_state.x.quat_);
            Vector3d ab = global_to_body_velocity(pipeline_state.xd.ang_, pipeline_state.x.quat_);
            double reward_vel = -(vb.head<2>() - new_state.info.vel_tar.head<2>()).squaredNorm();
            double reward_ang_vel = -std::pow(ab[2] - new_state.info.ang_vel_tar[2], 2.0);

            double reward_height = -std::pow((pipeline_state.x.pos_[2] - new_state.info.pos_tar[2]), 2);

            double reward = 0.1 * reward_gaits + 0.5 * reward_upright + 0.3 * reward_yaw + 1.0 * reward_vel + 1.0 * reward_ang_vel + 1.0 * reward_height;

            bool done = false;

            if (up_body.dot(up_global) < 0.0 || pipeline_state.x.pos_[2] < 0.18)
            {
                done = true;
            }

            for (int i = 0; i < joint_range_.rows(); i++)
            {
                double angle = pipeline_state.qpos[7 + i];
                double lower = joint_range_(i, 0);
                double upper = joint_range_(i, 1);
                if (angle < lower || angle > upper)
                {
                    done = true;
                }
            }

            new_state.info.step += 1;

            new_state.pipeline_state = pipeline_state;
            new_state.reward = reward;
            new_state.done = done;

            return new_state;
        }

        EnvState reset(std::mt19937_64 &rng)
        {
            Vector18d zero_dq = Vector18d::Zero();
            PipelineState pipeline_state;

            mj_resetData(m_, d_main_);
            pipeline_state = pipelineInit(d_main_, init_q_, zero_dq);
            for (int i = 0; i < BATCH_SIZE; i++)
            {
                mj_resetData(m_, d_[i]);
                PipelineState tmp = pipelineInit(i, init_q_, zero_dq);
            }

            StateInfo info;
            info.rng = rng;
            info.pos_tar = Vector3d(0.0, 0.0, 0.27);
            info.vel_tar = Vector3d(0.0, 0.0, 0.0);
            info.ang_vel_tar = Vector3d(0.0, 0.0, 0.0);
            info.yaw_tar = 0.0;
            info.step = 0;
            info.z_feet = Vector4d::Zero();
            info.z_feet_tar = Vector4d::Zero();
            info.last_contact = std::vector<bool>(4, false);
            info.feet_air_time = std::vector<double>(4, 0.0);

            EnvState s;
            s.pipeline_state = pipeline_state;
            s.reward = 0.0;
            s.done = false;
            s.info = info;

            return s;
        }

        // return the mjModel pointer
        mjModel *model() { return m_; }

        // this doesnt really make sense anymore
        // mjData *data() { return d_; }

        double dt() const { return config_.timestep; }

        size_t action_size() const { return action_size_; }

        Matrix12Bounds joint_range() const { return joint_range_; }

        Matrix12Bounds joint_torque_range() const { return joint_torque_range_; }

    protected:
        PipelineState pipelineInit(size_t data_index, const Vector19d &qpos, const Vector18d &qvel)
        {
            return pipelineInit(d_[data_index], qpos, qvel);
        }

        PipelineState pipelineInit(mjData *d, const Vector19d &qpos, const Vector18d &qvel)
        {
            copy_eigen_to_mujoco(d->qpos, qpos, 19);
            copy_eigen_to_mujoco(d->qvel, qvel, 18);

            mj_forward(m_, d);
            return createTransformedState(d);
        }

        PipelineState pipelineStep(size_t data_index, const Vector19d &qpos, const Vector18d &qvel, const Vector12d &ctrl)
        {
            return pipelineStep(d_[data_index], qpos, qvel, ctrl);
        }

        PipelineState pipelineStep(mjData *d, const Vector19d &qpos, const Vector18d &qvel, const Vector12d &ctrl)
        {
            // copy_eigen_to_mujoco(d->qpos, qpos, 19);
            // copy_eigen_to_mujoco(d->qvel, qvel, 18);

            copy_eigen_to_mujoco(d->ctrl, ctrl, 12);

            mj_step(m_, d);
            return createTransformedState(d);
        }

        PipelineState createTransformedState(size_t data_index)
        {
            return createTransformedState(d_[data_index]);
        }

        PipelineState createTransformedState(mjData *d)
        {
            Vector19d qpos_out = Vector19d::Zero();
            Vector18d qvel_out = Vector18d::Zero();
            copy_mujoco_to_eigen(d->qpos, qpos_out, 19);
            copy_mujoco_to_eigen(d->qvel, qvel_out, 18);

            Vector3d xpos = Vector3d::Zero();
            Vector4d xquat = Vector4d::Zero();
            copy_mujoco_to_eigen(&d->xpos[torso_idx_ * 3], xpos, 3);
            copy_mujoco_to_eigen(&d->xquat[torso_idx_ * 4], xquat, 4);
            Transform x(xpos, xquat);

            Vector3d cvel_lin(d->cvel[3], d->cvel[4], d->cvel[5]);
            Vector3d cvel_ang(d->cvel[0], d->cvel[1], d->cvel[2]);
            Motion cvel(cvel_lin, cvel_ang);

            Vector3d offset(d->xpos[torso_idx_ * 3 + 0] - d->subtree_com[m_->body_rootid[torso_idx_] * 3 + 0],
                            d->xpos[torso_idx_ * 3 + 1] - d->subtree_com[m_->body_rootid[torso_idx_] * 3 + 1],
                            d->xpos[torso_idx_ * 3 + 2] - d->subtree_com[m_->body_rootid[torso_idx_] * 3 + 2]);
            Transform offset_t(offset);

            Motion xd = offset_t.apply(cvel);

            return PipelineState{qpos_out, qvel_out, x, xd};
        }

        Vector4d computeFootStep(const StateInfo &info)
        {
            Vector3d gp = kGaitParams_[gait_];
            double duty_ratio = gp[0];
            double cadence = gp[1];
            double amplitude = gp[2];

            Vector4d phases = kGaitPhases_[gait_];

            double time_sec = info.step * dt();
            return get_foot_step(duty_ratio, cadence, amplitude, phases, time_sec);
        }

        template <typename UDerived>
        Vector12d act2joint(const Eigen::MatrixBase<UDerived> &act) const
        {
            assert(act.rows() == 12 && "Action must be 12x1");
            size_t N = joint_range_.rows(); // e.g. 12
            Vector12d result = Vector12d::Zero();
            Vector12d act_normalized = (act * config_.action_scale + Vector12d::Constant(1.0)) / 2.0;
            for (size_t i = 0; i < N; i++)
            {
                // scale to joint_range
                double low = joint_range_(i, 0);
                double high = joint_range_(i, 1);
                double jt = low + act_normalized[i] * (high - low);

                // clip to physical_joint_range_
                double p_low = physical_joint_range_(i, 0);
                double p_high = physical_joint_range_(i, 1);
                jt = clamp(jt, p_low, p_high);

                result[i] = jt;
            }
            return result;
        }

        template <typename QJDerived, typename QdJDerived, typename UDerived>
        Vector12d act2tau(const Eigen::MatrixBase<QJDerived> &qj, const Eigen::MatrixBase<QdJDerived> &qdj, const Eigen::MatrixBase<UDerived> &act) const
        {
            assert(qj.rows() == 12 && "qj must be 12x1");
            assert(qdj.rows() == 12 && "qdj must be 12x1");
            assert(act.rows() == 12 && "Action must be 12x1");
            const size_t N = joint_range_.rows();
            Vector12d joint_target = act2joint(act);

            // PD
            Vector12d qj_err = joint_target - qj;
            Vector12d tau = Vector12d::Zero();
            for (size_t i = 0; i < N; i++)
            {
                double val = config_.kp * qj_err[i] - config_.kd * qdj[i];
                // clip to joint_torque_range_
                double tmin = joint_torque_range_(i, 0);
                double tmax = joint_torque_range_(i, 1);
                tau[i] = clamp(val, tmin, tmax);
            }
            return tau;
        }

        std::pair<Vector3d, Vector3d> sample_command(std::mt19937_64 &rng) const
        {
            std::uniform_real_distribution<double> dist_lin_x(-1.5, 1.5);
            std::uniform_real_distribution<double> dist_lin_y(-0.5, 0.5);
            std::uniform_real_distribution<double> dist_yaw(-1.5, 1.5);

            double lx = dist_lin_x(rng);
            double ly = dist_lin_y(rng);
            double yw = dist_yaw(rng);

            Vector3d lin(lx, ly, 0.0);
            Vector3d ang(0.0, 0.0, yw);
            return {lin, ang};
        }

        // --------------------------------------------------
        // fill in the gait tables (phases, params)
        // --------------------------------------------------
        void setupGaitTables()
        {
            kGaitPhases_["stand"] = (Vector4d() << 0.0, 0.0, 0.0, 0.0).finished();
            kGaitPhases_["walk"] = (Vector4d() << 0.0, 0.5, 0.75, 0.25).finished();
            kGaitPhases_["trot"] = (Vector4d() << 0.0, 0.5, 0.5, 0.0).finished();
            kGaitPhases_["canter"] = (Vector4d() << 0.0, 0.33, 0.33, 0.66).finished();
            kGaitPhases_["gallop"] = (Vector4d() << 0.0, 0.05, 0.4, 0.35).finished();

            // (duty_ratio, cadence, amplitude)
            kGaitParams_["stand"] = (Vector3d() << 1.0, 1.0, 0.0).finished();
            kGaitParams_["walk"] = (Vector3d() << 0.75, 1.0, 0.08).finished();
            kGaitParams_["trot"] = (Vector3d() << 0.45, 2.0, 0.08).finished();
            kGaitParams_["canter"] = (Vector3d() << 0.4, 4.0, 0.06).finished();
            kGaitParams_["gallop"] = (Vector3d() << 0.3, 3.5, 0.10).finished();
        }

        UnitreeGo2EnvConfig config_;

        mjModel *m_ = nullptr;
        mjData *d_main_ = nullptr;
        mjData *d_[BATCH_SIZE];

        int torso_idx_;
        std::vector<int> feet_site_id_;

        // From the Python code
        Matrix12Bounds joint_range_;
        Matrix12Bounds physical_joint_range_;
        Matrix12Bounds joint_torque_range_;

        Vector19d init_q_;
        Vector12d default_pose_;
        double foot_radius_;

        std::string gait_;

        // Gait phase table
        std::map<std::string, Vector4d> kGaitPhases_;
        // Gait param table: (duty_ratio, cadence, amplitude)
        std::map<std::string, Vector3d> kGaitParams_;

        size_t action_size_;
    };

} // namespace go2env