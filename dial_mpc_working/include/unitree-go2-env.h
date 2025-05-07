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
#include <cmath>
#include <chrono>

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
        std::cout<<"Euler"<<euler<<std::endl;
        double yaw = euler[0];
        return yaw;
    }
    Eigen::Vector3d quatToEuler(Vector4d q_vec) {
        // Assuming q_vec is in [w, x, y, z] order
        double w = q_vec[0];
        double x = q_vec[1];
        double y = q_vec[2];
        double z = q_vec[3];
        
        Eigen::Vector3d euler;
        
        // Roll (x-axis rotation)
        double sinr_cosp = 2.0 * (w * x + y * z);
        double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
        euler[0] = std::atan2(sinr_cosp, cosr_cosp);
        
        // Pitch (y-axis rotation)
        double sinp = 2.0 * (w * y - z * x);
        if (std::abs(sinp) >= 1)
            euler[1] = std::copysign(M_PI / 2, sinp); // Use 90 degrees if out of range
        else
            euler[1] = std::asin(sinp);
        
        // Yaw (z-axis rotation)
        // double siny_cosp = 2.0 * (w * z + x * y);
        // double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
        // euler[2] = std::atan2(siny_cosp, cosy_cosp);

        // Match Python's yaw calculation
        double sin_yaw = -2.0 * (x * y) + 2.0 * (w * z);
        double cos_yaw = x * x + w * w - z * z - y * y;
        euler[2] = std::atan2(sin_yaw, cos_yaw);
        
        return euler;
    }

    //-----------------------------------------------------
    // Compute the quaternion inverse
    //-----------------------------------------------------
    static Vector4d quat_inv(const Vector4d &q)
    {
        Vector4d q_inv = q;
        q_inv.segment(1, 3) *= -1;
        return q_inv;
        // std::cout << "q inside quat_inv: " << q <<"\n";
        // return Vector4d(q[0], -q[1], -q[2], -q[3]);
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

        // Motion apply(const Motion &m)
        std::tuple<Vector4d, Motion, Vector4d, Vector3d, Vector3d, Motion> apply(const Motion &m)
        {
            Vector4d quat_inv1 = quat_inv(quat_);
            Vector3d ang = rotate(m.ang_, quat_inv1);
            Vector3d lin = rotate(m.lin_ - pos_.cross(m.ang_), quat_inv1);
            return std::make_tuple(quat_, m, quat_inv1, ang, lin, Motion(lin, ang));
            // return Motion(lin, ang);
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

    // struct JaxRNG {
    //     std::mt19937_64 rng;
    
    //     // Constructor initializes with a seed
    //     explicit JaxRNG(uint64_t seed) : rng(seed) {}
    
    //     // Split function: creates two new independent RNGs
    //     std::tuple<JaxRNG, JaxRNG> split() {
    //         std::uniform_int_distribution<uint64_t> dist;
    //         uint64_t new_seed1 = dist(rng);
    //         uint64_t new_seed2 = dist(rng);
    //         return {JaxRNG(new_seed1), JaxRNG(new_seed2)};
    //     }
    // };

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
        // std::vector<std::array<double, 3>> contactForces;
        std::map<std::string, std::array<double, 3>> footContacts;
        std::vector<double> jointAccelerations;
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
            // std::cout << "Torso index: " << torso_idx_ << std::endl;

            // Identify feet site indices
            std::vector<std::string> feet_site_names = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"};
            for (auto &site_name : feet_site_names)
            {
                int sid = mj_name2id(m_, mjOBJ_SITE, site_name.c_str());
                if (sid < 0)
                {
                    std::cerr << "[Warning] Site not found: " << site_name << std::endl;
                }
                // std::cout << "Site id for " << site_name << ": " << sid << std::endl;
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
            // std::cout << "------ INSIDE STEP TRAJECTORY ------" << std::endl;
            assert(actions.cols() == 12 && "Actions must be Nx12");
            std::vector<EnvState> states;
            states.reserve(actions.rows());
            EnvState state = state_init;
            copy_eigen_to_mujoco(d_[data_index]->qpos, state_init.pipeline_state.qpos, 19);
            copy_eigen_to_mujoco(d_[data_index]->qvel, state_init.pipeline_state.qvel, 18);
            for (size_t i = 0; i < actions.rows(); i++)
            {
                Vector12d transposed_action = actions.row(i).transpose();
                // if (i == 0)
                // {
                //     transposed_action << -1.1175871e-08, -1.1175871e-08, -1.1175871e-08, -1.1175871e-08,
                //          -1.1175871e-08, -1.1175871e-08, -1.1175871e-08, -1.1175871e-08,
                //          -1.1175871e-08, -1.1175871e-08, -1.1175871e-08, -1.1175871e-08;
                // }
                // std::cout << "State Going In: " << std::endl;
                // std::cout << "state qpos: " << state.pipeline_state.qpos << std::endl;
                // std::cout << "state qvel: " << state.pipeline_state.qvel << std::endl;
                // std::cout << "state x (pos): " << state.pipeline_state.x.pos_<< std::endl;
                // std::cout << "state reward: " << state.reward << std::endl;
                // std::cout << "Action Going in: " << std::endl;
                // std::cout << transposed_action << std::endl;
                // std::cin.get();
                state = step(data_index, state, transposed_action);
                // std::cout << "State Coming out: " << std::endl;
                // std::cout << "state qpos: " << state.pipeline_state.qpos << std::endl;
                // std::cout << "state qvel: " << state.pipeline_state.qvel << std::endl;
                // std::cout << "state x (pos): " << state.pipeline_state.x.pos_<< std::endl;
                // std::cout << "state reward: " << state.reward << std::endl;
                // std::cin.get();
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
            // std::cout << "Control after act2tau: " << ctrl << std::endl;
            // std::cin.get();

            PipelineState pipeline_state = pipelineStep(d, qpos_init, qvel_init, ctrl);

        //     pipeline_state.qpos << -1.1257210e-03, -3.5538673e-07,  2.6583159e-01,  9.9987209e-01,
        //     2.0311712e-05, -1.5999353e-02, -1.0907480e-06, -7.4294247e-03,
        //     9.3215001e-01, -1.7977694e+00,  7.3568495e-03,  9.3213862e-01,
        //    -1.7977433e+00, -4.3462031e-03,  9.6818620e-01, -1.8815672e+00,
        //     4.2731212e-03,  9.6817535e-01, -1.8815459e+00;
            
        //     pipeline_state.qvel << -3.7132964e-02, -1.2411534e-05, -1.4208087e-01,  1.2068768e-03,
        //     -1.0404927e+00, -7.1485731e-05, -2.8329131e-01,  8.7310266e-01,
        //      6.3354045e-01,  2.8121385e-01,  8.7275994e-01,  6.3433033e-01,
        //     -1.9744571e-01,  1.8684653e+00, -1.9873904e+00,  1.9544220e-01,
        //      1.8681194e+00, -1.9867712e+00;
            
        //     pipeline_state.x.pos_ << -3.8306182e-04, -1.0715606e-07, 2.6867321e-01;
            // PipelineState pipeline_state = createTransformedState(d);
            // std::cout << "Pipeline State: " << std::endl;
            // std::cout << "Pipeline state qpos: " << pipeline_state.qpos << std::endl;
            // std::cout << "Pipeline state qvel: " << pipeline_state.qvel << std::endl;
            // std::cout << "Len Pipeline state x (pos): " << pipeline_state.x.pos_.size()<< std::endl;
            // std::cin.get();

            // // Test values
            // pipeline_state.x.pos_ << -1.1257210e-03, -3.5538673e-07, 2.6583159e-01;
            // pipeline_state.x.quat_ << 9.9987209e-01, 2.0311712e-05, -1.5999353e-02, -1.0907480e-06;
            // pipeline_state.xd.lin_ << -3.7132964e-02, -1.2411538e-05, -1.4208087e-01;
            // pipeline_state.xd.ang_ << 1.2069531e-03, -1.0404928e+00, -7.5134871e-05;
            EnvState new_state = state_init;

            double scaled = std::min(config_.default_vx * new_state.info.step * dt() / config_.ramp_up_time, config_.default_vx);
            new_state.info.vel_tar[0] = scaled;
            scaled = std::min(config_.default_vy * new_state.info.step * dt() / config_.ramp_up_time, config_.default_vy);
            new_state.info.vel_tar[1] = scaled;
            double yaw_scaled = std::min(config_.default_vyaw * new_state.info.step * dt() / config_.ramp_up_time, config_.default_vyaw);
            new_state.info.ang_vel_tar[2] = yaw_scaled;

            Vector4d z_feet;
            // std::cout<<"feet_site_id: \n";
            for (size_t i = 0; i < 4; i++)
            {
                z_feet[i] = d->site_xpos[feet_site_id_[i] * 3 + 2];
                // std::cout<<feet_site_id_[i]<<std::endl;
            }
            // std::cout<<z_feet<<std::endl;
            Vector4d z_feet_tar = computeFootStep(new_state.info);
            // std::cout << "z_feet: " << z_feet << std::endl;
            // std::cout << "z_feet_tar: " << z_feet_tar << std::endl;

            // Updating z_feet and z_feet_tar in state info
            new_state.info.z_feet = z_feet;
            new_state.info.z_feet_tar = z_feet_tar;

            double reward_gaits = 0.0;
            for (size_t i = 0; i < 4; i++)
            {
                double diff = (z_feet_tar[i] - z_feet[i]) / 0.05;
                reward_gaits -= diff * diff;
            }
            // std::cout<< "Reward_gaits: "<< reward_gaits <<std::endl;
            Vector3d up_global(0, 0, 1);
            Vector3d up_body = rotate(up_global, pipeline_state.x.quat_);
            double reward_upright = -(up_body - up_global).squaredNorm();
            // std::cout<< "Reward_upright: "<< reward_upright <<std::endl;

            double yaw_tar = new_state.info.yaw_tar + new_state.info.ang_vel_tar[2] * dt() * new_state.info.step;
            // std::cout<< "Yaw_tar: "<<yaw_tar<<std::endl;
            Vector3d euler1 = quatToEuler(pipeline_state.x.quat_);
            // double yaw = quat_to_yaw(pipeline_state.x.quat_);
            double yaw =  euler1[2];
            // std::cout<<"Quat: " << pipeline_state.x.quat_ <<std::endl;
            // std::cout<< "Euler: "<<euler1<<std::endl;
            double yaw_error = yaw - yaw_tar;
            double wrapped = atan2(std::sin(yaw_error), std::cos(yaw_error));
            // std::cout<< "Yaw error wrapped: "<<wrapped<<std::endl;
            double reward_yaw = -(wrapped * wrapped);
            // std::cout<< "Reward_yaw: "<< reward_yaw <<std::endl;

            Vector3d vb = global_to_body_velocity(pipeline_state.xd.lin_, pipeline_state.x.quat_);
            Vector3d ab = global_to_body_velocity(pipeline_state.xd.ang_ * M_PI/ 180.0, pipeline_state.x.quat_);
            // std::cout<<"xd.lin: "<<pipeline_state.xd.lin_<<std::endl;
            // std::cout<<"vb: "<<vb<<std::endl;
            // std::cout<<"vel_tar"<< new_state.info.vel_tar<<std::endl;
            // std::cout<<"xd.ang: "<<pipeline_state.xd.ang_<<std::endl;
            double reward_vel = -(vb.head<2>() - new_state.info.vel_tar.head<2>()).squaredNorm();
            // std::cout<< "Reward_vel: "<< reward_vel <<std::endl;
            // std::cout << "ab: " << ab << std::endl;
            // double reward_ang_vel = -(ab.head<2>() - new_state.info.ang_vel_tar.head<2>()).squaredNorm();
            double reward_ang_vel = -std::pow(ab[2] - new_state.info.ang_vel_tar[2], 2.0);
            
            // std::cout<< "Reward_ang_vel: "<< reward_ang_vel <<std::endl;

            double reward_height = -std::pow((pipeline_state.x.pos_[2] - new_state.info.pos_tar[2]), 2);
            // std::cout<< "Reward_height: "<< reward_height <<std::endl;
            // std::cout << "step: " << new_state.info.step << "\nrew_gaits: " << reward_gaits <<
            // "\nrew_upright: " << reward_upright << "\nrew_yaw: " << reward_yaw << "\nrew_vel: " <<
            // reward_vel << "\nrew_ang_vel: " << reward_ang_vel << "rew_height: " << reward_height << std::endl;

            double reward = 0.1 * reward_gaits + 0.5 * reward_upright + 0.3 * reward_yaw + 1.0 * reward_vel + 1.0 * reward_ang_vel + 1.0 * reward_height;
            // std::cout<< "Total_reward: "<< reward <<std::endl; 
            // std::cin.get();

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

            // std::vector<std::array<double, 3>> contactForces;
            std::map<std::string, std::array<double, 3>> footContacts;
            for (int i = 0; i < d->ncon; i++) {
                mjContact* con = &d->contact[i];
        
                // Get geom IDs
                // int geom1 = con->geom1;
                int geom2 = con->geom2;
                
                // Get geom names
                // const char* geom1_name = m_->names + m_->name_geomadr[geom1];
                const char* geom2_name = m_->names + m_->name_geomadr[geom2];
                // std::cout << "Geom 1" << ": " << geom1_name << std::endl;
                // std::cout << "Geom 2" << ": " << geom2_name << std::endl;

                // Check if this is a foot contact and identify which foot
                std::string footName = geom2_name;
                
                // Assuming your foot geoms have names like "foot_FL", "foot_FR", etc.
                // if (strstr(geom1_name, "foot_") != nullptr) {
                //     footName = geom1_name;
                // } else if (strstr(geom2_name, "foot_") != nullptr) {
                    // footName = geom2_name;
                // }

                mjtNum result[6];
                
                mj_contactForce(m_, d, i, result);
                
                // Store the force
                std::array<double, 3> forceArray = {result[0], result[1], result[2]};
                footContacts[footName] = forceArray;
                // std::cout << "Force on " << footName << ": []" << forceArray[0] << ", " << forceArray[1] << ", " << forceArray[2] << "]" << std::endl;
                // contactForces.push_back(forceArray);
            }
            // new_state.info.contactForces = contactForces;

            new_state.info.footContacts = footContacts;

            std::vector<double> jointAccelerations(m_->nu);
            // Copy the accelerations from mjData
            for (int i = 0; i < m_->nu; i++) {
                jointAccelerations[i] = d->qacc[6 + i];
            }
            new_state.info.jointAccelerations = jointAccelerations;
            // std::cin.get();
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
            info.pos_tar = Vector3d(0.282, 0.0, 0.3); // 0.0, 0.0, 0.27
            info.vel_tar = Vector3d(0.0, 0.0, 0.0);
            info.ang_vel_tar = Vector3d(0.0, 0.0, 0.0);
            info.yaw_tar = 0.0;
            info.step = 0;
            info.z_feet = Vector4d::Zero();
            info.z_feet_tar = Vector4d::Zero();
            info.last_contact = std::vector<bool>(4, false);
            info.feet_air_time = std::vector<double>(4, 0.0);
            // info.contactForces.push_back({0.0, 0.0, 0.0});
            // List of foot names
            std::vector<std::string> footNames = {"FL", "FR", "RL", "RR"};

            // Initialize all to zero
            for (const auto& footName : footNames) {
                info.footContacts[footName] = {0.0, 0.0, 0.0};
            }
            info.jointAccelerations = std::vector<double>(m_->nu, 0.0);

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

        double foot_radius_;

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
            auto result = createTransformedState(d);
            auto [motion_tuple, ps] = result;
            return ps;
            // return createTransformedState(d);
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
            // std::cout <<"Before step before assigning: " <<std::endl;
            // std::cout<< "q_pos: " <<std::endl;
            // for (int i = 0; i<19; i++){
            //     std::cout << d->qpos[i]<<", ";
            // }
            // std::cout<< "\nq_vel: " <<std::endl;
            // for (int i = 0; i<18; i++){
            //     std::cout << d->qvel[i]<<", ";
            // }
            // std::cin.get();
            // Vector19d possss;
            // possss << -3.8306182e-04, -1.0715606e-07, 2.6867321e-01, 9.9998444e-01,
            //     8.2435199e-06, -5.5950787e-03, -3.5768699e-07, -1.7635984e-03,
            //     9.1468793e-01, -1.8104402e+00, 1.7325724e-03, 9.1468340e-01,
            //     -1.8104299e+00, -3.9728897e-04, 9.3081689e-01, -1.8418194e+00,
            //     3.6427772e-04, 9.3081295e-01, -1.8418105e+00;
            // copy_eigen_to_mujoco(d->qpos, possss, 19);
            // Vector18d velllll;
            // velllll << -1.9153092e-02, -5.3578033e-06, -6.6340007e-02, 8.2435622e-04,
            //     -5.5951077e-01, -3.5768880e-05, -8.8179924e-02, 7.3439878e-01,
            //     -5.2201146e-01, 8.6628623e-02, 7.3417050e-01, -5.2150005e-01,
            //     -1.9864449e-02, 1.5408450e+00, -2.0909727e+00, 1.8213887e-02,
            //     1.5406494e+00, -2.0905232e+00;
            // copy_eigen_to_mujoco(d->qvel, velllll, 18);
            // std::cout <<"Before step after assigning: " <<std::endl;
            // std::cout<< "q_pos: " <<std::endl;
            // for (int i = 0; i<19; i++){
            //     std::cout << d->qpos[i]<<", ";
            // }
            // std::cout<< "\nq_vel: " <<std::endl;
            // for (int i = 0; i<18; i++){
            //     std::cout << d->qvel[i]<<", ";
            // }
            // std::cin.get();
            // mj_fwdPosition(m_, d);
            mj_step(m_, d);
            
            // std::cout <<"After step before assigning: " << std::endl;
            // std::cout<< "q_pos: " <<std::endl;
            // for (int i = 0; i<19; i++){
            //     std::cout << d->qpos[i]<<", ";
            // }
            // std::cout<< "\nq_vel: " <<std::endl;
            // for (int i = 0; i<18; i++){
            //     std::cout << d->qvel[i]<<", ";
            // }
            // std::cin.get();

            // Vector19d posssss1;
            // posssss1 << -1.1257210e-03, -3.5538503e-07, 2.6583159e-01, 9.9987209e-01,
            //     2.0311869e-05, -1.5999353e-02, -1.0906903e-06, -7.4294251e-03,
            //     9.3215001e-01, -1.7977694e+00, 7.3568486e-03, 9.3213862e-01,
            //     -1.7977433e+00, -4.3462031e-03, 9.6818620e-01, -1.8815672e+00,
            //     4.2731203e-03, 9.6817535e-01, -1.8815459e+00;

            // copy_eigen_to_mujoco(d->qpos, posssss1, 19);

            // Vector18d velllll1;
            // velllll1 << -3.7132964e-02, -1.2411449e-05, -1.4208087e-01, 1.2068923e-03,
            //     -1.0404927e+00, -7.1480041e-05, -2.8329134e-01, 8.7310266e-01,
            //         6.3354045e-01, 2.8121382e-01, 8.7276000e-01, 6.3433033e-01,
            //     -1.9744571e-01, 1.8684654e+00, -1.9873908e+00, 1.9544214e-01,
            //         1.8681195e+00, -1.9867715e+00;
            // copy_eigen_to_mujoco(d->qvel, velllll1, 18);

            // std::cout <<"After step after assigning: " << std::endl;
            // std::cout<< "q_pos: " <<std::endl;
            // for (int i = 0; i<19; i++){
            //     std::cout << d->qpos[i]<<", ";
            // }
            // std::cout<< "\nq_vel: " <<std::endl;
            // for (int i = 0; i<18; i++){
            //     std::cout << d->qvel[i]<<", ";
            // }
            // std::cout << "nbody len: " << m_->nbody << "\n";
            // std::cout << "torso_idx: " << torso_idx_ << "\n";
            // // std::cout << "cvel[0] len: " << sizeof(d->cvel[0]) << "\n";
            // for (int i = 0; i < 6; i++) {
            //     std::cout << "cvel[" << i << "]: " << d->cvel[torso_idx_*6+i] << "\n";
            // }
            // std::cin.get();

            auto result = createTransformedState(d);
            auto [motion_tuple, ps] = result;
            auto [cvel_ang, cvel_lin, offset, quat_, original_motion, quat_inverse, rotated_ang, rotated_lin] = motion_tuple;
            // std::cout << "cvel ang: " << cvel_ang << "\ncvel lin: " << cvel_lin << "\n";
            // std::cout << "offset: " << offset << "\n";
            // std::cout << "input quat_: " << quat_ << "\n";
            // std::cout << "input motion (ang): " << original_motion.ang_ << "\ninput motion (lin): " << original_motion.lin_ << "\n";
            // std::cout << "quat_inv: " << quat_inverse << "\n";
            // std::cout << "rotated ang: " << rotated_ang << "\n";
            // std::cout << "rotated lin: " << rotated_lin << "\n";
            // std::cout << "Pipeline State Out: " << std::endl;
            // std::cout << "Pipeline state qpos: " << ps.qpos << std::endl;
            // std::cout << "Pipeline state qvel: " << ps.qvel << std::endl;
            // std::cout << "Pipeline state x (pos): " << ps.x.pos_<< std::endl;
            // std::cout << "Pipeline state x (quat): " << ps.x.quat_<< std::endl;
            // std::cout << "Pipeline state xd (lin): " << ps.xd.lin_<< std::endl;
            // std::cout << "Pipeline state xd (ang): " << ps.xd.ang_<< std::endl;
            // std::cin.get();

            return ps;
        }

        PipelineState createTransformedState(size_t data_index)
        {   
            auto result = createTransformedState(d_[data_index]);
            auto [motion_tuple, ps] = result;
            return ps;
        }

        // PipelineState createTransformedState(mjData *d)
        std::tuple<std::tuple<Vector3d, Vector3d, Vector3d, Vector4d, Motion, Vector4d, Vector3d, Vector3d>, PipelineState> createTransformedState(mjData *d)
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

            Vector3d cvel_lin(d->cvel[torso_idx_*6+3], d->cvel[torso_idx_*6+4], d->cvel[torso_idx_*6+5]);
            Vector3d cvel_ang(d->cvel[torso_idx_*6+0], d->cvel[torso_idx_*6+1], d->cvel[torso_idx_*6+2]);
            Motion cvel(cvel_lin, cvel_ang);
            // std::cout << "cvel ang: " << cvel.ang_ << "\ncvel lin: " << cvel.lin_ << "\n";
            Vector3d offset(d->xpos[torso_idx_ * 3 + 0] - d->subtree_com[m_->body_rootid[torso_idx_] * 3 + 0],
                            d->xpos[torso_idx_ * 3 + 1] - d->subtree_com[m_->body_rootid[torso_idx_] * 3 + 1],
                            d->xpos[torso_idx_ * 3 + 2] - d->subtree_com[m_->body_rootid[torso_idx_] * 3 + 2]);
            // std::cout << "offset: " << offset << "\n";
            Transform offset_t(offset);

            // Motion xd = offset_t.apply(cvel);
            auto [quat_, original_motion, quat_inverse, rotated_ang, rotated_lin, xd] = offset_t.apply(cvel);

            // std::cin.get();
            return std::make_tuple(
                std::make_tuple(cvel_ang, cvel_lin, offset, quat_, original_motion, quat_inverse, rotated_ang, rotated_lin), 
                PipelineState{qpos_out, qvel_out, x, xd}
            );
            // return PipelineState{qpos_out, qvel_out, x, xd};
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

            // std::cout << "qj inside act2tau: " << qj << std::endl;
            // std::cout << "qdj inside act2tau: " << qdj << std::endl;
            // std::cout << "act inside act2tau: " << act << std::endl;

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
        // double foot_radius_;

        std::string gait_;

        // Gait phase table
        std::map<std::string, Vector4d> kGaitPhases_;
        // Gait param table: (duty_ratio, cadence, amplitude)
        std::map<std::string, Vector3d> kGaitParams_;

        size_t action_size_;
    };

} // namespace go2env