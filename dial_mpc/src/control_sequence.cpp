// #include <iostream>
// #include <cmath>
// #include <stdexcept>
// #include <cassert>
// #include <random>
// #include <algorithm>
// #include <tuple>
// #include <filesystem>

// #include <Eigen/Dense>
// #include <rclcpp/rclcpp.hpp>
// #include <std_msgs/msg/float64.hpp>
// #include <std_msgs/msg/float64_multi_array.hpp>
// #include <mujoco/mujoco.h>

// // Base System Class
// class System {
// public:
//     int state_dim;
//     int control_dim;
//     Eigen::VectorXd target_state;

// protected:
    
//     Eigen::MatrixXd Q;
//     Eigen::MatrixXd R;
//     Eigen::MatrixXd Q_terminal;

// public:
//     System(int state_dim, int control_dim, const Eigen::VectorXd& target_state = Eigen::VectorXd())
//         : state_dim(state_dim), control_dim(control_dim) {
//         if (target_state.size() == 0) {
//             this->target_state = Eigen::VectorXd::Zero(state_dim);
//         } else {
//             this->target_state = target_state;
//         }
//     }

//     virtual ~System() = default;

//     virtual Eigen::MatrixXd dynamics(const Eigen::MatrixXd& states,
//                                    const Eigen::MatrixXd& controls, 
//                                    double dt = 0.1) const = 0;

//     virtual double running_cost(const Eigen::MatrixXd& states,
//                               const Eigen::MatrixXd& controls) const = 0;

//     virtual double terminal_cost(const Eigen::VectorXd& state) const = 0;
// };

// // Inverted Pendulum Implementation
// class InvertedPendulum : public System {
// public:
//     InvertedPendulum(const Eigen::VectorXd& target_state = Eigen::VectorXd())
//         : System(2, 1, target_state) {
//         Q = Eigen::MatrixXd(2, 2);
//         Q << 10.0, 0.0,
//              0.0, 1.0;
        
//         R = Eigen::MatrixXd(1, 1);
//         R << 0.1;
        
//         Q_terminal = Eigen::MatrixXd(2, 2);
//         Q_terminal << 50.0, 0.0,
//                      0.0, 5.0;
//     }

//     Eigen::MatrixXd dynamics(const Eigen::MatrixXd& states,
//                             const Eigen::MatrixXd& controls,
//                             double dt = 0.1) const override {
//         const double g = 9.81; // gravity
//         const double l = 1.0;  // length of pendulum
//         const double m = 1.0;  // mass of pendulum

//         Eigen::MatrixXd next_states(states.rows(), states.cols());

//         for (int i = 0; i < states.rows(); ++i) {
//             double theta = states(i, 0);
//             double theta_dot = states(i, 1);
//             double torque = controls(i, 0);

//             double theta_ddot = (torque - m * g * l * std::sin(theta)) / (m * l * l);
//             next_states(i, 0) = theta + theta_dot * dt;
//             next_states(i, 1) = theta_dot + theta_ddot * dt;
//         }

//         return next_states;
//     }

//     double running_cost(const Eigen::MatrixXd& states,
//                        const Eigen::MatrixXd& controls) const override {
//         double total_cost = 0.0;

//         for (int i = 0; i < states.rows(); ++i) {
//             Eigen::VectorXd state_diff = states.row(i).transpose() - target_state;
//             total_cost += state_diff.transpose() * Q * state_diff;
//             total_cost += controls.row(i) * R * controls.row(i).transpose();
//         }

//         return total_cost;
//     }

//     double terminal_cost(const Eigen::VectorXd& state) const override {
//         Eigen::VectorXd state_diff = state - target_state;
//         return state_diff.transpose() * Q_terminal * state_diff;
//     }
// };

// // Cartpole Implementation
// class Cartpole : public System {
// public:
//     Cartpole(const Eigen::VectorXd& target_state = Eigen::VectorXd())
//         : System(4, 1, target_state) {
//         Q = Eigen::MatrixXd(4, 4);
//         Q << 1.0, 0.0, 0.0, 0.0,
//              0.0, 1.0, 0.0, 0.0,
//              0.0, 0.0, 10.0, 0.0,
//              0.0, 0.0, 0.0, 1.0;
        
//         R = Eigen::MatrixXd(1, 1);
//         R << 0.1;
        
//         Q_terminal = Eigen::MatrixXd(4, 4);
//         Q_terminal << 10.0, 0.0, 0.0, 0.0,
//                      0.0, 10.0, 0.0, 0.0,
//                      0.0, 0.0, 50.0, 0.0,
//                      0.0, 0.0, 0.0, 5.0;
//     }

//     Eigen::MatrixXd dynamics(const Eigen::MatrixXd& states,
//                             const Eigen::MatrixXd& controls,
//                             double dt = 0.05) const override {
//         const double g = 9.81;     // gravity
//         const double m_cart = 1.0; // mass of the cart
//         const double m_pole = 0.1; // mass of the pole
//         const double l = 0.5;      // half-length of the pole
//         const double total_mass = m_cart + m_pole;
//         const double polemass_length = m_pole * l;

//         Eigen::MatrixXd next_states(states.rows(), states.cols());

//         for (int i = 0; i < states.rows(); ++i) {
//             double x = states(i, 0);
//             double x_dot = states(i, 1);
//             double theta = states(i, 2);
//             double theta_dot = states(i, 3);
//             double force = controls(i, 0);

//             double temp = (force + polemass_length * theta_dot * theta_dot * std::sin(theta)) / total_mass;
//             double theta_ddot = (g * std::sin(theta) - std::cos(theta) * temp) /
//                               (l * (4.0 / 3.0 - m_pole * std::cos(theta) * std::cos(theta) / total_mass));
//             double x_ddot = temp - polemass_length * theta_ddot * std::cos(theta) / total_mass;

//             next_states(i, 0) = x + x_dot * dt;
//             next_states(i, 1) = x_dot + x_ddot * dt;
//             next_states(i, 2) = theta + theta_dot * dt;
//             next_states(i, 3) = theta_dot + theta_ddot * dt;
//         }

//         return next_states;
//     }

//     double running_cost(const Eigen::MatrixXd& states,
//                        const Eigen::MatrixXd& controls) const override {
//         double total_cost = 0.0;

//         for (int i = 0; i < states.rows(); ++i) {
//             Eigen::VectorXd state_diff = states.row(i).transpose() - target_state;
//             total_cost += state_diff.transpose() * Q * state_diff;
//             total_cost += controls.row(i) * R * controls.row(i).transpose();
//         }

//         return total_cost;
//     }

//     double terminal_cost(const Eigen::VectorXd& state) const override {
//         Eigen::VectorXd state_diff = state - target_state;
//         return state_diff.transpose() * Q_terminal * state_diff;
//     }
// };

// //Legged Robot Implementation
// char error[1000] = "";
// mjModel* model = mj_loadXML("/home/laasya/go2.xml", nullptr, error, 1000);
// mjData* data = mj_makeData(model);


// class LeggedRobot : public System {
// public:
//     LeggedRobot(const Eigen::VectorXd& target_state = Eigen::VectorXd())
//         : System(37, 12, target_state) {
//         // State Vector consists of 37 values - 19 position values + 18 velocity values
//         Q = Eigen::MatrixXd(37, 37);
//         for (int i = 0; i < 3; ++i) {
//     	    Q(i, i) = 50.0;  
// 	}
// 	for (int i = 3; i < 19; ++i) {
// 	    Q(i, i) = 5.0;  
// 	}
// 	for (int i = 19; i < 37; ++i) {
// 	    Q(i, i) = 1.0;  
// 	}
        
//         // Control vector consists of 12 actuator values
//         R = Eigen::MatrixXd::Identity(12, 12) * 0.1;
        
//         Q_terminal = Eigen::MatrixXd(37, 37);
//         for (int i = 0; i < 3; ++i) {
//     	    Q_terminal(i, i) = 50.0;  
// 	}
// 	for (int i = 3; i < 19; ++i) {
// 	    Q_terminal(i, i) = 10.0;  
// 	}
// 	for (int i = 19; i < 37; ++i) {
// 	    Q_terminal(i, i) = 5.0;  
// 	}
//     }

//     Eigen::MatrixXd dynamics(const Eigen::MatrixXd& states,
//                             const Eigen::MatrixXd& controls,
//                             double dt = 0.1) const override {
        
//         Eigen::MatrixXd next_states(states.rows(), states.cols());

//         for (int i = 0; i < states.rows(); ++i) {
//             // Assigning the current state and control values to data
//             for (int j = 0; j < model->nq; ++j) {
// 		data->qpos[j] = states(i,j);
// 	    }
// 	    for (int j = 0; j < model->nv; ++j) {
// 		data->qvel[j] = states(i,model->nq + j);
// 	    }
// 	    for (int j = 0; j < model->nu; ++j) {
// 		data->ctrl[j] = controls(i,j);
// 	    }
	    
// 	    // Running the simulation to get the next state value 
// 	    mj_step(model, data);
	    
// 	    // Assigning the next state value to next_states
// 	    for (int j = 0; j < model->nq; ++j) {
// 		next_states(i,j) = data->qpos[j];
// 	    }
// 	    for (int j = 0; j < model->nv; ++j) {
// 		next_states(i,model->nq + j) = data->qvel[j];
// 	    }
//         }

//         return next_states;
//     }

//     double running_cost(const Eigen::MatrixXd& states,
//                        const Eigen::MatrixXd& controls) const override {
//         double total_cost = 0.0;

//         for (int i = 0; i < states.rows(); ++i) {
//             Eigen::VectorXd state_diff = states.row(i).transpose() - target_state;
//             total_cost += state_diff.transpose() * Q * state_diff;
//             total_cost += controls.row(i) * R * controls.row(i).transpose();
//         }

//         return total_cost;
//     }

//     double terminal_cost(const Eigen::VectorXd& state) const override {
//         Eigen::VectorXd state_diff = state - target_state;
//         return state_diff.transpose() * Q_terminal * state_diff;
//     }
// };

// class publishControlSeq;

// std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>>
// DIAL_MPC(publishControlSeq& control_seq_node, System& system, Eigen::VectorXd& initial_state,
//          int horizon, int steps, int diffusion_levels, int num_samples, double dt);

// class publishControlSeq : public rclcpp::Node {
// public:
//     publishControlSeq() : Node("publish_control_sequence") {
//         publisher_state_control = this->create_publisher<std_msgs::msg::Float64MultiArray>("current_state_control", 10);
//         //publisher_state = this->create_publisher<std_msgs::msg::Float32MultiArray>("current_state", 10);
//     }

//     void publish_state_control_value(Eigen::MatrixXd state_value, Eigen::MatrixXd control_value) {
//     	Eigen::VectorXd state(37);
// 	state = state_value.row(0).transpose();

// 	Eigen::VectorXd control(12);
// 	control = control_value.row(0).transpose();
//         /*
//         for (int i = 0; i < value.cols(); ++i) {
//             msg.data.push_back(static_cast<float>(value(0, i)));
//     	}
//     	*/
//     	Eigen::VectorXd combined(state.size() + control.size());
//     	combined << state, control;
//     	auto msg = std_msgs::msg::Float64MultiArray();
//     	msg.data.resize(combined.size());
// 	for (int i = 0; i < combined.size(); ++i) {
// 	    msg.data[i] = static_cast<float>(combined[i]); 
// 	}
//         publisher_state_control->publish(msg);
//         RCLCPP_INFO(this->get_logger(), "Published control value");
//     }
//     /*
//     void publish_state_value(Eigen::MatrixXd value) {
//         auto msg = std_msgs::msg::Float32MultiArray();
//         for (int i = 0; i < value.cols(); ++i) {
//             msg.data.push_back(static_cast<float>(value(0, i)));
//     	}
//         publisher_state->publish(msg);
//         RCLCPP_INFO(this->get_logger(), "Published state value");
//     }
//     */
//     void select_and_simulate() {
//         std::cout << "Select the system to simulate:\n";
//         std::cout << "1. Inverted Pendulum\n";
//         std::cout << "2. Cartpole\n";
//         std::cout << "3. Legged Robot\n";
//         int choice;
//         std::cin >> choice;

//         if (choice == 1) {
//             Eigen::VectorXd target(2);
//             target << M_PI / 4, 0.0;
//             InvertedPendulum pendulum(target);
            
//             Eigen::VectorXd initial_state(2);
//             initial_state << M_PI / 4, 0.0;

//             int diffusionLevels = 100;
//             auto result = DIAL_MPC(*this, pendulum, initial_state, 10, 100, diffusionLevels, 100, 0.1);

//             std::cout << "Trajectory\n";
//             const auto& trajectory = std::get<0>(result);
//             for (int i = 0; i < trajectory.rows(); ++i) {
//                 std::cout << "State: [" << trajectory(i, 0) << ", " << trajectory(i, 1) << "]\n";
//             }

//             std::cout << "\nControl History\n";
//             const auto& controlHistory = std::get<1>(result);
//             for (int i = 0; i < controlHistory.rows(); ++i) {
//                 std::cout << "Control Input: [" << controlHistory(i, 0) << "]\n";
//             }

//         } else if (choice == 2) {
//             Eigen::VectorXd target(4);
//             target << 0.0, 0.0, M_PI / 18, 0.0;
//             Cartpole cartpole(target);
            
//             Eigen::VectorXd initial_state(4);
//             initial_state << 0.0, 0.0, M_PI / 18, 0.0;

//             int diffusionLevels = 50;
//             auto result = DIAL_MPC(*this, cartpole, initial_state, 15, 200, diffusionLevels, 100, 0.05);

//             std::cout << "Trajectory\n";
//             const auto& trajectory = std::get<0>(result);
//             for (int i = 0; i < trajectory.rows(); ++i) {
//                 std::cout << "State: [" << trajectory(i, 0) << ", " << trajectory(i, 1) 
//                          << ", " << trajectory(i, 2) << ", " << trajectory(i, 3) << "]\n";
//             }

//             std::cout << "\nControl History\n";
//             const auto& controlHistory = std::get<1>(result);
//             for (int i = 0; i < controlHistory.rows(); ++i) {
//                 std::cout << "Control Input: [" << controlHistory(i, 0) << "]\n";
//             }

//         } else if (choice == 3) {
//             Eigen::VectorXd target(37);
//             target.setZero(); 
//             //target(1) = 0.1;
//             //target.head(19) << 0.000387204, -0.99806, 0.44498, 0.999988, 0.00491659, -0.000810121, -0.000123243,                    0.00736236, 0.00622736, -0.00695497,       0.00730722, 0.00581688, -0.0054609,          0.00730117, 0.00586242, -0.00558119,                0.00734579, 0.00623534, -0.00701207;
//             //target.head(19) << 0.000604053, -4.99032, 0.444978, 0.999734, 0.0230346, -0.0012668, -0.000612313, 0.0398097, 0.0124107, -0.0167069,     0.0397264, 0.00643107, -0.00274828,                0.0396762, 0.00655416, -0.00300612,                0.0397493, 0.0123066, -0.0165843;
//             //target.head(19) << -4.9893, -6.06736e-07, 0.444816,                     0.999569, 2.35124e-05, -0.0293552, -7.79631e-07,    0.0011356, -0.0345642, 0.0195182,               -0.00122462, -0.0345616, 0.0195092,          -0.00120136, -0.0341621, 0.0185292,                0.00111227, -0.0341646, 0.0185381;
//             //target.head(19) << 0.221051, -1.57645e-06, 0.284385,                     0.998598, -0.00139272, 0.0529071, 7.06741e-05,      0.00436996, 1.26421, -1.32682,                      0.00119127, 1.26538, -1.32814,                       0.00318151, 1.17054, -1.21895,                       0.00243347, 1.17162, -1.22041;
//             target.head(19) << 0.0774193, -2.56248e-06, 0.427565, 0.999088, -0.000412944, -0.0427079, -1.4878e-05,   0.00083223, 0.576419, -0.583108,                       0.000841615, 0.576466, -0.583057,                       0.00061411, 0.574115, -0.577141,                      0.00103737, 0.574253, -0.577406;

//             LeggedRobot leggedrobot(target);
            
//             Eigen::VectorXd initial_state(37);
//             initial_state.setZero();
//             initial_state(1) = 0.0;
//             initial_state(2) = 0.455;
//             /*
//             initial_state(2) = 0.27;
//             initial_state(3) = 1;
//             initial_state(8) = 0.9;
//             initial_state(9) = -1.8;
//             initial_state(11) = 0.9;
//             initial_state(12) = -1.8;
//             initial_state(14) = 0.9;
//             initial_state(15) = -1.8;
//             initial_state(17) = 0.9;
//             initial_state(18) = -1.8;
//             */

//             int diffusionLevels = 10;
//             auto result = DIAL_MPC(*this, leggedrobot, initial_state, 10, 100, diffusionLevels, 100, 0.1);
// 	    /*
//             std::cout << "Trajectory\n";
//             const auto& trajectory = std::get<0>(result);
//             for (int i = 0; i < trajectory.rows(); ++i) {
//                 std::cout << "State: [" << trajectory(i, 0) << ", " << trajectory(i, 1) 
//                          << ", " << trajectory(i, 2) << ", " << trajectory(i, 3) << "]\n";
//             }
// 	    */
//             std::cout << "\nControl History\n";
//             const auto& controlHistory = std::get<1>(result);
//             for (int i = 0; i < controlHistory.rows(); ++i) {
//             	std::cout << "Control Input: [" << controlHistory(i, 0);
//             	for (int j=1; j<12; ++j) {
//             	    std::cout << ", " << controlHistory(i, j);
//             	}
//                 std::cout << "]\n";
//             }

//         } else {
//             std::cout << "Invalid choice. Please select 1 or 2.\n";
//         }
//     }
    
// private:
//     rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_state_control;
//     //rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr publisher_state;
// };

// // DIAL-MPC Algorithm
// std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>>
// DIAL_MPC(publishControlSeq& control_seq_node, System& system, Eigen::VectorXd& initial_state,
//          int horizon, int steps, int diffusion_levels, int num_samples, double dt) {
//     int state_dim = system.state_dim;
//     int control_dim = system.control_dim;
    
//     Eigen::MatrixXd trajectory(steps + 1, state_dim);
//     trajectory.row(0) = initial_state;

//     Eigen::MatrixXd control_history(steps, control_dim);
//     std::vector<Eigen::MatrixXd> sampled_trajectories;
//     std::vector<Eigen::MatrixXd> sampled_control_trajectories;

//     Eigen::MatrixXd control_sequence = Eigen::MatrixXd::Zero(horizon, control_dim);

//     double sigma_initial = 1.0, sigma_final = 0.1;
//     double beta_inner = std::log(sigma_initial/sigma_final)/horizon;
//     double beta_outer = std::log(sigma_initial/sigma_final)/horizon + 0.2;
//     beta_inner = beta_inner/10;
//     beta_outer = beta_outer/10;

//     std::mt19937 gen(std::random_device{}());

//     for (int step = 0; step < steps; ++step) {
//         std::vector<Eigen::MatrixXd> sampled_controls(num_samples, Eigen::MatrixXd::Zero(horizon, control_dim));
        
//         for (int i = 0; i < diffusion_levels; ++i) {
//             double sigma_outer = sigma_initial * std::exp(-beta_outer * i);
//             Eigen::VectorXd sigma_inner(horizon);
            
//             // Compute sigma inner for all horizon steps
//             for (int t = 0; t < horizon; ++t) {
//                 sigma_inner(t) = sigma_outer * std::exp(-(static_cast<double>(t) / horizon) / beta_inner);
//             }
	    
// 	    // Generate control samples
//             for (int j = 0; j < num_samples; ++j) {
//                 for (int t = 0; t < horizon; ++t) {
//                     for (int d = 0; d < control_dim; ++d) {
//                         std::normal_distribution<double> dist(control_sequence(t, d), sigma_inner(t));
//                         sampled_controls[j](t, d) = dist(gen);
//                     }
//                 }
//             }

//             Eigen::VectorXd costs = Eigen::VectorXd::Zero(num_samples);

//             for (int j = 0; j < num_samples; ++j) {
//                 Eigen::MatrixXd state_trajectory(horizon + 1, state_dim);
//                 state_trajectory.row(0) = initial_state;
                
//                 for (int t = 0; t < horizon; ++t) {
//                     Eigen::MatrixXd current_state = state_trajectory.row(t);
//                     Eigen::MatrixXd current_control = sampled_controls[j].row(t);
//                     state_trajectory.row(t + 1) = system.dynamics(current_state, current_control, dt);
//                     costs(j) += system.running_cost(current_state, current_control);
//                 }
//                 costs(j) += system.terminal_cost(state_trajectory.row(horizon));
//                 sampled_trajectories.push_back(state_trajectory);
//                 sampled_control_trajectories.push_back(sampled_controls[j]);
//             }
            
//             double costs_mean = costs.mean();
//             double costs_std = std::sqrt((costs.array() - costs_mean).square().mean());
            
//             // Normalize costs
//             Eigen::VectorXd normalized_costs = (costs.array() - costs_mean) / (costs_std + 1e-6);
            
//             // Compute weights
//             Eigen::VectorXd weights = (-normalized_costs.array() / sigma_outer).exp();
            
//             // Normalize weights
//             weights /= weights.sum();

//             // Update control sequence using weighted average
//             control_sequence.setZero();
//             for (int i = 0; i < num_samples; ++i) {
//                 control_sequence += weights(i) * sampled_controls[i];
//             }
//         }

//         // Update trajectory and control history
//         Eigen::MatrixXd current_state = trajectory.row(step);
//         Eigen::MatrixXd current_control = control_sequence.row(0);
//         trajectory.row(step + 1) = system.dynamics(current_state, current_control, dt);
//         control_history.row(step) = current_control;
        
//         //Print current state value
//         std::cout <<"State\n";
//         for (int j = 0; j<37; ++j){
//             std::cout << j << ":" << current_state(0, j) << "\n"; 
//         } 
        
//         // Publish current state and control values
//         control_seq_node.publish_state_control_value(current_state, current_control);
//         //control_seq_node.publish_control_value(current_control);

//         // Shift control sequence
//         for (int t = 0; t < horizon - 1; ++t) {
//             control_sequence.row(t) = control_sequence.row(t + 1);
//         }
//         control_sequence.row(horizon - 1).setZero();

//         // Update initial state for next iteration
//         initial_state = trajectory.row(step + 1);
//     }

//     return {trajectory, control_history, sampled_trajectories, sampled_control_trajectories};
// }

// int main(int argc, char *argv[]) {
//     rclcpp::init(argc, argv);
//     auto publish_control_seq_node = std::make_shared<publishControlSeq>();
//     publish_control_seq_node->select_and_simulate();
//     rclcpp::shutdown();
//     return 0;
// }
