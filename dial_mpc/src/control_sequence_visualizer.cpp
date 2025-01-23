// #include <rclcpp/rclcpp.hpp>
// #include <visualization_msgs/msg/marker_array.hpp>
// #include <std_msgs/msg/float64_multi_array.hpp>
// #include <geometry_msgs/msg/point.hpp>

// class ControlVisualizer : public rclcpp::Node {
// public:
//     ControlVisualizer() : Node("control_visualizer") {
//         // Create publisher for marker array
//         marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
//             "control_markers", 10);

//         // Subscribe to combined state and control topic
//         state_control_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
//             "current_state_control", 10,
//             std::bind(&ControlVisualizer::state_control_callback, this, std::placeholders::_1));
        
//         // Store joint names and their indices in the state vector
//         // Format: {joint_name, qpos_index}
//         joint_indices_ = {
//             {"FR_hip_joint", 7},    // Starting from 7 because first 7 values are base pose (3 pos + 4 quat)
//             {"FR_thigh_joint", 8},
//             {"FR_calf_joint", 9},
//             {"FL_hip_joint", 10},
//             {"FL_thigh_joint", 11},
//             {"FL_calf_joint", 12},
//             {"RR_hip_joint", 13},
//             {"RR_thigh_joint", 14},
//             {"RR_calf_joint", 15},
//             {"RL_hip_joint", 16},
//             {"RL_thigh_joint", 17},
//             {"RL_calf_joint", 18}
//         };
//     }

// private:
//     void state_control_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
//         if (msg->data.size() != 49) {  // 37 state + 12 control values
//             RCLCPP_ERROR(this->get_logger(), "Expected 49 values (37 state + 12 control), got %zu", msg->data.size());
//             return;
//         }
//         // Extract state and control from combined message
//         std::vector<double> state(msg->data.begin(), msg->data.begin() + 37);
//         std::vector<double> controls(msg->data.begin() + 37, msg->data.end());

//         publish_markers(state, controls);
//     }

//     geometry_msgs::msg::Point get_actuator_position(const std::string& joint_name, const std::vector<double>& state) {
//         geometry_msgs::msg::Point position;
//         int idx = joint_indices_[joint_name];
//         // Base position from state vector
//         double base_x = state[0];
//         double base_y = state[1];
//         double base_z = state[2];
        
//         // Add offsets based on the joint positions
//         // These offsets are from the MJCF file
//         if (joint_name.find("FR") != std::string::npos) {
//             position.x = base_x + 0.1934;
//             position.y = base_y - 0.0465;
//             position.z = base_z + 0.426; // This + 0.426 offset is not there in the MJCF file (have to check again). It has been added for visualization purposes.
//             if (joint_name.find("thigh") != std::string::npos) {
//                 position.y -= 0.0955;
//             } else if (joint_name.find("calf") != std::string::npos) {
//                 position.y -= 0.0955;
//                 position.z -= 0.213;
//             }
//         } else if (joint_name.find("FL") != std::string::npos) {
//             position.x = base_x + 0.1934;
//             position.y = base_y + 0.0465;
//             position.z = base_z + 0.426; // This + 0.426 offset is not there in the MJCF file (have to check again). It has been added for visualization purposes.
//             if (joint_name.find("thigh") != std::string::npos) {
//                 position.y += 0.0955;
//             } else if (joint_name.find("calf") != std::string::npos) {
//                 position.y += 0.0955;
//                 position.z -= 0.213;
//             }
//         } else if (joint_name.find("RR") != std::string::npos) {
//             position.x = base_x - 0.1934;
//             position.y = base_y - 0.0465;
//             position.z = base_z + 0.426; // This + 0.426 offset is not there in the MJCF file (have to check again). It has been added for visualization purposes.
//             if (joint_name.find("thigh") != std::string::npos) {
//                 position.y -= 0.0955;
//             } else if (joint_name.find("calf") != std::string::npos) {
//                 position.y -= 0.0955;
//                 position.z -= 0.213;
//             }
//         } else if (joint_name.find("RL") != std::string::npos) {
//             position.x = base_x - 0.1934;
//             position.y = base_y + 0.0465;
//             position.z = base_z + 0.426; // This + 0.426 offset is not there in the MJCF file (have to check again). It has been added for visualization purposes.
//             if (joint_name.find("thigh") != std::string::npos) {
//                 position.y += 0.0955;
//             } else if (joint_name.find("calf") != std::string::npos) {
//                 position.y += 0.0955;
//                 position.z -= 0.213;
//             }
//         }
//         return position;
//     }

//     void publish_markers(const std::vector<double>& state, const std::vector<double>& controls) {
//         visualization_msgs::msg::MarkerArray marker_array;
//         int i = 0;
//         std::cout <<"State\n";
//         for (int j = 0; j<37; ++j){
//             std::cout << j << ":" << state[j] << "\n"; 
//         } 
//         for (const auto& [joint_name, state_idx] : joint_indices_) {
//             visualization_msgs::msg::Marker marker;
//             marker.header.frame_id = "map";
//             marker.header.stamp = this->now();
//             marker.ns = "control_vectors";
//             marker.id = i;
//             marker.type = visualization_msgs::msg::Marker::ARROW;
//             marker.action = visualization_msgs::msg::Marker::ADD;

//             // Get the actuator position based on current state
//             geometry_msgs::msg::Point actuator_pos = get_actuator_position(joint_name, state);
//             // Set the start point of the arrow
//             marker.points.resize(2);
//             marker.points[0] = actuator_pos;

//             // Calculate the end point based on the control value
//             double scale = 0.02;  // Adjust scale as needed
//             if (joint_name.find("hip_joint") != std::string::npos) {
//                 // Abduction joints - arrows point along X axis
//                 marker.points[1].x = actuator_pos.x + controls[i] * scale;
//                 marker.points[1].y = actuator_pos.y;
//                 marker.points[1].z = actuator_pos.z;
//             } else {
//                 // Hip and knee joints - arrows point along Y axis
//                 marker.points[1].x = actuator_pos.x;
//                 marker.points[1].y = actuator_pos.y + controls[i] * scale;
//                 marker.points[1].z = actuator_pos.z;
//             }

//             // Set the arrow properties
//             marker.scale.x = 0.02;  // shaft diameter
//             marker.scale.y = 0.04;  // head diameter
//             marker.scale.z = 0.2;   // head length

//             // Color based on magnitude (red for positive, blue for negative)
//             marker.color.a = 1.0;
//             if (controls[i] > 0) {
//                 marker.color.r = 1.0;
//                 marker.color.g = 0.0;
//                 marker.color.b = 0.0;
//             } else {
//                 marker.color.r = 0.0;
//                 marker.color.g = 0.0;
//                 marker.color.b = 1.0;
//             }

//             marker.lifetime = rclcpp::Duration::from_seconds(1.0);
//             marker_array.markers.push_back(marker);
//             i++;
//         }

//         marker_pub_->publish(marker_array);
//         std::cout<<"Published Markers\n";
//     }

//     rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
//     rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr state_control_sub_;
//     std::map<std::string, int> joint_indices_;
// };

// int main(int argc, char** argv) {
//     rclcpp::init(argc, argv);
//     auto node = std::make_shared<ControlVisualizer>();
//     rclcpp::spin(node);
//     rclcpp::shutdown();
//     return 0;
// }
