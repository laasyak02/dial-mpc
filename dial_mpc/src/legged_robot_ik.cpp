/*
#include <mujoco/mujoco.h>
#include <vector>
#include <iostream>
#include <cstring>
#include <map>

class IKSolver {
public:
    IKSolver(const char* xml_path) {
        // Load model from XML
        char error[1000];
        m = mj_loadXML(xml_path, nullptr, error, 1000);
        if (!m) {
            std::cerr << "Failed to load model: " << error << std::endl;
            return;
        }
        
        // Initialize data
        d = mj_makeData(m);
        
        // Initialize to home position from keyframe
        mj_resetDataKeyframe(m, d, 0);  // 0 for the first keyframe (home)
        
        // Store initial foot positions
        saveFootPositions();
    }
    
    ~IKSolver() {
        if (d) mj_deleteData(d);
        if (m) mj_deleteModel(m);
    }
    
    void saveFootPositions() {
        // Store initial foot positions
        for (int i = 0; i < m->nsite; i++) {
            std::string site_name(m->names + m->name_siteadr[i]);
            if (site_name == "FR" || site_name == "FL" || 
                site_name == "RR" || site_name == "RL") {
                // Get foot position in world coordinates
                double* pos = d->site_xpos + 3*i;
                initial_foot_pos[site_name] = std::vector<double>(pos, pos + 3);
            }
        }
    }
    
    std::vector<double> solveIK(const std::vector<double>& target_base_pos) {
        // Save current state
        std::vector<double> original_qpos(m->nq);
        mju_copy(original_qpos.data(), d->qpos, m->nq);
        
        // Set target base position while keeping orientation
        d->qpos[0] = target_base_pos[0];
        d->qpos[1] = target_base_pos[1];
        d->qpos[2] = target_base_pos[2];
        
        // Create optimization variables
        double tolerance = 1e-6;
        int max_iterations = 100;
        double step_size = 1.0;
        
        // Temporary vectors for optimization
        std::vector<double> jacp(3 * m->nv);    // Position jacobian
        std::vector<double> jacr(3 * m->nv);    // Rotation jacobian
        std::vector<double> qpos_update(m->nv);
        std::vector<double> qfrc_constraint(m->nv);
        
        // Iterative IK solving
        for (int iter = 0; iter < max_iterations; iter++) {
            // Compute forward kinematics
            mj_forward(m, d);
            
            // Reset update vector
            std::fill(qfrc_constraint.begin(), qfrc_constraint.end(), 0.0);
            
            // For each foot
            double max_error = 0.0;
            for (const auto& foot : initial_foot_pos) {
                int site_id = mj_name2id(m, mjOBJ_SITE, foot.first.c_str());
                if (site_id < 0) continue;
                
                // Get current foot position
                double* curr_pos = d->site_xpos + 3*site_id;
                const std::vector<double>& target_pos = foot.second;
                
                // Compute position error
                double error[3];
                for (int i = 0; i < 3; i++) {
                    error[i] = target_pos[i] - curr_pos[i];
                    max_error = std::max(max_error, std::abs(error[i]));
                }
                
                // Get Jacobian for this foot
                mj_jacSite(m, d, jacp.data(), jacr.data(), site_id);
                
                // Compute position correction
                for (int i = 0; i < m->nv; i++) {
                    for (int j = 0; j < 3; j++) {
                        qfrc_constraint[i] += jacp[j * m->nv + i] * error[j];
                    }
                }
            }
            
            // Check convergence
            if (max_error < tolerance) {
                break;
            }
            
            // Update positions
            for (int i = 0; i < m->nv; i++) {
                d->qvel[i] = step_size * qfrc_constraint[i];
            }
            
            // Integrate one step
            mj_integratePos(m, d->qpos, d->qvel, 1.0);
        }
        
        // Get final configuration
        std::vector<double> result_qpos(d->qpos, d->qpos + m->nq);
        return result_qpos;
    }
    
    std::vector<double> getInitialConfig() const {
        return std::vector<double>(d->qpos, d->qpos + m->nq);
    }

private:
    mjModel* m = nullptr;
    mjData* d = nullptr;
    std::map<std::string, std::vector<double>> initial_foot_pos;
};

int main() {
    
    IKSolver solver("/home/laasya/go2.xml");
    
    // Get initial configuration
    auto initial_config = solver.getInitialConfig();
    
    // Define target base position (e.g., move 0.1m in x direction)
    std::vector<double> target_base_pos = {
        initial_config[0] + 0.1,  // x + 0.1
        initial_config[1],        // same y
        initial_config[2]         // same z
    };
    
    // Solve IK
    std::vector<double> target_qpos = solver.solveIK(target_base_pos);
    
    // Print results
    std::cout << "Target configuration:\n";
    for (size_t i = 0; i < target_qpos.size(); i++) {
        std::cout << i << ": " << target_qpos[i] << std::endl;
    }
    
    return 0;
}
*/

#include <mujoco/mujoco.h>
#include <iostream>
#include <thread>
#include <chrono>

void modifyBasePositionAndSimulate(const std::string& model_file) {
    char error[1000] = "";

    // Load the model from the MJCF file
    mjModel* model = mj_loadXML(model_file.c_str(), nullptr, error, 1000);
    if (!model) {
        std::cerr << "Error loading model: " << error << std::endl;
        return;
    }

    // Create data for the model
    mjData* data = mj_makeData(model);
    
    // Print the original base position
    std::cout << "Original base position: \n";
    for (int i = 0; i < model->nq; ++i) {  
	std::cout << "i: " << data->qpos[i] << "\n";
    }

    // Modify the base position (e.g., move it to the left along the y-axis)
    data->qpos[0] -= 5.0;  // Move base position in y-direction by -0.05

    // Print the modified base position
    /*
    std::cout << "Modified base position: "
              << "x=" << data->qpos[0] << ", "
              << "y=" << data->qpos[1] << ", "
              << "z=" << data->qpos[2] << std::endl;
    */
    std::cout << "Modified base position: \n";
    for (int i = 0; i < model->nq; ++i) {  
	std::cout << "i: " << data->qpos[i] << "\n";
    }              

    // Simulate for a few steps to let MuJoCo compute new joint angles
    for (int step = 0; step < 1000; ++step) {
        mj_step(model, data);

        // Print joint positions (qpos) every 100 steps
        if (step % 100 == 0) {
            std::cout << "Step: " << step << ", Joint angles: ";
            for (int i = 0; i < model->nq; ++i) {  // Skip the first 7 qpos values (base position and orientation)
                std::cout << data->qpos[i] << " ";
            }
            std::cout << std::endl;
        }

        // Add a small delay to simulate real-time
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Free resources
    mj_deleteData(data);
    mj_deleteModel(model);
}

int main(int argc, char** argv) {
    // Run the simulation with the modified base position
    modifyBasePositionAndSimulate("/home/laasya/go2.xml");

    return 0;
}

