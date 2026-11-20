# DIAL-MPC for Legged Robots

A **C++/Eigen** implementation of **DIAL-MPC** (Diffusion-Inspired Annealing for Legged MPC) - a sampling-based model predictive controller for legged locomotion - with MuJoCo simulation and RViz visualization.

<p align="center">
  <img src="media/go2_walk.gif" width="560" alt="Unitree Go2 walking in MuJoCo under DIAL-MPC">
  <br><i>Unitree Go2 walking in MuJoCo, controlled by the C++ DIAL-MPC.</i>
</p>

DIAL-MPC is a training-free, sampling-based controller: at each planning step it samples a batch of control sequences, rolls them out, and refines them over several annealed "diffusion" iterations (an MPPI-style update with a decaying noise schedule), producing smooth control trajectories for dynamic legged motion.

This repo ports the algorithm from Python to C++/Eigen and runs it on the **Unitree Go2** (quadruped) MuJoCo model.


---

## What's here

| Directory | Description |
|-----------|-------------|
| `dial_mpc/` | **ROS2 package** (`rclcpp`) - control-sequence generation, an RViz visualizer (`visualization_msgs`), and legged-robot inverse kinematics. |
| `dial_mpc_eigen/` | **Standalone C++/Eigen** implementation of the DIAL-MPC core and the Unitree Go2 environment, with MuJoCo + GLFW simulation. |
| `dial_mpc_eigen_plotting/` | Same as above, with added plotting of the control sequences. |
| `models/` | MuJoCo models for **Unitree Go2** (quadruped) and **Unitree H1** (humanoid), including MJX scene variants. |

---

## Method

The core (`dial_mpc_eigen/include/dial-core.h`) is a C++/Eigen translation of the reference Python DIAL-MPC, matching its math:

- **Sampling + annealing.** Each planning step samples `Nsample` candidate control sequences, rolls them out, and runs `reverse_once` several times (`Ndiffuse` iterations, more on the first step) to refine toward the best sequence. The sampling noise `σ_control` is scaled by a diffusion factor that decays over iterations (annealing).
- **MPPI update.** Rollout rewards are normalized across time and turned into softmax weights (temperature `temp_sample`) to form the updated control sequence.
- **Spline-parameterized controls.** Control sequences are represented at a small set of node points and expanded to the full horizon with **piecewise cubic Hermite interpolation**, keeping the commands smooth.

Default configuration (from `DialConfig`): horizon `Hsample = 16`, `Hnode = 4` node points, `Nsample = 20` samples, `ctrl_dt = 0.02 s`.

### Environment

The Unitree Go2 environment (`unitree-go2-env.h`) wraps MuJoCo physics with a PD joint controller (`kp = 30`, `kd = 1`) and tracks commanded body velocities (`vx, vy, vyaw`). Rendering is done through GLFW.

---

## Results

DIAL-MPC was run for three gaits - **stand**, **walk** and **trot** - for 400 timesteps each, with a commanded forward velocity of 1.0 m/s for walking and trotting.

<table align="center">
  <tr>
    <td align="center"><img src="media/go2_trot.gif" width="380" alt="Go2 trotting"><br><b>Trot</b></td>
    <td align="center"><img src="media/go2_stand.gif" width="380" alt="Go2 standing"><br><b>Stand</b></td>
  </tr>
</table>

The base position over time shows the expected behavior for each gait: the robot holds its position while standing, and moves steadily forward along X with little drift in Y while walking and trotting. Z stays near the nominal base height, with small oscillations from the gait. Walking comes out smoother than trotting.

<table align="center">
  <tr>
    <td align="center"><img src="media/base_position_stand.png" width="270" alt="Base position, standing"><br><b>Stand</b></td>
    <td align="center"><img src="media/base_position_walk.png" width="270" alt="Base position, walking"><br><b>Walk</b></td>
    <td align="center"><img src="media/base_position_trot.png" width="270" alt="Base position, trotting"><br><b>Trot</b></td>
  </tr>
</table>
<p align="center"><i>Base position (m) over 400 timesteps - Position 0, 1, 2 = x, y, z.</i></p>

---

## Galileo integration

DIAL-MPC's solution is also used to warm-start **[Galileo](https://arxiv.org/abs/2409.12465)**, a C++ pseudospectral trajectory optimizer for legged robots. At each step, the joint accelerations, foot contact forces and joint torques from the DIAL-MPC solution are published over ROS, and Galileo's whole-body controller uses them as the initial guess for its QP solve (qpOASES) instead of the solver's default start.

<p align="center"><img src="media/dial_galileo_pipeline.png" width="680" alt="DIAL-MPC to Galileo pipeline"></p>

This work lives on the **[`devel/galileo-v2-integration`](https://github.com/laasyak02/dial-mpc/tree/devel/galileo-v2-integration)** branch, as a ROS1 catkin workspace with the `dial_mpc` and `galileo` packages - see its README for details and how to run it.

---

## Dependencies

- **MuJoCo** (physics + rendering)
- **Eigen 3.3+** (linear algebra, cubic-spline interpolation)
- **GLFW3** (visualization window)
- **ROS2** + `rclcpp`, `std_msgs`, `geometry_msgs`, `visualization_msgs` - for the `dial_mpc` ROS2 package only
- A C++ toolchain with CMake

---

## Build & run

**Standalone Eigen version:**
```sh
cd dial_mpc_eigen
mkdir build && cd build
cmake ..
make
# run the DIAL-MPC simulation / tests, e.g.
./dial-core-simulate
```

**ROS2 package:**
```sh
# from your ROS2 workspace (with dial_mpc in src/)
colcon build --packages-select dial_mpc
source install/setup.bash
ros2 run dial_mpc control_sequence            # generate the control sequence
ros2 run dial_mpc control_sequence_visualizer # visualize in RViz
```
---

## References

- Xue et al. *Full-order sampling-based MPC for torque-level locomotion control via diffusion-style annealing.* [arXiv:2409.15610](https://arxiv.org/abs/2409.15610)
- Chandler et al. *Galileo: A pseudospectral collocation framework for legged robots.* [arXiv:2409.12465](https://arxiv.org/abs/2409.12465)
- Unitree Go2 / H1 MuJoCo models — [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)

---
