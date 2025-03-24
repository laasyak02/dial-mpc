from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union, List

import numpy as np

import jax
import jax.numpy as jnp
from functools import partial

from brax import math
import brax.base as base
from brax.base import System
from brax import envs as brax_envs
from brax.envs.base import PipelineEnv, State
from brax.io import html, mjcf, model

import mujoco
from mujoco import mjx
import inspect

from dial_mpc.envs.base_env import BaseEnv, BaseEnvConfig
from dial_mpc.utils.function_utils import global_to_body_velocity, get_foot_step
from dial_mpc.utils.io_utils import get_model_path
# from jax.experimental import host_callback
from dataclasses import replace

@dataclass
class UnitreeGo2EnvConfig(BaseEnvConfig):
    kp: Union[float, jax.Array] = 30.0
    kd: Union[float, jax.Array] = 0.0
    default_vx: float = 1.0
    default_vy: float = 0.0
    default_vyaw: float = 0.0
    ramp_up_time: float = 2.0
    gait: str = "trot"

def debug_pause():
    input("Press Enter to continue...")
    return None

class UnitreeGo2Env(BaseEnv):
    def __init__(self, config: UnitreeGo2EnvConfig):
        super().__init__(config)

        self._foot_radius = 0.0175

        self._gait = config.gait
        self._gait_phase = {
            "stand": jnp.zeros(4),
            "walk": jnp.array([0.0, 0.5, 0.75, 0.25]),
            "trot": jnp.array([0.0, 0.5, 0.5, 0.0]),
            "canter": jnp.array([0.0, 0.33, 0.33, 0.66]),
            "gallop": jnp.array([0.0, 0.05, 0.4, 0.35]),
        }
        self._gait_params = {
            #                  ratio, cadence, amplitude
            "stand": jnp.array([1.0, 1.0, 0.0]),
            "walk": jnp.array([0.75, 1.0, 0.08]),
            "trot": jnp.array([0.45, 2, 0.08]),
            "canter": jnp.array([0.4, 4, 0.06]),
            "gallop": jnp.array([0.3, 3.5, 0.10]),
        }

        self._torso_idx = mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "base"
        )

        self._init_q = jnp.array(self.sys.mj_model.keyframe("home").qpos)
        self._default_pose = self.sys.mj_model.keyframe("home").qpos[7:]

        self.joint_range = jnp.array(
            [
                [-0.5, 0.5],
                [0.4, 1.4],
                [-2.3, -0.85],
                [-0.5, 0.5],
                [0.4, 1.4],
                [-2.3, -0.85],
                [-0.5, 0.5],
                [0.4, 1.4],
                [-2.3, -1.3],
                [-0.5, 0.5],
                [0.4, 1.4],
                [-2.3, -1.3],
            ]
        )
        feet_site = [
            "FL_foot",
            "FR_foot",
            "RL_foot",
            "RR_foot",
        ]
        feet_site_id = [
            mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        ]
        assert not any(id_ == -1 for id_ in feet_site_id), "Site not found."
        self._feet_site_id = jnp.array(feet_site_id)
        print("Joint Range", self.joint_range)
        print("Joint Torque Range:", self.joint_torque_range)

    def make_system(self, config: UnitreeGo2EnvConfig) -> System:
        model_path = get_model_path("unitree_go2", "mjx_scene_force.xml")
        sys = mjcf.load(model_path)
        sys = sys.tree_replace({"opt.timestep": config.timestep})
        return sys

    def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "pos_tar": jnp.array([0.282, 0.0, 0.3]),
            "vel_tar": jnp.array([0.0, 0.0, 0.0]),
            "ang_vel_tar": jnp.array([0.0, 0.0, 0.0]),
            "yaw_tar": 0.0,
            "step": 0,
            "z_feet": jnp.zeros(4),
            "z_feet_tar": jnp.zeros(4),
            "randomize_target": self._config.randomize_tasks,
            "last_contact": jnp.zeros(4, dtype=jnp.bool),
            "feet_air_time": jnp.zeros(4),
            "rewards": [[0.0, 0.0, 0.0, 0.0],0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }

        obs = self._get_obs(pipeline_state, state_info)
        reward, done = jnp.zeros(2)
        metrics = {}
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state


    def step(self, state: State, action: jax.Array) -> State:
        rng, cmd_rng = jax.random.split(state.info["rng"], 2)

        jax.debug.print("------ INSIDE STEP ------")
        # jax.debug.print("Incoming state: ")
        # jax.debug.print("state q: {}", state.pipeline_state.q)
        # jax.debug.print("Len q: {}", len(state.pipeline_state.q))
        # jax.debug.callback(debug_pause)
        # jax.debug.print("state qd: {}", state.pipeline_state.qd)
        # jax.debug.callback(debug_pause)
        # jax.debug.print("state xpos: {}", state.pipeline_state.x.pos)
        # jax.debug.callback(debug_pause)
        # jax.debug.print("state reward: {}", state.reward)

        jax.debug.print("Incoming control: ")
        jax.debug.print("{}", action)
        # jax.debug.callback(debug_pause)
        
        # physics step
        joint_targets = self.act2joint(action)
        if self._config.leg_control == "position":
            ctrl = joint_targets
        elif self._config.leg_control == "torque":
            ctrl = self.act2tau(action, state.pipeline_state)
        
        # jax.debug.print("Control after act2tau: {}", ctrl)
        # jax.debug.callback(debug_pause)

        # ctrl = ctrl.block_until_ready()
        # print("\n\nctrl:\n", np.array(ctrl))
        # input("Press Enter to continue...")

        try:
            print("pipeline_step module:", self.pipeline_step.__module__)
            print("pipeline_step file:", inspect.getfile(self.pipeline_step))
        except (TypeError, AttributeError) as e:
            print("Could not get source due to:", e)
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        
        
        print("_pipeline type:", type(self._pipeline))
        print("_pipeline module:", self._pipeline.__class__.__module__)
        jax.debug.callback(debug_pause)

        # Try to get the source
        # try:
        #     print("_pipeline.step source:", inspect.getsource(self._pipeline.step))
        # except Exception as e:
        #     print("Couldn't get source:", e)

        # x_new = jnp.array([
        #     -1.1257210e-03, -3.5538673e-07,  2.6583159e-01,  9.9987209e-01,
        #     2.0311712e-05, -1.5999353e-02, -1.0907480e-06, -7.4294247e-03,
        #     9.3215001e-01, -1.7977694e+00,  7.3568495e-03,  9.3213862e-01,
        #     -1.7977433e+00, -4.3462031e-03,  9.6818620e-01, -1.8815672e+00,
        #     4.2731212e-03,  9.6817535e-01, -1.8815459e+00
        # ])
        # pipeline_state = replace(pipeline_state, q=x_new)

        # xd_new = jnp.array([
        #     -3.7132964e-02, -1.2411534e-05, -1.4208087e-01,  1.2068768e-03,
        #     -1.0404927e+00, -7.1485731e-05, -2.8329131e-01,  8.7310266e-01,
        #     6.3354045e-01,  2.8121385e-01,  8.7275994e-01,  6.3433033e-01,
        #     -1.9744571e-01,  1.8684653e+00, -1.9873904e+00,  1.9544220e-01,
        #     1.8681194e+00, -1.9867712e+00
        # ])
        # pipeline_state = replace(pipeline_state, qd=xd_new)
        
        # x_pos = jnp.array([-3.8306182e-04, -1.0715606e-07, 2.6867321e-01])
        # new_x = replace(pipeline_state.x, pos=x_pos)
        # pipeline_state = replace(pipeline_state, x=new_x)

        # new_x = pipeline_state.x.at[self._torso_idx - 1].set(x_pos)
        # pipeline_state = replace(pipeline_state, x=new_x)
        x, xd = pipeline_state.x, pipeline_state.xd

        # jax.debug.print("Pipeline State: ")
        # jax.debug.print("Pipeline state q: {}", pipeline_state.q)
        # jax.debug.callback(debug_pause)
        # jax.debug.print("Pipeline state qd: {}", pipeline_state.qd)
        # jax.debug.callback(debug_pause)
        # jax.debug.print("Pipeline state x: {}", x)
        # jax.debug.print("Len Pipeline state x_pos: {}", len(x.pos))
        # jax.debug.print("Torso idx: {}", self._torso_idx)
        # jax.debug.print("xpos: {}", x.pos[self._torso_idx - 1])
        # jax.debug.print("xrot: {}", x.rot[self._torso_idx - 1])
        # jax.debug.print("xdlin: {}", xd.vel[self._torso_idx - 1])
        # jax.debug.print("xdang: {}", xd.ang[self._torso_idx - 1])

        # jax.debug.callback(debug_pause)
        # observation data
        obs = self._get_obs(pipeline_state, state.info)

        # switch to new target if randomize_target is True
        def dont_randomize():
            return (
                jnp.array([self._config.default_vx, self._config.default_vy, 0.0]),
                jnp.array([0.0, 0.0, self._config.default_vyaw]),
            )

        def randomize():
            return self.sample_command(cmd_rng)

        vel_tar, ang_vel_tar = jax.lax.cond(
            (state.info["randomize_target"]) & (state.info["step"] % 500 == 0),
            randomize,
            dont_randomize,
        )
        # print("vel_tar inside step: ", [self._config.default_vx, self._config.default_vy, 0.0])
        # print("step: ", state.info["step"])
        # print("dt: ", self.dt)
        # print("ramp_up_time: ", self._config.ramp_up_time)
        # print("Config: \n", self._config)
        state.info["vel_tar"] = jnp.minimum(
            vel_tar * state.info["step"] * self.dt / self._config.ramp_up_time, vel_tar
        )
        state.info["ang_vel_tar"] = jnp.minimum(
            ang_vel_tar * state.info["step"] * self.dt / self._config.ramp_up_time,
            ang_vel_tar,
        )

        # reward
        # gaits reward
        # print("Size of feet_site_id:", self._feet_site_id.shape[0])  # Print size
        # print("Values:", self._feet_site_id)  # Print values
        
        z_feet = pipeline_state.site_xpos[self._feet_site_id][:, 2]
        # for i, value in enumerate(z_feet):
        #     print(f"Index: {i}", ) # Value: {value}")
        # print("z_feet: ", z_feet)
        # input("Press Enter to continue...")

        duty_ratio, cadence, amplitude = self._gait_params[self._gait]
        phases = self._gait_phase[self._gait]
        z_feet_tar = get_foot_step(
            duty_ratio, cadence, amplitude, phases, state.info["step"] * self.dt
        )
        # jax.debug.print("z_feet: {}\nz_feet_tar: {}", z_feet, z_feet_tar)
        # jax.debug.callback(debug_pause)
        
        reward_gaits = -jnp.sum(((z_feet_tar - z_feet) / 0.05) ** 2)
        # foot contact data based on z-position
        foot_pos = pipeline_state.site_xpos[
            self._feet_site_id
        ]  # pytype: disable=attribute-error
        foot_contact_z = foot_pos[:, 2] - self._foot_radius
        contact = foot_contact_z < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info["last_contact"]
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0) * contact_filt_mm
        state.info["feet_air_time"] += self.dt
        reward_air_time = jnp.sum((state.info["feet_air_time"] - 0.1) * first_contact)
        # position reward
        pos_tar = (
            state.info["pos_tar"] + state.info["vel_tar"] * self.dt * state.info["step"]
        )
        pos = x.pos[self._torso_idx - 1] # changed
        # pos = x_pos

        R = math.quat_to_3x3(x.rot[self._torso_idx - 1])
        head_vec = jnp.array([0.285, 0.0, 0.0])
        head_pos = pos + jnp.dot(R, head_vec)
        reward_pos = -jnp.sum((head_pos - pos_tar) ** 2)
        # stay upright reward
        vec_tar = jnp.array([0.0, 0.0, 1.0])
        vec = math.rotate(vec_tar, x.rot[0])
        reward_upright = -jnp.sum(jnp.square(vec - vec_tar))
        # yaw orientation reward
        yaw_tar = (
            state.info["yaw_tar"]
            + state.info["ang_vel_tar"][2] * self.dt * state.info["step"]
        )
        yaw = math.quat_to_euler(x.rot[self._torso_idx - 1])[2]
        # print("Quat: ", x.rot[self._torso_idx - 1])
        d_yaw = yaw - yaw_tar
        reward_yaw = -jnp.square(jnp.atan2(jnp.sin(d_yaw), jnp.cos(d_yaw)))
        # stay to norminal pose reward
        # reward_pose = -jnp.sum(jnp.square(joint_targets - self._default_pose))
        # velocity reward
        vb = global_to_body_velocity(
            xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1]
        )
        ab = global_to_body_velocity(
            xd.ang[self._torso_idx - 1] * jnp.pi / 180.0, x.rot[self._torso_idx - 1]
        )
        reward_vel = -jnp.sum((vb[:2] - state.info["vel_tar"][:2]) ** 2)
        reward_ang_vel = -jnp.sum((ab[2] - state.info["ang_vel_tar"][2]) ** 2)
        # height reward
        reward_height = -jnp.sum(
            (x.pos[self._torso_idx - 1, 2] - state.info["pos_tar"][2]) ** 2
        ) # changed
        # reward_height = -jnp.sum(
        #     (x_pos[2] - state.info["pos_tar"][2]) ** 2
        # )
        # energy reward
        reward_energy = -jnp.sum(
            jnp.maximum(ctrl * pipeline_state.qvel[6:] / 160.0, 0.0) ** 2
        )
        # stay alive reward
        reward_alive = 1.0 - state.done
        # reward
        # jax.debug.print("step: {}, \nrew_gaits: {}, \nrew_upright: {}, \nrew_yaw: {}, \nrew_vel: {}, \nrew_ang_vel: {}, rew_height: {}", 
        #     state.info["step"], reward_gaits, reward_upright, reward_yaw,
        #     reward_vel, reward_ang_vel, reward_height)

        reward = (
            reward_gaits * 0.1
            + reward_air_time * 0.0
            + reward_pos * 0.0
            + reward_upright * 0.5
            + reward_yaw * 0.3
            # + reward_pose * 0.0
            + reward_vel * 1.0
            + reward_ang_vel * 1.0
            + reward_height * 1.0
            + reward_energy * 0.00
            + reward_alive * 0.0
        )
        # print("Reward_gaits: ",jax.device_get(reward_gaits), "\nReward_upright: ", jax.device_get(reward_upright), 
        # "\nReward_yaw: ", reward_yaw, "\nReward_vel: ", reward_vel, "\nReward_ang_vel: ", reward_ang_vel, 
        # "\nReward_height: ", reward_height, "\nTotal_reward: ", reward)
        # print("\nreward:\n", np.array(reward))
        # input("Press Enter to continue...")
        
        # done
        up = jnp.array([0.0, 0.0, 1.0])
        joint_angles = pipeline_state.q[7:]
        done = jnp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        done |= jnp.any(joint_angles < self.joint_range[:, 0])
        done |= jnp.any(joint_angles > self.joint_range[:, 1])
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18
        done = done.astype(jnp.float32)

        # state management
        state.info["step"] += 1
        state.info["rng"] = rng
        state.info["z_feet"] = z_feet
        state.info["z_feet_tar"] = z_feet_tar
        state.info["feet_air_time"] *= ~contact_filt_mm
        state.info["last_contact"] = contact
        state.info["rewards"] = [xd.ang[self._torso_idx - 1] * jnp.pi / 180.0, reward_gaits, reward_upright, reward_yaw, reward_vel, reward_ang_vel, reward_height, reward]

        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

        # jax.debug.print("Outgoing state: ")
        # jax.debug.print("state q: {}", state.pipeline_state.q)
        # jax.debug.print("Len q: {}", len(state.pipeline_state.q))
        # jax.debug.callback(debug_pause)
        # jax.debug.print("state qd: {}", state.pipeline_state.qd)
        # jax.debug.callback(debug_pause)
        # jax.debug.print("state xpos: {}", state.pipeline_state.x.pos)
        # jax.debug.callback(debug_pause)
        # jax.debug.print("state reward: {}", state.reward)
        # jax.debug.callback(debug_pause)
        return state

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
    ) -> jax.Array:
        x, xd = pipeline_state.x, pipeline_state.xd
        vb = global_to_body_velocity(
            xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1]
        )
        ab = global_to_body_velocity(
            xd.ang[self._torso_idx - 1] * jnp.pi / 180.0, x.rot[self._torso_idx - 1]
        )
        obs = jnp.concatenate(
            [
                state_info["vel_tar"],
                state_info["ang_vel_tar"],
                pipeline_state.ctrl,
                pipeline_state.qpos,
                vb,
                ab,
                pipeline_state.qvel[6:],
            ]
        )
        return obs

    def render(
        self,
        trajectory: List[base.State],
        camera: str | None = None,
        width: int = 240,
        height: int = 320,
    ) -> Sequence[np.ndarray]:
        camera = camera or "track"
        return super().render(trajectory, camera=camera, width=width, height=height)

    def sample_command(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        lin_vel_x = [-1.5, 1.5]  # min max [m/s]
        lin_vel_y = [-0.5, 0.5]  # min max [m/s]
        ang_vel_yaw = [-1.5, 1.5]  # min max [rad/s]

        _, key1, key2, key3 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(
            key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
        )
        new_lin_vel_cmd = jnp.array([lin_vel_x[0], lin_vel_y[0], 0.0])
        new_ang_vel_cmd = jnp.array([0.0, 0.0, ang_vel_yaw[0]])
        return new_lin_vel_cmd, new_ang_vel_cmd


@dataclass
class UnitreeGo2SeqJumpEnvConfig(UnitreeGo2EnvConfig):
    jump_dt: float = 1.0
    contact_targets: jax.Array = None
    contact_target_radius: jax.Array = None
    pose_target_sequence: jax.Array = None
    yaw_target_sequence: jax.Array = None


class UnitreeGo2SeqJumpEnv(UnitreeGo2Env):
    def __init__(
        self, config: UnitreeGo2SeqJumpEnvConfig = UnitreeGo2SeqJumpEnvConfig()
    ):
        super().__init__(config)
        if config.contact_targets is None or config.contact_target_radius is None:
            (
                self._contact_targets,
                self._contact_target_radius,
                self._pose_target_sequence,
                self._yaw_target_sequence,
            ) = UnitreeGo2SeqJumpEnv.generate_jumping_sequence(
                config.pose_target_sequence, config.yaw_target_sequence, 0.1
            )
        else:
            self._contact_targets = config.contact_targets
            self._contact_target_radius = config.contact_target_radius
            self._pose_target_sequence = config.pose_target_sequence
            self._yaw_target_sequence = config.yaw_target_sequence
        self.joint_range = jnp.array(
            [
                [-0.5, 0.5],
                [0.4, 2.0],
                [-2.3, -1.3],
                [-0.5, 0.5],
                [0.4, 2.0],
                [-2.3, -1.3],
                [-0.5, 0.5],
                [0.4, 1.4],
                [-2.3, -1.3],
                [-0.5, 0.5],
                [0.4, 1.4],
                [-2.3, -1.3],
            ]
        )

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)
        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "pos_tar": jnp.array([0.0, 0.0, 0.27]),
            "vel_tar": jnp.array([0.0, 0.0, 0.0]),
            "ang_vel_tar": jnp.array([0.0, 0.0, 0.0]),
            "yaw_tar": 0.0,
            "step": 0,
            "z_feet": jnp.zeros(4),
            "z_feet_tar": jnp.zeros(4),
            "randomize_target": self._config.randomize_tasks,
            "last_contact": jnp.zeros(4, dtype=jnp.bool),
            "feet_air_time": jnp.zeros(4),
            "last_ctrl": jnp.zeros(12),
        }

        state_info["contact_stage"] = 0
        if not self._config.randomize_tasks:
            state_info["contact_targets"] = self._contact_targets
            state_info["contact_target_radius"] = self._contact_target_radius
            state_info["pose_target_sequence"] = self._pose_target_sequence
            state_info["yaw_target_sequence"] = self._yaw_target_sequence
        else:
            (
                state_info["contact_targets"],
                state_info["contact_target_radius"],
                state_info["pose_target_sequence"],
                state_info["yaw_target_sequence"],
            ) = self.sample_command(rng)

        obs = self._get_obs(pipeline_state, state_info)
        reward, done = jnp.zeros(2)
        metrics = {}
        state = State(pipeline_state, obs, reward, done, metrics, state_info)

        return state

    def step(
        self, state: State, action: jax.Array
    ) -> State:  # pytype: disable=signature-mismatch
        rng, cmd_rng = jax.random.split(state.info["rng"], 2)

        # physics step
        if self._config.leg_control == "position":
            ctrl = self.act2joint(action)
        elif self._config.leg_control == "torque":
            ctrl = self.act2tau(action, state.pipeline_state)
        else:
            raise ValueError("Invalid leg control type.")
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        x, xd = pipeline_state.x, pipeline_state.xd

        # observation data
        obs = self._get_obs(pipeline_state, state.info)

        # done
        done = 0.0

        # reward
        # gaits reward
        z_feet = pipeline_state.site_xpos[self._feet_site_id][:, 2]
        duty_ratio, cadence, amplitude = self._gait_params[self._gait]
        phases = self._gait_phase[self._gait]
        z_feet_tar = get_foot_step(
            duty_ratio, cadence, amplitude, phases, state.info["step"] * self.dt
        )
        reward_gaits = -jnp.sum(((z_feet_tar - z_feet) / 0.05) ** 2)
        # position reward
        pose_target_sequence = state.info["pose_target_sequence"]
        pos_tar = pose_target_sequence[state.info["contact_stage"]]
        pos = x.pos[self._torso_idx - 1]
        reward_pos = -jnp.sum((pos - pos_tar) ** 2)
        # stay upright reward
        vec_tar = jnp.array([0.0, 0.0, 1.0])
        vec = math.rotate(vec_tar, x.rot[0])
        reward_upright = -jnp.sum(jnp.square(vec - vec_tar))
        # yaw orientation reward
        yaw_target_sequence = state.info["yaw_target_sequence"]
        yaw_tar = yaw_target_sequence[state.info["contact_stage"]]
        yaw = math.quat_to_euler(x.rot[self._torso_idx - 1])[2]
        reward_yaw = -jnp.square(yaw - yaw_tar)
        # stay to norminal pose reward
        # reward_pose = -jnp.sum(jnp.square(joint_targets - self._default_pose))

        # contact reward
        reward_contact = 0.0
        penalty_contact = pipeline_state.contact.dist <= 0.001
        reward_1 = lambda x: 1.0 * x
        reward_0 = lambda x: 0.0
        contact_targets = state.info["contact_targets"]
        contact_target_radius = state.info["contact_target_radius"]
        for i in range(4):
            for j in range(len(contact_targets)):
                contact_dist = pipeline_state.contact.dist[i]
                contact_pt = pipeline_state.contact.pos[i]
                cond = (
                    jnp.sum((contact_pt[:2] - contact_targets[j, i, :2]) ** 2)
                    <= contact_target_radius[j, i] ** 2
                )  # & (z_feet[i] < 0.001)
                reward_contact += jax.lax.cond(
                    cond,
                    reward_1,
                    reward_0,
                    (j == state.info["contact_stage"])
                    * jnp.clip(contact_dist * -1.0 + 1.0, 0.0, 1.0),
                )
                penalty_contact = penalty_contact.at[i].set(
                    penalty_contact[i] & (~cond)
                )
        penalty_contact = jnp.sum(penalty_contact)
        # energy reward
        reward_energy = -jnp.sum(
            jnp.maximum(ctrl * pipeline_state.qvel[6:] / 160.0, 0.0) ** 2
        )
        # control rate reward
        reward_ctrl_rate = -jnp.sum((ctrl - state.info["last_ctrl"]) ** 2)
        # alive reward
        reward_alive = 1.0
        # reward
        reward = (
            reward_gaits * 0.0
            + reward_pos * 1.0
            + reward_upright * 1.0
            + reward_yaw * 0.3
            # + reward_pose * 0.0
            + reward_contact * 0.1
            - penalty_contact * 0.1
            + reward_energy * 0.0
            + reward_ctrl_rate * 0.0
            + reward_alive * 10.0
        )

        # done
        up = jnp.array([0.0, 0.0, 1.0])
        joint_angles = pipeline_state.q[7:]
        done = jnp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        done |= jnp.any(joint_angles < self.joint_range[:, 0])
        done |= jnp.any(joint_angles > self.joint_range[:, 1])
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.1
        done = done.astype(jnp.float32)

        # state management
        state.info["step"] += 1
        state.info["rng"] = rng
        state.info["z_feet"] = z_feet
        state.info["z_feet_tar"] = z_feet_tar
        state.info["contact_stage"] = jnp.minimum(
            jnp.floor(state.info["step"] * self.dt / self._config.jump_dt),
            len(contact_targets) - 1,
        ).astype(jnp.int32)
        state.info["last_ctrl"] = ctrl

        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
        return state

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
    ) -> jax.Array:
        x, xd = pipeline_state.x, pipeline_state.xd
        vb = global_to_body_velocity(
            xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1]
        )
        ab = global_to_body_velocity(
            xd.ang[self._torso_idx - 1] * jnp.pi / 180.0, x.rot[self._torso_idx - 1]
        )
        quat = pipeline_state.qpos[3:7]
        rpy = math.quat_to_euler(quat)
        pose_target = state_info["pose_target_sequence"][state_info["contact_stage"]]
        yaw_target = state_info["yaw_target_sequence"][state_info["contact_stage"]]

        diff_position = x.pos[self._torso_idx - 1] - pose_target
        diff_yaw = rpy[2] - yaw_target
        diff_yaw = jnp.arctan2(jnp.sin(diff_yaw), jnp.cos(diff_yaw)).reshape(1)
        obs = jnp.concatenate(
            [
                state_info["vel_tar"],
                state_info["ang_vel_tar"],
                state_info["last_ctrl"],
                diff_position,
                rpy[:2],
                diff_yaw,
                pipeline_state.qpos[7:],
                vb,
                ab,
                pipeline_state.qvel[6:],
            ]
        )
        return obs

    def generate_jumping_sequence(
        com_pos: Sequence, com_heading: Sequence, foot_place_radius: float
    ):
        n_steps = com_pos.shape[0]
        contact_targets = []
        contact_target_radius = jnp.full((n_steps, 4), foot_place_radius)
        pose_target_sequence = jnp.array(com_pos)
        yaw_target_sequence = jnp.array(com_heading)
        assert n_steps == len(com_heading)

        for i in range(n_steps):
            contact_target = jnp.repeat(jnp.array([com_pos[i]]), 4, axis=0)
            offsets = jnp.array(
                [
                    [0.2, -0.135, 0.0],  # FR
                    [0.2, 0.135, 0.0],  # FL
                    [-0.2, -0.135, 0.0],  # RR
                    [-0.2, 0.135, 0.0],  # RL
                ]
            )
            R = math.quat_to_3x3(
                math.euler_to_quat(jnp.array([0.0, 0.0, com_heading[i] * 180 / jnp.pi]))
            )
            offsets = jnp.dot(offsets, R.T)
            contact_target = contact_target + offsets
            contact_targets.append(contact_target)
        contact_targets = jnp.array(contact_targets)

        return (
            contact_targets,
            contact_target_radius,
            pose_target_sequence,
            yaw_target_sequence,
        )

    def sample_command(
        self, rng: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        com_pos_begin = jnp.array([0.0, 0.0, 0.27])
        com_yaw_begin = jnp.array([0.0])

        def randomize_com_pos(last_com_pos, rng):
            next_com_pos = last_com_pos.at[:2].add(
                jax.random.uniform(rng, (2,), minval=-0.65, maxval=0.65)
            )
            return next_com_pos, next_com_pos

        def randomize_com_yaw(last_com_yaw, rng):
            next_com_yaw = last_com_yaw + jax.random.uniform(
                rng, (1,), minval=-0.5, maxval=0.5
            )
            return next_com_yaw, next_com_yaw

        n_steps = 10
        keys = jax.random.split(rng, n_steps * 2)
        _, com_pos = jax.lax.scan(randomize_com_pos, com_pos_begin, keys[:n_steps])
        _, com_yaw = jax.lax.scan(randomize_com_yaw, com_yaw_begin, keys[n_steps:])
        com_pos = jnp.concatenate([com_pos_begin.reshape(1, 3), com_pos], axis=0)
        com_yaw = jnp.concatenate(
            [com_yaw_begin.reshape(1, 1), com_yaw], axis=0
        ).flatten()
        (
            contact_targets,
            contact_target_radius,
            pose_target_sequence,
            yaw_target_sequence,
        ) = UnitreeGo2SeqJumpEnv.generate_jumping_sequence(com_pos, com_yaw, 0.1)
        return (
            contact_targets,
            contact_target_radius,
            pose_target_sequence,
            yaw_target_sequence,
        )

    def update_viewer(self, viewer):
        cnt = viewer.user_scn.ngeom
        for i in range(self._contact_targets.shape[0]):
            for j in range(4):
                color = np.array([0.0, 1.0, 0.0, 0.5])
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[cnt],
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=np.array([self._contact_target_radius[i, j], 0.01]),
                    rgba=color,
                    pos=self._contact_targets[i, j],
                    mat=np.eye(3).flatten(),
                )
                cnt += 1


class UnitreeGo2CrateEnvConfig(UnitreeGo2EnvConfig):
    pass


class UnitreeGo2CrateEnv(UnitreeGo2Env):
    def __init__(self, config: UnitreeGo2CrateEnvConfig = UnitreeGo2CrateEnvConfig()):
        super().__init__(config)
        self.joint_range = jnp.array(
            [
                [-0.25, 0.25],
                [-1.0, 1.4],
                [-2.7, -1.0],
                [-0.25, 0.25],
                [-1.0, 1.4],
                [-2.7, -1.0],
                [-0.25, 0.25],
                [0.0, 1.8],
                [-2.7, -1.0],
                [-0.25, 0.25],
                [0.0, 1.8],
                [-2.7, -1.0],
            ]
        )

    def make_system(self, config: UnitreeGo2EnvConfig) -> System:
        model_path = get_model_path("unitree_go2", "mjx_scene_force_crate.xml")
        sys = mjcf.load(model_path)
        sys = sys.tree_replace({"opt.timestep": config.timestep})
        return sys

    def step(
        self, state: State, action: jax.Array
    ) -> State:  # pytype: disable=signature-mismatch
        rng, cmd_rng = jax.random.split(state.info["rng"], 2)

        # physics step
        if self._config.leg_control == "position":
            ctrl = self.act2joint(action)
        elif self._config.leg_control == "torque":
            ctrl = self.act2tau(action, state.pipeline_state)
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        x, xd = pipeline_state.x, pipeline_state.xd

        # observation data
        obs = self._get_obs(pipeline_state, state.info)

        # done
        done = 0.0

        # reward
        # gaits reward
        z_feet = pipeline_state.site_xpos[self._feet_site_id][:, 2]
        duty_ratio, cadence, amplitude = self._gait_params[self._gait]
        phases = self._gait_phase[self._gait]
        z_feet_tar = get_foot_step(
            duty_ratio, cadence, amplitude, phases, state.info["step"] * self.dt
        )
        reward_gaits = -jnp.sum(((z_feet_tar - z_feet) / 0.05) ** 2)
        # position reward
        pos_tar = (
            state.info["pos_tar"] + state.info["vel_tar"] * self.dt * state.info["step"]
        )
        pos = x.pos[self._torso_idx - 1]
        R = math.quat_to_3x3(x.rot[self._torso_idx - 1])
        head_vec = jnp.array([0.285, 0.0, 0.0])
        head_pos = pos + jnp.dot(R, head_vec)
        reward_pos = -jnp.sum((head_pos - pos_tar) ** 2)
        # stay upright reward
        vec_tar = jnp.array([0.0, 0.0, 1.0])
        vec = math.rotate(vec_tar, x.rot[0])
        reward_upright = -jnp.sum(jnp.square(vec - vec_tar))
        # yaw orientation reward
        yaw_tar = state.info["yaw_tar"]
        yaw = math.quat_to_euler(x.rot[self._torso_idx - 1])[2]
        reward_yaw = -jnp.square(yaw - yaw_tar)
        # stay to norminal pose reward
        # reward_pose = -jnp.sum(jnp.square(joint_targets - self._default_pose))
        # velocity reward
        reward_vel = -jnp.sum(
            (xd.vel[self._torso_idx - 1] - state.info["vel_tar"]) ** 2
        )
        # height reward
        reward_height = -jnp.sum(
            (x.pos[self._torso_idx - 1, 2] - state.info["pos_tar"][2]) ** 2
        )
        # energy reward
        reward_energy = -jnp.sum(
            jnp.maximum(ctrl * pipeline_state.qvel[6:] / 160.0, 0.0) ** 2
        )
        # pitch reward
        rpy = math.quat_to_euler(x.rot[self._torso_idx - 1])
        pitch_tar = -0.7854
        pitch = rpy[1]
        reward_pitch = -jnp.square(pitch - pitch_tar)
        reward_roll = -jnp.square(rpy[0])

        # contact reward
        reward_contact = 0.0
        penalty_contact = pipeline_state.contact.dist <= 0.001
        reward_1 = lambda x: 1.0 * x
        reward_0 = lambda x: 0.0
        contact_indices = [16, 17, 18, 19]
        for i in range(4):
            # contact_idx = 26 + 4 + 2 + 2 * 2 * (i+1) + i
            # contact_idx = (4 + 2 + 2 * 2 * (i+1) + i) * 2 + 1
            contact_idx = contact_indices[i]
            contact_dist = pipeline_state.contact.dist[contact_idx]
            contact_pt = pipeline_state.contact.pos[contact_idx]
            cond = (
                (contact_pt[0] > 1.0)
                & (contact_pt[0] < 1.6)
                & (contact_pt[1] > -0.45)
                & (contact_pt[1] < 0.45)
                & (contact_pt[2] > 0.59)
                & (contact_pt[2] < 0.61)
            )
            reward_contact += jax.lax.cond(cond, reward_1, reward_0, 1.0)
            penalty_contact = penalty_contact.at[i].set(penalty_contact[i] & (~cond))
        penalty_contact = jnp.sum(penalty_contact)

        # reward
        reward = (
            reward_gaits * 0.0
            + reward_pos * 0.5
            + reward_upright * 0.01
            + reward_yaw * 0.3
            # + reward_pose * 0.0
            + reward_vel * 0.0
            + reward_height * 0.0
            + reward_energy * 0.0000
            + reward_pitch * 0.0
            + reward_roll * 0.0
            + reward_contact * 0.1
            - penalty_contact * 0.0
        )
        # jax.debug.print("{geom}", geom=pipeline_state.contact.geom)

        # state management
        state.info["step"] += 1
        state.info["rng"] = rng
        state.info["z_feet"] = z_feet
        state.info["z_feet_tar"] = z_feet_tar

        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
        return state

    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)
        state.info["pos_tar"] = jnp.array([1.45, 0.0, 0.87])
        state.info["vel_tar"] = jnp.array([0.0, 0.0, 0.0])
        state.info["ang_vel_tar"] = jnp.array([0.0, 0.0, 0.0])
        state.info["yaw_tar"] = 0.0
        return state


brax_envs.register_environment("unitree_go2_walk", UnitreeGo2Env)
brax_envs.register_environment("unitree_go2_seq_jump", UnitreeGo2SeqJumpEnv)
brax_envs.register_environment("unitree_go2_crate_climb", UnitreeGo2CrateEnv)
