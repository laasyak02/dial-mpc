from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union, List

import jax
import jax.numpy as jnp
from functools import partial

from brax.base import System
from brax.envs.base import PipelineEnv

from dial_mpc.config.base_env_config import BaseEnvConfig


class BaseEnv(PipelineEnv):
    def __init__(self, config: BaseEnvConfig):
        assert config.dt % config.timestep == 0, "timestep must be divisible by dt"
        self._config = config
        n_frames = int(config.dt / config.timestep)
        print("n_frames", n_frames)
        input("Press Enter to continue")
        sys = self.make_system(config)
        super().__init__(sys, config.backend, n_frames, config.debug)

        # joint limit definitions
        self.physical_joint_range = self.sys.jnt_range[1:]
        self.joint_range = self.physical_joint_range
        self.joint_torque_range = self.sys.actuator_ctrlrange
        print("Joint Range", self.joint_range)
        print("Joint Torque Range:", self.joint_torque_range)

        # number of everything
        self._nv = self.sys.nv
        self._nq = self.sys.nq

    def make_system(self, config: BaseEnvConfig) -> System:
        """
        Make the system for the environment. Called in BaseEnv.__init__.
        """
        raise NotImplementedError

    # @partial(jax.jit, static_argnums=(0,))
    def act2joint(self, act: jax.Array) -> jax.Array:
        act_normalized = (
            act * self._config.action_scale + 1.0
        ) / 2.0  # normalize to [0, 1]
        joint_targets = self.joint_range[:, 0] + act_normalized * (
            self.joint_range[:, 1] - self.joint_range[:, 0]
        )  # scale to joint range
        joint_targets = jnp.clip(
            joint_targets,
            self.physical_joint_range[:, 0],
            self.physical_joint_range[:, 1],
        )
        return joint_targets

    # @partial(jax.jit, static_argnums=(0,))
    def act2tau(self, act: jax.Array, pipline_state) -> jax.Array:
        joint_target = self.act2joint(act)

        q = pipline_state.qpos[7:]
        q = q[: len(joint_target)]
        qd = pipline_state.qvel[6:]
        qd = qd[: len(joint_target)]

        # Test Values
        # q = jnp.array([
        #     -0.00688,  0.93321,  -1.81252,
        #     0.00683147,  0.933203,  -1.8125,
        #     -0.00443277,  0.95772,  -1.87624,
        #     0.00438561,  0.957714,  -1.87623
        # ])
        # qd = jnp.array([
        #     -0.255823,  0.926206, -0.10389,
        #     0.254946,  0.926056, -0.103565,
        #     -0.201774,  1.34531,  -1.7214,
        #     0.201066,  1.3452,  -1.72124
        # ])

        # jax.debug.print("qj inside act2tau: {}", q)
        # jax.debug.print("qdj inside act2tau: {}", qd)
        # jax.debug.print("action inside act2tau: {}", act)
        q_err = joint_target - q
        tau = self._config.kp * q_err - self._config.kd * qd

        tau = jnp.clip(
            tau, self.joint_torque_range[:, 0], self.joint_torque_range[:, 1]
        )
        return tau
