import os
import time
from dataclasses import dataclass
import importlib
import sys

import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import scienceplots
import art
import emoji

import jax
from jax import numpy as jnp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
import functools
import pprint

from brax.io import html
import brax.envs as brax_envs

import dial_mpc.envs as dial_envs
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict
from dial_mpc.examples import examples
from dial_mpc.core.dial_config import DialConfig

from jax import disable_jit

plt.style.use("science")

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

def debug_pause():
    input("Press Enter to continue...")
    return None

def rollout_us(step_env, state, us):
    def step(state, u):
        jax.debug.print("------ INSIDE STEP BEFORE STEP------")
        jax.debug.print("Incoming state: ")
        jax.debug.print("state q: {}", state.pipeline_state.q)
        jax.debug.print("Len q: {}", len(state.pipeline_state.q))
        jax.debug.callback(debug_pause)
        jax.debug.print("state qd: {}", state.pipeline_state.qd)
        jax.debug.callback(debug_pause)
        jax.debug.print("state xpos: {}", state.pipeline_state.x.pos)
        jax.debug.callback(debug_pause)
        jax.debug.print("state reward: {}", state.reward)

        jax.debug.print("Incoming control: ")
        jax.debug.print("{}", u)
        jax.debug.callback(debug_pause)
        state = step_env(state, u)
        return state, (state.reward, state.pipeline_state)

    states, (rews, pipline_states) = jax.lax.scan(step, state, us)
    return states.info, rews, pipline_states


# @jax.jit
def softmax_update(weights, Y0s, sigma, mu_0t):
    mu_0tm1 = jnp.einsum("n,nij->ij", weights, Y0s)
    return mu_0tm1, sigma


class MBDPI:
    def __init__(self, args: DialConfig, env):
        self.args = args
        self.env = env
        self.nu = env.action_size

        self.update_fn = {
            "mppi": softmax_update,
        }[args.update_method]

        sigma0 = 1e-2
        sigma1 = 1.0
        A = sigma0
        B = jnp.log(sigma1 / sigma0) / args.Ndiffuse
        self.sigmas = A * jnp.exp(B * jnp.arange(args.Ndiffuse))
        self.sigma_control = (
            args.horizon_diffuse_factor ** jnp.arange(args.Hnode + 1)[::-1]
        )

        # node to u
        self.ctrl_dt = 0.02
        print("Sampleeeee", args.Hsample)
        print(args.Hnode)
        
        self.step_us = jnp.linspace(0, self.ctrl_dt * args.Hsample, args.Hsample + 1)
        self.step_nodes = jnp.linspace(0, self.ctrl_dt * args.Hsample, args.Hnode + 1)
        self.node_dt = self.ctrl_dt * (args.Hsample) / (args.Hnode)

        # setup function
        # self.rollout_us = jax.jit(functools.partial(rollout_us, self.env.step)) # change
        self.rollout_us = functools.partial(rollout_us, self.env.step)         
        # self.rollout_us_vmap = jax.jit(jax.vmap(self.rollout_us, in_axes=(None, 0))) # change
        self.rollout_us_vmap = jax.vmap(self.rollout_us, in_axes=(None, 0))

        self.node2u_vmap = jax.jit(
            jax.vmap(self.node2u, in_axes=(1), out_axes=(1))
        )  # process (horizon, node)
        # self.node2u_vmap = jax.vmap(self.node2u, in_axes=(1), out_axes=(1))

        # self.u2node_vmap = jax.jit(jax.vmap(self.u2node, in_axes=(1), out_axes=(1)))
        self.u2node_vmap = jax.vmap(self.u2node, in_axes=(1), out_axes=(1))

        self.node2u_vvmap = jax.jit(
            jax.vmap(self.node2u_vmap, in_axes=(0))
        )  # process (batch, horizon, node)
        # self.node2u_vvmap = jax.vmap(self.node2u_vmap, in_axes=(0))
        # self.u2node_vvmap = jax.jit(jax.vmap(self.u2node_vmap, in_axes=(0)))
        self.u2node_vvmap = jax.vmap(self.u2node_vmap, in_axes=(0))


    @functools.partial(jax.jit, static_argnums=(0,))
    def node2u(self, nodes):
        print("Step Nodes: ", self.step_nodes)
        spline = InterpolatedUnivariateSpline(self.step_nodes, nodes, k=2)
        us = spline(self.step_us)
        return us

    # @functools.partial(jax.jit, static_argnums=(0,))
    def u2node(self, us):
        spline = InterpolatedUnivariateSpline(self.step_us, us, k=2)
        nodes = spline(self.step_nodes)
        return nodes

    # @functools.partial(jax.jit, static_argnums=(0,))
    def reverse_once(self, state, rng, Ybar_i, noise_scale):
        print("---------- INSIDE REVERSE ONCE ----------")
        print("Ybar_i being sent inside: ", Ybar_i)
        print("noise_scale being sent inside: ", noise_scale)
        # sample from q_i
        rng, Y0s_rng = jax.random.split(rng)
        # eps_Y = jax.random.normal(
        #     Y0s_rng, (self.args.Nsample, self.args.Hnode + 1, self.nu)
        # )
        print(self.args.Nsample, self.args.Hnode + 1, self.nu)
        eps_Y = jnp.full((self.args.Nsample, self.args.Hnode + 1, self.nu), 0.5)
        Y0s = eps_Y * noise_scale[None, :, None] + Ybar_i
        # print("Y0s",  )
        # we can't change the first control
        Y0s = Y0s.at[:, 0].set(Ybar_i[0, :])
        # append Y0s with Ybar_i to also evaluate Ybar_i
        Y0s = jnp.concatenate([Y0s, Ybar_i[None]], axis=0)
        print("1st Y0 before clipping: ", Y0s[0])
        print("2nd Y0 before clipping: ", Y0s[1])
        print("Last 2nd Y0 before clipping: ", Y0s[-2])
        print("Last Y0 before clipping: ", Y0s[-1])
        Y0s = jnp.clip(Y0s, -1.0, 1.0)
        print("1st Y0 after clipping: ", Y0s[0])
        print("2nd Y0 after clipping: ", Y0s[1])
        print("Last 2nd Y0 after clipping: ", Y0s[-2])
        print("Last Y0 after clipping: ", Y0s[-1])
        # convert Y0s to us
        us = self.node2u_vvmap(Y0s)
        print("1st u : ", us[0])
        print("2nd u : ", us[1])
        print("Last 2nd u : ", us[-2])
        print("Last u : ", us[-1])

        print("state q: ", state.pipeline_state.q)
        print("state qd: ", state.pipeline_state.qd)
        print("state reward: ", state.reward)
        # esitimate mu_0tm1
        infoss, rewss, pipeline_statess = self.rollout_us_vmap(state, us)
        print("us length: ", len(us), len(us[0]), len(us[0][0]))
        print("Pipeline states length: ", len(pipeline_statess.q), len(pipeline_statess.q[0]), len(pipeline_statess.q[0][0]))

        # for i, vec in enumerate(rewss):
        #     print(f"Vector {i} (size: {vec.size}): {vec}")
        # print("Size of rewss:", len(rewss), len(rewss[0]), rewss[0][0])

        rew_Ybar_i = rewss[-1].mean()
        qss = pipeline_statess.q
        qdss = pipeline_statess.qd
        xss = pipeline_statess.x.pos
        for index in range(len(qss)):
            if index == 0 or index == 1 or index == len(qss)-2 or index == len(qss)-1:
                print(index)
                print("qss: ", qss[index])
                print("qdss: ", qdss[index])
                print("x pos: ", xss[index])
                print("Rews:", rewss[index])
                input("Press enter to continue")

        rews = rewss.mean(axis=-1)
        logp0 = (rews - rew_Ybar_i) / rews.std(axis=-1) / self.args.temp_sample

        weights = jax.nn.softmax(logp0)
        Ybar, new_noise_scale = self.update_fn(weights, Y0s, noise_scale, Ybar_i)

        # NOTE: update only with reward
        Ybar = jnp.einsum("n,nij->ij", weights, Y0s)
        qbar = jnp.einsum("n,nij->ij", weights, qss)
        qdbar = jnp.einsum("n,nij->ij", weights, qdss)
        xbar = jnp.einsum("n,nijk->ijk", weights, xss)

        # infoss = pipeline_statess.info
        info = {
            "rews": rews,
            "qbar": qbar,
            "qdbar": qdbar,
            "xbar": xbar,
            "new_noise_scale": new_noise_scale,
        }

        return rng, Ybar, info, infoss

    def reverse(self, state, YN, rng):
        Yi = YN
        with tqdm(range(self.args.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for i in pbar:
                t0 = time.time()
                rng, Yi, rews, infoss = self.reverse_once(
                    state, rng, Yi, self.sigmas[i] * jnp.ones(self.args.Hnode + 1)
                )
                Yi.block_until_ready()
                freq = 1 / (time.time() - t0)
                pbar.set_postfix({"rew": f"{rews.mean():.2e}", "freq": f"{freq:.2f}"})
        return Yi

    # @functools.partial(jax.jit, static_argnums=(0,))
    def shift(self, Y):
        u = self.node2u_vmap(Y)
        u = jnp.roll(u, -1, axis=0)
        u = u.at[-1].set(jnp.zeros(self.nu))
        Y = self.u2node_vmap(u)
        return Y

    def shift_Y_from_u(self, u, n_step):
        u = jnp.roll(u, -n_step, axis=0)
        u = u.at[-n_step:].set(jnp.zeros_like(u[-n_step:]))
        Y = self.u2node_vmap(u)
        return Y


def main():

    def reverse_scan(rng_Y0_state, factor):
        rng, Y0, state = rng_Y0_state
        rng, Y0, info, infoss = mbdpi.reverse_once(state, rng, Y0, factor)

        # # Force evaluation of the arrays
        # factor_val = jax.device_get(factor)
        # Y0_val = jax.device_get(Y0)
        # info_val = jax.device_get(info)

        # print("Factor:\n", factor_val)
        # print("Y0\n:", Y0_val)
        # print("Info:\n", info_val)
        # input("Press Enter to continue ...")

        return (rng, Y0, state), (factor, Y0, info, infoss)

    art.tprint("LeCAR @ CMU\nDIAL-MPC", font="big", chr_ignore=True)
    parser = argparse.ArgumentParser()
    config_or_example = parser.add_mutually_exclusive_group(required=True)
    config_or_example.add_argument("--config", type=str, default=None)
    config_or_example.add_argument("--example", type=str, default=None)
    config_or_example.add_argument("--list-examples", action="store_true")
    parser.add_argument(
        "--custom-env",
        type=str,
        default=None,
        help="Custom environment to import dynamically",
    )
    args = parser.parse_args()

    if args.list_examples:
        print("Examples:")
        for example in examples:
            print(f"  {example}")
        return

    if args.custom_env is not None:
        sys.path.append(os.getcwd())
        importlib.import_module(args.custom_env)

    if args.example is not None:
        config_dict = yaml.safe_load(open(get_example_path(args.example + ".yaml")))
    else:
        config_dict = yaml.safe_load(open(args.config))

    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    print("Dial_Config", dial_config)
    input("Enter")
    rng = jax.random.PRNGKey(seed=dial_config.seed)

    # find env config
    env_config_type = dial_envs.get_config(dial_config.env_name)
    env_config = load_dataclass_from_dict(
        env_config_type, config_dict, convert_list_to_array=True
    )

    print(emoji.emojize(":rocket:") + "Creating environment")
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)
    # reset_env = jax.jit(env.reset)
    reset_env = env.reset

    # step_env = jax.jit(env.step)
    step_env = env.step

    mbdpi = MBDPI(dial_config, env)

    mbdpi.args.Nsample = 50 # changed for checking

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env(rng_reset)

    YN = jnp.zeros([dial_config.Hnode + 1, mbdpi.nu])

    rng_exp, rng = jax.random.split(rng)
    # Y0 = mbdpi.reverse(state_init, YN, rng_exp)
    Y0 = YN

    Nstep = dial_config.n_steps
    rews = []
    rews_plan = []
    rollout = []
    state = state_init
    us = []
    infos = []
    print("Inside main\n")
    with tqdm(range(Nstep), desc="Rollout") as pbar:
        for t in pbar:
            # forward single step
            print("Step: ", t)
            print("Current State Info: ")
            pprint.pprint(state.info)
            print("Action (Y0)", Y0)
            state = step_env(state, Y0[0])
            print("Next State Info: ")
            pprint.pprint(state.info)
            input("Press Enter to continue...")

            rollout.append(state.pipeline_state)
            rews.append(state.reward)
            us.append(Y0[0])
            # print("State attributes", dir(state))
            # print("Pipline State attributes", dir(state.pipeline_state))
            # print("Info Attributes", state.info.keys())
            
            # update Y0
            # Y0 = jnp.array([
            #     [ 0,          0,          0,          0,          0,          0,          0,          0,          0,          0,          0,          0],
            #     [-0.220489,   0.13616,   -0.894216,   0.0327878,  -0.146311,  -0.0289915,  0.0324404,  0.0369636,   0.312936,  -0.189875,   0.27813,  -0.893601],
            #     [ 0.206555,   0.0200812,  0.0921673, -0.119499,  -0.131378,   0.431911,  -0.124762,   0.0546106,   0.384184,  -0.0836074,  0.213481,   0.211173],
            #     [ 0.362011,  -0.102883,   0.414179,   0.0877391, -0.194232,   0.0380637,   0.149895,  -0.279662,   0.0662897,  -0.305643, -0.0989681,   0.431775],
            #     [ 0.186118,  -0.0756217,  0.187157,  -0.0679896,  0.282145,  -0.207351,   0.142919,  -0.22163,   -0.302543,  -0.347605, -0.0250917,  0.0755507]
            # ])
            # print("Before shifting", Y0)
            # print("Step Nodes: ", mbdpi.step_nodes)
            # print("Interpolated spline: ")
            # for i in range(Y0.shape[1]):  # Loop over columns
            #     column = Y0[:, i]  # Extract column i
            #     print(f"Column {i}:", (InterpolatedUnivariateSpline(mbdpi.step_nodes, column, k=2)(mbdpi.step_us)))
            
            # print("shape: ", mbdpi.node2u_vmap(Y0).shape, " node to u conversion: ", mbdpi.node2u_vmap(Y0))
            print("Y0 before shifting: ", Y0)
            Y0 = mbdpi.shift(Y0)
            print("Y0 after shifting: ", Y0)
            # print("After shifting", Y0)

            n_diffuse = dial_config.Ndiffuse
            if t == 0:
                n_diffuse = dial_config.Ndiffuse_init
                print("Performing JIT on DIAL-MPC")

            t0 = time.time()
            print("Initial Y0 going in: \n", Y0)
            print("NSample", mbdpi.args.Nsample)
            traj_diffuse_factors = (
                mbdpi.sigma_control * dial_config.traj_diffuse_factor ** (jnp.arange(n_diffuse))[:, None]
            )
            # with disable_jit():
            #     (rng, Y0, _), info = jax.lax.scan(
            #         reverse_scan, (rng, Y0, state), traj_diffuse_factors
            #     )
            

            # Print results after scan
            if t == 0:
                for kkk in range(10):
                    print("Going in to reverse_once")
                    print("Factor: ", traj_diffuse_factors[kkk])
                    print("Y0:", Y0)
                    (rng, Y0, _), (factor, Y0_single, info, info_single) = reverse_scan((rng, Y0, state), traj_diffuse_factors[kkk])
                    print("Factor: ", factor)
                    print("Y0:", Y0)
                    print("Reward Len:", len(info["rews"]))
                    print("Rewards: ", info["rews"])
                    print("New Scale Factor: ", info["new_noise_scale"])
                    input("Press Enter to continue ...")
                '''
                for i, (f, y) in enumerate(zip(factors, Y0s)):
                    print(f"\nStep {i}:")
                    print("Factor:\n", jax.device_get(f))
                    print("Y0:\n", jax.device_get(y))
                    # for key, value in inf.items():
                    #     print(f"{key}:\n", value)
                    inf = jax.device_get(info)
                    print("\n--- Info Dictionary --- at step ", i)
                    for key, value in inf.items():
                        if key == "new_noise_scale" or key == "rews":
                            print(f"{key}:\n")
                            for val in value[i]:
                                print(len(val), val)
                    
                    inf1 = jax.device_get(infoss)
                    print("\n--- Reward Info Dictionary --- at step ", i)
                    print(len(infoss))
                    
                    for key, value in inf1.items():
                        if key == "rewards":
                            print(f"{key}:\n")
                            print(len(value))
                            print(len(value[i]))
                            print(len(value[i][0]))
                            for val in value:
                                print(val[0])
                '''
                        # if key == "rews":
                        #     print(f"{key}:\n")
                        #     print(len(value))
                        #     print(len(value[i]))
                        # print(f"{key}:\n", value[i])
                
                
            (rng, Y0, _), (factors, Y0s, info, infoss) = jax.lax.scan(
                reverse_scan, (rng, Y0, state), traj_diffuse_factors
            )

            rews_plan.append(info["rews"][-1].mean())
            infos.append(info)
            freq = 1 / (time.time() - t0)
            pbar.set_postfix({"rew": f"{state.reward:.2e}", "freq": f"{freq:.2f}"})

    rew = jnp.array(rews).mean()
    print(f"mean reward = {rew:.2e}")

    # save us
    # us = jnp.array(us)
    # jnp.save("./results/us.npy", us)

    # create result dir if not exist
    if not os.path.exists(dial_config.output_dir):
        os.makedirs(dial_config.output_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # plot rews_plan
    # plt.plot(rews_plan)
    # plt.savefig(os.path.join(dial_config.output_dir,
    #             f"{timestamp}_rews_plan.pdf"))

    # host webpage with flask
    print("Processing rollout for visualization")
    import flask

    app = flask.Flask(__name__)
    webpage = html.render(
        env.sys.tree_replace({"opt.timestep": env.dt}), rollout, 1080, True
    )

    # save the html file
    with open(
        os.path.join(dial_config.output_dir, f"{timestamp}_brax_visualization.html"),
        "w",
    ) as f:
        f.write(webpage)

    # save the rollout
    data = []
    xdata = []
    for i in range(len(rollout)):
        pipeline_state = rollout[i]
        data.append(
            jnp.concatenate(
                [
                    jnp.array([i]),
                    pipeline_state.qpos,
                    pipeline_state.qvel,
                    pipeline_state.ctrl,
                ]
            )
        )
        xdata.append(infos[i]["xbar"][-1])
    data = jnp.array(data)
    xdata = jnp.array(xdata)
    jnp.save(os.path.join(dial_config.output_dir, f"{timestamp}_states"), data)
    jnp.save(os.path.join(dial_config.output_dir, f"{timestamp}_predictions"), xdata)

    @app.route("/")
    def index():
        return webpage

    app.run(port=5000)


if __name__ == "__main__":
    main()
