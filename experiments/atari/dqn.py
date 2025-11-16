import os
import sys

import jax

from experiments.base.dqn import train
from experiments.base.utils import prepare_logs
from slimdqn.environments.atari import AtariEnv
from slimdqn.algorithms.dqn import DQN


def run(argvs=sys.argv[1:]):
    env_name, algo_name = os.path.abspath(__file__).split("/")[-2], os.path.abspath(__file__).split("/")[-1][:-3]
    p = prepare_logs(env_name, algo_name, argvs)

    q_key, train_key = jax.random.split(jax.random.PRNGKey(p["seed"]))

    env = AtariEnv(p["experiment_name"].split("_")[-1])
    agent = DQN(
        q_key,
        (env.state_height, env.state_width, env.n_stacked_frames),
        env.n_actions,
        features=p["features"],
        architecture_type=p["architecture_type"],
        gamma=p["gamma"],
        lambda_trace=p["lambda_trace"],
        target_update_period=p["target_update_period"],
    )
    train(train_key, p, agent, env)


if __name__ == "__main__":
    run()
