import os
import sys

import jax

from experiments.base.dqn import train
from experiments.base.utils import prepare_logs
from slimstreamq.environments.atari import AtariEnv
from slimstreamq.algorithms.qlambda import QLambda


def run(argvs=sys.argv[1:]):
    env_name, algo_name = os.path.abspath(__file__).split("/")[-2], os.path.abspath(__file__).split("/")[-1][:-3]
    p = prepare_logs(env_name, algo_name, argvs)

    q_key, train_key = jax.random.split(jax.random.PRNGKey(p["seed"]))

    env = AtariEnv(p["experiment_name"].split("_")[-1])
    agent = QLambda(
        q_key,
        (env.state_height, env.state_width, env.n_stacked_frames),
        env.n_actions,
        features=p["features"],
        architecture_type=p["architecture_type"],
        gamma=p["gamma"],
        lambda_trace=p["lambda_trace"],
    )
    train(train_key, p, agent, env)


if __name__ == "__main__":
    run()
