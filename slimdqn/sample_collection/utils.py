import jax
import jax.numpy as jnp
from dataclasses import dataclass
from functools import partial
import numpy as np


@dataclass
class Sample:
    state: np.ndarray[np.int8]
    action: np.uint
    reward: np.float32
    next_state: np.ndarray[np.int8]
    is_terminal: bool


@partial(jax.jit, static_argnames=("best_action_fn", "n_actions", "epsilon_fn"))
def select_action(best_action_fn, params, state, key, n_actions, epsilon_fn, n_training_steps):
    uniform_key, action_key = jax.random.split(key)
    return jnp.where(
        jax.random.uniform(uniform_key) <= epsilon_fn(n_training_steps),  # if uniform < epsilon,
        jax.random.randint(action_key, (), 0, n_actions),  # take random action
        best_action_fn(params, state),  # otherwise, take a greedy action
    )


def collect_single_sample(key, env, agent, p, epsilon_schedule, n_training_steps: int):
    normalized_state = self.normalized_next_state

    action = select_action(
        agent.best_action, agent.params, env.state, key, env.n_actions, epsilon_schedule, n_training_steps
    ).item()

    reward, absorbing = env.step(action)
    episode_end = absorbing or env.n_steps >= p["horizon"]

    self.normalized_reward = self.normalize(reward)
    self.normalized_next_state = self.normalize(env.state)

    if episode_end:
        env.reset()
        self.normalized_next_state = self.normalize(env.state)  # avoid

    return (
        reward,
        episode_end,
        Sample(
            state=normalized_state,
            action=action,
            reward=normalized_reward,
            next_state=normalized_next_state,
            is_terminal=absorbing,
        ),
    )
