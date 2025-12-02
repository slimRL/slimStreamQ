import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
from flax import struct


class Sample(struct.PyTreeNode):
    state: np.ndarray[np.int8]
    action: np.uint
    reward: np.float32
    next_state: np.ndarray[np.int8]
    is_terminal: bool
    random_action: bool


def update_normalize_params(x, mean, stat, count):
    count += 1
    new_mean = mean + (x - mean) / count
    stat += (x - mean) * (x - new_mean)
    variance = stat / (count - 1)
    return new_mean, stat, count, variance


@jax.jit
def normalize_observation(observation_stats, observation):
    observation_stats["mean"], observation_stats["stat"], observation_stats["count"], variance = (
        update_normalize_params(
            observation, observation_stats["mean"], observation_stats["stat"], observation_stats["count"]
        )
    )
    return observation_stats, (observation - observation_stats["mean"]) / jnp.sqrt(variance + 1e-8)


@jax.jit
def normalize_reward(reward_stats, reward, gamma, absorbing):
    reward_stats["trace"] = gamma * (1 - absorbing) * reward_stats["trace"] + reward
    reward_stats["mean"], reward_stats["stat"], reward_stats["count"], variance = update_normalize_params(
        reward_stats["trace"], reward_stats["mean"], reward_stats["stat"], reward_stats["count"]
    )
    return reward_stats, reward / jnp.sqrt(variance + 1e-8)


class SampleCollector:
    def __init__(self, env, best_action_fn, epsilon_schedule, horizon, gamma):
        self.env = env
        self.best_action_fn = best_action_fn
        self.epsilon_schedule = epsilon_schedule
        self.horizon = horizon
        self.gamma = gamma

        self.env.reset_with_noop(jax.random.PRNGKey(0))

        self.observation_stats = {
            "mean": self.env.observation,
            "stat": jnp.zeros_like(self.env.observation),
            "count": 1,
        }
        self.env.observation = jnp.zeros_like(self.env.observation)
        self.reward_stats = {"trace": 0.0, "stat": 0.0, "count": 0}

    def __call__(self, key, params, n_training_steps: int):
        state = self.env.state
        action, random_action = self.select_action(params, state, key, n_training_steps)

        reward, absorbing, game_over = self.env.step(action)
        is_truncation = self.env.n_steps >= self.horizon
        # episode_end = absorbing or self.env.n_steps >= self.horizon

        # On absorbing (life loss), the episode continues. We reset and log only when game over or truncation
        if game_over or is_truncation:
            self.env.reset_with_noop(jax.random.split(key)[1])
        elif absorbing:
            _, _, game_over_ = self.env.step(0)
            if game_over_:
                self.env.reset_with_noop(jax.random.split(key)[1])

        self.observation_stats, self.env.observation = normalize_observation(
            self.observation_stats, self.env.observation
        )
        if self.reward_stats["count"] == 0:
            self.reward_stats = {"trace": reward, "mean": reward, "stat": 0.0, "count": 1}
            normalized_reward = reward
        else:
            self.reward_stats, normalized_reward = normalize_reward(self.reward_stats, reward, self.gamma, absorbing)

        return (
            reward,
            game_over or is_truncation,
            Sample(
                state=state,
                action=action,
                reward=normalized_reward,
                next_state=self.env.state,
                is_terminal=absorbing,
                random_action=random_action,
            ),
        )

    @partial(jax.jit, static_argnames="self")
    def select_action(self, params, state, key, n_training_steps):
        uniform_key, action_key = jax.random.split(key)
        do_random_action = jax.random.uniform(uniform_key) <= self.epsilon_schedule(n_training_steps)
        random_action = jax.random.randint(action_key, (), 0, self.env.n_actions)
        best_action = self.best_action_fn(params, state)
        return jnp.where(do_random_action, random_action, best_action), do_random_action & (
            random_action != best_action
        )
