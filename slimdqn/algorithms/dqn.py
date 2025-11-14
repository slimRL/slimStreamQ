from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict

from slimdqn.algorithms.architectures.dqn import DQNNet
from slimdqn.sample_collection.utils import Sample


class DQN:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        features: list,
        architecture_type: str,
        learning_rate: float,
        gamma: float,
        lambda_trace: float,
        update_horizon: int,
        data_to_update: int,
        target_update_period: int,
    ):
        self.network = DQNNet(features, architecture_type, n_actions)
        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.trace = jax.tree.map(lambda w: jnp.zeros_like(w))(self.params)
        self.learning_rate = learning_rate

        self.gamma = gamma
        self.lambda_trace = lambda_trace
        self.update_horizon = update_horizon
        self.data_to_update = data_to_update
        self.target_update_period = target_update_period
        self.cumulated_loss = 0

    def update_online_params(self, step: int, sample: Sample):
        if step % self.data_to_update == 0:
            self.params, self.trace, self.learning_rate, loss = self.learn_on_sample(
                self.params, self.trace, self.learning_rate, sample
            )
            self.cumulated_loss += loss
            self.learning_rate

    def update_target_params(self, step: int):
        if step % self.target_update_period == 0:
            logs = {
                "loss": self.cumulated_loss / (self.target_update_period / self.data_to_update),
                "learning_rate": self.learning_rate,
            }
            self.cumulated_loss = 0

            return True, logs
        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_sample(self, params: FrozenDict, trace: FrozenDict, learning_rate, sample: Sample):
        q_value, grad_q_value = jax.value_and_grad(lambda p: self.network.apply(p, sample.state)[sample.action])(params)
        td_error = self.compute_target(params, sample) - q_value

        trace = optax.apply_updates(trace, grad_q_value)
        norm_trace = jnp.sum(jnp.abs(w.flatten()) for w in jax.tree.leaves(trace))

        learning_rate = jnp.minimum(1 / (2 * jnp.maximum(abs(td_error), 1) * norm_trace), learning_rate)
        params = jax.tree.map(lambda p, t: p + learning_rate * td_error * t, params, trace)

        return params, trace, learning_rate, jnp.square(td_error)

    def compute_target(self, params: FrozenDict, sample: Sample):
        # computes the target value for single sample
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(
            self.network.apply(params, sample.next_state)
        )

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        # computes the best action for a single state
        return jnp.argmax(self.network.apply(params, state))

    def get_model(self):
        return {"params": self.params}
