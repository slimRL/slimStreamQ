from functools import partial

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from slimstreamq.algorithms.architectures.dqn import DQNNet
from slimstreamq.sample_collection.utils import Sample


class QLambda:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        features: list,
        architecture_type: str,
        gamma: float,
        lambda_trace: float,
    ):
        self.network = DQNNet(features, architecture_type, n_actions)
        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.trace = jax.tree.map(lambda w: jnp.zeros_like(w), self.params)

        self.gamma = gamma
        self.lambda_trace = lambda_trace
        self.cumulated_loss = 0
        self.avg_learning_rate = 0

    def update_online_params(self, sample: Sample):
        self.params, self.trace, learning_rate, loss = self.learn_on_sample(self.params, self.trace, sample)
        self.cumulated_loss += loss
        self.avg_learning_rate += learning_rate

    @partial(jax.jit, static_argnames="self")
    def learn_on_sample(self, params: FrozenDict, trace: FrozenDict, sample: Sample):
        q_value, grad_q_value = jax.value_and_grad(lambda p: self.network.apply(p, sample.state)[sample.action])(params)
        td_error = self.compute_target(params, sample) - q_value

        trace = jax.tree.map(lambda t, g: self.gamma * self.lambda_trace * t - g, trace, grad_q_value)
        norm_trace = sum(jnp.sum(jnp.abs(w.flatten())) for w in jax.tree.leaves(trace))
        learning_rate = jnp.minimum(1 / (2 * jnp.maximum(jnp.abs(td_error), 1.0) * norm_trace), 1.0)

        params = jax.tree.map(lambda p, t: p - learning_rate * td_error * t, params, trace)

        trace = jax.lax.cond(
            sample.random_action | sample.is_terminal,
            lambda t: jax.tree.map(lambda w: jnp.zeros_like(w), t),
            lambda t: t,
            trace,
        )

        return params, trace, learning_rate, jnp.square(td_error)

    def compute_target(self, params: FrozenDict, sample: Sample):
        # computes the target value for single sample
        return sample.reward + (1 - sample.is_terminal) * self.gamma * jnp.max(
            self.network.apply(params, sample.next_state)
        )

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        # computes the best action for a single state
        return jnp.argmax(self.network.apply(params, state))

    def get_logs(self):
        logs = {
            "loss": self.cumulated_loss,
            "avg_learning_rate": self.avg_learning_rate,
        }
        self.cumulated_loss = 0
        self.avg_learning_rate = 0

        return logs

    def get_model(self):
        return {"params": self.params}
