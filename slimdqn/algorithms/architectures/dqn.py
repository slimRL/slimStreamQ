from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class DQNNet(nn.Module):
    features: Sequence[int]
    architecture_type: str
    n_actions: int

    @nn.compact
    def __call__(self, x):
        if self.architecture_type == "cnn":
            idx_feature_start = 3
            x = nn.Conv(features=self.features[0], kernel_size=(8, 8), strides=(5, 5), kernel_init=sparse_cnn_uniform)(
                jnp.array(x, ndmin=4)
            )
            x = nn.LayerNorm()(x)
            x = nn.leaky_relu(x)

            x = nn.Conv(features=self.features[1], kernel_size=(4, 4), strides=(3, 3), kernel_init=sparse_cnn_uniform)(
                x
            )
            x = nn.LayerNorm()(x)
            x = nn.leaky_relu(x)

            x = nn.Conv(features=self.features[2], kernel_size=(3, 3), strides=(2, 2), kernel_init=sparse_cnn_uniform)(
                x
            )
            x = nn.LayerNorm()(x)
            x = nn.leaky_relu(x)

            x = x.reshape((x.shape[0], -1))
        elif self.architecture_type == "fc":
            idx_feature_start = 0

        x = jnp.squeeze(x)

        for idx_layer in range(idx_feature_start, len(self.features)):
            x = nn.Dense(self.features[idx_layer], kernel_init=sparse_mlp_uniform)(x)
            x = nn.LayerNorm()(x)
            x = nn.leaky_relu(x)

        return nn.Dense(self.n_actions, kernel_init=sparse_mlp_uniform)(x)


def sparse_cnn_uniform(key, shape, dtype, out_sharding):
    uniform_key, mask_key = jax.random.split(key)
    kernel = nn.initializers.variance_scaling(1 / 3, "fan_in", "uniform", in_axis=(0, 1, 2), out_axis=(0, 1, 3))(
        uniform_key, shape, dtype, out_sharding
    )
    return kernel * jax.random.bernouilli(mask_key, 1 - 0.9, shape)


def sparse_mlp_uniform(key, shape, dtype, out_sharding):
    uniform_key, mask_key = jax.random.split(key)
    kernel = nn.initializers.variance_scaling(1 / 3, "fan_in", "uniform")(uniform_key, shape, dtype, out_sharding)
    return kernel * jax.random.bernouilli(mask_key, 1 - 0.9, shape)
