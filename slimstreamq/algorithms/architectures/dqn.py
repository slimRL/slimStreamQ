from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class DQNNet(nn.Module):
    features: Sequence[int]
    architecture_type: str
    n_actions: int
    n_heads: int = None

    @nn.compact
    def __call__(self, x):
        if self.architecture_type == "cnn":
            idx_feature_start = 3
            x = nn.Conv(
                features=self.features[0],
                kernel_size=(8, 8),
                strides=(5, 5),
                kernel_init=sparse_cnn_uniform,
                padding="VALID",
            )(jnp.array(x, ndmin=4))
            x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
            x = nn.leaky_relu(x)

            x = nn.Conv(
                features=self.features[1],
                kernel_size=(4, 4),
                strides=(3, 3),
                kernel_init=sparse_cnn_uniform,
                padding="VALID",
            )(x)
            x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
            x = nn.leaky_relu(x)

            x = nn.Conv(
                features=self.features[2],
                kernel_size=(3, 3),
                strides=(2, 2),
                kernel_init=sparse_cnn_uniform,
                padding="VALID",
            )(x)
            x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
            x = nn.leaky_relu(x)

            x = x.reshape((x.shape[0], -1))
        elif self.architecture_type == "fc":
            idx_feature_start = 0

        x = jnp.squeeze(x)

        for idx_layer in range(idx_feature_start, len(self.features)):
            x = nn.Dense(self.features[idx_layer], kernel_init=sparse_mlp_uniform)(x)
            x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
            x = nn.leaky_relu(x)

        if self.n_heads is None:
            return nn.Dense(self.n_actions, name="Dense_final", kernel_init=sparse_mlp_uniform)(x)
        else:
            return nn.Dense(self.n_heads * self.n_actions, name="Dense_final", kernel_init=sparse_mlp_uniform)(
                x
            ).reshape((self.n_heads, self.n_actions))


def sparse_cnn_uniform(key, shape, dtype):
    key, uniform_key = jax.random.split(key)
    h, w, channels_in, channels_out = shape
    fan_in = h * w * channels_in

    kernel = jax.random.uniform(uniform_key, shape, dtype, -jnp.sqrt(1.0 / fan_in), jnp.sqrt(1.0 / fan_in))

    flat_kernel = kernel.reshape((-1, channels_out))
    for out_channel_idx in range(channels_out):
        key, mask_key = jax.random.split(key)
        mask = jax.random.bernoulli(mask_key, 1 - 0.9, (fan_in,))
        flat_kernel = flat_kernel.at[:, out_channel_idx].set(mask * flat_kernel[:, out_channel_idx])

    return flat_kernel.reshape(kernel.shape)


def sparse_mlp_uniform(key, shape, dtype):
    key, uniform_key = jax.random.split(key)
    fan_in, fan_out = shape

    kernel = jax.random.uniform(uniform_key, shape, dtype, -jnp.sqrt(1.0 / fan_in), jnp.sqrt(1.0 / fan_in))

    for col_idx in range(fan_out):
        key, mask_key = jax.random.split(key)
        mask = jax.random.bernoulli(mask_key, 1 - 0.9, (fan_in,))
        kernel = kernel.at[:, col_idx].set(mask * kernel[:, col_idx])

    return kernel
