import flax.linen as fnn
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
import time


class WakeDeficitModelFlax(fnn.Module):
    @fnn.compact
    def __call__(self, x):
        x = fnn.tanh(fnn.Dense(70)(x))
        x = fnn.sigmoid(fnn.Dense(102)(x))
        x = fnn.sigmoid(fnn.Dense(102)(x))
        x = fnn.sigmoid(fnn.Dense(102)(x))
        x = fnn.Dense(1)(x)
        return x


class WakeAddedTIModelFlax(fnn.Module):
    @fnn.compact
    def __call__(self, x):
        x = fnn.sigmoid(fnn.Dense(118)(x))
        x = fnn.sigmoid(fnn.Dense(118)(x))
        x = fnn.sigmoid(fnn.Dense(118)(x))
        x = fnn.sigmoid(fnn.Dense(118)(x))
        x = fnn.Dense(1)(x)
        return x


if __name__ == "__main__":
    input_dim = 6
    model = WakeDeficitModelFlax()
    dummy_input = jnp.ones((1, input_dim))
    variables = model.init(jax.random.PRNGKey(0), dummy_input)

    # Load the parameters
    with open("./data/rans_deficit_surrogate.msgpack", "rb") as f:
        bytes_data = f.read()
    restored_variables = serialization.from_bytes(variables, bytes_data)

    def predict(params, x):
        return model.apply(params, x).mean()

    value_and_grad_func = jax.jit(jax.value_and_grad(predict, argnums=1))

    # Inference
    real_input = np.ones((1, input_dim)).astype(np.float32)
    real_input_jax = jnp.array(real_input)

    # Warmup (compile)
    output = value_and_grad_func(restored_variables, real_input_jax)
    # print("Model output:", output)

    # time it with large batch size
    batch_size = 10_000
    real_input_large = np.random.randn(batch_size, input_dim).astype(np.float32)
    real_input_large_jax = jnp.array(real_input_large)

    # Measure inference time
    start_time = time.time()
    output_large = value_and_grad_func(restored_variables, real_input_large_jax)
    output_large[0].block_until_ready()
    output_large[1].block_until_ready()
    end_time = time.time()
    print(
        "Inference time for batch size",
        batch_size,
        ":",
        end_time - start_time,
        "seconds",
    )
