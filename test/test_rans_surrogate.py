import flax.linen as fnn
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
import time
from numpy import newaxis as na


class WakeDeficitModelFlax(fnn.Module):
    scale_x = jnp.array(
        [21.21759238, 3.60546819, 0.31714823, 0.09218609, 18.70851079, 0.25810896]
    )
    mean_x = jnp.array(
        [
            3.34995157e1,
            3.63567130e-04,
            2.25024289e-02,
            1.43747711e-01,
            -1.45229452e-03,
            6.07149107e-01,
        ]
    )
    scale_y = jnp.array([0.02168894])
    mean_y = jnp.array([0.00614207])

    @fnn.compact
    def __call__(self, x):
        x = (x - self.mean_x) / self.scale_x
        x = fnn.tanh(fnn.Dense(70)(x))
        x = fnn.sigmoid(fnn.Dense(102)(x))
        x = fnn.sigmoid(fnn.Dense(102)(x))
        x = fnn.sigmoid(fnn.Dense(102)(x))
        x = fnn.Dense(1)(x)
        return (x * self.scale_y) + self.mean_y


class WakeAddedTIModelFlax(fnn.Module):
    scale_x = jnp.array(
        [21.21759238, 3.60546819, 0.31714823, 0.09218609, 18.70851079, 0.25810896]
    )
    mean_x = jnp.array(
        [
            3.34995157e1,
            3.63567130e-04,
            2.25024289e-02,
            1.43747711e-01,
            -1.45229452e-03,
            6.07149107e-01,
        ]
    )
    scale_y = jnp.array([0.00571155])
    mean_y = jnp.array([0.0014295])

    @fnn.compact
    def __call__(self, x):
        x = (x - self.mean_x) / self.scale_x
        x = fnn.sigmoid(fnn.Dense(118)(x))
        x = fnn.sigmoid(fnn.Dense(118)(x))
        x = fnn.sigmoid(fnn.Dense(118)(x))
        x = fnn.sigmoid(fnn.Dense(118)(x))
        x = fnn.Dense(1)(x)
        return (x * self.scale_y) + self.mean_y


# def get_input_from_pywake(
#     dw_ijlk,
#     hcw_ijlk,
#     z_ijlk,
#     ct_ilk,
#     TI_ilk,
#     h_ilk,
#     D_src_il,
#     TI_eff_ilk,
#     use_eff_ti=False,
#     **kwargs,
# ):
#     x, y = dw_ijlk / D_src_il[:, na, :, na], hcw_ijlk / D_src_il[:, na, :, na]
#     z = (z_ijlk - h_ilk[:, na]) / D_src_il[:, na, :, na]
#     ti = TI_eff_ilk if use_eff_ti else TI_ilk
#     return x, y, z, ti, np.zeros_like(ct_ilk), ct_ilk


if __name__ == "__main__":
    input_dim = 6
    model = WakeAddedTIModelFlax()
    dummy_input = jnp.ones((1, input_dim))
    variables = model.init(jax.random.PRNGKey(0), dummy_input)

    # Load the parameters
    with open("./data/rans_addedti_surrogate.msgpack", "rb") as f:
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
    print("Model output:", output)

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
