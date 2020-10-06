from jax.experimental.stax import Dense, serial
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap, value_and_grad, jacfwd, jacrev, jacobian, hessian
from jax import random
from jax.experimental import stax
from jax.experimental.optimizers import adam, sgd
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt

"""

Utils

"""


def mse(y_true, y_pred):
    assert (jnp.squeeze(y_true).shape == jnp.squeeze(y_pred).shape)
    diff = jnp.squeeze(y_true) - np.squeeze(y_pred)
    diff = jnp.sqrt(jnp.conjugate(diff)*diff)
    return jnp.abs(jnp.mean(diff))


"""

Modelo Físico

"""


def initialize_params(rng, dims, scale=1):
    """ Inicializa massa k e c """
    keys = random.split(rng, 3)
    params = [random.normal(keys[i], [dims, dims]) * scale for i in range(2)]
    return params


def get_batch_forward_pass(mass,  freqs, g=9.81):
    def forward_pass(params, q, freqs=freqs, mass=jnp.array(mass)):
        """
        Um forward pass estima a força no dominio da frequencia

        """
        q = q.reshape((-1, 1))
        M = mass
        C = params[1]
        K = params[0]
        w = np.pi*2*freqs
        return jnp.squeeze(jax.lax.complex(K - jnp.square(w)*M, w*C) @ q)

    batch_forward_pass = vmap(forward_pass, in_axes=(None, 0, 0), out_axes=0)
    return batch_forward_pass


def get_loss_function(batch_forward_pass):
    @jit
    def loss(params, q, freqs, f):
        pred = batch_forward_pass(params, q, freqs)
        return mse(pred, f)

    return loss


def train_step(q, freqs, f, opt_state, opt_update, get_params, loss):
    params = get_params(opt_state)
    mse, grad = value_and_grad(loss)(params, q, freqs, f)
    opt_state = opt_update(0, grad, opt_state)
    params = get_params(opt_state)
    return get_params(opt_state), opt_state, mse


def train(params, q, freqs, f, batch_size, optimizer, step_size, batch_forward_pass, epochs=1, callback=None):
    init_fun, opt_update, get_params = optimizer(step_size=step_size)
    opt_state = init_fun(params)

    loss = get_loss_function(batch_forward_pass)
    epoch_errors = []
    params_history = []
    for epoch in range(epochs):

        n_batchs = len(q) // batch_size
        errors = []

        print("Epoch", epoch)
        for i in tqdm(range(n_batchs)):
            q_batch = jnp.array(q[i * batch_size:((i + 1) * batch_size)])
            freqs_batch = jnp.array(freqs[i * batch_size:((i + 1) * batch_size)])
            f_batch = jnp.array(f[i * batch_size:((i + 1) * batch_size)])
            params, opt_state, error = train_step(q_batch, freqs_batch,
                                                  f_batch, opt_state,
                                                  opt_update,
                                                  get_params, loss)
            errors.append(error)
            params_history.append(params.copy())

        mean_error = np.mean(np.array(errors))
        print("Epoch", epoch, ", mean error:", mean_error, "params:", params)
        epoch_errors.append(mean_error)

        if callback:
            y_pred = np.array(batch_forward_pass(params, q, freqs, f))
            callback(y_pred=y_pred, y_true=f)

