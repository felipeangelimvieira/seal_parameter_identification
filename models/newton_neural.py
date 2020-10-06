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
    return jnp.mean(jnp.square(diff))


"""

Modelo Físico

"""

def neural_network_layer(key, dim_in, dim_out, scale=1):
    w_key, b_key = random.split(key, 2)
    return random.normal(w_key, (dim_out, dim_in))*scale, random.normal(b_key, (dim_out,))*scale

def neural_network_params(key, layers, scale=1):
    keys = random.split(key, len(layers))
    return [neural_network_layer(keys[i], layers[i], layers[i+1], scale) for i in range(len(layers)-1)]


def forward_neural_network(params, x, activation_fun=jax.nn.sigmoid, scale=1):
    for w, b in params[:-1]:
        x = jnp.dot(w, x) + b
        x = activation_fun(x/1)
    w, b = params[-1]
    return jnp.dot(w, x) + b



def initialize_params(rng, layers = [4, 50, 50, 4], scale=1):
    """ Inicializa massa k e c """
    keys = random.split(rng, 3)

    params = {}

    params["K"] = neural_network_params(rng, layers=layers, scale=scale)
    params["C"] = neural_network_params(rng, layers=layers, scale=scale)
    return params

def forward_pass(params, q, q_dot, f, mass=jnp.array([[1, 0],[0, 1]])):
    """
    Uma instancia de x é do formato [x, x_dot]
    Um forward pass estima a aceleração do sistema

    """
    q = q.reshape((-1, 1))
    q_dot = q_dot.reshape((-1, 1))

    q_concat = jnp.concatenate([q, q_dot], axis=0).reshape((-1,))

    f = f.reshape((-1, 1))
    M_inv = np.linalg.pinv(mass)
    C = forward_neural_network(params["C"], q_concat, activation_fun=jax.nn.relu).reshape((2, 2))
    K = forward_neural_network(params["K"], q_concat, activation_fun=jax.nn.relu).reshape((2, 2))

    return jnp.squeeze(M_inv @ (f - C @ q_dot - K @ q), -1)

batch_forward_pass = vmap(forward_pass, in_axes=(None, 0, 0, 0), out_axes=0)


def get_loss_function(batch_forward_pass):
    @jit
    def loss(params, q, q_dot, q_dot2, f):
        pred = batch_forward_pass(params, q, q_dot, f)
        return mse(q_dot2, pred)

    return loss

def train_step(q, q_dot, q_dot2, f, opt_state, opt_update, get_params, loss):
    params = get_params(opt_state)
    mse, grad = value_and_grad(loss)(params, q, q_dot, q_dot2, f)
    opt_state = opt_update(0, grad, opt_state)
    params = get_params(opt_state)
    return get_params(opt_state), opt_state, mse


def train(params, q, q_dot, q_dot2, f, batch_size, optimizer, step_size, batch_forward_pass, epochs=1, callback=None):

    init_fun, opt_update, get_params = optimizer(step_size=step_size)
    opt_state = init_fun(params)

    loss = get_loss_function(batch_forward_pass)
    epoch_errors = []
    params_history = []
    for epoch in range(epochs):

        n_batchs = len(q)//batch_size
        errors = []


        print("Epoch", epoch)
        for i in tqdm(range(n_batchs)):
            q_batch      = jnp.array(q[i*batch_size:((i+1)*batch_size)])
            q_dot_batch  = jnp.array(q_dot[i*batch_size:((i+1)*batch_size)])
            q_dot2_batch = jnp.array(q_dot2[i*batch_size:((i+1)*batch_size)])
            f_batch = jnp.array(f[i*batch_size:((i+1)*batch_size)])
            params, opt_state, error = train_step(q_batch, q_dot_batch, q_dot2_batch, f_batch, opt_state, opt_update,
                                                  get_params, loss)
            errors.append(error)
            params_history.append(params.copy())

        mean_error = np.mean(np.array(errors))
        print("Epoch", epoch, ", mean error:",mean_error)
        epoch_errors.append(mean_error)

        if callback:
            y_pred = np.array(batch_forward_pass(params, q, q_dot, f))
            callback(y_pred=y_pred, y_true=q_dot2)

    return params
