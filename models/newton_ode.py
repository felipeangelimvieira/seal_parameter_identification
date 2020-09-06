from jax.experimental.stax import Dense, serial
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap, value_and_grad, jacfwd, jacrev, jacobian, hessian
from jax import random
from jax.experimental import stax
from jax.experimental.ode import odeint
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


def initialize_params(rng, dims, scale=1):
    """ Inicializa massa k e c """
    keys = random.split(rng, 3)
    params = [random.normal(keys[i], [dims, dims])*scale for i in range(2)]
    return params

mass = jnp.array([[1, 0], [0, 1]])
g = jnp.array([[0], [0]])

def dynamics_fun(y, t, f, params):
    """

    :param y: inputs [position, velocity]
    :param t: time
    :param f: force
    :return: outputs dydt [velocity, acceleration]
    """

    q, q_dot = jnp.split(y.reshape((-1)), 2)

    q = q.reshape((-1, 1))
    q_dot = q_dot.reshape((-1, 1))
    f = f.reshape((-1, 1))
    M_inv = np.linalg.pinv(mass)
    C = params[1] @ q_dot
    K = params[0] @ q

    q_dot2 = jnp.squeeze(M_inv @ (f - C - K) - g, -1)

    dydt = jnp.concatenate([q_dot.flatten(), q_dot2.flatten()], axis=0)

    return dydt

def forward_pass(params, q, q_dot, f):
    """
    Uma instancia de x é do formato [x, x_dot]
    Um forward pass estima a aceleração do sistema

    """
    q = q.reshape((-1,))
    q_dot = q_dot.reshape((-1,))
    y =  jnp.concatenate([q, q_dot])

    ysol = odeint(dynamics_fun, y, jnp.linspace(0., 0.000195, num=50), f, params)[-1]

    q, q_dot = jnp.split(ysol, 2)

    #return ysol.flatten()
    return q_dot.flatten()

batch_forward_pass = vmap(forward_pass, in_axes=(None, 0, 0, 0), out_axes=0)



def get_loss_function(batch_forward_pass):
    @jit
    def loss(params, q, q_dot, q_dot2, f, q_t1, q_dot_t1):
        pred = batch_forward_pass(params, q, q_dot, f)
        return mse(q_dot_t1.flatten(), pred.flatten())

    return loss

def train_step(q, q_dot, q_dot2, f, q_t1, q_dot_t1, opt_state, opt_update, get_params, loss):
    params = get_params(opt_state)
    mse, grad = value_and_grad(loss)(params, q, q_dot, q_dot2, f, q_t1, q_dot_t1)
    opt_state = opt_update(0, grad, opt_state)
    params = get_params(opt_state)
    return get_params(opt_state), opt_state, mse


def train(params, q, q_dot, q_dot2, f, q_t1, q_dot_t1,  batch_size, optimizer, step_size, batch_forward_pass, epochs=1, callback=None):

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

            q_batch_t1 = jnp.array(q_t1[i * batch_size:((i + 1) * batch_size)])
            q_dot_batch_t1 = jnp.array(q_dot_t1[i * batch_size:((i + 1) * batch_size)])
            params, opt_state, error = train_step(q_batch, q_dot_batch, q_dot2_batch, f_batch, q_batch_t1,
                                                  q_dot_batch_t1,
                                                  opt_state, opt_update,
                                                  get_params, loss)
            errors.append(error)
            params_history.append(params.copy())

        mean_error = np.mean(np.array(errors))
        print("Epoch", epoch, ", mean error:",mean_error, "params:", params)
        epoch_errors.append(mean_error)

        if callback:
            y_pred = np.array(batch_forward_pass(params, q, q_dot, f))
            callback(y_pred=y_pred, y_true=q_dot_t1)

    return params
