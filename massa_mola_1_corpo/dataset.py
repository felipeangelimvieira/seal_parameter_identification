#from utils import first_order_diff, second_order_diff
from tqdm import tqdm
from scipy.integrate import solve_ivp
import numpy as np

def system_fun(m, k, c, g):
    def dxdt(t, x, f):
        x, x_dot = x
        x_dot2 = (f(t) - k * x - c * x_dot) / m - g
        return np.array([x_dot, x_dot2])

    return dxdt


def get_position_and_force(m=1, k=2, c=1, g=9.81, dt=0.01, f=np.sin, steps=10000):
    dxdt = system_fun(m, k, c, g)

    y_buffer = []
    f_buffer = [0]
    y0 = np.array([0, 0])

    y_buffer.append(y0)
    for i in tqdm(range(steps)):
        sol = solve_ivp(dxdt, t_span=(dt * i, dt * (i + 1)), y0=y0, args=(f,), t_eval=[dt * (i + 1)])
        y0 = np.squeeze(sol["y"])
        y_buffer.append(y0)
        f_buffer.append(f(dt * (i + 1)))

    return np.array(y_buffer), np.array(f_buffer)

def second_order_diff(x, dt):
    return  np.convolve(x, [1, - 2, 1], mode="same")[1:-1]/(dt**2)

def first_order_diff(x, dt):
    return  np.convolve(x, [1, -1], mode="same")[1:-1]/dt

def build_dataset(y, f, dt):
  """
  Return a tuple q, q_dot, q_dot2, f
  """
  x_dot = np.expand_dims(first_order_diff(y[:,0], dt), 1)
  x_dot2 = np.expand_dims(second_order_diff(y[:,0], dt), 1)
  x = np.expand_dims(y[1:-1,0], 1)
  f= np.expand_dims(f[1:-1], 1)
  return np.concatenate([x, x_dot, x_dot2, f], axis=1)

def f(t):
    return 2 * np.sin(2 * np.pi * 0.004 * (t // 1) ** 2 * t)


