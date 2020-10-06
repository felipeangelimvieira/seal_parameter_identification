import numpy as np
from scipy.integrate import odeint, solve_ivp
import time
import math


class SimpleSystem:

    def __init__(self):

        self.gravity = 9.80665
        self.gravity = 0

        self.dt = 0
        self.q = np.array([[0],
                           [0]])
        self.q_dot = np.array([[0],
                               [0]])

        self.M = np.array([[1, 0],
                           [0, 1]])
        self.K = np.array([[0, 0],
                           [0, 0]])
        self.C = np.array([[0, 0],
                           [0, 0]])

        self.kp = 0
        self.kd = 0
        self.ki = 0

        self.q_ref = np.array([[0],
                               [0]])

    def solve(self, dt, t, force_fun):

        t_eval = np.arange(0, t, step=dt)

        v = self.M @ self.q_dot + self.C @ self.q - self.kd * (self.q - self.q_ref)
        E = np.array([[0],
                      [0]])
        y = np.concatenate([v.flatten(), self.q.flatten(), E.flatten()], axis=0).flatten()
        sol = solve_ivp(fun=self.dynamics_fn, t_span=[0, t], y0=y, method='RK45',
                        args=(force_fun,), t_eval=t_eval)

        if sol['success']:

            f = np.array(list(map(force_fun, sol["t"])))
            y = sol["y"]

            return {"t": sol["t"],
                    "x": y[2, :],
                    "y": y[3, :],
                    "fx": f[:, 0],
                    "fy": f[:, 1]}

        else:
            raise ValueError('Solver error')

    def dynamics_fn(self, t, y, f):
        """

        :param t: time
        :param y:  array of shape (12,) with position and velocity
        :param f:  force
        :param mass: mass
        :param gravity: gravity
        :param Icm:
        :return:
        """

        v, q, E = np.split(y, [2, 4])

        v = v.reshape((-1, 1))
        q = q.reshape((-1, 1))
        E = E.reshape((-1, 1))
        f = f(t).reshape((-1, 1))
        e = self.q_ref - q

        v_dot = - self.K @ q + f + self.kp * e + self.ki * E
        q_dot = np.linalg.inv(self.M) @ (v - self.C @ q + self.kd * e)

        dydt = np.concatenate([v_dot.flatten(), q_dot.flatten(), e.flatten()])

        return dydt


def parse_axis(axis):
    if axis not in ["both", "x", "y"]:
        raise ValueError()
    mask = np.array([0, 0])
    if axis in ("y", "both"):
        mask += np.array([0, 1])
    if axis in ("x", "both"):
        mask += np.array([1, 0])
    return mask


def sinusoidal_fun(freq, axis="both", **kwargs):
    mask = parse_axis(axis)

    def force(t):
        f = np.sin(2 * np.pi * freq * t)
        return mask * f

    return force


def sweep_fun(T, f1, f2, axis="both", **kwargs):
    mask = parse_axis(axis)

    f0 = 1 / T
    k1 = f1 / f0
    k2 = f2 / f0
    a = np.pi * (k2 - k1) * (f0 ** 2)
    b = 2 * np.pi * f0

    def sweep(t):
        f = np.sin((a * t * b) * t)
        return f * mask

    return sweep