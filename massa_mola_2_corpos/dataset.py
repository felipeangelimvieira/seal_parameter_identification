from utils import first_order_diff, second_order_diff
from tqdm import tqdm
from scipy.integrate import solve_ivp
import numpy as np

def k_matrix(k):
    return np.array([[k[0] - k[1], k[1]],
                     [-k[1], k[1]]])


def c_matrix(c):
    return np.array([[c[0] - c[1], c[1]],
                     [-c[1], c[1]]])


def m_matrix(m):
    return np.array([[m[0], 0],
                     [0, m[1]]])


def system_fun(m, k, c, g):
    K = k_matrix(k)
    C = c_matrix(c)
    M = m_matrix(m)

    def dxdt(t, x, F):

        x, x_dot = x.reshape((2, -1))
        x = x.reshape((-1, 1))
        x_dot = x_dot.reshape((-1, 1))

        x_dot2 = np.linalg.inv(M) @ (F(t).reshape((-1, 1)) - K @ x - C @ x_dot) - g * np.array([[1], [1]])
        return np.array([x_dot.reshape((-1,)), x_dot2.reshape((-1,))]).reshape((-1,))

    return dxdt


def get_position_and_force(m=1, k=2, c=1, g=9.81, dt=0.01, f=np.sin, steps=10000):
    dxdt = system_fun(m, k, c, g)

    y_buffer = []
    f_buffer = [0]
    y0 = np.array([0, 1, 0, 0])

    y_buffer.append(y0)

    for i in tqdm(range(steps)):
        sol = solve_ivp(dxdt, t_span=(dt * i, dt * (i + 1)), y0=y0, args=(f,), t_eval=[dt * (i + 1)])
        y0 = np.squeeze(sol["y"])
        y_buffer.append(y0)
        f_buffer.append(f(dt * (i + 1)))

    y_buffer = np.array(y_buffer)
    return y_buffer[:, :2], np.array(f_buffer)


def f(t):
    return np.array([1, 1]) * 5 * np.sin(2 * np.pi * 0.01 * (t // 1) ** 2 * t) * np.sin(
        2 * np.pi * 0.004 * (t // 1) ** 2 * t)


def build_dataset(x, f, dt):
    x_dot = first_order_diff(x.transpose(), dt).transpose()
    x_dot2 = second_order_diff(x.transpose(), dt).transpose()
    return np.array(x[1:-1], dtype=np.float32), np.array(x_dot, dtype=np.float32), np.array(x_dot2, dtype=np.float32), np.array(f[1:-1], dtype=np.float32)


if __name__ == "__main__":
    dt = 0.001
    k = [2, 1]
    m = [10, 5]
    c = [9, 7]
    pos, force = get_position_and_force(dt=dt, k=k, m=m, c=c)