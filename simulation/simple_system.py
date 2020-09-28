import numpy as np
from scipy.integrate import odeint, solve_ivp
import time
import math


class SimpleSystem:
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Define basic variables and constants

        - gravity
        - mass (of the axis)
        - inertia : moment of inertia
        - radius: tha radius of the axis at point A and B
        - bearing constant: the constant of the magnetic bearing (Fm = bearing_constant*i^2/d^2)
        - L : the distance between the center of mass of the axis and the magnetic bearing action point
        - dt : the time between iterations

        """

        self.reset_count = 1
        self.gravity = 9.80665
        self.mass = 1  # axis mass

        self.dt = 0.000001
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

    def step(self, action=np.array([0, 0])):
        action = action.reshape((2, 2)).transpose().sum(axis=1).flatten()
        
        dt = self.dt
        """
        State matrix
    
        state = [ x       x_dot,      x_dot2,
                  y,      y_dot,      y_dot2,
                  z,      z_dot,      z_dot2,
                  theta,   theta_dot,  theta_dot2,
                  beta,   beta_dot,   beta_dot2,
                  omega,   omega_dot,  omega_dot2,
                  ]
    
        """

        f = action.copy()

        y = np.concatenate([self.q.flatten(), self.q_dot.flatten()], axis=0).flatten()
        sol = solve_ivp(fun=self.dynamics_fn, t_span=[0, dt], y0=y, method='RK45',
                        args=(f,), t_eval=[dt])

        if sol['success']:
            y = sol['y'].reshape((2, 2, 1))
            self.q, self.q_dot = y
        else:
            raise ValueError('Solver error')
        
        return self._get_obs(), False

    def reset(self):
        self.q = np.zeros_like(self.q)
        self.q_dot = np.zeros_like(self.q_dot)
        return self._get_obs()

   

    def _get_obs(self):
        """
        Return observations (current x y position and velocities at A and B)
        """
        x, y = self.q.flatten()

        return np.array([x, y, x, y],
                         dtype=np.float64)


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
        q, q_dot = y[:2], y[2:]

        

        
        q = q.reshape((-1, 1))
        q_dot = q_dot.reshape((-1, 1))


        bearing_force = self.K @ q + self.C @ q_dot
        assert bearing_force.shape == (2, 1)

        q_dot2 = - np.linalg.inv(self.M) @ ( bearing_force - f.reshape((2, 1)) ) - self.gravity*np.array([[1],
                                                                                                          [0]])
        assert q_dot2.shape == (2, 1)
        dydt = np.concatenate([q_dot.flatten(), q_dot2.flatten()], axis=0)


        return dydt


if __name__ == '__main__':

    env = MagneticBearing3D()

    num_episodes = 3
    for ep in range(num_episodes):
        env.reset()

        for _ in range(1000):
            obs, done = env.step()  # take a random action ([0,1.135/2,10,0, 0,1.135/2,0,0.1])
    env.close()