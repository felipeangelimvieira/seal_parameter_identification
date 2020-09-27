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
        

    def step(self, action=np.array([0, 0, 0, 0])):
        
        
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

        
        sol = solve_ivp(fun=self.dynamics_fn, t_span=[0, dt], y0=y, method='RK45',
                        args=(f, self.mass, self.gravity, self.Icm), t_eval=[dt])

       
        
        return self._get_obs(), done

    def reset(self):

        
        return self._get_obs()

   

    def _get_obs(self):
        """
        Return observations (current x y position and velocities at A and B)
        """

        return np.array([self.rotor_position_A[0, 0],  # x1 distance
                          self.rotor_position_A[1, 0],  # x2 distance
                          self.rotor_position_B[0, 0],  # x1 distance
                          self.rotor_position_B[1, 0],  # x2 distance
                          self.rotor_velocity_A[0, 0],  # x1 speed
                          self.rotor_velocity_A[1, 0],  # x2 speed
                          self.rotor_velocity_B[0, 0],  # x1 speed
                          self.rotor_velocity_B[1, 0]],  # x2 speed
                         dtype=np.float64)


    def dynamics_fn(self, t, y, f, mass, gravity, Icm):
        """

        :param t: time
        :param y:  array of shape (12,) with position and velocity
        :param f:  force
        :param mass: mass
        :param gravity: gravity
        :param Icm:
        :return:
        """
        q, q_dot = y[:6], y[6:]

        
        f_ax, f_ay, f_bx, f_by = f
                

        R = self.get_R(theta=q[3], beta=q[4], omega=q[5])

        state = np.concatenate([np.expand_dims(q, axis=-1), np.expand_dims(q_dot, axis=-1)], axis=-1)

        trans_dot2 = self._get_acc(f_ax=f_ax,
                                   f_bx=f_bx,
                                   f_ay=f_ay,
                                   f_by=f_by,
                                   mass=mass,
                                   gravity=gravity)
        
        ang_dot2 = self._get_angular_acc(f_ax=f_ax,
                                         f_bx=f_bx,
                                         f_ay=f_ay,
                                         f_by=f_by,
                                         state=state)

        dydt = np.concatenate([q_dot.flatten(), trans_dot2.flatten(), ang_dot2.flatten()])
        return dydt


if __name__ == '__main__':

    env = MagneticBearing3D()

    num_episodes = 3
    for ep in range(num_episodes):
        env.reset()

        for _ in range(1000):
            obs, done = env.step()  # take a random action ([0,1.135/2,10,0, 0,1.135/2,0,0.1])
    env.close()