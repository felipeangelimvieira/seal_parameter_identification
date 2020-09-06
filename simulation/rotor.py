import numpy as np
from scipy.integrate import odeint, solve_ivp
import time
import math


class MagneticBearing3D:
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

        radius = 14.28 / 2 * 0.001  # radius of the axis
        self.rotor_radius = 36.4 / 2 * 0.001  # radius of the rotors

        # TODO:
        radius = self.rotor_radius
        self.radius = radius
        # self.inertia = 1/2*self.mass*(radius**2) #moment of inertia
        # self.bearing_constant = 2.15846*1e-5 #0.039925 # magnetic constant. Fm = bearing_constant*i^2/d^2

        # bearing n number
        self.n = 260
        # area of transversal section
        self.area = 2 * 0.0077 * 0.0165 * np.cos(np.pi / 8)
        # axis length
        self.axis_length = 0.4
        self.L = self.axis_length / 2
        # gap between rotor and support
        self.gap_rotor_support = 0.4 * 0.001  # 0.4 mm
        # gap between rotor and bearing
        self.gap_rotor_bearing = 1 * 0.001  # 1.0 mm
        # axial position from center C to bearings
        self.pm = 0.150
        # radial position from center C to bearings
        self.radial_bearing_position = self.rotor_radius + self.gap_rotor_bearing
        # radial position from center C to support
        self.radial_support_position = self.rotor_radius + self.gap_rotor_support

        # TODO: add rotor and axis inertia
        self.inertia_x = 1 / 4 * self.mass * radius ** 2 + 1 / 12 * self.mass * (self.L * 2) ** 2
        self.inertia_y = self.inertia_x
        self.inertia_z = 1 / 8 * self.mass * (radius * 2) ** 2

        self.Icm = np.array([[self.inertia_x, 0, 0],
                             [0, self.inertia_y, 0],
                             [0, 0, self.inertia_z]])

        # Magnetic constants
        self.l_metal = 0.115  # m
        self.mi_r = 8e3
        self.mi = np.pi * 4e-7

        self.current_inf_A = 0
        self.current_sup_A = 0
        self.current_left_A = 0
        self.current_right_A = 0

        self.current_inf_B = 0
        self.current_sup_B = 0
        self.current_left_B = 0
        self.current_right_B = 0

        self.rotor_position_A = None
        # bearing A (positive rz axis)
        # position = R.self.pm*sz + [x, y, z]T
        self.rotor_position_B = None

        self.Tq = 0.01
        self.init_rpm_min_ratio = 0.5
        self.init_rpm_max_ratio = 1
        self.max_rpm = 0  # rpm
        self.init_from_support = False
        self.init_with_velocity = False
        self.init_with_rpm = 0
        self.steps_before_collision = 1000000000
        self.forced_initial_omega_dot = None
        self.init_from_center = True
        self.speed_reward = True

        self.dt = 1e-4 / 3
        self.dt_std = 0

        self.theta = np.zeros((3,))
        self.beta = np.zeros((3,))
        self.omega = np.zeros((3,))
        self.y = np.zeros((3,))
        self.x = np.zeros((3,))
        self.z = np.zeros((3,))
        self.state = np.array([self.x, self.y, self.z, self.theta, self.beta, self.omega])

        self.max_current = 4

        self.steps_before_done = 0
        self.past_y_a = []
        self.past_y_b = []
        self.past_y = []
        self.past_beta = []

        self.y_a = 0
        self.y_b = 0
        self.y_a_dot = 0
        self.y_b_dot = 0
        self.x_a = 0
        self.x_b = 0
        self.x_a_dot = 0
        self.x_b_dot = 0

        self.viewer = None

        self.force_inf_A = 0
        self.force_sup_A = 0
        self.force_left_A = 0
        self.force_right_A = 0

        self.force_inf_B = 0
        self.force_sup_B = 0
        self.force_left_B = 0
        self.force_right_B = 0

        self.K = np.array([[4300, 43],[-43, 430]])
        self.C = np.array([[43, 0],
                           [0, 43]])

        self.gravity = 0
        k = 4300
        c = 43
        self.K = np.array([[k, k / 10],
                          [-k / 10, k]])
        self.C = np.array([[c, 0],
                          [0, c]])


    def get_force(self, q, q_dot):
        q = q.reshape((-1, 1))[:2]
        q_dot = q_dot.reshape((-1, 1))[:2]

        return - (self.K @ q + self.C @ q_dot)


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

        x, y, z, beta, theta, omega = self.state
        # print(omega, self.steps_before_done)
        # Rotation along rx (axis = 0) r->a
        R_theta = self.rotation_matrix(theta[0], 0)
        # Rotation along ay (axis = 1) a->b
        R_beta = self.rotation_matrix(beta[0], 1)
        # Rotation along sz (axis = 2) b->s
        R_omega = self.rotation_matrix(omega[0], 2)
        # r->b
        R_beta_theta = np.matmul(R_beta, R_theta)
        R = np.matmul(R_omega, R_beta_theta)

        self.R = R

        q = self.state[:, :1]
        q_dot = self.state[:, 1:2]


        force_a = self.get_force(q=self.rotor_position_A, q_dot=self.rotor_velocity_A)
        force_b = self.get_force(q=self.rotor_position_B, q_dot=self.rotor_velocity_B)

        f = np.array([force_a[0], force_a[1], force_b[0], force_b[1]]).flatten()
        f += action
        y = self.state[:, :2].transpose().flatten()
        self.dynamics_fn(dt, y, f, self.mass, self.gravity, self.Icm)
        # new_y = odeint(func=self.dynamics_fn, y0=y, t=[0,dt], args=(f, self.mass, self.gravity, self.Icm))[1]
        sol = solve_ivp(fun=self.dynamics_fn, t_span=[0, dt], y0=y, method='RK45',
                        args=(f, self.mass, self.gravity, self.Icm), t_eval=[dt])

        if sol['success']:
            new_y = sol['y']
        else:
            raise ValueError('Solver error')

        new_y = new_y.reshape((-1, 6)).transpose()
        self.state[:, :2] = new_y
        # self.state = self.use_rungekutta2_method(self.state)
        # self.state = self.use_euler_method(x, y, z, theta, beta, omega)
        x, y, z, theta, beta, omega = self.state

        # Rotation along rx (axis = 0) r->a
        R_theta = self.rotation_matrix(theta[0], 0)
        # Rotation along ay (axis = 1) a->b
        R_beta = self.rotation_matrix(beta[0], 1)
        # Rotation along sz (axis = 2) b->s
        R_omega = self.rotation_matrix(omega[0], 2)
        # r->b
        R_beta_theta = np.matmul(R_beta, R_theta)
        R = np.matmul(R_omega, R_beta_theta)

        self.R = R

        q = self.state[:, :1]
        q_dot = self.state[:, 1:2]

        self.rotor_position_A_bef = self.rotor_position_A.copy()
        self.rotor_position_B_bef = self.rotor_position_B.copy()

        # bearing A (negative rz axis)
        # position = R.-self.pm*sz + [x, y, z]T
        self.rotor_position_A = np.matmul(R.transpose(), np.array([[0], [0], [-self.pm]])) + q[:3]
        # bearing B (positive rz axis)
        # position = R.self.pm*sz + [x, y, z]T
        self.rotor_position_B = np.matmul(R.transpose(), np.array([[0], [0], [self.pm]])) + q[:3]
        self.rotor_velocity_A = (self.rotor_position_A - self.rotor_position_A_bef) / dt
        self.rotor_velocity_B = (self.rotor_position_B - self.rotor_position_B_bef) / dt

        self.rotor_velocity_A = np.cross(self._w_r(theta, beta, omega), np.array([[0], [0], [-self.pm]]),
                                         axis=0) + q_dot[:3]
        self.rotor_velocity_B = np.cross(self._w_r(theta, beta, omega), np.array([[0], [0], [self.pm]]),
                                         axis=0) + q_dot[:3]

        rotor_distance_A = np.linalg.norm(self.rotor_position_A - np.array([[0], [0], [-self.pm]]))
        rotor_distance_B = np.linalg.norm(self.rotor_position_B - np.array([[0], [0], [self.pm]]))

        done = False
        if not (rotor_distance_A < self.gap_rotor_support and rotor_distance_B < self.gap_rotor_support):
            done = True
        return self._get_obs(), done

    def reset(self):

        self.reset_count += 1
        self.state = np.zeros((6, 3))

        ra, rb = np.random.rand(2) * self.gap_rotor_support
        angle_a, angle_b = np.random.rand(2) * np.pi * 2
        self.state[:, 0] = self.get_position_state_from_polar(ra=ra, angle_a=angle_a,
                                                              rb=rb, angle_b=angle_b)

        # self.state[:3, 0] = pos_center
        # self.state[3:, 0] = [ang_theta, ang_beta, 0]

        if self.init_from_support:
            self.state[:6, 0] = 0
            self.state[0, 0] = -self.gap_rotor_support * 0.99

        if self.init_from_center:
            self.state[:6, 0] = 0

        # if np.random.uniform(low=0, high=1) > 0.9:
        #    self.state[:,0] = 0

        x, y, z, theta, beta, omega = self.state
        # Rotation along rx (axis = 0) r->a
        R_theta = self.rotation_matrix(theta[0], 0)
        # Rotation along ay (axis = 1) a->b
        R_beta = self.rotation_matrix(beta[0], 1)
        # Rotation along sz (axis = 2) b->s
        R_omega = self.rotation_matrix(omega[0], 2)
        # r->b
        R_beta_theta = np.matmul(R_beta, R_theta)
        R = np.matmul(R_omega, R_beta_theta)

        theta_dot, beta_dot = self.get_init_angular_speed(self.state, R_theta, R_beta_theta)

        if self.init_with_velocity:
            self.state[:, 1] = self.sample_velocity_from_polar(ra, angle_a,
                                                               rb, angle_b, self.state,
                                                               steps_before_collision=self.steps_before_collision)

        else:
            theta_dot = np.zeros_like(theta_dot)
            beta_dot = np.zeros_like(beta_dot)
            self.state[3:5, 1] = np.array([theta_dot, beta_dot])

        q = self.state[:, :1]
        q_dot = self.state[:, 1:2]

        self.state[-1, 1] = (np.random.uniform(self.init_rpm_min_ratio,
                                               self.init_rpm_max_ratio)) * self.max_rpm / 60 * 3.14

        if self.forced_initial_omega_dot:
            self.state[-1, 1] = self.forced_initial_omega_dot

        self.R = R
        self.rotor_position_A = np.matmul(R.transpose(), np.array([[0], [0], [-self.pm]])) + q[:3]
        # bearing B (positive rz axis)
        # position = R.self.pm*sz + [x, y, z]T
        self.rotor_position_B = np.matmul(R.transpose(), np.array([[0], [0], [self.pm]])) + q[:3]

        self.rotor_velocity_A = np.cross(self._w_r(theta, beta, omega), np.array([[0], [0], [-self.pm]]),
                                         axis=0) + q_dot[:3]
        self.rotor_velocity_B = np.cross(self._w_r(theta, beta, omega), np.array([[0], [0], [self.pm]]),
                                         axis=0) + q_dot[:3]
        self.steps_before_done = 0
        return self._get_obs()

    def get_init_angular_speed(self, state, R_theta, R_beta_theta):
        # suppose omega_dot = 0
        omega_dot = 0.001
        min_steps_before_collision = np.random.uniform(low=1, high=1.2) * 1e4
        # min_steps_before_collision = 1e7
        w_r_max = self.gap_rotor_support / self.dt / (min_steps_before_collision) / self.pm
        w_r = np.random.rand(2) - 0.5
        w_r = w_r * w_r_max / np.linalg.norm(w_r)

        theta_dot = w_r[0] - omega_dot * np.sin(state[4, 1])
        beta_dot = (w_r[1] - omega_dot * np.cos(state[4, 1]) * np.sin(state[4, 0])) / np.cos(state[3, 0])

        return theta_dot, beta_dot


    def rotation_matrix(self, angle, axis):
        """Returns rotation matrix of angle along axis

        Arguments:
            angle {float} -- angle
            axis {int} -- axis
        """

        if axis == 0:
            return np.array([[1, 0, 0],
                             [0, np.cos(angle), np.sin(angle)],
                             [0, -np.sin(angle), np.cos(angle)]])
        if axis == 1:
            return np.array([[np.cos(angle), 0, np.sin(angle)],
                             [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        if axis == 2:
            return np.array([[np.cos(angle), np.sin(angle), 0],
                             [-np.sin(angle), np.cos(angle), 0],
                             [0, 0, 1]])
        raise ValueError("Axis must be 0, 1 or 2")

    def rotation_matrix_compose(self, theta, beta, inverse=False):
        """Returns rotation matrix along rx and ay

        Arguments:
            theta {float} -- rotation angle around axis rx
            beta {float} -- rotation angle around axis ay

        Keyword Arguments:
            inverse {bool} -- whether to return rotation matrix r->b or its inverse (default: {False})

        Returns:
            np.array -- rotation matrix
        """
        matrix = np.array([[np.cos(beta), -np.sin(theta) * np.sin(beta), np.cos(theta) * np.sin(beta)],
                           0, np.cos(theta), np.sin(theta),
                           -np.sin(beta), -np.sin(theta) * np.cos(beta), np.cos(beta) * np.cos(beta)])
        if inverse:
            self.transpose = np.transpose(matrix)
            return self.transpose
        return matrix

    def _get_obs(self):
        """
        Return observations (current x y position and velocities at A and B)
        """

        return (np.array([self.rotor_position_A[0, 0],  # x1 distance
                          self.rotor_position_A[1, 0],  # x2 distance
                          self.rotor_position_B[0, 0],  # x1 distance
                          self.rotor_position_B[1, 0],  # x2 distance
                          self.rotor_velocity_A[0, 0],  # x1 speed
                          self.rotor_velocity_A[1, 0],  # x2 speed
                          self.rotor_velocity_B[0, 0],  # x1 speed
                          self.rotor_velocity_B[1, 0]],  # x2 speed
                         dtype=np.float64)).tolist()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def magnetic_force(self, current, d):
        return self.mi * (self.n ** 2) * self.area * (current ** 2) / (self.l_metal / self.mi_r + 2 * d) ** 2

    def _w_s(self, theta, beta, omega):
        """
        >>> self._w_s(np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)))
        array([[0.],
               [0.],
               [0.]])

        """

        return np.array([[theta[1] * np.cos(omega[0]) * np.cos(beta[0]) + beta[1] * np.sin(omega[0])],
                         [beta[1] * np.cos(omega[0]) - theta[1] * np.sin(omega[0]) * np.cos(beta[0])],
                         [omega[1] - theta[1] * np.sin(beta[0])]])

    def _w_r(self, theta, beta, omega):
        """

        Velocidade ^R w ^S
        >>> self._w_s(np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)))
        array([[0.],
               [0.],
               [0.]])

        """
        return np.array([[theta[1] - np.cos(theta[0]) * np.sin(beta[0]) * omega[1]],
                         [np.cos(theta[0]) * beta[1] - np.sin(theta[0]) * omega[1]],
                         [np.sin(theta[1]) * beta[1] + np.cos(theta[0]) * np.cos(beta[0]) * omega[1]]
                         ])

    def _w_r_dot(self, theta, beta, omega):
        return np.array([[theta[2] + np.sin(theta[0]) * np.cos(beta[0]) * theta[1] * omega[1] - np.cos(
            theta[0]) * np.cos(beta[0]) * beta[1] * omega[1] - np.cos(theta[0]) * np.sin(beta[1]) * omega[2]],
                         [-np.sin(theta[0]) * theta[1] * beta[1] + np.cos(theta[0]) * beta[2] - np.cos(theta[0]) *
                          theta[1] * omega[1] - np.sin(theta[0]) * omega[2]],
                         [np.cos(theta[0]) * theta[1] * beta[1] + np.sin(theta[0]) * beta[2] - np.sin(
                             theta[0]) * np.cos(beta[0]) * theta[1] * omega[1] - np.cos(theta[0]) * np.sin(beta[0]) *
                          beta[1] * omega[1] + np.cos(theta[0]) * np.cos(beta[0]) * omega[2]]
                         ])

    def _B(self, theta, beta, omega):

        return np.array([[beta[1] * omega[1] * np.cos(omega[0]) - theta[1] * omega[1] * np.sin(omega[0]) * np.cos(
            beta[0]) - theta[1] * beta[1] * np.cos(omega[0]) * np.sin(beta[0])],
                         [- beta[1] * omega[1] * np.sin(omega[0]) - theta[1] * omega[1] * np.cos(omega[0]) * np.cos(
                             beta[0]) + theta[1] * beta[1] * np.sin(omega[0]) * np.sin(beta[0])],
                         [- theta[1] * beta[1] * np.cos(beta[0])]])

    def _A(self, theta, beta, omega, inertia):

        t = np.array([[np.cos(omega[0]) * np.cos(beta[0]), np.sin(omega[0]), 0],
                      [- np.sin(omega[0]) * np.cos(beta[0]), np.cos(omega[0]), 0],
                      [np.sin(beta[0]), 0, 1]])
        return np.matmul(inertia, t)

    def _get_acc(self,
                 f_ay,
                 f_by,
                 f_ax,
                 f_bx,
                 mass,
                 gravity):


        x_dot2 = (f_ax + f_bx) / mass - gravity
        y_dot2 = (f_ay + f_by) / mass
        z_dot2 = 0

        return np.array([x_dot2, y_dot2, z_dot2])

    def _get_angular_acc(self,
                         f_ax,
                         f_bx,
                         f_ay,
                         f_by,
                         state):

        Icm = self.Icm
        R = self.R
        x, y, z, beta, theta, omega = state

        M_r = np.cross([[0], [0], [-self.pm]],
                       np.array([[f_ax], [f_ay], [0]]), axis=0) \
              + np.cross([[0], [0], [self.pm]],
                         np.array([[f_bx], [f_by], [0]]), axis=0)

        M = np.matmul(R, M_r) + np.array([[0], [0], [self.Tq]])
        # M = M_r + np.array([[0],[0],[self.Tq]])
        w_s = self._w_s(theta, beta, omega)
        B = self._B(theta, beta, omega)
        A = self._A(theta, beta, omega, Icm)

        return np.matmul(np.linalg.inv(A), M - np.cross(w_s, np.matmul(self.Icm, w_s), axis=0) - B).reshape((3,))

    def get_dot2(self,
                 state,
                 f_ay,
                 f_by,
                 f_ax,
                 f_bx):

        x, y, z, theta, beta, omega = state

        R = self.get_R(theta=theta[0], beta=beta[0], omega=omega[0])


        trans_dot2 = self._get_acc(f_ay=f_ay,
                                   f_by=f_by,
                                   f_ax=f_ax,
                                   f_bx=f_bx,
                                   mass=self.mass,
                                   gravity=self.gravity
                                   )




        ang_dot2 = self._get_angular_acc(f_ay=f_ay,
                                           f_by=f_by,
                                           f_ax=f_ax,
                                           f_bx=f_bx,
                                         state=state)

        return np.array([trans_dot2, ang_dot2]).flatten()

    def get_R(self, theta: float, beta: float, omega: float):

        # Rotation along rx (axis = 0) r->a
        R_theta = self.rotation_matrix(theta, 0)
        # Rotation along ay (axis = 1) a->b
        R_beta = self.rotation_matrix(beta, 1)
        # Rotation along sz (axis = 2) b->s
        R_omega = self.rotation_matrix(omega, 2)
        # r->b
        R_beta_theta = np.matmul(R_beta, R_theta)
        return np.matmul(R_omega, R_beta_theta)

    def get_momentum(self):
        M = np.eye(5) * self.mass
        M[2:, 2:] = self.Icm

        return M.dot(np.delete(self.state[:, 2].flatten(), 2))

    def dynamics_fn(self, t, y, f, mass, gravity, Icm):

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

    def set_max_rpm(self, rpm):
        self.max_rpm = rpm
        return self.max_rpm

    def set_Tq(self, Tq):
        self.Tq = Tq
        return self.Tq

    def set_dt(self, dt):
        self.dt = dt
        return self.dt

    def set_dt_std(self, dt_std):
        self.dt_std = dt_std

    def get_position_state_from_polar(self, ra, angle_a, rb, angle_b):
        xa = ra * np.sin(angle_a)
        ya = ra * np.cos(angle_a)
        xb = rb * np.sin(angle_b)
        yb = rb * np.cos(angle_b)

        xc = 0.5 * (xa + xb)
        yc = 0.5 * (ya + yb)

        theta = np.arcsin((yb - yc) / self.pm)
        assert (np.abs(theta) < np.pi / 2)
        beta = np.arcsin((xb - xc) / self.pm / np.cos(theta))

        omega = 0

        return np.array([xc, yc, 0, theta, beta, 0])

    def sample_velocity_from_polar(self, ra, angle_a,
                                   rb, angle_b, state,
                                   steps_before_collision=1000):
        r = np.max([ra, rb])
        angle = np.random.uniform(0, 2 * np.pi)
        xc_dot = np.random.uniform(0, self.gap_rotor_support / self.dt / steps_before_collision)
        yc_dot = np.random.uniform(0, self.gap_rotor_support / self.dt / steps_before_collision)

        max_norm = (self.gap_rotor_support - r) / self.dt / steps_before_collision

        beta_dot = max_norm / self.pm * np.sin(angle) - xc_dot
        theta_dot = max_norm / self.pm * np.cos(angle) / (-np.cos(state[4, 0])) - yc_dot

        return np.array([xc_dot, yc_dot, 0, theta_dot, beta_dot, 0])


if __name__ == '__main__':

    env = MagneticBearing3D()

    num_episodes = 3
    for ep in range(num_episodes):
        env.reset()

        for _ in range(1000):
            obs, done = env.step()  # take a random action ([0,1.135/2,10,0, 0,1.135/2,0,0.1])
    env.close()