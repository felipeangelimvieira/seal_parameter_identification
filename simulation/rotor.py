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


        ## Rotor parameters
        # Considering constant cross-section area
        #radius = 14.28 / 2 * 0.001  # radius of the axis
        self.rotor_radius = 36.4 / 2 * 0.001  # radius of the rotors
        radius = self.rotor_radius
        self.radius = radius
        # axis length
        self.axis_length = 0.4
        self.L = self.axis_length / 2
        # axial position from center C to bearings
        self.pm = 0.150

        ## Bearing parameters
        # bearing n number
        self.n = 260
        # Magnetic constants
        self.l_metal = 0.115  # m
        self.mi_r = 8e3
        self.mi = np.pi * 4e-7

        # area of transversal section
        self.area = 2 * 0.0077 * 0.0165 * np.cos(np.pi / 8)
        # gap between rotor and support
        self.gap_rotor_support = 4e-3  # 0.4 mm
        # gap between rotor and bearing
        self.gap_rotor_bearing = 1 * 0.001  # 1.0 mm
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



        # Init parameters
        self.Tq = 0.00
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


        self.dt = 0.000195


        # State
        self.theta = np.zeros((3,))
        self.beta = np.zeros((3,))
        self.omega = np.zeros((3,))
        self.y = np.zeros((3,))
        self.x = np.zeros((3,))
        self.z = np.zeros((3,))
        self.state = np.array([self.x, self.y, self.z, self.theta, self.beta, self.omega])


        self.max_current = 4
        self.steps_before_done = 0



        self.force_inf_A = 0
        self.force_sup_A = 0
        self.force_left_A = 0
        self.force_right_A = 0

        self.force_inf_B = 0
        self.force_sup_B = 0
        self.force_left_B = 0
        self.force_right_B = 0

        self.K = np.array([[.0, .0],[.0, .0]])
        self.C = np.array([[.0, .0],
                           [.0, .0]])

        
        #k = 4300
        #c = 43
        #self.K = np.array([[k, k / 10],
        #                  [-k / 10, k]])
        #self.C = np.array([[c, 0],
        #                  [0, c]])


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

        f = 0

        force_a = self.get_force(q=self.rotor_position_A, q_dot=self.rotor_velocity_A)
        force_b = self.get_force(q=self.rotor_position_B, q_dot=self.rotor_velocity_B)
        #f += np.array([force_a[0], force_a[1], force_b[0], force_b[1]]).flatten()
        
        
        f += action
        
        y = self.state[:, :2].transpose().flatten()
        #self.dynamics_fn(dt, y, f, self.mass, self.gravity, self.Icm)
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


        # bearing A (negative rz axis)
        # position = R.-self.pm*sz + [x, y, z]T
        #self.rotor_position_A = np.matmul(R.transpose(), np.array([[0], [0], [-self.pm]])) + q[:3]
        #self.rotor_position_A = self.position_to_inertial(np.array([[0], [0], [-self.pm]]), q)
        # bearing B (positive rz axis)
        # position = R.self.pm*sz + [x, y, z]T
        #self.rotor_position_B = np.matmul(R.transpose(), np.array([[0], [0], [self.pm]])) + q[:3]
        #self.rotor_position_B = self.position_to_inertial(np.array([[0], [0], [self.pm]]), q)


        #self.rotor_velocity_A = np.cross(self._w_r(theta, beta, omega), np.array([[0], [0], [-self.pm]]),
        #                                 axis=0) + q_dot[:3]
        #self.rotor_velocity_B = np.cross(self._w_r(theta, beta, omega), np.array([[0], [0], [self.pm]]),
        #                                 axis=0) + q_dot[:3]

        self.rotor_position_A, self.rotor_position_B = self.get_rotor_position(q)
        self.rotor_velocity_A, self.rotor_velocity_B = self.get_rotor_velocity(q, q_dot)

        rotor_distance_A = np.linalg.norm(self.rotor_position_A - np.array([[0], [0], [-self.pm]]))
        rotor_distance_B = np.linalg.norm(self.rotor_position_B - np.array([[0], [0], [self.pm]]))

        done = False
        if not (rotor_distance_A < self.gap_rotor_support and rotor_distance_B < self.gap_rotor_support):
            done = True
        return self._get_obs(), done

    def get_rotor_position(self, q):
        rotorA = self.position_to_inertial(np.array([[0], [0], [-self.pm]]), q)
        rotorB = self.position_to_inertial(np.array([[0], [0], [self.pm]]), q)

        return rotorA, rotorB

    def get_rotor_velocity(self, q, q_dot):
        rotorA = self.velocity_to_inertial(np.array([[0], [0], [-self.pm]]), q, q_dot)
        rotorB = self.velocity_to_inertial(np.array([[0], [0], [self.pm]]), q, q_dot)

        return rotorA, rotorB

    def position_to_inertial(self, x, q):
        q = q.reshape((-1, 1))

        _, _, _, theta, beta, omega = q.flatten()
        # Rotation along rx (axis = 0) r->a
        R_theta = self.rotation_matrix(theta, 0)
        # Rotation along ay (axis = 1) a->b
        R_beta = self.rotation_matrix(beta, 1)
        # Rotation along sz (axis = 2) b->s
        R_omega = self.rotation_matrix(omega, 2)
        # r->b
        R_beta_theta = np.matmul(R_beta, R_theta)
        R = np.matmul(R_omega, R_beta_theta)

        x = x.reshape((-1, 1))
        return R.transpose() @ x + q[:3]

    def velocity_to_inertial(self, x, q, q_dot):
        x = x.reshape((-1, 1))
        q = q.reshape((-1, 1))
        q_dot = q_dot.reshape((-1, 1))

        state =  np.concatenate([q, q_dot], axis=-1)
        try:
            assert state.shape == (6, 2)
        except Exception as e:
            print("x:", x.shape)
            print("State:", state.shape)
            raise e
        _, _, _, theta, beta, omega = state
        return np.cross(self._w_r(theta, beta, omega), x, axis=0) + q_dot[:3]

    def reset(self):

        self.reset_count += 1
        self.state = np.zeros((6, 3))


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


    def get_R(self, q):

        R_theta = self.rotation_matrix(theta[0], 0)
        # Rotation along ay (axis = 1) a->b
        R_beta = self.rotation_matrix(beta[0], 1)
        # Rotation along sz (axis = 2) b->s
        R_omega = self.rotation_matrix(omega[0], 2)
        # r->b
        R_beta_theta = np.matmul(R_beta, R_theta)
        R = np.matmul(R_omega, R_beta_theta)

        return R

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
        """
        Acceleration of center of mass

        :param f_ay: force on bearing A along y axis
        :param f_by: force on bearing B along y axis
        :param f_ax: force on bearing A along x axis
        :param f_bx: force on bearing B along x axis
        :param mass: mass
        :param gravity: gravity
        :return:
        """

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
        """
        Angular acceleration

        :param f_ay: force on bearing A along y axis
        :param f_by: force on bearing B along y axis
        :param f_ax: force on bearing A along x axis
        :param f_bx: force on bearing B along x axis
        :param state: state 6x3 matrix (coordinates position, coords velocity, coords acceleration)
        :return: vector of shape (3, 1)
        """

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

        f = f.copy()
        # Seal

        rotor_position_A, rotor_position_B = self.get_rotor_position(q)
        rotor_velocity_A, rotor_velocity_B = self.get_rotor_velocity(q, q_dot)
        force_a = self.get_force(q=rotor_position_A, q_dot=rotor_velocity_A)
        force_b = self.get_force(q=rotor_position_B, q_dot=rotor_velocity_B)
        f += np.array([force_a[0], force_a[1], force_b[0], force_b[1]]).flatten()

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