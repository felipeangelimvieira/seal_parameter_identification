class PID:
    def __init__(self, kp, kd, ki, dt):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.dt = dt
        self.e_prev = 0
        self.integral = 0
        
    def __call__(self, error):
        error = - error
        self.integral += error*self.dt 
        u = self.kp * error + self.kd / self.dt * (error - self.e_prev) + self.ki  * self.integral
        self.e_prev = error
        return u