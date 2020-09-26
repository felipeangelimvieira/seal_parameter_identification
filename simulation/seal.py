class Seal:
    def __init__(self, K, C):
        self.K = K
        self.C = C
        
    def __call__(self, q, q_dot):
        
        q = q.reshape((-1, 1))[:2]
        q_dot = q_dot.reshape((-1, 1))[:2]
        return - (self.K @ q + self.C @ q_dot)