import numpy as np

def parse_axis(axis):
    if axis not in ["both", "x", "y"]:
        raise ValueError()
    mask = np.array([0, 0, 0, 0])
    if axis in ("y", "both"):
        mask += np.array([0, 1, 0, 1])
    if axis in ("x", "both"):
        mask += np.array([1, 0, 1, 0])
    return mask    
    
def sinusoidal(freq, axis="both", **kwargs):
    
    mask = parse_axis(axis)
    
    def force(t):
        f = np.sin(2*np.pi*freq*t)
        return mask * f
    return force


def sweep_fun(T, f1, f2, axis="both", **kwargs):
    
    mask = parse_axis(axis)
    
    f0 = 1/T
    k1 = f1/f0
    k2 = f2/f0
    a = np.pi * (k2 - k1) * (f0**2)
    b = 2*np.pi*f0
    
    def sweep(t):
        f = np.sin(2*np.pi*(a*t * b)*t)
        return f*mask
    return sweep