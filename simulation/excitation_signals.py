import numpy as np


def parse_axis(axis):
    if axis not in ["both", "x", "y"]:
        raise ValueError()
    if axis == "y":
        mask = np.array([0, 1])
    if axis  == "x":
        mask = np.array([1, 0])
    if axis == "both":
        mask = np.random.uniform(0.3, 1, size=2)
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
        t = t % T

        f = np.sin((a * t + b) * t)
        return f * mask

    return sweep


def _multisine(t, phases, amplitudes, frequencies):
  result = 0
  for i in range(len(amplitudes)):
    result += amplitudes[i]*np.sin(2*np.pi*frequencies[i]*t + phases[i])
  return result

def multisine_fun(phases, amplitudes, frequencies, axis, *args, **kwargs):
    from functools import partial
    mask = parse_axis(axis)

    multisine = partial(_multisine, phases=phases, amplitudes=amplitudes, frequencies=frequencies)
    def masked_multisine(t):
        return multisine(t)*mask

    return masked_multisine