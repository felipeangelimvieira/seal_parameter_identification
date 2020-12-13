import argparse
import numpy as np
from numpy.fft import fftshift, fftfreq, fft
import matplotlib.pyplot as plt
import pandas as pd
import sys
from utils import *
from utils import Shuffler
import jax
from jax import numpy as jnp
from jax.experimental.ode import odeint
from jax import grad, jit, vmap, value_and_grad, jacfwd, jacrev, jacobian, hessian
from jax import random
from jax.experimental import stax
from jax.experimental.optimizers import adam, sgd
from models.newton import mse, initialize_params, get_batch_forward_pass, get_loss_function, train


def add_derivatives(df):
    all_data = pd.DataFrame()

    for name, group in df.groupby(["seal", "episode", "freq"]):
        dt = (df["t"] - df["t"].shift()).median()
        group = append_derivatives_to_dataframe(group, "x", dt=dt)
        group = append_derivatives_to_dataframe(group, "y", dt=dt)
        group = append_derivatives_to_dataframe(group, "fx", dt=dt)
        group = append_derivatives_to_dataframe(group, "fy", dt=dt)

        all_data = pd.concat([all_data, group])

    return all_data


def linreg_estimate(df):
    # df["x_dot2"] =  df["x_dot2"].shift()
    # df["y_dot2"] =  df["y_dot2"].shift()

    df = df[df.t > 0.5]
    X = df[["x_dot", "y_dot", "x", "y"]].values
    df["fx_"] = df["fx"] - 1 * df["x_dot2"]
    df["fy_"] = df["fy"] - 1 * df["y_dot2"]
    Y = df[["fx_", "fy_"]].values
    params = np.linalg.inv(X.transpose() @ X) @ X.transpose() @ Y

    C = params[:2].transpose()
    K = params[2:].transpose()

    return {"C": C,
            "K": K}

def to_frequency_domain(df):
    data = {}

    dt = (df["t"] - df["t"].shift()).median()
    for col in ["x", "y", "fy", "fx"]:
        data[col + "f"] = fftshift(fft(df[col].values))
    data["freqs"] = fftshift(fftfreq(data["xf"].shape[0], d=dt))
    return pd.DataFrame(data)

def select_frequency(df, freq, tol=1e-2):
    return df[np.abs(df.freqs - freq) <= tol]


def eiv_estimate(df, freq):
    df_freq = pd.DataFrame()

    for (episode, axis), group in df.groupby(["episode", "axis"]):
        _df = to_frequency_domain(group)
        _df["axis"] = axis
        _df["episode"] = episode

        df_freq = pd.concat([df_freq, _df])

    df_freq = select_frequency(df_freq, freq=freq)

    Us = []
    Ys = []
    for episode, group in df_freq.groupby("episode"):
        U = group[["xf", "yf"]].values.transpose()
        Y = group[["fxf", "fyf"]].values.transpose()
        Us.append(U)
        Ys.append(Y)

    Us = np.array(Us)
    Ys = np.array(Ys)

    G = np.mean(Ys, axis=0) @ np.linalg.inv(np.mean(Us, axis=0))
    C = np.imag(G) / (2 * np.pi * freq)
    K = np.real(G) + (2 * np.pi * freq) ** 2 * np.array([[1, 0], [0, 1]])

    return {"freq": freq,
            "G": G,
            "C": C,
            "K": K}
    return df_freq


def optimize(q, q_dot, q_dot2, f, batch_size=10000, step_size=1e3, epochs=7):
    rng = random.PRNGKey(15)
    params = initialize_params(rng, dims=2, scale=1)
    batch_forward_pass = get_batch_forward_pass(mass=[[1, 0], [0, 1]], g=jnp.array([[0], [0]]))

    shuffler = Shuffler(len(q))
    q_shuffled = shuffler.shuffle(q)
    q_dot_shuffled = shuffler.shuffle(q_dot)
    q_dot2_shuffled = shuffler.shuffle(q_dot2)
    f_shuffled = shuffler.shuffle(f)

    def callback(y_pred, y_true, window=14000):
        plt.figure(figsize=(10, 8))
        random_ind = int(np.random.uniform(0, len(y_pred) - window - 1))
        plt.plot(shuffler.undo_shuffle(np.squeeze(y_pred))[random_ind:(random_ind + window)], linestyle="dotted")
        plt.plot(shuffler.undo_shuffle(np.squeeze(y_true))[random_ind:(random_ind + window)])
        plt.show()

    params = train(params, q_shuffled, q_dot_shuffled, q_dot2_shuffled, f_shuffled, batch_size=batch_size,
                   optimizer=adam, step_size=step_size, epochs=epochs, callback=callback,
                   batch_forward_pass=batch_forward_pass)
    return params


def optimization_estimate(df):
    q = df[["x", "y"]].values.reshape((-1, 2, 1))
    q_dot = df[["x_dot", "y_dot"]].values.reshape((-1, 2, 1))
    q_dot2 = df[["x_dot2", "y_dot2"]].values.reshape((-1, 2, 1))
    f = df[["fx", "fy"]].values.reshape((-1, 2, 1))
    params = optimize(q, q_dot, q_dot2, f, batch_size=8192, step_size=1e3, epochs=7)
    K, C = params
    return {"K": K,
            "C": C}



def get_coefficients(df, estimate_fun):
    Cs = []
    Ks = []
    Fs = []

    for freq in df.freq.unique():
        sel_df = df[(df.seal == True) & (df.freq == freq)]
        with_seal = estimate_fun(sel_df)

        sel_df = df[(df.seal == False) & (df.freq == freq)]
        wo_seal = estimate_fun(sel_df)

        C = with_seal["C"] - wo_seal["C"]
        K = with_seal["K"] - wo_seal["K"]

        Cs.append(C)
        Ks.append(K)
        Fs.append(freq)

    Cs = np.array(Cs)
    Ks = np.array(Ks)
    Fs = np.array(Fs)

    return Fs, Ks, Cs


def to_dataframe(Fs, Ks, Cs):
    all_data = []
    for i, freq in enumerate(Fs):
        K = Ks[i]
        C = Cs[i]

        data = {"freq": freq,
                "kxx": K[0, 0],
                "kxy": K[0, 1],
                "kyy": K[1, 1],
                "kyx": K[1, 0],
                "cxx": C[0, 0],
                "cxy": C[0, 1],
                "cyy": C[1, 1],
                "cyx": C[1, 0]}
        all_data.append(data)

    df = pd.DataFrame(all_data)

    return df

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_file', type=str, help=".csv filepath that must have x, y, fx, fy, t, freq, axis,"
                                                         "and episode columns.")
    argparser.add_argument('--save_dir', type=str)
    argparser.add_argument('--model', type=str)
    argparser.add_argument('--savgol', action="store_true")
    args = argparser.parse_args()

    df = pd.read_csv("../simulation/data/debug/excitation_sinusoidal.csv")df = pd.read_csv("../simulation/data/debug/excitation_sinusoidal.csv")
