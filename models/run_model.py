import argparse

import numpy as np
from numpy.fft import fftshift, fftfreq, fft
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(Path(__file__).parent))
from tqdm import tqdm
from scipy.stats import trim_mean
from utils import *
from utils import Shuffler
from models.sin_projection_linreg import sin_projection_linreg_estimate

try:
    import jax
    from jax import numpy as jnp
    from jax.experimental.ode import odeint
    from jax import grad, jit, vmap, value_and_grad, jacfwd, jacrev, jacobian, hessian
    from jax import random
    from jax.experimental import stax
    from jax.experimental.optimizers import adam, sgd
    from models.newton import mse, initialize_params, get_batch_forward_pass, get_loss_function, train
except:
    pass



def add_derivatives(df):
    all_data = pd.DataFrame()

    for name, group in df.groupby(df.columns.intersection(["seal", "episode", "freq", "axis"]).tolist()):
        dt = (df["t"] - df["t"].shift()).median()
        group = append_derivatives_to_dataframe(group, "x", dt=dt)
        group = append_derivatives_to_dataframe(group, "y", dt=dt)
        group = append_derivatives_to_dataframe(group, "fx", dt=dt)
        group = append_derivatives_to_dataframe(group, "fy", dt=dt)

        all_data = pd.concat([all_data, group])

    return all_data

def _linreg_estimate(df, axis, *args, **kwargs):
    X = df[["x_dot", "y_dot", "x", "y"]].values
    df["fx_"] = df["fx"] - 1 * df["x_dot2"]
    df["fy_"] = df["fy"] - 1 * df["y_dot2"]
    sel_cols = []
    if axis in ("x", "both"):
        sel_cols.append("fx_")
    if axis in ("y", "both"):
        sel_cols.append("fy_")
    for null_col in np.where(np.abs(X.sum(axis=0)) < 1e-9)[0]:
        X[:, null_col] = np.random.normal(size=(X.shape[0]))*1e10

    Y = df[sel_cols].values
    params = np.linalg.inv(X.transpose() @ X) @ X.transpose() @ Y

    if axis == "x":
        params = np.concatenate([params, np.zeros_like(params)*np.nan], axis=1)
    elif axis == "y":
        params = np.concatenate([np.zeros_like(params) * np.nan, params], axis=1)

    C = params[:2].transpose()
    K = params[2:].transpose()

    return {"C": C,
            "K": K}

def linreg_estimate(df, *args, **kwargs):
    # df["x_dot2"] =  df["x_dot2"].shift()
    # df["y_dot2"] =  df["y_dot2"].shift()

    #df = df[df.t > 0.5]
    X = df[["x_dot", "y_dot", "x", "y"]].values
    df["fx_"] = df["fx"] - 1 * df["x_dot2"]
    df["fy_"] = df["fy"] - 1 * df["y_dot2"]
    Y = df[["fx_", "fy_"]].values
    params = np.linalg.inv(X.transpose() @ X) @ X.transpose() @ Y

    C = params[:2].transpose()
    K = params[2:].transpose()

    return {"C": C,
            "K": K}


def linreg_sweep_estimate(df, fmin, fmax, period, window=None, *args, **kwargs):
    # df["x_dot2"] =  df["x_dot2"].shift()
    # df["y_dot2"] =  df["y_dot2"].shift()

    params_history = []

    if not window:

        group = df[(df.episode == 0) & (df.axis == df.axis.values[0])]
        freq_per_s = (fmax - fmin) / period
        dt = (group["t"] - group["t"].shift()).median()
        window = int(4/(dt*freq_per_s))
        print(f"Window: {window}")
        
        print(f"Frequency per s: {freq_per_s}")
        print(f"Frequency per window: { window*dt * freq_per_s}")

    Cs_episodes = []
    Ks_episodes = []
    #df = df[df["episode"] < 2]
    episode_length = -np.inf
    for (episode, axis), group in df.groupby(["episode", "axis"]):
        if group.shape[0]//window < episode_length:
            continue
        episode_length = group.shape[0]//window

        Cs = []
        Ks = []
        for i in tqdm(range(group.shape[0] - window)):
            params = _linreg_estimate(group.iloc[i:(i + window)], axis=axis)
            K = params["K"]
            C = params["C"]
            Ks.append(K)
            Cs.append(C)

        Ks_episodes.append(np.array(Ks))
        Cs_episodes.append(np.array(Cs))

    Cs_episodes = np.array(Cs_episodes)
    Ks_episodes = np.array(Ks_episodes)

    freq_per_s = (fmax - fmin) / 1
    print(f"Frequency per s: {freq_per_s}")
    dt = (group["t"] - group["t"].shift()).median()
    window_t = window*dt
    freq_per_window = window_t * freq_per_s
    
    print(f"Frequency per window: {freq_per_window}")
    print(f"""size: {Ks_episodes.shape[1]}, freq_per_s = {freq_per_s},
          dt = {dt}, fmin={fmin}, freq_per_window={freq_per_window}""")

    Fs = np.arange(start=1, stop=Ks_episodes.shape[1], step=1)*freq_per_s*dt + fmin + freq_per_window/2
    print(Fs)
    Ks = np.nanmean(Ks_episodes, axis=0)
    Cs = np.nanmean(Cs_episodes, axis=0)
    return {"Fs" : Fs,
           "Cs": Cs,
            "Ks": Ks}

def to_frequency_domain(df):
    data = {}

    dt = (df["t"] - df["t"].shift()).median()
    for col in ["x", "y", "fy", "fx"]:
        data[col + "f"] = fftshift(fft(df[col].values))
    data["freqs"] = fftshift(fftfreq(data["xf"].shape[0], d=dt))
    return pd.DataFrame(data)

def select_frequency(df, freq, tol=5e-2):
    return df[np.abs(df.freqs - freq) <= tol]


def eiv_estimate(df, freq, *args, **kwargs):

    print(freq)
    period = 1/freq
    df["period"] = df["t"].apply(lambda x: x // (5*period))
    df = df[df["period"] > 1]

    if "episode" not in df.columns:
        df["episode"] = 0

    df_freq = pd.DataFrame()
    for (_period, episode, axis), group in df.groupby(["period", "episode", "axis"]):
        _df = to_frequency_domain(group)
        _df["axis"] = axis
        _df["episode"] = episode
        _df["period"] = _period

        df_freq = pd.concat([df_freq, _df])

    #df_freq = select_frequency(df_freq[df_freq["freqs"]], freq=freq)

    nearest_freq = np.ceil(df_freq.iloc[(df_freq["freqs"] - freq).abs().argmin(),]["freqs"])
    df_freq = df_freq[np.ceil(df_freq["freqs"]) == nearest_freq]
    #df_freq = df_freq[]

    Us = []
    Ys = []
    for (period, episode), group in df_freq.groupby(["period", "episode"]):
        if group.shape[0] < 2:
            continue

        U = group[["xf", "yf"]].values.transpose()
        Y = group[["fxf", "fyf"]].values.transpose()
        Us.append(U)
        Ys.append(Y)

    Us = np.array(Us)
    Ys = np.array(Ys)


    G = np.mean(Ys, axis=0) @ np.linalg.inv(np.mean(Us, axis=0))
    C = np.imag(G) / (2 * np.pi * freq)
    K = np.real(G) + (2 * np.pi * freq) ** 2 * np.array([[1, 0], [0, 1]])

    print(f"K: {K}")
    print(f"C: {C}")

    return {"freq": freq,
            "G": G,
            "C": C,
            "K": K}
    return df_freq


def optimize(q, q_dot, q_dot2, f, batch_size=10000, step_size=1e3, epochs=7):
    from jax import random
    
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


def optimization_estimate(df, *args, **kwargs):
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

        if "seal" in df.columns:
            sel_df = df[(df.seal == True) & (df.freq == freq)]
            with_seal = estimate_fun(sel_df, freq)

            sel_df = df[(df.seal == False) & (df.freq == freq)]
            wo_seal = estimate_fun(sel_df, freq)

            C = with_seal["C"] - wo_seal["C"]
            K = with_seal["K"] - wo_seal["K"]

        else:
            sel_df = df[(df.freq == freq)]
            coefs = estimate_fun(sel_df, freq)

            C = coefs["C"]
            K = coefs["K"]

        Cs.append(C)
        Ks.append(K)
        Fs.append(freq)

    Cs = np.array(Cs)
    Ks = np.array(Ks)
    Fs = np.array(Fs)

    return Fs, Ks, Cs


def get_coefficients_sweep(df, estimate_fun, *args, **kwargs):

    if "seal" in df.columns:
        wo_seal = estimate_fun(df[df.seal == False], *args, **kwargs)
        with_seal = estimate_fun(df[df.seal == True], *args, **kwargs)
        Cs = np.array(with_seal["Cs"]) - np.array(wo_seal["Cs"])
        Ks = np.array(with_seal["Ks"]) - np.array(wo_seal["Ks"])


        coefs = {"Fs" : wo_seal["Fs"],
                "Cs" : Cs,
                "Ks" : Ks}

    else:
        coefs = estimate_fun(df, *args, **kwargs)

    return coefs["Fs"], coefs["Ks"], coefs["Cs"]


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


models = {
    "linreg" : linreg_estimate,
    "gradient" : optimization_estimate,
    "eiv" : eiv_estimate,
    "linreg_sweep" : linreg_sweep_estimate,
    "sin_projection_linreg" : sin_projection_linreg_estimate
}


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', type=str, help=".csv filepath that must have x, y, fx, fy, t, freq, axis,"
                                                         "and episode columns.")
    argparser.add_argument('--save_path', type=str)
    argparser.add_argument('--model', type=str)
    argparser.add_argument('--sweep_fmin', type=float, default=5)
    argparser.add_argument('--sweep_fmax', type=float, default=69)
    argparser.add_argument('--sweep_period', type=float, default=2)
    argparser.add_argument('--frequencies', nargs="+",
                           default=[5,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,69], #experimento
                           #default = list(range(4,70,5)),
                           )
    argparser.add_argument('--savgol', action="store_true")
    args = argparser.parse_args()

    data_path = Path(args.data_path)
    save_path = Path(args.save_path) if args.save_path else None
    savgol_path = save_path.parent / Path(data_path.stem + "_savgol.csv") if args.save_path else None

    df = pd.read_csv(args.data_path)
    estimate_fun = models[args.model]

    if args.savgol:
        df = add_derivatives(df)


    if "sweep" in args.model or "sin_projection" in args.model:
        Fs, Ks, Cs = get_coefficients_sweep(df, estimate_fun, fmin=args.sweep_fmin, fmax=args.sweep_fmax,
                                            period=args.sweep_period, **vars(args))
    else:
        Fs, Ks, Cs = get_coefficients(df, estimate_fun)

    results_df = to_dataframe(Fs, Ks, Cs)
    print(results_df)
    results_df["model"] = args.model
    results_df["savgol"] = args.savgol

    if args.save_path:
        df.to_csv(savgol_path, index=False)
        results_df.to_csv(save_path, index=False)



