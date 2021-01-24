import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def build_x_for_frequency_index(*args, index):

    vectors_to_concat = []
    for v in args:
        vectors_to_concat.append(v[:, index].reshape((-1, 1)))
    return np.concatenate(vectors_to_concat, axis=1)


def fit_sin(X, t, frequencies):
    ones_matrix = np.ones(shape=(X.shape[0], len(frequencies)))
    frequencies = np.array(frequencies).reshape((1, -1))
    freq_ts = ones_matrix * 2 * np.pi * frequencies * np.array(t).reshape((-1, 1))
    S = np.concatenate([np.sin(freq_ts), np.cos(freq_ts)], axis=1)


    amplitudes = np.linalg.lstsq(S, X, rcond=None)[0]  # np.linalg.inv(S.transpose() @ S) @ S.transpose() @ np.array(X)

    # V = ( S * amplitudes ).reshape((-1, 2, len(frequencies))).sum(axis=1)
    return S, amplitudes.reshape((1 ,-1))

def calc_basis_matrix(S, amplitudes, index=None):
    num_frequencies = S.shape[1 ]//2
    V = ( S * amplitudes.reshape((1, -1)) ).reshape((-1, 2, num_frequencies))
    if index:
        return V[:, index:index +1]

    return V.sum(axis=1)

def first_derivative_basis(frequencies, t, num_observations=0):
    frequencies = np.array(frequencies).reshape((1, -1))
    ones_matrix = np.ones(shape=(num_observations, len(frequencies)))

    freq_ts = ones_matrix * 2 * np.pi * frequencies * np.array(t).reshape((-1, 1))
    S = np.concatenate([np.cos(freq_ts ) *2 * np.pi * np.array(frequencies) ,
                        -np.sin(freq_ts ) *2 * np.pi * np.array(frequencies)], axis=1)
    return S

def second_derivative_basis(frequencies, t, num_observations=0):
    frequencies = np.array(frequencies).reshape((1, -1))
    ones_matrix = np.ones(shape=(num_observations, len(frequencies)))

    freq_ts = ones_matrix * 2 * np.pi * frequencies * np.array(t).reshape((-1, 1))
    S = np.concatenate([-np.sin(freq_ts) * (2 * np.pi * np.array(frequencies) )**2 ,
                        -np.cos(freq_ts) * (2 * np.pi * np.array(frequencies) )**2], axis=1)

    return S

def build_model_one_axis(sel_df, frequencies, index):

    X =sel_df["x"].values.reshape((-1, 1))
    Y = sel_df["y"].values.reshape((-1, 1))
    Fx = sel_df["fx"].values.reshape((-1, 1))
    Fy = sel_df["fy"].values.reshape((-1, 1))
    F = sel_df[["fx", "fy"]].values.reshape((-1, 2))
    Q = sel_df[["x", "y"]].values.reshape((-1, 1))
    t = sel_df["t"].values.reshape((-1, 1))

    S, amplitudes_x = fit_sin(X, t, frequencies)
    _, amplitudes_y = fit_sin(Y, t, frequencies)
    _, amplitudes_fx = fit_sin(Fx, t, frequencies)
    _, amplitudes_fy = fit_sin(Fy, t, frequencies)
    _, amplitudes_f = fit_sin(F, t, frequencies)
    S_f, amplitudes_f = fit_sin(F, t, frequencies)
    S_dot = first_derivative_basis(frequencies, t, num_observations=X.shape[0])
    S_dot2 = second_derivative_basis(frequencies, t, num_observations=X.shape[0])

    X_project = calc_basis_matrix(S, amplitudes_x)
    Y_project = calc_basis_matrix(S, amplitudes_y)
    X_dot_project = calc_basis_matrix(S_dot, amplitudes_x)
    Y_dot_project = calc_basis_matrix(S_dot, amplitudes_y)
    X_dot2_project = calc_basis_matrix(S_dot2, amplitudes_x)
    Y_dot2_project = calc_basis_matrix(S_dot2, amplitudes_y)
    Fx_project = calc_basis_matrix(S, amplitudes_fx)
    Fy_project = calc_basis_matrix(S, amplitudes_fy)

    X_input = build_x_for_frequency_index(X_project, Y_project, X_dot_project, Y_dot_project, index=index)
    X_dot2_input = build_x_for_frequency_index(X_dot2_project, Y_dot2_project, index=index)
    # F_output = build_x_for_frequency_index(F, index=index)
    F_output = build_x_for_frequency_index(Fx_project, Fy_project, index=index)

    Y_output = F_output - X_dot2_input @ np.array([[1, 0], [0, 1]])

    return X_input, Y_output


def sin_projection_linreg_estimate(df, frequencies, *args, **kwargs):
    Fs = []
    params_history = []


    for index in range(len(frequencies)):
        Fs.append(frequencies[index])
        X_x, y_x = build_model_one_axis(df[df.axis == "x"], frequencies, index=index)
        X_y, y_y = build_model_one_axis(df[df.axis == "y"], frequencies, index=index)

        X = np.concatenate([X_x, X_y], axis=0)
        y = np.concatenate([y_x, y_y], axis=0)

        params, res, _, _ = np.linalg.lstsq(X, y)
        print(params)
        print("Res:")
        print(res)
        params_history.append(params)

    params_history = np.array(params_history)
    Ks = params_history[:, :2].transpose((0, 2, 1))
    Cs = params_history[:, 2:].transpose((0, 2, 1))

    return {"Fs" : Fs,
            "Ks" : Ks,
            "Cs" : Cs}
