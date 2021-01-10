import numpy as np
import pandas as pd
from scipy import signal
savgol_filter = signal.savgol_filter

def second_order_diff(x, dt):

    diff = []
    for dim in x:
        diff.append(np.convolve(dim, np.array([1, - 2, 1]), mode="same")[1:-1]/(dt**2))
    return np.array(diff)

def first_order_diff(x, dt):

    diff = []
    for dim in x:
        diff.append(np.convolve(dim,  np.array([1, -1]), mode="same")[1:-1]/dt)
    return np.array(diff)


def first_order_diff(x, dt):
    diff = []
    for dim in x:
        diff.append(np.convolve(dim,  np.array([1, -1]), mode="same")[1:-1]/dt)
    return np.array(diff)

def first_order_diff_five_points(x, dt):
    diff = []
    for dim in x:
        diff.append(np.convolve(dim,  np.array([1, -8, 0, 8, -1]), mode="same")[1:-1]/dt)
    return np.array(diff)


def load_amb_sin_data(file):
    df = pd.read_csv(file, sep="\t")
    df = df.iloc[3:]
    old_cols = ["waveform", "Pos_Ax [um]", "Pos_Ay [um]", "Pos_Bx [um]", "Pos_By [um]", "F_Ax [N]",
                "F_Ay [N]", "F_Bx [N]", "F_By [N]", "Ex_Ax", "Ex_Ay", 'Ex_Bx', "Ex_By"]
    new_cols = ["datetime", "ax", "ay", "bx", "by", "f_ax", "f_ay", "f_bx", "f_by", "e_ax", "e_ay", "e_bx", "e_by"]
    df = df[old_cols]
    df.rename(columns = dict(zip(old_cols, new_cols)), inplace=True)
    df["datetime"] = df["datetime"].apply(pd.to_datetime)
    
    def to_float(x):
        if not isinstance(x, str):
            return x
        return float(x.replace(",", "."))

    for col in [ "ax", "ay", "bx", "by", "f_ax", "f_ay", "f_bx", "f_by", "e_ax", "e_ay", "e_bx", "e_by"]:
        df[col] = df[col].apply(to_float)
    
    df = df.drop(df[pd.isnull(df["datetime"])].index)
    
    return df



def append_derivatives_to_dataframe(df, column, dt, window_length=71, polyorder=5):

    df = df.copy()
    x = df[column].values.flatten()
    x = savgol_filter(x,
                       window_length=window_length,
                       polyorder=polyorder, deriv=0)

    dx = savgol_filter(x,
                       window_length=window_length,
                       polyorder=polyorder, deriv=1)/(dt)

    ddx = savgol_filter(x,
                       window_length=window_length,
                       polyorder=polyorder, deriv=2)/(dt**2)
    df[column] = x
    df[column + "_dot"] = dx
    df[column + "_dot2"] = ddx

    return df

def append_derivative_to_dataframe(df, column, dt, window_length, polyorder):
    
    df = df.copy()
    x = df[column].values.flatten()

    dx = savgol_filter(x,
                       window_length=window_length,
                       polyorder=polyorder, deriv=1)/(dt)

    df[column + "_dot"] = dx

    return df


class Shuffler:

    def __init__(self, length):
        self.indexes = [i for i in range(length)]
        np.random.shuffle(self.indexes)


    def shuffle(self, x):
        return x[self.indexes]

    def undo_shuffle(self, x):
        return x[np.argsort(self.indexes)]
