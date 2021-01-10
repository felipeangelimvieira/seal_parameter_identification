"""
This file organizes experimental data to be fed to estimators.
The following functions will read a csv file containing position and force
data on both sides of the rotor.
"""

import pandas as pd
import numpy as np
import argparse
import glob
import os


def get_frequency_from_filename(filename):
    freq = filename.split("Hz")[0].split("_")[-1]
    return int(freq)


def prepare_df(df):
    df["f_ay"] -= df["f_ay"].mean()
    df["f_ax"] -= df["f_ax"].mean()
    df["f_bx"] -= df["f_bx"].mean()
    df["f_by"] -= df["f_by"].mean()
    df["x"] = (df["ax"] + df["bx"]) / 2
    # df["x"] = df["ax"]
    df["y"] = (df["ay"] + df["by"]) / 2
    # df["y"] = df["ay"]
    df['fx'] = df["f_ax"] + df["f_bx"]
    # df["fx"] = df["f_ax"]
    df['fy'] = df["f_ay"] + df["f_by"]
    # df["fx"] = df["f_ay"]
    df = df[df.columns.intersection(["datetime", "x", "y", "fx", "fy", "episode"])]

    start_datetime = df["datetime"].min()
    df["t"] = df["datetime"].apply(lambda x: (x - start_datetime).delta * 1e-9)
    return df

def get_data_sin(data_path="../amb_sin/*"):
    all_data = pd.DataFrame()
    data_files = list(
        filter(lambda x: "Hz" in x and x.endswith(".txt"), glob.glob(data_path))
    )

    for file in data_files:
        print(file)
        df = load_amb_sin_data(file).iloc[10000:]
        df = prepare_df(df)

        if "fx" in file:
            df["axis"] = "x"
        elif "fy" in file:
            df["axis"] = "y"
        df["freq"] = get_frequency_from_filename(file)

        all_data = pd.concat([all_data, df], ignore_index=True)

    return all_data


def load_amb_sin_data(file):
    df = pd.read_csv(file, sep="\t")
    df = df.iloc[3:]
    old_cols = ["waveform", "Pos_Ax [um]", "Pos_Ay [um]", "Pos_Bx [um]", "Pos_By [um]", "F_Ax [N]",
                "F_Ay [N]", "F_Bx [N]", "F_By [N]", "Ex_Ax", "Ex_Ay", 'Ex_Bx', "Ex_By"]
    new_cols = ["datetime", "ax", "ay", "bx", "by", "f_ax", "f_ay", "f_bx", "f_by", "e_ax", "e_ay", "e_bx", "e_by"]
    df = df[old_cols]
    df.rename(columns=dict(zip(old_cols, new_cols)), inplace=True)
    df["datetime"] = df["datetime"].apply(pd.to_datetime)

    def to_float(x):
        if not isinstance(x, str):
            return x
        return float(x.replace(",", "."))

    for col in ["ax", "ay", "bx", "by", "f_ax", "f_ay", "f_bx", "f_by", "e_ax", "e_ay", "e_bx", "e_by"]:
        df[col] = df[col].apply(to_float)

    df = df.drop(df[pd.isnull(df["datetime"])].index)

    return df

def load_amb_sweep_data(file):
    dfs = read_sweep_txt(file)

    all_data = pd.DataFrame()
    for i, df in enumerate(dfs):
        df = df.iloc[3:]
        old_cols = ["waveform", "Pos_Ax [um]", "Pos_Ay [um]", "Pos_Bx [um]", "Pos_By [um]", "F_Ax [N]",
                    "F_Ay [N]", "F_Bx [N]", "F_By [N]", "Ex_Ax", "Ex_Ay", 'Ex_Bx', "Ex_By"]
        new_cols = ["datetime", "ax", "ay", "bx", "by", "f_ax", "f_ay", "f_bx", "f_by", "e_ax", "e_ay", "e_bx", "e_by"]
        df = df[old_cols]
        df.rename(columns=dict(zip(old_cols, new_cols)), inplace=True)
        df["datetime"] = df["datetime"].apply(pd.to_datetime)

        def to_float(x):
            if not isinstance(x, str):
                return x
            return float(x.replace(",", "."))

        for col in ["ax", "ay", "bx", "by", "f_ax", "f_ay", "f_bx", "f_by", "e_ax", "e_ay", "e_bx", "e_by"]:
            df[col] = df[col].apply(to_float)

        df = df.drop(df[pd.isnull(df["datetime"])].index)
        df["episode"] = i

        all_data = pd.concat([all_data, df], ignore_index=True)
    return all_data


def read_sweep_txt(file):
    df = pd.read_csv(file, sep="\t", header=None)

    file_break_indexes = list(df[df[0]=="waveform"].index) + [df.index.max()]

    files = []
    for i in range(len(file_break_indexes) - 1):
        file = df.loc[file_break_indexes[i]:file_break_indexes[i+1]]
        file.columns = file.iloc[0]
        file = file.drop(file.index[0])
        file = file.iloc[:-1].dropna(how="all")
        files.append(file)
    return files

def get_data_sweep(data_path):
    all_data = pd.DataFrame()
    data_files = list(
        filter(lambda x: "Hz" in x and x.endswith(".txt"), glob.glob(data_path))
    )

    for file in data_files:
        print(file)
        df = load_amb_sweep_data(file)
        df = prepare_df(df)

        if "fx" in file:
            df["axis"] = "x"
        elif "fy" in file:
            df["axis"] = "y"

#        df["freq"] = np.nan

        all_data = pd.concat([all_data, df], ignore_index=True)

    # Verificado na mão, é um intervalo onde os dados estão com boa qualidade
    all_data = all_data[all_data["episode"] == 0]
    all_data = all_data[all_data["t"] >= 0.771289]#
    all_data["t"] = all_data["t"] -  0.771289
    all_data["episode"] = all_data["t"].apply(lambda t: t // 1)
    all_data["t"] = all_data["t"].apply(lambda t: t % 1)

    return all_data

if __name__ == "__main__":



    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', type=str, help=".csv filepath that must have x, y, fx, fy, t, freq, axis,"
                                                         "and episode columns.")
    argparser.add_argument('--save_path', type=str)
    argparser.add_argument('--signal', type=str, default="sweep")
    args = argparser.parse_args()

    if args.signal == "sin":
        df = get_data_sin(args.data_path)
    elif args.signal == "sweep":
        df = get_data_sweep(args.data_path)

    df.to_csv(args.save_path, index=False)