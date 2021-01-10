import argparse    
import sys
sys.path.append("..")
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from simulation.simple_system import SimpleSystem, sinusoidal_fun, sweep_fun
from simulation.excitation_signals import multisine_fun
from pathlib import Path
import logging 
from tqdm import tqdm

log = logging.getLogger()
log.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
log.addHandler(handler)


def generate_data_episode(axis,
                          config,
                          use_seal=True):
    dt = config["dt"]
    t = config["time_per_episode"]


    env = SimpleSystem()

    if use_seal:
        env.K = np.array(config["K"])
        env.C = np.array(config["C"])
    env.kp = config["kp"]
    env.kd = config["kd"]
    env.ki = config["ki"]

    if config["excitation"].lower() == "sweep":
        excitation_params = config["excitation_params"]
        force_fun = sweep_fun(T=config["time_per_episode"], f1=excitation_params["fmin"], f2=excitation_params["fmax"], axis=axis)
    elif config["excitation"].lower() == "sinusoidal":
        force_fun = sinusoidal_fun(config["f"],  axis=axis)
    elif config["excitation"].lower() == "multisin":
        force_fun = multisin_fun(config["f"],  axis=axis)
    else:
        raise ValueError(config["excitation"])

    res = env.solve(dt=dt, t=t*2, force_fun=force_fun)
    df = pd.DataFrame(res)
    df = df.iloc[df.shape[0]//2:]

    if config["excitation"].lower() == "sinusoidal":
        df["freq"] = config["f"]

    df["axis"] = axis
    df["seal"] = use_seal
    return df

def add_noise_to_data(df, config):

    # Excitation
    df["x"] += np.random.normal(0, config["position_std"], df.shape[0])
    df["y"] += np.random.normal(0, config["position_std"], df.shape[0])
    df["fx"] += np.random.normal(0, config["force_std"], df.shape[0])
    df["fy"] += np.random.normal(0, config["force_std"], df.shape[0])

    return df


def post_processing_data(df, noise):
    if noise:
        df = add_noise_to_data(df, config)
    return df


def generate_dataset_sin(config):
    all_data = pd.DataFrame()

    excitation_params = config["excitation_params"]

    freqs = np.arange(excitation_params["fmin"], excitation_params["fmax"] + 1, excitation_params["fstep"])

    for freq in freqs:
        logging.info(f"{freq}Hz")
        config["f"] = freq
        for use_seal in [True, False]:
            for i in tqdm(range(config.get("episodes_x", 0))):
                df = generate_data_episode(axis="x", config=config, use_seal=use_seal)
                df["episode"] = i
                all_data = pd.concat([all_data, df], ignore_index=False)
            for i in tqdm(range(config.get("episodes_y", 0))):
                df = generate_data_episode(axis="y", config=config, use_seal=use_seal)
                df["episode"] = i
                all_data = pd.concat([all_data, df], ignore_index=False)
            for i in tqdm(range(config.get("episodes_both", 0))):
                df = generate_data_episode(axis="both", config=config, use_seal=use_seal)
                df["episode"] = i
                all_data = pd.concat([all_data, df], ignore_index=False)

    return all_data


def generate_dataset_sweep(config):
    all_data = pd.DataFrame()

    for use_seal in [True, False]:
        logging.info(f"\n Use seal: {use_seal}")
        for i in tqdm(range(config.get("episodes_x", 0))):
            df = generate_data_episode(axis="x", config=config, use_seal=use_seal)
            df["episode"] = i
            all_data = pd.concat([all_data, df], ignore_index=False)
        for i in tqdm(range(config.get("episodes_y", 0))):
            df = generate_data_episode(axis="y", config=config, use_seal=use_seal)
            df["episode"] = i
            all_data = pd.concat([all_data, df], ignore_index=False)
        for i in tqdm(range(config.get("episodes_both", 0))):
            df = generate_data_episode(axis="both", config=config, use_seal=use_seal)
            df["episode"] = i
            all_data = pd.concat([all_data, df], ignore_index=False)

    return all_data

def generate_dataset(config):

    if config["excitation"] == "sweep":
        data = generate_dataset_sweep(config)
    elif config["excitation"] == "sinusoidal":
        data = generate_dataset_sin(config)
    else:
        raise ValueError(config["excitation"])

    data["fx"] = data["fx"] + np.random.normal(0, scale=config["force_std"], size=data.shape[0])
    data["fy"] = data["fy"] + np.random.normal(0, scale=config["force_std"], size=data.shape[0])
    data["x"] = data["x"] + np.random.normal(0, scale=config["position_std"], size=data.shape[0])
    data["y"] = data["y"] + np.random.normal(0, scale=config["position_std"], size=data.shape[0])

    return data

        
if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', type=str)
    argparser.add_argument('--save_dir', type=str)
    argparser.add_argument('--noise', action="store_true")
    args = argparser.parse_args()
    
    config_file = Path(args.config_file)
    with open(config_file, "rb") as f:
        config = json.load(f)

    logging.info(config)


    df = generate_dataset(config)
    df.to_csv(config_file.parent / Path(f"excitation_{config['excitation']}.csv"))

    
    