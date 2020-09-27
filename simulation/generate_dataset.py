import argparse    
import sys
sys.path.append("..")
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from simulation.rotor import MagneticBearing3D
from simulation.pid import PID
from simulation.seal import Seal
from simulation import excitation_signals
from pathlib import Path
import logging 
from tqdm import tqdm

log = logging.getLogger()
log.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
log.addHandler(handler)


def generate_data_episode(env,
                          axis,
                          config,
                          use_seal=True):

    if use_seal:
        K = np.array(config["K"])
        C = np.array(config["C"])
        env.K = K
        env.C = C
    else:
        env.K = np.array([[0, 0],
                          [0, 0]])
        env.C = np.array([[0, 0],
                          [0, 0]])

    # PID
    pid = PID(kp=config["kp"], kd=config["kd"], ki=config["ki"], dt=env.dt)

    # Excitation signal
    signal_fun = getattr(excitation_signals, config["excitation"] + "_fun")
    excitation_signal = signal_fun(axis=axis, T=config["time_per_episode"], **config["excitation_params"])
    
    # Seal
    K = np.array(config["K"])
    C = np.array(config["C"])
    seal = Seal(K=K, C=C)
    
    history = []
    excitation_history = []
    pid_history = []
    t = []
    
    obs = env.reset()
    for _ in tqdm(range(int(config["time_per_episode"]/env.dt))):
        
        
        t.append(_*env.dt)
        # Excitation_signal
        excitation_force = excitation_signal(_*env.dt)

        
        # Pid
        pid_force = pid(obs[:4])
        

        # Action
        action = np.array([.0, .0, .0, .0])
        action += pid_force
        action += np.array([env.gravity, 0, env.gravity, 0])/2
        action += excitation_force
        
        obs, done = env.step(action) 
    
    
        excitation_history.append(excitation_force)
        pid_history.append(pid_force)
        history.append(obs)

        #if done:
        #    break
            
    return {
        "excitation" : np.array(excitation_history),
        "pid" : np.array(pid_history),
        "observation" : np.array(history),
        "t" : np.array(t)
    }


def parse_data_to_csv(data):
    df = pd.DataFrame()
    
    # Excitation
    df["f_ax"] = data["excitation"][:, 0]
    df["f_ay"] = data["excitation"][:, 1]
    df["f_bx"] = data["excitation"][:, 2]
    df["f_by"] = data["excitation"][:, 3]
    
    #
    df["ax"] = data["observation"][:, 0]
    df["ay"] = data["observation"][:, 1]
    df["bx"] = data["observation"][:, 2]
    df["by"] = data["observation"][:, 3]
    
    df["t"]  = data["t"]
    
    return df
    
def add_noise_to_data(df, config):
    # Excitation
    df["f_ax"] += np.random.normal(0, config["force_std"], df.shape[0])
    df["f_ay"] += np.random.normal(0, config["force_std"], df.shape[0])
    df["f_bx"] += np.random.normal(0, config["force_std"], df.shape[0])
    df["f_by"] += np.random.normal(0, config["force_std"], df.shape[0])
    
    # Position
    df["ax"] += np.random.normal(0, config["position_std"], df.shape[0])
    df["ay"] += np.random.normal(0, config["position_std"], df.shape[0])
    df["bx"] += np.random.normal(0, config["position_std"], df.shape[0])
    df["by"] += np.random.normal(0, config["position_std"], df.shape[0])
    
    return df

def add_center_of_mass_columns(df):
    
    df["fx"] = (df["f_ax"] + df["f_bx"])/2
    df["fy"] = (df["f_ay"] + df["f_by"])/2
    
    df["y"] = (df["by"] + df["ay"])/2
    df["x"] = (df["bx"] + df["ax"])/2
    
    return df
    

def post_processing_data(df, noise):
    df = parse_data_to_csv(data)
    if noise:
        df = add_noise_to_data(df, config)
    df = add_center_of_mass_columns(df)
    return df

def get_data_sweep(env, config, config_file):
    
    for use_seal in [True, False]:
        
        
        for i in range(config.get("episodes_x", 0)):
            log.info("X episodes")

            data = generate_data_episode(env, axis="x", config=config, use_seal=use_seal)
            df = post_processing_data(df, args.noise)
            df.to_csv(config_file.parent / Path(f"excitation_x_episode_{i}_seal_{use_seal}.csv"), index=False)



        for i in range(config.get("episodes_y", 0)):
            log.info("Y episodes")
            data = generate_data_episode(env, axis="y", config=config, use_seal=use_seal)
            df = post_processing_data(df, args.noise)

            df.to_csv(config_file.parent / Path(f"excitation_y_episode_{i}_seal_{use_seal}.csv"), index=False)

        for i in range(config.get("episodes_both", 0)):
            log.info("Both - episodes")
            data = generate_data_episode(env, axis="both", config=config, use_seal=use_seal)
            df = post_processing_data(df, args.noise)

            df.to_csv(config_file.parent / Path(f"excitation_both_episode_{i}_seal_{use_seal}.csv"), index=False)
        
def get_data_sin(env, config, config_file):
    
    fmax = config["excitation_params"]["fmax"]
    fmin = config["excitation_params"]["fmin"]
    fstep = config["excitation_params"]["fstep"]
    
    def loop_axis(env, config, axis, use_seal, freq):
        df = pd.DataFrame()
        for i in range(config.get(f"episodes_{axis}", 0)):
            data = generate_data_episode(env, axis=axis, config=config, use_seal=use_seal)
            _df = parse_data_to_csv(data)
            if args.noise:
                _df = add_noise_to_data(_df, config)
            _df = add_center_of_mass_columns(_df)
            _df["episode"] = i
            df = pd.concat([df, _df], ignore_index=True)
        if df.shape[0] > 0:
            df.to_csv(config_file.parent / Path(f"excitation_{axis}_freq_{freq}Hz_seal_{use_seal}.csv"),
                  index=False)

    for use_seal in [True, False]:
        
        for freq in tqdm(range(fmin, fmax+fstep, fstep)):

            log.info(f" {freq} Hz")

            config["excitation_params"]["freq"] = freq

            loop_axis(env, config, axis="x", use_seal=use_seal, freq=freq)
            loop_axis(env, config, axis="y", use_seal=use_seal, freq=freq)
            loop_axis(env, config, axis="both", use_seal=use_seal, freq=freq)


        
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

    env = MagneticBearing3D()
    env.dt = config["dt"]
    
    
    if config["excitation"] == "sweep":
        log.info("Sweep excitation")
        get_data_sweep(env, config, config_file)
        
    elif config["excitation"] == "sinusoidal":
        log.info("Sinusoidal excitation")
        get_data_sin(env, config, config_file)
    
    else:
        ValueError(config["excitation"])
        
    
    