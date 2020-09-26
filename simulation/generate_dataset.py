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
from tqdm import tqdm



def generate_data_episode(env,
                          axis,
                          config):

    
    # PID
    pid = PID(kp=config["kp"], kd=config["kd"], ki=config["ki"], dt=env.dt)

    # Excitation signal
    signal_fun = getattr(excitation_signals, config["excitation"] + "_fun")
    excitation_signal = signal_fun(axis=axis , **config["excitation_params"])
    
    # Seal
    K = np.array(config["K"])
    C = np.array(config["C"])
    seal = Seal(K=K, C=C)
    
    history = []
    excitation_history = []
    pid_history = []
    seal_history = []
    t = []
    
    obs = env.reset()
    for _ in tqdm(range(int(config["time_per_episode"]/env.dt))):
        
        t.append(_*env.dt)
        # Excitation_signal
        excitation_force = excitation_signal(_*env.dt)
        
        # Seal force
        f_ax, f_ay = seal(q=obs[[0,1]], q_dot=obs[[4,5]]).flatten()/2
        f_bx, f_by = seal(q=obs[[2,3]], q_dot=obs[[6,7]]).flatten()/2
        seal_force = np.array([f_ax, f_ay, f_bx, f_by])
        
        # Pid
        pid_force = pid(obs[:4])
        

        # Action
        action = pid_force
        action += seal_force
        action += np.array([env.gravity, 0, env.gravity, 0])/2
        action += excitation_force
        
        obs, done = env.step(action) 
    
    
        excitation_history.append(excitation_force)
        pid_history.append(pid_force)
        seal_history.append(seal_force)
        history.append(obs)

        if done:
            break
            
    return {
        "excitation" : np.array(excitation_history),
        "pid" : np.array(pid_history),
        "seal" : np.array(seal_history),
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

if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', type=str)
    argparser.add_argument('--save_dir', type=str)
    argparser.add_argument('--noise', action="store_true")
    args = argparser.parse_args()
    
    config_file = Path(args.config_file)
    with open(config_file, "rb") as f:
        config = json.load(f)
    
    env = MagneticBearing3D()
    env.dt = config["dt"]
    
    
    
    for i in range(config.get("episodes_x", 0)):
        data = generate_data_episode(env, axis="x", config=config)
        df = parse_data_to_csv(data)
        if args.noise:
            df = add_noise_to_data(df, config)
            
        df.to_csv(config_file.parent / Path(f"excitation_x_episode_{i}.csv"), index=False)
        
    for i in range(config.get("episodes_y", 0)):
        data = generate_data_episode(env, axis="y", config=config)
        df = parse_data_to_csv(data)
        if args.noise:
            df = add_noise_to_data(df, config)
            
        df.to_csv(config_file.parent / Path(f"excitation_y_episode_{i}.csv"), index=False)
            
    for i in range(config.get("episodes_both", 0)):
        data = generate_data_episode(env, axis="both", config=config)
        df = parse_data_to_csv(data)
        if args.noise:
            df = add_noise_to_data(df, config)
            
        df.to_csv(config_file.parent / Path(f"excitation_both_episode_{i}.csv"), index=False)
        
    
    