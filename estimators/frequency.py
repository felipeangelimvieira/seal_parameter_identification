import numpy as np
from numpy.fft import fft, fftshift, fftfreq
import pandas as pd
from collections import Counter

class EIVSin:



    def __init__(self):
        pass

    def estimate(self, df_x, df_y, freq):

        dt = (df_x["t"] - df_y["t"].shift()).median()

        dfs = []
        for episode, group in df_x.groupby("episode"):
            temp = self.get_frequency_domain_data(df=group,
                                                  freq=freq,
                                                  dt=dt)
            temp["episode"] = episode
            dfs.append(temp)
        df_x = pd.concat(dfs, ignore_index=True)

        dfs = []
        for episode, group in df_y.groupby("episode"):
            temp = self.get_frequency_domain_data(df=group,
                                                  freq=freq,
                                                  dt=dt)
            temp["episode"] = episode
            dfs.append(temp)
        df_y = pd.concat(dfs, ignore_index=True)


        df = pd.concat([df_x, df_y], ignore_index=False)

        # Apenas frequencia de interesse
        sel_df = df[(df["freqs"] > freq - 2) & (df["freqs"] < freq + 1)].sort_index()
        # Apenas indices que exisitam no dataframe x e y
        inds = list(map(lambda x: x[0], (filter(lambda x: x[1] == 2, Counter(df.index).most_common()))))


        inds = list(filter(lambda  x: x in sel_df.index, inds))

        print(len(inds))
        sel_df = sel_df.loc[inds]

        Us = []
        Ys = []
        for name, group in sel_df.groupby(sel_df.index):
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

    @staticmethod
    def get_frequency_domain_data(df, freq, dt):
        """
        Recebe um dataframe com colunas x, y, fy e fx e retorna um dataframe com as mesmas colunas + sufixo "f"
        no dominio do tempo, alem da coluna "freqs"

        :param df: pd.DataFrame
        :param freq: float
        :param dt: float: sample interval
        :return:
        """
        N = int(np.ceil(1 / freq / dt * 2))
        
        data = {}
        data["freqs"] = []
        for n in range(df.shape[0] // N):
            data["freqs"].extend(fftshift(fftfreq(N, dt)))
        for col in ["x", "y", "fy", "fx"]:
            data[col + "f"] = []
            for n in range(df.shape[0] // N):
                data[col + "f"].extend(fftshift(fft(df[col].iloc[N * n:N * (n + 1)].values)))

        return pd.DataFrame(data)