# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np

# - Hyperparameter
# task = "rgb/a=80.0, b=0.03125, r=3.0"
# task = 'rgb/a=80.0, b=0.0625, r=3.0'
# task = 'rgb/a=80.0, b=0.125, r=3.0'
# task = 'rgb/a=80.0, b=0.25, r=3.0'
# task = 'rgb/a=120.0, b=0.03125, r=3.0'
# task = 'rgb/a=40.0, b=0.03125, r=3.0'
# task = 'rgb/a=10.0, b=0.03125, r=3.0'

# - Multiband input
# task = 'multiband/a=80.0, b=0.03125, r=3.0'

# - Ablation study
# task = 'wobilateral/a=80.0, b=0.03125, r=3.0'
# task = "wolinear2/a=80.0, b=0.03125, r=3.0"

class Config:
    def __init__(self):
        # Variables wrt write, should be checked carefully.
        metrics = "f1s"
        # Var wrt read
        theta_alpha = [80.0, 40.0, 10.0, 120.0][0] # 0~3
        theta_beta = [0.03125, 0.0625, 0.125, 0.25][0] # 0~3
        theta_gamma = 3.0
        n_processes = [8, 4, 2, 1][0] # 0~2
        channels = ["rgb", "multiband"][0] # 0~1

        # Params in terms of vars.
        params = "{}/n_processes={}/a={}, b={}, r={}".format(channels, n_processes, theta_alpha, theta_beta, theta_gamma)
        self.file_path = os.path.join(params)
        self.save_name = os.path.join(params, "{}.csv".format(metrics))

        self.files = [
            "LC08_L1TP_113026_20160327_20170327_01_T1_mask", 
            "LC08_L1TP_113026_20160412_20170326_01_T1_mask", 
            "LC08_L1TP_113026_20160428_20170326_01_T1_mask", 
            "LC08_L1TP_113026_20160514_20170324_01_T1_mask", 
            "LC08_L1TP_113026_20160530_20170324_01_T1_mask", 
            "LC08_L1TP_113026_20160615_20170324_01_T1_mask", 
            "LC08_L1TP_113026_20160717_20170323_01_T1_mask", 
            "LC08_L1TP_113026_20160802_20170322_01_T1_mask", 
            "LC08_L1TP_113026_20160818_20170322_01_T1_mask", 
            "LC08_L1TP_113026_20161021_20170319_01_T1_mask", 
            "LC08_L1TP_113026_20161106_20170318_01_T1_mask", 
        ]

config = Config()

f1s = list()

for f in config.files:
    df = pd.read_excel(os.path.join(config.file_path, f, 'f1.xlsx'))
    data = df.values[:, 1]
    f1s += [data]

f1s = np.asarray(f1s)
f1s = f1s.transpose()

dframe = pd.DataFrame(f1s)
dframe.to_csv(config.save_name, header=False, index=False)
print("Write to `{}`.".format(config.save_name))
os.chmod(config.save_name, mode=0o444)
print("Read-only `{}`".format(config.save_name))
