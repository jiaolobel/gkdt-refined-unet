"""
Categorical indicators to latex code.
"""

import os
import numpy as np
import pandas as pd

# tasks = [
#     "rgb/a=80.0, b=0.03125, r=3.0",
#     "rgb/a=80.0, b=0.0625, r=3.0",
#     "rgb/a=80.0, b=0.125, r=3.0",
#     "rgb/a=80.0, b=0.25, r=3.0",
#     "rgb/a=120.0, b=0.03125, r=3.0",
#     "rgb/a=40.0, b=0.03125, r=3.0",
#     # "rgb/a=10.0, b=0.03125, r=3.0",
#     # "multiband/a=80.0, b=0.03125, r=3.0",
#     "wobilateral/a=80.0, b=0.03125, r=3.0",
#     "wolinear2/a=80.0, b=0.03125, r=3.0",
# ]

theta_alphas = [80.0, 40.0, 10.0, 120.0] # 0~3
theta_betas = [0.03125, 0.0625, 0.125, 0.25] # 0~3
theta_gamma = 3.0
ns_processes = [8, 4, 2, 1] # 0~2
channels = ["rgb", "multiband"] # 0~1

# Params in terms of vars.
tasks = [
    "{}/n_processes={}/a={}, b={}, r={}".format(channels[0], ns_processes[0], theta_alphas[0], theta_betas[0], theta_gamma), 

    # theta_alpha
    "{}/n_processes={}/a={}, b={}, r={}".format(channels[0], ns_processes[0], theta_alphas[1], theta_betas[0], theta_gamma), 
    "{}/n_processes={}/a={}, b={}, r={}".format(channels[0], ns_processes[0], theta_alphas[2], theta_betas[0], theta_gamma), 
    "{}/n_processes={}/a={}, b={}, r={}".format(channels[0], ns_processes[0], theta_alphas[3], theta_betas[0], theta_gamma), 

    # theta_beta
    "{}/n_processes={}/a={}, b={}, r={}".format(channels[0], ns_processes[0], theta_alphas[0], theta_betas[1], theta_gamma), 
    "{}/n_processes={}/a={}, b={}, r={}".format(channels[0], ns_processes[0], theta_alphas[0], theta_betas[2], theta_gamma), 
    "{}/n_processes={}/a={}, b={}, r={}".format(channels[0], ns_processes[0], theta_alphas[0], theta_betas[3], theta_gamma), 

    # n_processes
    "{}/n_processes={}/a={}, b={}, r={}".format(channels[0], ns_processes[1], theta_alphas[0], theta_betas[0], theta_gamma), 
    "{}/n_processes={}/a={}, b={}, r={}".format(channels[0], ns_processes[2], theta_alphas[0], theta_betas[0], theta_gamma), 

    # channels
    "{}/n_processes={}/a={}, b={}, r={}".format(channels[1], ns_processes[0], theta_alphas[0], theta_betas[0], theta_gamma), 
] 

out_name = "categorical_indicators.txt"

with open(out_name, "w") as fwriter:
    fwriter.writelines("model & precision & recall & f1 & precision & recall & f1 & precision & recall & f1 & precision & recall & f1 \\\\\n")

    for task in tasks:
        line = "{}".format(task)

        means_p, stds_p = list(), list()
        means_r, stds_r = list(), list()
        means_f1, stds_f1 = list(), list()

        fname = os.path.join(task, "precisions.csv")
        dframe = pd.read_csv(fname, header=None)
        dframe = dframe.T

        for idx in range(4): # 0, 1, 2, and 3 for background, fill value, shadow, and cloud
            means_p.append(np.round(dframe[idx].mean() * 100, 2))
            stds_p.append(np.round(dframe[idx].std() * 100, 2))

        fname = os.path.join(task, "recalls.csv")
        dframe = pd.read_csv(fname, header=None)
        dframe = dframe.T

        for idx in range(4):
            means_r.append(np.round(dframe[idx].mean() * 100, 2))
            stds_r.append(np.round(dframe[idx].std() * 100, 2))

        fname = os.path.join(task, "f1s.csv")
        dframe = pd.read_csv(fname, header=None)
        dframe = dframe.T

        for idx in range(4):
            means_f1.append(np.round(dframe[idx].mean() * 100, 2))
            stds_f1.append(np.round(dframe[idx].std() * 100, 2))

        for mean_p, std_p, mean_r, std_r, mean_f1, std_f1 in zip(means_p, stds_p, means_r, stds_r, means_f1, stds_f1):
            line += " & ${} \\pm {}$ & ${} \\pm {}$ & ${} \\pm {}$".format(mean_p, std_p, mean_r, std_r, mean_f1, std_f1)

        line += "\\\\\n"

        fwriter.writelines(line)

print("Write to `{}`".format(out_name))
os.chmod(out_name, mode=0o444)
print("Read-only `{}`".format(out_name))