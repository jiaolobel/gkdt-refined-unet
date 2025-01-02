"""
Global indicators to latex code.
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

out_name = "global_indicators.txt"

with open(out_name, "w") as fwriter:
    fwriter.writelines("model & time & acc & kappa & miou\\\\\n")
    for task in tasks:
        line = "{} & ".format(task)
        # - time
        fname = "E:/Research/experiment_results/gkdt_rfn_unet/l8/{}/log.csv".format(task)
        dframe = pd.read_csv(fname)
        mean_time, std_time = dframe[" time"].mean(), dframe[" time"].std()
        mean_time, std_time = np.round(mean_time, 2), np.round(std_time, 2)
        # times.append("${} \\pm {}$".format(mean_time, std_time))
        line += "${} \\pm {}$ & ".format(mean_time, std_time)

        # - acc
        fname = os.path.join(task, "accs.csv")
        dframe = pd.read_csv(fname, header=None)
        mean_acc, std_acc = dframe[0].mean(), dframe[0].std()
        mean_acc, std_acc = np.round(mean_acc * 100, 2), np.round(std_acc * 100, 2)
        # accs.append("${} \\pm {}$".format(mean_acc, std_acc))
        line += "${} \\pm {}$ & ".format(mean_acc, std_acc)

        # - kappa
        fname = os.path.join(task, "kappas.csv")
        dframe = pd.read_csv(fname, header=None)
        mean_kappa, std_kappa = dframe[0].mean(), dframe[0].std()
        mean_kappa, std_kappa = np.round(mean_kappa * 100, 2), np.round(std_kappa * 100, 2)
        # kappas.append("${} \\pm {}$".format(mean_kappa, std_kappa))
        line += "${} \\pm {}$ & ".format(mean_kappa, std_kappa)

        # - miou
        fname = os.path.join(task, "mious.csv")
        dframe = pd.read_csv(fname, header=None)
        mean_miou, std_miou = dframe[0].mean(), dframe[0].std()
        mean_miou, std_miou = np.round(mean_miou * 100, 2), np.round(std_miou * 100, 2)
        # mious.append("${} \\pm {}$".format(mean_miou, std_miou))
        line += "${} \\pm {}$ ".format(mean_miou, std_miou)

        fwriter.writelines(line + "\\\\\n")

print("Write to `{}`".format(out_name))
os.chmod(out_name, mode=0o444)
print("Read-only ``{}".format(out_name))