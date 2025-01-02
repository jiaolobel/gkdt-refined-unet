# -*- coding: utf-8 -*-

import multiprocessing
import os
from multiprocessing import Pool, Process

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
# import tensorflow_addons as tfa
# from sklearn.metrics import cohen_kappa_score

tf.keras.backend.set_floatx('float64')

# - Hyperparameter
# task = 'rgb/a=80.0, b=0.03125, r=3.0'
# task = 'rgb/a=80.0, b=0.0625, r=3.0'
# task = 'rgb/a=80.0, b=0.125, r=3.0'
# task = 'rgb/a=80.0, b=0.25, r=3.0'
# task = 'rgb/a=120.0, b=0.03125, r=3.0'
# task = 'rgb/a=40.0, b=0.03125, r=3.0'

# - Ablation study
# task = 'wobilateral/a=80.0, b=0.03125, r=3.0'
# task = "wolinear2/a=80.0, b=0.03125, r=3.0"

# num_multiprocessing = 6

# qa_path = '../../../data/l8/qa/'
# result_path = os.path.join("../../../output_v4/l8/", task)
# save_root = os.path.join(task)

theta_alpha = [80.0, 40.0, 10.0, 120.0][0]
theta_beta = [0.03125, 0.0625, 0.125, 0.25][0]
theta_gamma = 3.0
n_processes = [8, 4, 2, 1][0]
channels = ["rgb", "multiband"][0]

task = "{}/n_processes={}/a={}, b={}, r={}".format(channels, n_processes, theta_alpha, theta_beta, theta_gamma)

qa_path = "E:/Research/experiment_data/l8/qa/"
result_path = os.path.join("E:/Research/experiment_results/gkdt_rfn_unet/l8/", task)
save_root = os.path.join(task)

compute_miou = tf.keras.metrics.MeanIoU(num_classes=4)

def categorize(image):
    return np.uint8(np.rint(image / 255.0 * 3))

def eval(f):
    if f == None:
        return
    
    # ==== Create result folder ====
    if not os.path.exists(os.path.join(save_root, os.path.splitext(f)[0])):
        os.makedirs(os.path.join(save_root, os.path.splitext(f)[0]))
        print("Create `{}`".format(os.path.join(save_root, os.path.splitext(f)[0])))

    # ==== Load qa and fn out ====
    qa = np.asarray(Image.open(os.path.join(qa_path, f)).convert('L'))
    qa_categorical = categorize(qa)
    out = np.load(os.path.join(result_path, f.replace('mask.png', 'rfn.npz')))['arr_0']

    # - Confusion matrix
    cm = tf.math.confusion_matrix(labels=qa_categorical.flatten(), predictions=out.flatten()).numpy()

    # Kappa score
    cm = cm.astype(np.float64)
    pe_rows = cm.sum(axis=0)
    pe_cols = cm.sum(axis=1)
    sum_total = cm.sum()
    pe = (pe_rows * pe_cols).sum() / (sum_total ** 2)
    po = cm.trace() / sum_total
    kappa = (po - pe) / (1 - pe)
    with open(os.path.join(save_root, os.path.splitext(f)[0], 'kappa.txt'), 'w') as fp:
        fp.writelines(str(kappa))
        print("Write to `{}`.".format(os.path.join(save_root, os.path.splitext(f)[0], 'kappa.txt')))

    # Mean IoU
    compute_miou.reset_states()
    compute_miou.update_state(y_true=qa_categorical.flatten(), y_pred=out.flatten())
    miou = compute_miou.result().numpy()
    with open(os.path.join(save_root, os.path.splitext(f)[0], 'miou.txt'), 'w') as fp:
        fp.writelines(str(miou))
        print("Write to `{}`.".format(os.path.join(save_root, os.path.splitext(f)[0], 'miou.txt')))
  

def main():
    files = os.listdir(qa_path)

    # - Create root
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for f in files:
        eval(f)
        print(f, ' Done.')


if __name__ == "__main__":
    main()
