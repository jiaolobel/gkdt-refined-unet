# -*- coding: utf-8 -*-

import multiprocessing
import os
from multiprocessing import Pool, Process

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score, recall_score, f1_score)

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

theta_alpha = [80.0, 40.0, 10.0, 120.0][0]
theta_beta = [0.03125, 0.0625, 0.125, 0.25][0]
theta_gamma = 3.0
n_processes = [8, 4, 2, 1][0]
channels = ["rgb", "multiband"][1]

task = "{}/n_processes={}/a={}, b={}, r={}".format(channels, n_processes, theta_alpha, theta_beta, theta_gamma)

qa_path = "E:/Research/experiment_data/l8/qa/"
result_path = os.path.join("E:/Research/experiment_results/gkdt_rfn_unet/l8/", task)
save_root = os.path.join(task)

num_multiprocessing = 11

def categorize(image):
    return np.uint8(np.rint(image / 255.0 * 3))

def calculate_mean_std(indicators):
    return np.mean(indicators, axis=0), np.std(indicators, axis=0, ddof=1)

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

    # - Accuracy
    acc = accuracy_score(y_true=qa_categorical.flatten(), y_pred=out.flatten())
    with open(os.path.join(save_root, os.path.splitext(f)[0], 'accuracy.txt'), 'w') as fp:
        fp.writelines(str(acc))
    print("Write to `{}`".format(os.path.join(save_root, os.path.splitext(f)[0], 'accuracy.txt')))

    # - Confusion matrix
    cm = confusion_matrix(y_true=qa_categorical.flatten(), y_pred=out.flatten())
    with pd.ExcelWriter(os.path.join(save_root, os.path.splitext(f)[0], 'confusion_matrix.xlsx')) as writer:
        data = pd.DataFrame(cm)
        data.to_excel(writer)
        writer.save()
    print("Write to `{}`".format(os.path.join(save_root, os.path.splitext(f)[0], 'confusion_matrix.xlsx')))

    # - Precision and recall
    precision = precision_score(y_true=qa_categorical.flatten(), y_pred=out.flatten(), average=None)
    recall = recall_score(y_true=qa_categorical.flatten(), y_pred=out.flatten(), average=None)
    with pd.ExcelWriter(os.path.join(save_root, os.path.splitext(f)[0], 'precision.xlsx')) as writer:
        data = pd.DataFrame(precision)
        data.to_excel(writer)
        writer.save()
    print("Write to `{}`".format(os.path.join(save_root, os.path.splitext(f)[0], 'precision.xlsx')))
    
    with pd.ExcelWriter(os.path.join(save_root, os.path.splitext(f)[0], 'recall.xlsx')) as writer:
        data = pd.DataFrame(recall)
        data.to_excel(writer)
        writer.save()
    print("Write to `{}`".format(os.path.join(save_root, os.path.splitext(f)[0], 'recall.xlsx')))

    # - Classification report
    cr = classification_report(y_true=qa_categorical.flatten(), y_pred=out.flatten(), digits=8)
    with open(os.path.join(save_root, os.path.splitext(f)[0], 'classification_report.txt'), 'w') as fp:
        fp.writelines(cr)
    print("Write to `{}`".format(os.path.join(save_root, os.path.splitext(f)[0], 'classification_report.txt')))

    # - F1 score
    f1 = f1_score(y_true=qa_categorical.flatten(), y_pred=out.flatten(), average=None)
    with pd.ExcelWriter(os.path.join(save_root, os.path.splitext(f)[0], 'f1.xlsx')) as writer:
        data = pd.DataFrame(f1)
        data.to_excel(writer)
        writer.save() 
    print("Write to `{}`".format(os.path.join(save_root, os.path.splitext(f)[0], 'f1.xlsx')))
  

def main():
    files = os.listdir(qa_path)

    # - Create root
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    if not len(files) % num_multiprocessing == 0:
        num_none = num_multiprocessing - len(files) % num_multiprocessing
    else:
        num_none = 0
    files += [None] * num_none
    print('Add {} None in list of file name'.format(num_none))

    for i in range(len(files) // num_multiprocessing):
        pool = Pool(processes=num_multiprocessing)
        print(files[(i * num_multiprocessing):((i + 1) * num_multiprocessing)])
        for f in files[(i * num_multiprocessing):((i + 1) * num_multiprocessing)]:
            pool.apply_async(eval, args=(f, ))
        print('======  apply_async  ======')
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
