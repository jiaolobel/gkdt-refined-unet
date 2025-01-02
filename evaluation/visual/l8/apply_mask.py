import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# - Hyperparameter
# task = 'rgb/a=80.0, b=0.03125, r=3.0'
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

task_idx = 9 # 0 ~ 9

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

task = tasks[task_idx]

input_path = "E:/Research/experiment_results/gkdt_rfn_unet/l8/{}".format(task)
output_path = "{}".format(task)
image_path = "E:/Research/experiment_data/l8/false/"

# create output folder
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print("Create {}.".format(output_path))

""" Apply the given mask to the image. """

def apply_mask(image, mask, mask_value, color, alpha=.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == mask_value, image[:, :, c] * (1 - alpha) + alpha * color[c], image[:, :, c])

    return image


for f in os.listdir(input_path):
    if os.path.splitext(f)[-1] == '.npz':
        print("Load {}.".format(f))
        
        mask_name = os.path.join(input_path, f)
        image_name = os.path.join(image_path, f.replace('rfn.npz', 'sr_bands.png'))
        save_name = os.path.join(output_path, f.replace('rfn.npz', 'sr_bands_masked.png'))

        image = np.array(Image.open(image_name), dtype=np.float32)
        mask = np.load(mask_name)['arr_0']

        cloud, cloud_color = 3, [115, 223, 255]
        shadow, shadow_color = 2, [38, 115, 0]

        mask_image = apply_mask(image, mask, cloud, cloud_color)
        mask_image = apply_mask(mask_image, mask, shadow, shadow_color)

        mask_image = np.uint8(mask_image)

        plt.imsave(save_name, mask_image)
        print("Write to {}.".format(save_name))
        os.chmod(save_name, mode=0o444)
        print("Change mode of {} to read-only".format(save_name))