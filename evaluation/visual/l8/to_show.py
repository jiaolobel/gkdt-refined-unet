import os
from PIL import Image

task_idx = 9

from eval_config import EvalConfig
config = EvalConfig()
task = config.tasks[task_idx]

inp_root = "tile"
out_root = "jpg"

inp_path = os.path.join(inp_root, task)
out_path = os.path.join(out_root, task)

fullres_names = [
    'LC08_L1TP_113026_20160412_20170326_01_T1_sr_bands_masked', 
    'LC08_L1TP_113026_20160514_20170324_01_T1_sr_bands_masked', 
    'LC08_L1TP_113026_20160530_20170324_01_T1_sr_bands_masked', 
    'LC08_L1TP_113026_20160717_20170323_01_T1_sr_bands_masked', 
]

if not os.path.exists(out_path):
    os.makedirs(out_path)
    print("Create {}.".format(out_path))

for name in fullres_names:
    fullname = os.path.join(inp_path, name, '{}_0.1size.png'.format(name))
    outname = os.path.join(out_path, '{}_0.1size.jpg'.format(name))
    im = Image.open(fullname).convert("RGB")
    im.save(outname)
    print('Write to {}.'.format(outname))
    os.chmod(outname, mode=0o444)
    print("Change mode of {} to read-only".format(outname))

inp_path = os.path.join(inp_root, task, "LC08_L1TP_113026_20160412_20170326_01_T1_sr_bands_masked")
out_path = os.path.join(out_root, task, "LC08_L1TP_113026_20160412_20170326_01_T1_sr_bands_masked")

tile_names = [
    '_rfn_20.jpg', 
    '_rfn_21.jpg',
    '_rfn_37.jpg', 
    '_rfn_52.jpg', 
    '_rfn_87.jpg', 
    '_rfn_90.jpg', 
    '_rfn_91.jpg', 
    '_rfn_172.jpg', 
]

if not os.path.exists(out_path):
    os.makedirs(out_path)
    print("Create {}.".format(out_path))

for name in tile_names:
    fullname = os.path.join(inp_path, name.replace(".jpg", ".png"))
    outname = os.path.join(out_path, name)
    im = Image.open(fullname).convert("RGB")
    im.save(outname)
    print('Write to {}.'.format(outname))
    os.chmod(outname, mode=0o444)
    print("Change mode of {} to read-only".format(outname))