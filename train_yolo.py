import argparse
import torch
import sys
import os

yolov9_dir = os.path.dirname(os.path.abspath(__file__)) + "/yolov9"
sys.path.append(yolov9_dir)

from yolov9.train_dual import train  
from yolov9.utils.general import increment_path
from yolov9.utils.callbacks import Callbacks
from pathlib import Path
# Manually create the options namespace (simulating command-line arguments)
opt = argparse.Namespace()
opt.name = 'fish_1'
opt.project = "yolov9-hpo"  
opt.save_dir = str(increment_path(Path(opt.project) / opt.name))       # Where to save results
opt.epochs = 1                                 # Total training epochs
opt.batch_size = 4                               # Batch size
opt.weights = "./yolov9/weights/yolov9-c-converted.pt"    # Initial weights path
opt.cfg = "./yolov9/models/detect/yolov9-c-fish-od.yaml"  # Model configuration file
opt.hyp = "./yolov9/data/hyps/hyp.scratch-high.yaml"      # Hyperparameters file
opt.data = "./yolov9/data/dataset.yaml"                 # Dataset configuration file
opt.noval = False
opt.nosave = False
opt.workers = 8
opt.freeze = [0]
opt.rect = False
opt.image_weights = False
opt.single_cls = False
opt.evolve = False
opt.noautoanchor = False
opt.noplots = False
opt.cos_lr = False
opt.flat_cos_lr = False
opt.fixed_lr = False
opt.label_smoothing = 0.0
opt.patience = 100
opt.save_period = -1
opt.seed = 0
opt.resume = False
opt.imgsz = 640
opt.optimizer = 'SGD'
opt.sync_bn=False
opt.cache = None
opt.close_mosaic = 15
opt.quad = False
opt.min_items = 0
opt.multi_scale = False
# Set up the device (for example, use CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optionally, instantiate your callbacks if needed:
# from utils.callbacks import Callbacks
# callbacks = Callbacks()
callbacks = Callbacks()  # or create your Callbacks object

# Now call the train function directly.
results = train(opt.hyp, opt, device, callbacks)

print(results) # [P , R , mAP50, mAP50-95]