from ptflops import get_model_complexity_info
import torch
import torch.nn as nn

from main import ConvNet   # or ConvNet or ResNet

model = ConvNet()

with torch.cuda.device(0):
    macs, params = get_model_complexity_info(
        model, 
        (3, 32, 32),  # CIFAR-10 input size
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True,
    )

print("MACs:", macs)
print("Params:", params)
print(f"FLOPs: {2*macs}")
