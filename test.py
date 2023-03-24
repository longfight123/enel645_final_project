import torch
import datetime as dt
# check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(dt.datetime.now(), "Checking if we are on a CUDA machine.")
print(device)