"""
Created on Thursday Oct 12 20:51:33 2023

@author: Yoshi234
"""

"""
Instructions:
Please run from the root in package mode `python3 -m`
"""

import torch
import numpy as np
from models.compactcnn import CompactCNN

def load_model(model_path):
    my_net = CompactCNN().double().cuda()
    my_net.load_state_dict(torch.load(model_path))
    my_net_dict = my_net.state_dict().items()
    return my_net_dict

def main():
    subjnum = 9
    results_file = "dev_tools/model_params/sub{}_params.txt".format(subjnum)
    model_path = "pretrained/sub{}/model.pth".format(subjnum)
    model_dict = load_model(model_path)
    # save the parameters to text file for viewing
    with open(results_file, "w") as f:
        for name, value in model_dict:
            f.write(f"{name:20}:{value}\n")
            f.write(f"{'size':20}:{value.size()}\n\n")
            f.write(100*"="+"\n")

if __name__ == "__main__":
    main()