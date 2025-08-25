# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 17:41:44 2023

@author: ALARST13

Adapted on Oct 12 by @Yoshi234
"""

import torch
import numpy as np
from models.compactcnn import CompactCNN

def delphi_extract_weights(model_dict, save_path):
    interest_params = {"conv.weight":None, "conv.bias":None, 
                       "batch.gamma":None, "batch.beta":None, 
                       "batch.running_mean":None, "batch.running_var":None}
    params = []
    for name, param in model_dict:
        if name in interest_params:
            interest_params[name] = param

    # get necessary network parameters
    # variance is trained + saved as the square root value - no need to use sqrt op
    bias = interest_params["conv.bias"]
    beta = interest_params["batch.beta"]
    gamma2 = interest_params["batch.gamma"]
    gamma1 = interest_params["batch.gamma"]
    conv = interest_params["conv.weight"]
    mean = interest_params["batch.running_mean"]
    var1 = interest_params["batch.running_var"]
    var2 = interest_params["batch.running_var"]

    # transform necessary parameters
    gamma1 = interest_params["batch.gamma"].view([32,1,1,1])
    # gamma1 = gamma1.expand(int(conv.size(0)), int(conv.size(1)), int(conv.size(2)), int(conv.size(3)))
    gamma2 = gamma2.view([32])
    beta = beta.view([32])
    mean = mean.view([32])
    var1 = var1.view([32,1,1,1])
    var1 = var1.expand(int(conv.size(0)), int(conv.size(1)), int(conv.size(2)), int(conv.size(3)))
    var2 = var2.view([32])

    # perform parameter value transformations
    # conv_fold = conv * (torch.div(gamma1, var1)).view(-1,1,1,1)# torch.div(gamma1 * conv, var1) 
    # bias_fold = (bias - mean) * torch.div(gamma1, var1) + beta# gamma2 * torch.div((bias - mean), var2) + beta
    conv_fold = torch.div(gamma1 * conv, var1)
    bias_fold = gamma2 * torch.div((bias - mean), var2) + beta

    # generate new parameters
    params.append(conv_fold.view(-1))
    params.append(bias_fold.view(-1))
    for name, param in model_dict:
        if name not in interest_params:
            params.append(param.view(-1))
    params = torch.cat(params)
    model_weights = params.cpu().detach().numpy()
    np.save(save_path, model_weights.astype(np.float64))


def extract_weights(model_dict, save_path):
    params = []
    for name, param in model_dict:
        print(name)
        params.append(param.view(-1))
    # Concatenate the network parameters
    params = torch.cat(params)
    # Convert the tensor to a NumPy array
    model_weights = params.cpu().detach().numpy()
    np.save(save_path, model_weights.astype(np.float64))


def main():
    subjnum = 9
    model_path = "pretrained/sub{}/model.pth".format(subjnum)
    save_path = "pretrained/sub{}/model.npy".format(subjnum)

    # Load pretrained model
    my_net = CompactCNN().double().cuda()
    my_net.load_state_dict(torch.load(model_path))
    model_dict = my_net.state_dict().items()

    # extract weights in delphi format
    delphi_extract_weights(model_dict, save_path)
    

if __name__ == '__main__':
    main()
