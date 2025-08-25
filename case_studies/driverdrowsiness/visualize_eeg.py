import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

def set_graph_j(eeg, num, lbl):
    t = np.arange(len(eeg))/128
    color = 'r' if lbl==1 else 'g'
    plt.figure(figsize=(4,4))
    plt.xlim(t.min(),t.max())
    plt.ylim(-60,50)
    plt.plot(t,eeg,color=color)
    plt.savefig(f"dev_work/visuals/eeg_vis{num}.png")

def create_vis(features:str='dev_work/test_features_labels/9-crypten_features.pth',
               labels:str='dev_work/test_features_labels/9-crypten_labels.pth'):
    features = torch.load(features, map_location='cpu')
    labels = torch.load(labels, map_location='cpu')
    for i in range(1,314,100):
        set_graph_j(features[i].reshape(384),i,labels[i])
    
if __name__ == "__main__":
    create_vis()