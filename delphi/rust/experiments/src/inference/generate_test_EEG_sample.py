"""
Preliminary Evaluation of CompactCNN performance (rust model) in Delphi
"""

import argparse
import numpy as np
import os
import random
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from os import path
import scipy.io as sio

SUBJ_NUM = 9
SELECTED_CHANNEL = [28]
#just use the absolute data path to get the data file
ABSOLUTE_PATH = "/home/jjl20011/snap/snapd-desktop-integration/current/Lab/V2V-Delphi-Applications/python/data/dataset.mat"

def generate_eeg_data(data_path):
    """Sample and save a random set of (num_samples) eeg samples"""
    raw_data = sio.loadmat(ABSOLUTE_PATH)

    # x_data contains the actual EEGsample data (all 30 channels by default)
    x_data = np.array(raw_data['EEGsample'])
    # labels is an array (2022, 1) containing the corresponding labels (drowsy or awake)
    # for all 2022 samples in the dataset
    labels = np.array(raw_data['substate'])
    labels.astype(int)
    # subidx is an array (2022, 1) containing the corresponding subjnum of each
    # data sample
    subidx = np.array(raw_data['subindex'])
    subidx.astype(int)

    # set the indices of the test samples
    testindx = np.where(subidx == SUBJ_NUM)[0]

    # re-write x_data to only contain the 28th channel since this is the 
    # only data used for the model
    x_data = x_data[:,SELECTED_CHANNEL,:]
    # filter x_data to only contain items which have the subindex of 9 (for the
    # 9th subject)
    x_test = x_data[testindx]

    # initialize the results (actual values - labels)
    sample_num = labels.shape[0]
    y_data = np.zeros(sample_num, dtype=np.longlong)
    for i in range(sample_num):
        y_data[i] = labels[i]
    y_test = y_data[testindx]

    # randomly select a data sample
    sample_test_idx = 284 # random.choice(range(len(x_test)))
    eeg_data = x_test[sample_test_idx]
    classification = y_test[sample_test_idx]

    # test statements to ensure that the types are correct
    # print(classification.dtype)
    # assert(isinstance(classification, np.ndarray))
    # assert(isinstance(eeg_data, np.ndarray))

    with open("file_labels.txt", 'w') as f:
        f.write(f"test index is {sample_test_idx}\n")

    np.save(os.path.join(data_path, f"eeg_data.npy"), eeg_data.flatten().astype(np.float64))
    np.save(path.join(data_path, f"eeg_class.npy"), classification.flatten().astype(np.int64))

if __name__ == "__main__": 
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--image_path', required=False, type=str,
    #                     help='Path to place images (default cwd)')
    # args = parser.parse_args()

    data_path = "/home/jjl20011/snap/snapd-desktop-integration/current/Lab/V2V-Delphi-Applications/delphi/rust/experiments/src/inference"
    generate_eeg_data(data_path)







