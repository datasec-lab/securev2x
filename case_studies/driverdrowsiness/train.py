# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:17:29 2019
@author: JIAN

Adapted on Sep 23 by @ALARST13
"""
import torch
import scipy.io as sio
import numpy as np
from sklearn.metrics import accuracy_score
import torch.optim as optim
from models.compactcnn import CompactCNN
import os

torch.cuda.empty_cache()
torch.manual_seed(0)

"""
 This file performs leave-one-subject cross-subject classification on the driver drowsiness dataset.
 The data file contains 3 variables and they are EEGsample, substate, and subindex.
 "EEGsample" contains 2022 EEG samples of size 20x384 from 11 subjects. 
 Each sample is a 3s EEG data with 128Hz from 30 EEG channels.

 The names and their corresponding index are shown below:
 Fp1, Fp2, F7, F3, Fz, F4, F8, FT7, FC3, FCZ, FC4, FT8, T3, C3, Cz, C4, T4, TP7, CP3, CPz, CP4, TP8, T5, P3, PZ, P4, T6, O1, Oz  O2
 0,    1,  2,  3,  4,  5,  6,  7,   8,   9,   10,   11, 12, 13, 14, 15, 16, 17,  18,  19,  20,  21,  22,  23,24, 25, 26, 27, 28, 29

 Only the channel Oz is used.

 "subindex" is an array of 2022x1. It contains the subject indexes from 1-11 corresponding to each EEG sample. 
 "substate" is an array of 2022x1. It contains the labels of the samples. 0 corresponds to the alert state and 1 correspond to the drowsy state.
 
  This file prints leave-one-out accuracies for each subject and the overall accuracy.
  The overall accuracy for one run is expected to be 0.7364. However, the results will be slightly different for different computers.
  
  If you have met any problems, you can contact Dr. Cui Jian at cuij0006@ntu.edu.sg
"""


def run(
    save_torch_form = False
):

    # load data from the file
    filename = r'data/dataset.mat'

    tmp = sio.loadmat(filename)
    xdata = np.array(tmp['EEGsample'])
    label = np.array(tmp['substate'])
    subIdx = np.array(tmp['subindex'])

    label.astype(int)
    subIdx.astype(int)

    samplenum = label.shape[0]

    # there are 11 subjects in the dataset. Each sample is 3-seconds data from 30 channels with a sampling rate of 128Hz.
    channelnum = 30
    subjnum = 11
    samplelength = 3
    sf = 128

    # define the learning rate, batch size and epochs
    lr = 1e-2
    batch_size = 50
    n_epoch = 6

    # ydata contains the label of samples
    ydata = np.zeros(samplenum, dtype=np.longlong)

    for i in range(samplenum):
        ydata[i] = label[i]

    # only channel 28 is used, which corresponds to the Oz channel
    selectedchan = [28]

    # update the xdata and channel number
    xdata = xdata[:, selectedchan, :]
    channelnum = len(selectedchan)

    # the result stores accuracies of every subject
    results = np.zeros(subjnum)

    # it performs leave-one-subject-out training and classification
    # for each iteration, the subject i is the testing subject while all the other subjects are the training subjects.
    for i in range(1, subjnum+1):

        # form the training data
        trainindx = np.where(subIdx != i)[0]
        xtrain = xdata[trainindx]
        x_train = xtrain.reshape(
            xtrain.shape[0], 1, channelnum, samplelength*sf)
        y_train = ydata[trainindx]

        # form the testing data
        testindx = np.where(subIdx == i)[0]
        xtest = xdata[testindx]
            
        x_test = xtest.reshape(xtest.shape[0], 1, channelnum, samplelength*sf)
        y_test = ydata[testindx]
        
        if save_torch_form:
            xt_torch = torch.from_numpy(xtest)
            yt_torch = torch.from_numpy(y_test)
            f_x = "{}-drowsy_features.pth".format(i)
            f_y = "{}-drowsy_labels.pth".format(i)
            if f_x in 'dev_work/test_features_labels':
                continue
            else:
                torch.save(xt_torch, "dev_work/test_features_labels/{}".format(f_x))
                torch.save(yt_torch, "dev_work/test_features_labels/{}".format(f_y))
            continue

        train = torch.utils.data.TensorDataset(
            torch.from_numpy(x_train), torch.from_numpy(y_train))
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=True)

        # load the CNN model to deal with 1D EEG signals
        my_net = CompactCNN().double().cuda()

        optimizer = optim.Adam(my_net.parameters(), lr=lr)
        loss_class = torch.nn.NLLLoss().cuda()

        for p in my_net.parameters():
            p.requires_grad = True

        # train the classifier
        for epoch in range(n_epoch):
            for j, data in enumerate(train_loader, 0):
                inputs, labels = data

                input_data = inputs.cuda()
                class_label = labels.cuda()

                my_net.zero_grad()
                my_net.train()

                class_output = my_net(input_data)
                err_s_label = loss_class(class_output, class_label)
                err = err_s_label

                err.backward()
                optimizer.step()

        # test the results
        my_net.train(False)
        with torch.no_grad():
            x_test = torch.DoubleTensor(x_test).cuda()
            answer = my_net(x_test)
            probs = answer.cpu().numpy()
            preds = probs.argmax(axis=-1)
            acc = accuracy_score(y_test, preds)

            print("Subject " + f"{i}: " + str(acc))
            results[i-1] = acc
            
            base_folder = "pretrained"
            os.makedirs(base_folder, exist_ok=True)
            folder_path = "pretrained/" + "sub" + f"{i}/"
            os.makedirs(folder_path, exist_ok=True)
            torch.save(my_net.state_dict(), folder_path + "model.pth")

    print('mean accuracy:', np.mean(results))


if __name__ == '__main__':
    run(save_torch_form=True)
