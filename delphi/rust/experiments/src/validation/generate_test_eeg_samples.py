"""
Evaluate compact_cnn model accuracy on the 
generated test eeg_data of subject 9

Must be run from within the main repo folder using the command which follows
python -m delphi.rust.experiments.src.validation.generate_test_eeg_samples no_approx

"""

from os import path
import torch
import scipy.io as sio
import argparse
import numpy as np
import os
import random
import sys
from models.compactcnn import CompactCNN
# from python.python_models.compact_cnn import compact_cnn

# debugging function

torch.cuda.empty_cache()
torch.manual_seed(0) 

def print_torch_model_parameters(model):
    for name, value in model.state_dict().items():
        print(f"{name:20}:{value}")
        print(f"{'size':20}:{value.size()}\n")
        print(100*"=")

#  might want to come back to this - is important for generating the
# "plaintext.npy" file below


def build_model(type):
    """
    Construct model following the given architecture and approx layers
    and export architecture to an onnx pretrained model file
    """
    my_net = None
    if type == "no_approx":
        my_net = CompactCNN()
        my_net.load_state_dict(torch.load(
            "../../../../../case_studies/driverdrowsiness/pretrained/sub9/model.pth"))
    # elif type == "approx":
    #     my_net = compact_cnn_approximation().double().cuda()
    #     my_net.load_state_dict(torch.load(
    #         "/home/ala22014/V2V-Delphi-Applications/python/pretrained_model_weights/pretrained_torch_models/torch_models_with_poly_approx_relu/model_subj_9_seed0.pth"))
    my_net.train(False)
    
    from torch.onnx import OperatorExportTypes
    _OPSET_VERSION = 17

    dummy_input = torch.empty((1, 1, 1, 384))

    kwargs = {
        "do_constant_folding": False,
        "export_params": True,
        "input_names": ["input"],
        "operator_export_type": OperatorExportTypes.ONNX,
        "output_names": ["output"],
        "opset_version": _OPSET_VERSION
    }
    # export model to onnx as well
    torch.onnx.export(my_net, dummy_input, "compactCNN/onnx/compactcnn.onnx", **kwargs)

    return my_net


def generate_eeg_data(num_samples, dataset, eeg_data_path=None, flatten=True):
    """
    Get the 314 eeg test samples into scope
    """
    # load the dataset into scope
    if eeg_data_path == None:
        eeg_data_path = "compactCNN/Eeg_Samples_and_Validation"

    xdata = np.array(dataset['EEGsample'])
    label = np.array(dataset['substate'])
    subIdx = np.array(dataset['subindex'])

    label.astype(int)
    subIdx.astype(int)

    samplenum = label.shape[0]

    channelnum = 30
    subjnum = 11
    samplelength = 3
    selectedchan = [28]
    channelnum = len(selectedchan)
    sf = 128

    ydata = np.zeros(samplenum, dtype=np.longlong)
    for i in range(samplenum):
        ydata[i] = label[i]
    xdata = xdata[:, selectedchan, :]

    test_subj = 9
    testindx = np.where(subIdx == test_subj)[0]

    xtest = xdata[testindx]
    # x_test = images shape (314, 1, 1, 384)
    x_test = xtest.reshape(xtest.shape[0], 1, channelnum, samplelength*sf)
    # y_test = classes
    y_test = ydata[testindx]
    
    if num_samples > xdata.shape[0]: num_samples = xdata.shape[0]

    for i in range(num_samples):
        if flatten: 
            np.save(os.path.join(eeg_data_path,
                    f"eeg_sample_{i}.npy"), x_test[i].flatten().astype(np.float64))
            # print(x_test[i].flatten().astype(np.float64).shape)
        elif not flatten:
            np.save(os.path.join(eeg_data_path,
                    f"eeg_sample_{i}.npy"), x_test[i].astype(np.float64).reshape(1,1,1,384))
            # print(x_test[i].astype(np.float64).reshape(1,1,1,384).shape)
            
    if flatten:
        # print(y_test.flatten().astype(np.int64).shape)
        np.save(path.join(eeg_data_path, f"classes.npy"),
                y_test.flatten().astype(np.int64))
    else: # don't flatten numpy arrays
        # print(y_test.astype(np.int64).shape)
        np.save(path.join(eeg_data_path, f"classes.npy"),
                y_test.astype(np.int64))


def test_network(model, eeg_data_path=None):
    """Get inference results from the given network

    Arguments: 
    - Model --- pass the model object generate by `build_model` function
    to the function 
    - eeg_data_path --- pass the path to which the files should be saved after
    running this function

    Returns: 
    - None"""

    # print_torch_model_parameters(model)
    if eeg_data_path == None:
        eeg_data_path = "/home/jjl20011/snap/snapd-desktop-integration/current/Lab/Projects/Project1-V2X-Secure2PC/v2x-delphi-2pc/delphi/rust/experiments/src/validation/Eeg_Samples_and_Validation"
    # load image classes
    classes = np.load(path.join(eeg_data_path, "classes.npy"))
    correct = []

    with torch.no_grad():
        for i in range(len(classes)):
            eeg_sample = np.load(
                path.join(eeg_data_path, f"eeg_sample_{i}.npy")).reshape(1, 1, 1, 384)
            temp_test = torch.DoubleTensor(eeg_sample).cuda()
            answer = model(temp_test)
            probs = answer.cpu().numpy()
            preds = probs.argmax(axis=-1)
            correct += [1] if preds == classes[i] else [0]
    np.save(path.join(eeg_data_path, "plaintext.npy"), np.array(correct))
    return 100 * (sum(correct) / len(classes))


if __name__ == "__main__":
    # load dataset -- figure out which location to put the files in later
    # eeg_data_path = "/home/ala22014/V2V-Delphi-Applications/delphi/rust/experiments/src/validation/eeg_test_samples_subject9"
    dataset = sio.loadmat(
        "../../../../../case_studies/driverdrowsiness/data/dataset.mat")

    if len(sys.argv) < 2:
        print("Usage: {sys.agrv[0]} model_type")
        print("you can choose model_type=approx or no_approx")
        exit()

    type = sys.argv[1]

    # Build model
    model = build_model(type=type)

    # pass the loaded dataset as an argument to the data generation function
    generate_eeg_data(num_samples=314, dataset=dataset, eeg_data_path="compactCNN/cryptflow_eeg_samples", flatten=False) 
    # print(f"Accuracy: {test_network(model)}") 
