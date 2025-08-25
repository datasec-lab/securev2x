"""
Delphi Compact CNN - implements the folded convolution and batch norm layer

Created on October 14 2023

Author: @Yoshi234 
"""
import torch


class CryptenCompactCNN(torch.nn.Module):
    """
    The codes implement the CNN model proposed in the paper "A Compact and Interpretable Convolutional Neural Network for-
    Cross-Subject Driver Drowsiness Detection from Single-Channel EEG ".

    The network is designed to classify 1D drowsy and alert EEG signals for the purposed of driver drowsiness recognition.

    Parameters:

    classes      : number of classes to classify, the default number is 2 corresponding to the 'alert' and 'drowsy' labels.
    Channels     : number of channels output by the first convolutional layer.
    kernelLength : length of convolutional kernel in first layer
    sampleLength : the length of the 1D EEG signal. The default value is 384, which is 3s signal with sampling rate of 128Hz.

    """

    def __init__(self, classes=2, channels=32, kernelLength=64, sampleLength=384):
        super(CryptenCompactCNN, self).__init__()
        self.kernelLength = kernelLength

        self.conv = torch.nn.Conv2d(1, channels, (1, kernelLength))
        self.batch = Batchlayer(channels)
        #384 - 64 + 1 = 321
        # self.GAP = torch.nn.AvgPool2d((1, sampleLength-kernelLength+1))
        self.fc = torch.nn.Linear(channels, classes)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inputdata):
        intermediate = self.conv(inputdata)
        intermediate = self.batch(intermediate)
        intermediate = torch.nn.ReLU()(intermediate)
        # replace average pooling forward functionality with torch.mean implementation
        intermediate = torch.mean(intermediate.view(intermediate.size(0), intermediate.size(1), -1), dim=2)
        intermediate = intermediate.view(intermediate.size()[0], -1)
        intermediate = self.fc(intermediate)
        output = self.softmax(intermediate)

        return output


"""
We use the batch normalization layer implemented by ourselves for this model instead using the one provided by the Pytorch library.
In this implementation, we do not use momentum and initialize the gamma and beta values in the range (-0.1,0.1). 
We have got slightly increased accuracy using our implementation of the batch normalization layer.
"""


def normalizelayer(data, self):
    eps = 1e-05
    if self.training:
        mean = torch.mean(data, [0, 2, 3], True)
        self.running_mean = mean
        variance = torch.sqrt(torch.mean(
            (data - mean)**2, [0, 2, 3], True) + eps)
        self.running_var = variance
    else:
        mean = self.running_mean
        variance = self.running_var
    b = torch.div(data - mean, variance).expand(int(data.size(0)),
                                                int(data.size(1)), int(data.size(2)), int(data.size(3)))
    return b


class Batchlayer(torch.nn.Module):
    def __init__(self, dim):
        super(Batchlayer, self).__init__()
        self.gamma = torch.nn.Parameter(torch.Tensor(1, dim, 1, 1))
        self.beta = torch.nn.Parameter(torch.Tensor(1, dim, 1, 1))
        self.gamma.data.uniform_(-0.1, 0.1)
        self.beta.data.uniform_(-0.1, 0.1)

        # Register running_mean and running_var as buffers
        self.register_buffer('running_mean', torch.Tensor(1, dim, 1, 1))
        self.register_buffer('running_var', torch.Tensor(1, dim, 1, 1))

    def forward(self, input):
        data = normalizelayer(input, self)

        gammamatrix = self.gamma.expand(int(data.size(0)), int(
            data.size(1)), int(data.size(2)), int(data.size(3)))
        betamatrix = self.beta.expand(int(data.size(0)), int(
            data.size(1)), int(data.size(2)), int(data.size(3)))

        return data * gammamatrix + betamatrix