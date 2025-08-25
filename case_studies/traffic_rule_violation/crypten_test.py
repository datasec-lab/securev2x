'''
Script for testing functionality of custom CrypTen modules
'''

import crypten
import torch

# TODO:
# define basic torch models that do the same thing, and then 
# convert them automatically to CrypTensor modules 
# and test that way if possible

class SplitModel(crypten.nn.Module):
    def __init__(self):
        super(SplitModel, self).__init__()
        self.split = crypten.nn.Split(axis=-1)
        
    def forward(self, x):
        x, y = self.split([x, torch.tensor([2,2])])
        return x, y

class ResizeModel(crypten.nn.Module):
    def __init__(self):
        super(ResizeModel, self).__init__()
        self.upsample = crypten.nn.Resize()
    
    def forward(self, x):
        x = self.upsample([x,None,None,(10, 20)])
        return x
    
class ResizeModel_broken(crypten.nn.Module):
    def __init__(self):
        super(ResizeModel_broken, self).__init__()
        self.upsample = crypten.nn.Resize()
    
    def forward(self, x):
        x = self.upsample([x,'','',(10,20)])
        return x
    
def test_resize():
    ALICE = 0
    BOB = 1
    crypten.init()
    
    test_mod = ResizeModel_broken()
    test_mod.encrypt(src=ALICE)
    x = torch.rand((1,1,4,5))
    x = crypten.cryptensor(x)
    print("="*5,'Before Resize',"="*5)
    print(x)
    x = test_mod(x)
    print("="*5,'After Resize',"="*5)
    print(x)
    
def test_split():
    ALICE = 0
    BOB = 1
    crypten.init()

    test_mod = SplitModel()
    test_mod.encrypt(src=ALICE)
    x = torch.rand((1,1,5,4))
    x = crypten.cryptensor(x)
    print("="*5,'Before Split',"="*5)
    print(x)
    x,y = test_mod(x)
    print("="*5,'After Split',"="*5)
    print(x,y)
    
def main():
    test_split()
    
if __name__ == "__main__":
    main()