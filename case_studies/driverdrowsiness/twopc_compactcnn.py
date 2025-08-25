# utils
import logging
import os
import time
import pickle

# libs
from .models.crypten_compactcnn import CryptenCompactCNN
import crypten
import crypten.communicator as comm
import torch
from crypten.config import cfg
cfg.communicator.verbose = True

ALICE=0
BOB=1

def run_2pc_compactcnn(args:dict):
    '''
    Load the pytorch architecture and CompactCNN's pretrained weights
    to run secure inference
    '''
    # drowsy_mod = CryptenCompactCNN() # set arch
    # drowsy_mod.load_state_dict(torch.load(args['drowsy_mod'], map_location='cpu')) 
    
    crypten.init()
    dummy_input = torch.empty(args['batch_size'],1,*args['img_size'])

    # does moving the model to the gpu after encrypting mess it up?
    sec_model = crypten.nn.from_pytorch(args['model'], dummy_input).encrypt(src=ALICE).to(args['device'])
    
    # print("[DEBUG-twopc_yolo.py - 24]: type of sec_model = {}".format(type(sec_model)))
    sec_model.eval()
    
    # encrypt and reshape the data tensor into the proper format
    data_enc = crypten.load_from_party(args['data_path'], src=BOB)
    data_enc = data_enc.reshape(args['batch_size'],1,*args['img_size']).to(args['device'])
    
    # print("[DEBUG-{}]: Running on GPU? = {}".format(comm.get().get_rank(), data_enc.is_cuda))
    start = time.time()
    pred = sec_model(args['device'], data_enc) #device value is required positional argument for my code
    end = time.time()
    
    # decrypt output for the data holder, but not the server
    rank = comm.get().get_rank()
    
    pkl_str = "experiments/{}/run_{}.pkl".format(args['folder'], args['run_label'])
    pred_dec = pred.get_plain_text(dst=BOB) # only BOB should get the output set "dst=BOB"
    
    if rank == BOB:
        if args['debug']: 
            print("[DEBUG-twopc-compactcnn]: shape of preds = {}".format(pred_dec.shape))
        with open(pkl_str, 'wb') as pkl_file:
            pickle.dump([pred_dec, start, end], pkl_file)
            
    com_stats = comm.get().get_communication_stats() # get and write comm stats for current run
    with open("experiments/{}/comm_tmp_{}.pkl".format(args['folder'], rank), 'wb') as pkl_file:
        pickle.dump(com_stats, pkl_file)