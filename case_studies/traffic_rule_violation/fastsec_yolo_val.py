'''
Validation of YOLOv5 models on the COCO validation data set
Follows from the yolov5/val.py validation script. See 
https://github.com/ultralytics/yolov5/blob/master/val.py 
Inference time implementation of Secure-RLR utilizes CrypTen
running on a single GPU. Custom implementation of modules is 
required in order to use CrypTen primitives to privately 
compute the YOLOv5 architecture
'''

# handle module and non-module import strategies
try:
    from .utils.general import (
        cv2, non_max_suppression, scale_boxes, check_dataset, xywh2xyxy
    )
    from .utils.augmentations import letterbox
    from .utils.metrics import box_iou, ap_per_class
    from tqdm import tqdm
    # scripts
    from .fastsec_yolo_detect import multiproc_gpu, _run_sec_model
except ImportError: 
    from utils.general import (
        cv2, non_max_suppression, scale_boxes, check_dataset, xywh2xyxy
    )
    from utils.augmentations import letterbox
    from utils.metrics import box_iou, ap_per_class
    from tqdm import tqdm
    from fastsec_yolo_detect import multiproc_gpu, _run_sec_model
    
# utils
import threading
import pandas as pd
import os
# import argparse
# from utils.torch_utils import select_device
# module import 
import time
import pickle
import multiprocessing as mp
# from ultralytics.utils.plotting import Annotator, colors, save_one_box
import logging
import warnings

# libs
import onnx
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm
from crypten.config import cfg
import torch
# from models.common import DetectMultiBackend, AutoShape
import numpy as np
# from examples.multiprocess_launcher import MultiProcessLauncher
import torchvision

# NOT NECESSARY - DELETE AFTER VERIFYING THE FUNCTIONALITY OF VALIDATION
# class img_info:
#     def __init__(self, data, batch_idx, sub_idx, file, label, og_size, new_size):
#         self.dat = data             # stores idx in detection list
#         self.file = file            # stores img file (str)
#         self.label = label          # stores label file (str)
#         self.batch_idx = batch_idx  # stores batch index (the index of the batch which )
#         self.sub_idx = sub_idx      # stores index of element in the given batch
#         self.og_size = og_size
#         self.new_size = new_size

# globals
SUPPORTED_IMG_TYPES = {"png", "jpg", "jpeg"}
warnings.filterwarnings("ignore")
select_classes = {1,2,3,5,7}

def read_label_file(file_path, rearrange=False, device='cpu'):
    '''
    reads a coco .txt file and returns a tensor of bounding boxes/class labels
    corresponding to the output
    '''
    if file_path.split(".")[-1] != "txt": 
        raise ValueError("[ERROR-crypten_val 66]: only .txt label files can be read")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # print("[INFO]: initial lines - {}".format(lines))
        lines = [line.split(" ") for line in lines]
        for i in range(len(lines)):
            for j in range(len(lines[i])):
                lines[i][j] = float(lines[i][j])
        lines = [torch.tensor(line) for line in lines]
        lines = torch.stack(lines).to(device=device) # send the label tensor to the correct device
    
    for i in range(len(lines)):
        lines[i][1:] = xywh2xyxy(lines[i][1:]) # this tensor can be of whatever shape you want
    
    return lines[:,[1,2,3,4,0]] if rearrange else lines # rearrange the output columns if necessary

def read_img_file(file_path, img_size=None, stride=32, device='cpu'):
    '''
    reads in an image file, and returns the corresponding torch tensor
    and shape values for the original image and numpy image
    '''
    im0 = cv2.imread(file_path)
    # print("[DEBUG-82 - read_img_file] im0.shape = {}".format(im0.shape))
        
    if img_size is None:
        w = im0.shape[1]
        h = im0.shape[0]
        
        if w % stride  == 0: w = w # format width
        elif w % stride != 0: w = (w//stride) * stride
        
        if h % stride == 0: h = h # format height
        elif h % stride != 0: h = (h//stride) * stride
        
        img_size = (h,w)
    
    im = letterbox(im0, img_size, stride=stride, auto=False, scaleFill=False)[0]
    im = im.transpose((2,0,1))[::-1] # convert BGR -> RGB format for video frames
    im = np.ascontiguousarray(im)
    im_tensor = torch.from_numpy(im)/255
    # print("[DEBUG-90 - read_img_file] im0.shape = {}".format(im_tensor.shape))
    
    return im_tensor.to(device), tuple(im0.shape[-3:-1]), img_size # return the tensor to load and process

def load_and_process(img_folder, lbs_folder, load_size=None, start=0):
    '''
    loads and processes data from the specified dataset, creating 
    a single pytorch file, or set of files to hold a batched 
    set of numpy image arrays
    
    - load_size and start specify index values for loading a specific set
      of images from the folder structure so that not all images are loaded
      into RAM at the exact same time. This is important for much larger 
      batches of data. 
    '''
    if load_size is None: 
        load_size=len(os.listdir(lbs_folder))

    lbl_dir = os.listdir(lbs_folder)
    img_dir = os.listdir(img_folder)
    
    labels = [] # list of tensors 
    for i in range(start,len(lbl_dir)): 
        labels.append(read_label_file("{}/{}".format(lbs_folder, lbl_dir[i])))
        
    imgs = [] # list - tensor of images, separate into batches
    for j in range(start,len(img_dir)):
        imgs.append(read_img_file("{}/{}".format(img_folder, img_dir[j])))
        
    imgs = torch.stack(imgs) # convert tensor list into a single tensor    
    
    return imgs, labels # labels correspond index wise to the tensors
    
def sec_pred():
    # TODO: implement secure prediction (follow crypten detect)
    pass

def faster_rcnn_val(
    exp_folder='batched_val', 
    batch_size=32, 
    conf_thres=0.25, 
    iou_thres=0.45, 
    lbs_folder="/mnt/nvme1n1p1/data/coco128/coco128/labels/train2017",
    imgs_folder="/mnt/nvme1n1p1/data/coco128/coco128/images/train2017",
    plain=True, 
    results_name='plaintext_val', 
    img_size=None,
    num_batches=None, 
    vehicle_only=True, 
    device='cpu',
    classes=None,
    max_det=1000, 
    agnostic_nms=None
):
    yolo = torch.hub.load("ultralytics/yolov5", "yolov5n", force_reload=True, trust_repo=True)
    mod_names = yolo.names
    
    # initialize the model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights='DEFAULT', 
    ).to(device).eval() ## set to eval mode
    
    lbs_files = os.listdir(lbs_folder) # set file names
    imgs_files = os.listdir(imgs_folder)
    
    labels = [] # list of label tensors
    
    im_batches = [] # set empty list of image batches to run
    cur_batch = [] # reset every time you are finished filling a batch
    
    # compute prediction and stats for each image
    imgs_info = dict() # init empty img info dictionary
    
    i=0 # load images into scope
    for im in imgs_files:
        im_name = im.split(".")[0]
        lb = None
        for lb_name in lbs_files: # search for matching label
            if im_name in lb_name: 
                lb = lb_name
                break
        if lb is None: continue # go to next image if no corr. label
        
        label = read_label_file("{}/{}".format(lbs_folder,lb))
        veh = False
        for cls in label.T[0].tolist(): # iterate over all classes in bounding box list
            if cls in select_classes: veh=True
        
        if not veh: continue # got to next image if no vehicles in image
        
        # read and append label to list
        labels.append(label) 
        im_tensor, og_size, new_size = read_img_file(f"{imgs_folder}/{im}", 
                                                     img_size=img_size, 
                                                     stride=32) ## stride is always 32 px for FasterRCNN
        
        # this method doesn't preserve order, we need some kind of other identifier   
        if len(cur_batch) < batch_size:             
            cur_batch.append(im_tensor)
            # if debug: print("[INFO]: cur batch length is = {}".format(len(cur_batch)))
        else:
            im_batches.append(cur_batch)
            cur_batch = [] # reset current batch list
            cur_batch.append(im_tensor)
            
        imgs_info[i] = (og_size, new_size) # tuple of size tuples
        i+=1 # increment counter
      
    if len(cur_batch) < batch_size:       
        im_batches.append(cur_batch) # always append last batch (if not already appended)
      
    detections = [] # list of prediction tensors

    # run inference for each batch
    run_times = np.zeros(len(im_batches))
    com_costs = np.zeros(len(im_batches))
    round_nums = np.zeros(len(im_batches))
    
    # pass batches (lists of tensors) as input to the model  
    print("[DEBUG]: len(im_batches) = {}".format(len(im_batches)))
    n_batches = len(im_batches) if num_batches is None else num_batches
    if n_batches > len(im_batches): n_batches = len(im_batches)
    for batch_idx in range(n_batches):
        if plain: 
            start = time.time()
            with torch.no_grad():
                preds = model(im_batches[batch_idx])
            end = time.time()
        
        # format predictions output correctly
        # NMS is applied automatically
        
        for pred in preds:
            boxes = pred['boxes'] 
            lbls = pred['labels'] - 1
            confs = pred['scores']
            
            pred = torch.column_stack((boxes, confs, lbls))
            detections.append(pred)
            
        # for idx in range(len(detections)):
        #     print("pred = \n{}".format(detections[idx]))
        #     print("label = \n{}".format(labels[idx]))
            
                    
        run_times[batch_idx] = end - start
        # preds = non_max_suppression(
        #     prediction=preds, 
        #     conf_thres=conf_thres,
        #     iou_thres=iou_thres,
        #     classes=classes,
        #     agnostic=agnostic_nms,
        #     max_det=max_det
        # )
    
    metrics = []
    vals = []
    stats = []
    
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    
    inf_count=0
    for i in range(n_batches): inf_count += len(im_batches[i])
    assert len(detections) == inf_count, "[ERROR-264]: labels / detections don't match -> labels={}, detections={}".format(inf_count, len(detections))
    
    # for det in detections: print("[DEBUG]: det.shape = {}".format(det.shape)) 
    
    # print("[INFO]: labels = \n{}".format(labels))
    for i in range(len(detections)): # format detection result metrics
        lbl = labels[i]
    
        det = detections[i]
        og_size, new_size = imgs_info[i]
        
        correct = np.zeros((det.shape[0], iouv.shape[0])).astype(bool)
        
        # print("[DEBUG]: lbl.shape = {}".format(lbl.shape))

        correct_class = lbl[:,0:1] == det[:,5]
        
        det[:,:4] = scale_boxes(new_size, det[:,:4], og_size, device=device)
        lbl[:,1:] *= torch.tensor((og_size[1], og_size[0], og_size[1], og_size[0]), device=device)
        
        iou_score = box_iou(lbl[:,1:], det[:,:4])
        
        for j in range(len(iouv)):
            x = torch.where((iou_score > iouv[j]) & correct_class)
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x,1), iou_score[x[0], x[1]][:,None]), 1).cpu().numpy()
                if x[0].shape[0] > 1:
                    matches = matches[matches[:,2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:,1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:,0], return_index=True)[1]]
                correct[matches[:,1].astype(int), j] = True
        correct = torch.tensor(correct, dtype=torch.bool, device=iouv.device)
        
        # append results to stats list tuples=(correct, confidence (objectness score), pred class, true class)
        stats.append((correct, det[:,4], det[:,5], lbl[:,0]))
    
    # format the stats list and save results
    stats = [torch.cat(x,0).cpu().numpy() for x in zip(*stats)]
    
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir="experiments/batched_val", names=mod_names)
    ap50, ap = ap[:,0], ap.mean(1)
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    
    ap_metrics = pd.DataFrame({
        "veh ap class": ap_class,
        "ap": ap, 
        "ap50": ap50,
        "true positive": tp, 
        "false positive": fp, 
        "precision": p, 
        "recall": r, 
        "f1 score": f1, 
        "precision": p, 
        "recall": r
    })
    
    if vehicle_only: 
        ap_metrics = ap_metrics.loc[(ap_metrics['veh ap class'] == 1) | 
                                    (ap_metrics['veh ap class'] == 2) | 
                                    (ap_metrics['veh ap class'] == 3) | 
                                    (ap_metrics['veh ap class'] == 5) | 
                                    (ap_metrics['veh ap class'] == 7)]
    
    overall_metrics = {
        "mp": mp, 
        "mr": mr, 
        "map50": map50, 
        "veh_map": ap_metrics['ap'].mean(),
        "(avg) run time (s)": run_times.sum()/len(detections), # average over all detections
        "num threads": threading.active_count(),
        "communication": com_costs.sum()/len(detections), 
        "rounds": round_nums.sum()/len(detections)
    }
    
    for metr in overall_metrics:
        metrics.append(metr)
        vals.append(overall_metrics[metr])
    results = pd.DataFrame({'metric':metrics, 'score':vals})
    
    results.to_csv("experiments/{}/{}_{}-{}_{}_batch_{}_overall_metrics.csv".format(exp_folder,
                                                                         results_name, 
                                                                         img_size[0],
                                                                         img_size[1],
                                                                         'faster-rcnn',
                                                                         batch_size))
    ap_metrics.to_csv("experiments/{}/{}_{}-{}_{}_batch_{}_class_metrics.csv".format(exp_folder,
                                                                                results_name, 
                                                                                img_size[0],
                                                                                img_size[1],
                                                                                'faster-rcnn',
                                                                                batch_size))
        
def run_val(
    exp_folder='batched_val',
    batch_size=32,
    conf_thres=0.25, 
    iou_thres=0.45, 
    max_det=1000, 
    classes=None,
    agnostic_nms=None,
    lbs_folder="/mnt/nvme1n1p1/data/coco128/coco128/labels/train2017",
    imgs_folder="/mnt/nvme1n1p1/data/coco128/coco128/images/train2017",
    model_name='yolov5s',
    device='cpu',
    plain=True,
    results_name="plaintext_val", 
    img_size=None,
    num_batches=None,  # source folder to hold .pth versions of img batches
    debug=False,       # prints the prediction tensor outputs for debug purposes
    vehicle_only=True, # only record per-class ap metrics for vehicle classes
    get_stats=False,   # returns the stats object which is used to compute AP values if desired
    print_net=False
):
    '''
    Takes as input the yolov5 model of choice, and runs validation on the 
    COCO128 dataset for specified batch sizes. 
    '''
    if debug: 
        print(f"[INFO]: environ --- {os.getenv('CUDA_VISIBLE_DEVICES')}")
            
    # load model
    model = torch.hub.load('ultralytics/yolov5', model_name, force_reload=True, trust_repo=True)
    mod_names, mod_stride = model.names, model.stride
    model = model.to(device) # move model to correct device
    
    lbs_files = os.listdir(lbs_folder) # set file names
    imgs_files = os.listdir(imgs_folder)
    if debug:
        print('[DEBUG]: len lbs_files = {}'.format(len(lbs_files)))
        print('[DEBUG]: len imgs_files = {}'.format(len(imgs_files)))
    
    labels = [] # list of label tensors
    
    im_batches = [] # set empty list of image batches to run
    cur_batch = [] # reset every time you are finished filling a batch
    
    # compute prediction and stats for each image
    imgs_info = dict() # init empty img info dictionary
    
    i=0 # load images into scope
    for im in imgs_files:
        im_name = im.split(".")[0]
        lb = None
        for lb_name in lbs_files: # search for matching label
            if im_name in lb_name: 
                lb = lb_name
                break
        if lb is None: continue # go to next image if no corr. label
        
        label = read_label_file("{}/{}".format(lbs_folder,lb), device=device)
        veh = False
        for cls in label.T[0].tolist(): # iterate over all classes in bounding box list
            if cls in select_classes: veh=True
        
        if not veh: continue # got to next image if no vehicles in image
        
        # read and append label to list
        labels.append(label) 
        im_tensor, og_size, new_size = read_img_file(f"{imgs_folder}/{im}", 
                                                     img_size=img_size, 
                                                     stride=mod_stride,
                                                     device=device)
        
        # this method doesn't preserve order, we need some kind of other identifier   
        if len(cur_batch) < batch_size:             
            cur_batch.append(im_tensor)
            # if debug: print("[INFO]: cur batch length is = {}".format(len(cur_batch)))
        else:
            im_batches.append(cur_batch)
            cur_batch = [] # reset current batch list
            cur_batch.append(im_tensor)
            
        imgs_info[i] = (og_size, new_size) # tuple of size tuples
        i+=1 # increment counter
      
    if len(cur_batch) < batch_size:       
        im_batches.append(cur_batch) # always append last batch (if not already appended)
        
    # print("[DEBUG-250]: {} matches found".format(i))
    # print("[DEBUG-251]: batch size = {}".format(batch_size))
    # print("[DEBUG-252]: num batches = {}".format(len(im_batches)))
    # for i in range(len(im_batches)):
    #     print("\t[DEBUG-255 {}]: batch {} len = {}".format(i,i,len(im_batches[i])))
    
    # convert batches into tensors
    cryp_folder = None
    if not plain:                           ## secure inference
        cryp_folder = "crypten_source_batch{}_{}_{}-{}".format(batch_size, model_name, img_size[0], img_size[1])
        if cryp_folder not in os.listdir("source"): # if the folder doesn't exist, make it
            os.mkdir("source/{}".format(cryp_folder))
        for i in range(len(im_batches)): 
            im_batches[i] = torch.stack(im_batches[i])
            torch.save(im_batches[i], "source/{}/batch_{}.pth".format(cryp_folder, i))
    else:                                   ## plaintext inference
        for i in range(len(im_batches)): 
            im_batches[i] = torch.stack(im_batches[i])
    
    detections = [] # list of prediction tensors - issue is getting 32
    
    # run inference for each batch
    run_times = np.zeros(len(im_batches))
    com_costs = np.zeros(len(im_batches))
    round_nums = np.zeros(len(im_batches))
    
    n_batches = len(im_batches) if num_batches is None else num_batches # get number of batches to compute
    for batch_idx in range(n_batches): # compute one batch for checking purposes # range(len(im_batches)):
        if plain: ### PLAINTEXT INFERENCE
            start = time.time()
            preds = model(im_batches[batch_idx].to(device)) # make sure batch is on correct device
            end = time.time()
            
            if debug: 
                print("[DEBUG-plaintext]: tensor output = \n{}".format(preds))
                print(f"[DEBUG-plaintext]: output tensor shapes = \n{[pred.shape for pred in preds]}")
            
            # END PLAINTEXT INFERENCE --------------------------------------------------------------------
            
        elif not plain: ### SECURE INFERENCE
            assert cryp_folder is not None, "ERROR: cryp_folder should not be 'None'"
            batch_shape = im_batches[batch_idx].shape[0]
            print("[INFO]: batch size = {}".format(batch_shape))
            print("[DEBUG]: batch = source/{}/batch_{}.pth".format(cryp_folder, batch_idx))
            yolo_args = {
                "world_size":2,
                "img_size":img_size,
                "model":model, 
                "data_path":"source/{}/batch_{}.pth".format(cryp_folder, batch_idx),
                "run_label":batch_idx, # just use as a replacable tmp file
                "batch_size":batch_shape,
                "folder":"crypten_tmp",
                "device":device,
                "debug":True,
                "print_net":True
            }
            multiproc_gpu(_run_sec_model, f'{device}_val', args=yolo_args)
            with open("experiments/crypten_tmp/run_{}.pkl".format(batch_idx), "rb") as f: # read results from pickle file
                preds, start, end = pickle.load(f)
            with open("experiments/crypten_tmp/comm_tmp_0.pkl", "rb") as com_0:
                alice_com = pickle.load(com_0)
            with open("experiments/crypten_tmp/comm_tmp_1.pkl", "rb") as com_1:
                bob_com = pickle.load(com_1)
            cost = (alice_com['bytes'] + bob_com['bytes'])/(2e6)/yolo_args['batch_size'] # convert to MB
            round_vals = (alice_com['rounds'] + bob_com['rounds'])/2
            
            com_costs[batch_idx] = cost
            round_nums[batch_idx] = round_vals
            
            if debug: print("[DEBUG-secure]: tensor output = \n{}".format(preds))
            
            # clear out the tmp folder so no accidental reading of old results?
            for item in os.listdir("experiments/crypten_tmp"): os.system("rm experiments/crypten_tmp/{}".format(item))

            # END SECURE INFERENCE -----------------------------------------------------------------------------------
            
        run_times[batch_idx] = end - start # get run time of the batch
        
        preds = non_max_suppression(
            prediction=preds,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            classes=classes,
            agnostic=agnostic_nms, 
            max_det=max_det
        )
        if debug: print("[DEBUG-{}]: nms out = \n{}".format("secure" if not plain else "plain", preds))
        
        for pred in preds: # append each prediction to a single detections list
            detections.append(pred)
          
    # NOTE: THE ERROR IS HAPPENING ABOVE THIS LINE ...
      
    num_dets = []   # list for tracking the number of predictions made for each tensor
    metrics = []
    vals = []
    stats = []
    
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    
    inf_count=0
    for i in range(n_batches): inf_count += len(im_batches[i])
    assert len(detections) == inf_count, "[ERROR-574]: labels / detections don't match -> labels={}, detections={}".format(inf_count, len(detections))
    
    for i in range(len(detections)): # format detection result metrics
        num_dets.append(detections[i].shape[0]) # count the number of predictions from the input image
        lbl = labels[i] 
        det = detections[i]
        og_size, new_size = imgs_info[i]
        
        correct = np.zeros((det.shape[0], iouv.shape[0])).astype(bool)
        if debug: 
            print("[DEBUG]: lbl.device = {}".format(lbl[:,0:1].device))
            print("[DEBUG]: det.device = {}".format(det[:,5].device))
        correct_class = lbl[:,0:1] == det[:,5]
        
        det[:,:4] = scale_boxes(new_size, det[:,:4], og_size)
        lbl[:,1:] *= torch.tensor((og_size[1], og_size[0], og_size[1], og_size[0]), device=device)
        
        iou_score = box_iou(lbl[:,1:], det[:,:4])
        
        for j in range(len(iouv)):
            x = torch.where((iou_score > iouv[j]) & correct_class)
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x,1), iou_score[x[0], x[1]][:,None]), 1).cpu().numpy()
                if x[0].shape[0] > 1:
                    matches = matches[matches[:,2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:,1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:,0], return_index=True)[1]]
                correct[matches[:,1].astype(int), j] = True
        correct = torch.tensor(correct, dtype=torch.bool, device=iouv.device)
        
        # append results to stats list tuples=(correct, confidence (objectness score), pred class, true class)
        stats.append((correct, det[:,4], det[:,5], lbl[:,0]))
    
    # format the stats list and save results
    stats = [torch.cat(x,0).cpu().numpy() for x in zip(*stats)]
    
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir="experiments/batched_val", names=mod_names)
    ap50, ap = ap[:,0], ap.mean(1)
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    
    ap_metrics = pd.DataFrame({
        "veh ap class": ap_class,
        "ap": ap, 
        "ap50": ap50,
        "true positive": tp, 
        "false positive": fp, 
        "precision": p, 
        "recall": r, 
        "f1 score": f1, 
        "precision": p, 
        "recall": r
    })
    
    if vehicle_only: 
        ap_metrics = ap_metrics.loc[(ap_metrics['veh ap class'] == 1) | 
                                    (ap_metrics['veh ap class'] == 2) | 
                                    (ap_metrics['veh ap class'] == 3) | 
                                    (ap_metrics['veh ap class'] == 5) | 
                                    (ap_metrics['veh ap class'] == 7)]
    
    overall_metrics = {
        "mp": mp, 
        "mr": mr, 
        "map50": map50, 
        "map": ap_metrics['ap'].mean(),
        "(avg) run time (s)": run_times.sum()/len(detections), # average over all detections
        "num threads": threading.active_count(),
        "communication": com_costs.sum()/len(detections), 
        "rounds": round_nums.sum()/len(detections),
        "num_detections": np.array(num_dets).mean()
    }
    
    for metr in overall_metrics:
        metrics.append(metr)
        vals.append(overall_metrics[metr])
    results = pd.DataFrame({'metric':metrics, 'score':vals})
    
    results.to_csv("experiments/{}/{}_{}-{}_{}_batch_{}_overall_metrics.csv".format(exp_folder,
                                                                         results_name, 
                                                                         img_size[0],
                                                                         img_size[1],
                                                                         model_name,
                                                                         batch_size))
    ap_metrics.to_csv("experiments/{}/{}_{}-{}_{}_batch_{}_class_metrics.csv".format(exp_folder,
                                                                                results_name, 
                                                                                img_size[0],
                                                                                img_size[1],
                                                                                model_name,
                                                                                batch_size))
    return stats

def set_sizes(mult=32, start=5, max=20):
    sizes = []
    for i in range(start,max+1):
        sizes.append((mult*i, mult*i))
    return sizes

def debug_crypten_val():
    '''
    runs validation for crypten yolov5s and plaintext yolov5s on a 
    batch of 2, printing the torch output. 
    '''
    lbs_folder = "/mnt/nvme01/data/coco128/coco128/labels/train2017"
    imgs_folder = "/mnt/data/coco128/coco128/images/train2017"
    run_val(
        exp_folder='debug',
        batch_size=3,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        classes=None,
        agnostic_nms=None,
        lbs_folder=lbs_folder,
        imgs_folder=imgs_folder,
        model_name='yolov5s',
        device='cuda',
        plain=False,
        results_name="sec_debug",
        img_size=(192,192),
        num_batches=1,
        debug=True
    )
    run_val(
        exp_folder='debug',
        batch_size=3,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        classes=None,
        agnostic_nms=None,
        lbs_folder=lbs_folder,
        imgs_folder=imgs_folder,
        model_name='yolov5s',
        device='cpu',
        plain=True,
        results_name='plain_debug',
        img_size=(192,192),
        num_batches=1,
        debug=True
    )

def img_size_exps(lbs_folder, imgs_folder, device='cpu', dest='img_size_exps', debug=False):
    '''
    runs and saves the results / metrics of all specified experimental runs
    '''
    if device=='cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        
    # img_size experiments
    const_params = {"batch_size":32, "model_name":'yolov5n', 'folder':dest}
    res_names = {'plaintext_val':True, 'secure_val':False}
    img_sizes = [(160,160), (192,192), (256,256), (384,384), (640,640)]
    
    for bench in res_names:
        for i in tqdm(range(len(img_sizes)), # run one experiment set instead of multiple
                      ncols=100, 
                      total=len(img_sizes), 
                      desc='process {} batch'.format(bench), 
                      dynamic_ncols=True, 
                      leave=True):
            run_val(
                exp_folder=const_params['folder'],
                batch_size=const_params['batch_size'],
                conf_thres=0.25,
                iou_thres=0.45,
                max_det=1000,
                classes=None,
                agnostic_nms=None,
                lbs_folder=lbs_folder,
                imgs_folder=imgs_folder,
                model_name=const_params['model_name'],
                device=device,
                plain=res_names[bench], # bool: whether to perform secure or plaintext inference
                results_name=bench,
                img_size=img_sizes[i],
                debug=debug
                # num_batches=None
            )
    return

def batch_size_exps(lbs_folder, imgs_folder, device='cpu', dest='batch_size_exps', debug=False):
    if device=='cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        
    const_params = {"model_name":'yolov5n', "img_size":(288,288), 'folder':dest}
    res_names = {"plaintext_val":True, "secure_val":False}
    batch_sizes = [1,2,4,16,32,64,128]
    
    for bench in res_names:
        for i in tqdm(range(0,len(batch_sizes)), 
                      ncols=100, 
                      total=len(batch_sizes), 
                      desc='process {} batch'.format(bench), 
                      dynamic_ncols=True, 
                      leave=True):
            run_val(
                exp_folder=const_params['folder'],
                batch_size = batch_sizes[i],
                conf_thres=0.25,
                iou_thres=0.45, 
                max_det=1000,
                classes=None,
                agnostic_nms=None,
                lbs_folder=lbs_folder,
                imgs_folder=imgs_folder,
                model_name=const_params['model_name'],
                device=device,
                plain=res_names[bench],
                results_name=bench,
                img_size=const_params['img_size'],
                debug=debug
            )
    return

def model_type_exps(lbs_folder, imgs_folder, device='cpu', dest='model_type_exps', debug=False):
    if device=='cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        
    const_params = {"batch_size":32, "img_size":(288,288), 'folder':dest, "num_batches":1}
    res_names = {"plaintext_val":True, "secure_val":False}
    model_names = ["yolov5s",'yolov5n','yolov5x','yolov5m','yolov5l']
    
    for bench in res_names:
        for i in tqdm(range(0,len(model_names)), 
                      ncols=100, 
                      total=len(model_names), 
                      desc='process {} batch'.format(bench), 
                      dynamic_ncols=True, 
                      leave=True):
            run_val(
                exp_folder=const_params['folder'],
                batch_size=const_params['batch_size'],
                conf_thres=0.25,
                iou_thres=0.45,
                max_det=1000,
                classes=None,
                agnostic_nms=None,
                lbs_folder=lbs_folder,
                imgs_folder=imgs_folder,
                model_name=model_names[i],
                device=device,
                plain=res_names[bench],
                results_name=bench,
                img_size=const_params['img_size'],
                debug=debug
            )
    return
    
def main():
    device = 'cuda'
    if device == 'cuda':
        os.environ['CUDA_AVAILABLE_DEVICES'] = '0,1'
        
    lbs_folder = "/mnt/nvme1n1p1/data/coco128/coco128/labels/train2017"
    imgs_folder = "/mnt/nvme1n1p1/data/coco128/coco128/images/train2017"
    
    model_type_exps(lbs_folder,imgs_folder,device=device,dest='gpu_exps/model_type_exps',debug=True)
    img_size_exps(lbs_folder,imgs_folder,device=device,dest='gpu_exps/img_size_exps',debug=True)
    batch_size_exps(lbs_folder,imgs_folder,device=device,dest='gpu_exps/batch_size_exps',debug=True)
    
    return 0    

def test_single():
    conf_thres=0.25        # confidence threshold parameter
    iou_thres=0.45           # iou threshold parameter
    max_det=1000           # max detection number
    classes=None            # indicates how to filter the classes
    agnostic_nms=None       # perofrms class agnostic nms (if True)
    
    lbs_folder = "/mnt/data/coco128/coco128/labels/train2017"
    imgs_folder = "/mnt/data/coco128/coco128/images/train2017"
    
    lbl_file = os.listdir(lbs_folder)[0]
    lbl_tensor = read_label_file("{}/{}".format(lbs_folder, lbl_file)) # load labels
    im_name = lbl_file.split(".")[0]
    
    for im in os.listdir(imgs_folder):  
        if im_name in im: 
            im_file = im # get correct image
            break
        
    im_tensor, og_size, new_size = read_img_file("{}/{}".format(imgs_folder, im_file)) # load image
    print("[DEBUG]: file name = {}".format("{}/{}".format(imgs_folder, im_file)))
    print("[DEBUG]: label name = {}".format("{}/{}".format(lbs_folder, lbl_file)))
    
    pln_model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True, trust_repo=True)
    pln_names = pln_model.names
    
    start = time.time()
    pred = pln_model(im_tensor[None])
    end = time.time()
    
    pred = non_max_suppression( # obtain final bounding box predictions
        prediction=pred, 
        conf_thres=conf_thres, 
        iou_thres=iou_thres,
        classes=classes,
        agnostic=agnostic_nms, 
        max_det=max_det
    )
    
    metrics = []
    values = []
    
    stats = [] # initialize list for tracking metrics
    
    print("[DEBUG]: Pred output = {}".format(pred))

    iouv = torch.linspace(0.5, 0.95, 10, device="cpu")
    # pred[0][:,4] = pred[0][pred[0][:,4] > conf_thres] # get predictions above conf threshold (unecessary)
        
    correct = np.zeros((len(pred[0]), iouv.shape[0])).astype(bool)
    print("[INFO]: correct (og) = {}".format(correct))
    correct_class = lbl_tensor[:,0:1] == pred[0][:,5]
    
    print("[DEBUG] new_size={}, og_size={}".format(new_size, og_size))
    pred[0][:,:4] = scale_boxes(new_size, pred[0][:,:4], og_size).round() # rescale to proper image size
    #                               width     , height,   , width     , height
    print("[DEBUG] label tensor shape = {}".format(lbl_tensor.shape))
    print("[DEBUG] og size tensor = {}".format(og_size))
    lbl_tensor[:,1:] *= torch.tensor((og_size[1], og_size[0], og_size[1], og_size[0]))
    
    print("[DEBUG] type(lbl_tensor)={}".format(type(lbl_tensor[1:])))
    print("[DEBUG] type(pred)={}".format(type(pred[:3])))
    
    print("[DEBUG]: lbl_tensor = {}".format(lbl_tensor[:,1:]))
    print("[DEBUG]: pred = {}".format(pred[0][:,:4]))
    
    # if lbl_tensor[1:].shape != pred[0][:,:4].shape: 
    #     print("[!!ERROR!!]: mismatched tensor sizes. An inference error occurred")
    #     return
    
    iou_score = box_iou(lbl_tensor[:,1:], pred[0][:,:4])
    print("[INFO]: time = {} seconds".format(end-start))
    print("[INFO]: IOU = {}".format(iou_score))
    
    for i in range(len(iouv)):
        x = torch.where((iou_score > iouv[i]) & correct_class)
        print("[INFO] x = {}".format(x))
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x,1), iou_score[x[0],  x[1]][:, None]), 1).cpu().numpy()
            print("[DEBUG] (1) matches iter={} = {}".format(i, matches))
            if x[0].shape[0] > 1:
                matches = matches[matches[:,2].argsort()[::-1]]
                print("\t[DEBUG] (1a) matches = {}".format(matches))
                matches = matches[np.unique(matches[:,1], return_index=True)[1]]
                print("\t[DEBUG] (1b) matches = {}".format(matches))
                matches = matches[np.unique(matches[:,0], return_index=True)[1]]
                print("\t[DEBUG] (1c) matches = {}".format(matches))
            correct[matches[:,1].astype(int), i] = True
    correct = torch.tensor(correct, dtype=torch.bool, device=iouv.device) # convert numpy array into a tensor
    
    print("[INFO]: correct = {}".format(correct))
    stats.append((correct, pred[0][:,4], pred[0][:,5], lbl_tensor[:,0])) # (correct, conf, pcls, tcls)
    
    print("[DEBUG]: type(correct) = {}".format(type(correct)))
    print("[DEBUG]: type(pred[0][:,4]) = {}".format(pred[0][:,4]))
    print("[DEBUG]: type(pred[0][:,5]) = {}".format(pred[0][:,5]))
    print("[DEBUG]: type(lbl_tensor[:,0]) = {}".format(lbl_tensor[:,0]))
    
    # compute metrics
    stats = [torch.cat(x,0).cpu().numpy() for x in zip(*stats)] # convert these stats into tensors
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir="experiments/batched_val", names=pln_names)
    ap50, ap = ap[:,0], ap.mean(1) # AP@0.5, AP@0.5:0.95
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    
    # nc = 1 # number of image inferences is 1
    # nt = np.bincount(stats[3].astype(int), minlength=nc) 
    metric_res = {
        "true positive": tp, 
        "false positive": fp, 
        "precision": p, 
        "recall": r, 
        "f1 score": f1, 
        "average precision": ap, 
        "ap per class": ap_class, 
        "ap 50": ap50, 
        "mp": mp, 
        "mr": mr, 
        "map50": map50, 
        "map": map        
    }
    for metr in metric_res: 
        metrics.append(metr)
        values.append(metric_res[metr])
    results = pd.DataFrame({"metric":metrics, "score":values})
    results.to_csv("experiments/batched_val/single_eval.csv")
    
if __name__ == "__main__":
    import sys
    import pickle as pkl
    import multiprocessing as mp
    mp.set_start_method("spawn") # set thread start method for crypten     
    
    if len(sys.argv) < 2:
        print("[USAGE]: python3 {} experiment labels_folder_path imgs_folder_path ")
    elif len(sys.argv) == 4:
        # check device
        if torch.cuda.is_available():
                device = 'cuda'
                os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        else:
            device = 'cpu'
            
        # run correct experiment
        if sys.argv[1] == 'benchmark':
            model_type_exps(
                sys.argv[2],
                sys.argv[3],
                device=device,
                debug=False
            )
        elif sys.argv[1] == 'batch_size':
            batch_size_exps(
                sys.argv[2],
                sys.argv[3],
                device=device,
                debug=False
            )
        elif sys.argv[1] == 'img_size':
            img_size_exps(
                sys.argv[2],
                sys.argv[3],
                device=device,
                debug=False
            )
            
    # stats = run_val(
    #     exp_folder='gpu_exps', 
    #     img_size=(288,288), 
    #     debug=True,
    #     lbs_folder='/mnt/nvme1n1p1/data/coco128/coco128/labels/train2017',
    #     imgs_folder='/mnt/nvme1n1p1/data/coco128/coco128/images/train2017',
    #     device='cuda',
    #     model_name='yolov5n',
    #     plain=False,
    #     results_name='test_gpu_environ',
    #     num_batches=1,
    #     get_stats=True,
    #     batch_size=1,
    #     print_net=True
    # )
    # print("[INFO]: ... STATS ...")
    # with open("experiments/gpu_exps/acc_info.pkl", 'wb') as f:
    #     pkl.dump(stats, f)
    # print('[INFO]: stats saved ...')
    
    # const_params = {"batch_size":1, "img_size":(1000,300), 'folder':'model_type_exps', "num_batches":32}
    # faster_rcnn_val(
    #     img_size=const_params['img_size'],
    #     exp_folder=const_params['folder'], 
    #     batch_size=const_params['batch_size'], 
    #     num_batches=const_params['num_batches']
    # )
    
    # print(set_sizes())