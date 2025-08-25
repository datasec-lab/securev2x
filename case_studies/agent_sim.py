'''
Experiment for running multiple agents simultaneusly - each 
system is grabbing a separate video feed from the traffic lights
this is simulated by having a pre-stored set of individual frames, 
and reading the inference one by one in YOLO (for every five frames)

Purpose of experiment is to determine the following metrics
+ communication costs as more threads are running simultaneously
+ secure inference speed as more threads are run simultaneously
+ how do mixed simulations perform (both CompactCNN and YOLO 
  are run simultaneously). This might not work if we try to run 
  the experiments on the GPU since CompactCNN will not work if we 
  run the GPU actively. I wonder if there is a way to isolate the 
  environment variables to that particular thread. 
  
We will obtain separate simulation results - two 
separate servers control these multi-agent processes in 
our setup ... or at least two separate shell environments???
This kind of semantic is easy to argue away later in the paper
I think. Actually mixed CompactCNN / YOLO sims can be run since 
we can do this in the subprocess. The os.environ command needs
to be used in order to accomplish this

HYPOTHESIS: 
As long as the number of processes remains less than 64//3 = 21
then processing should remain optimal. But if we go over this 
amount, then we will encounter performance degradation - maybe this 
would be good to measure n>21

sources: https://docs.python.org/3/library/multiprocessing.html 
https://stackoverflow.com/questions/20886565/using-multiprocessing-process-with-a-maximum-number-of-simultaneous-processes 
'''

# multiprocess script
import multiprocessing as mp

# main stuff
import os
import sys
import logging
import traceback
import pandas as pd
import pickle
import cv2
import torch
import numpy as np
from tqdm import tqdm

try: 
    from driverdrowsiness.cryptodrowsy import multiproc_gpu_drowsy, _run_sec_drowsy_model # this one runs compactcnn
    from driverdrowsiness.models.crypten_compactcnn import CryptenCompactCNN
    from traffic_rule_violation.fastsec_yolo import multiproc_gpu, _run_sec_model # this one runs yolo
    from traffic_rule_violation.utils.augmentations import letterbox
    from traffic_rule_violation.utils.general import non_max_suppression
except: 
    print("[ERROR]: import error occurred for one of the above functions")
    print("[ERROR]: file path = {}".format(os.getcwd()))

def cryptodrowsy(call_id:int, params:dict, queue):
    '''
    pseudocode: setting the environment variables only works one call (process) above
    1. set up environment for the subprocess
       set the crypten folder parameter in the target function
    2. run the secure functionality
    3. record results in designated file
    
    the results have to be stored within the second level of an experiments file 
    within the directory from which the function is run
    '''
    drowsy_mod = CryptenCompactCNN() # set arch
    drowsy_mod.load_state_dict(torch.load(params['drowsy_mod'], map_location='cpu')) 
    params['model'] = drowsy_mod # set the model
    
    # fake func - load weights
    exp_run = params['run_number'] # label for the experiment batch eg. (1 drowsy, 2 rlr)
    results_label = params['run_label'] # file written to experiments/tmp_folder/
    tmp_folder = f'{exp_run}_crypten_tmp_drowsy_{call_id}' # each process has its own folder
    
    if params['device'] == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    if not tmp_folder in os.listdir("experiments"):
        os.mkdir(f'experiments/{tmp_folder}') # make the crypten holding dir
    params['folder'] = tmp_folder
    
    com_costs = []
    round_nums = []
    inf_times = []
    
    # nah, just do the first 32 images
    multiproc_gpu_drowsy(_run_sec_drowsy_model, run_val=call_id, args=params)
    
    # read results of inference run
    with open(f"experiments/{tmp_folder}/run_{results_label}.pkl", "rb") as f: # read results from pickle file
        preds, start, end = pickle.load(f)
    with open(f"experiments/{tmp_folder}/comm_tmp_0.pkl", "rb") as com_0:
        alice_com = pickle.load(com_0)
    with open(f"experiments/{tmp_folder}/comm_tmp_1.pkl", "rb") as com_1:
        bob_com = pickle.load(com_1)
    cost = (alice_com['bytes'] + bob_com['bytes'])/(2e6 * params['batch_size']) # convert to MB / inference
    round_vals = (alice_com['rounds'] + bob_com['rounds'])/(2 * params['batch_size']) #  rounds / inference
    
    com_costs.append(cost)
    round_nums.append(round_vals)
    inf_times.append((end-start)/params['batch_size'])
    
    queue.put(pd.DataFrame({"com_cost":com_costs,
                            "round_nums":round_nums,
                            "inf_time":inf_times}))
    
    # every run has its own folder for the crypten tmp stuff
    # in each run, there are a bunch
    
def fastsec_yolo(call_id:int, params:dict, queue, debug=True, ret_inf=False):
    '''
    pseudocode: 
    1. set up environment for the subprocess
       set the crypten folder parameter in the target function
       perform some logic checks to ensure that the batch size is compatible 
       with the dataset size
    2. run the secure functionality
    3. record results in designated file
    '''
    try:
        # set the 
        if debug: print("[DEBUG]: checkpoint 1")
        
        exp_run = params['run_number'] # label for the experiment batch eg. (1 drowsy, 2 rlr)
        results_label = params['run_label'] # file written to experiments/tmp_folder/
        tmp_folder = f'{exp_run}_crypten_tmp_yolo_{call_id}' # each process has its own folder
        
        if params['device'] == 'cuda':
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        if debug: print("[DEBUG]: checkpoint 2 - changed environment cuda")
        
        if not tmp_folder in os.listdir("experiments"):  
            os.mkdir(f'experiments/{tmp_folder}') # make the crypten holding dir
        params['folder'] = tmp_folder
        
        com_costs = []
        round_nums = []
        inf_times = []
        
        if debug: print("[DEBUG]: checkpoint 3 - about to start run")
        
        # nah, just do the first 32 images
        multiproc_gpu(_run_sec_model, run_val=call_id, args=params, agent=True)
        
        if debug: print("[DEBUG]: checkpoint 4 - finished run")
        
        # read results of inference run
        with open(f"experiments/{tmp_folder}/run_{results_label}.pkl", "rb") as f: # read results from pickle file
            preds, start, end = pickle.load(f)
        with open(f"experiments/{tmp_folder}/comm_tmp_0.pkl", "rb") as com_0:
            alice_com = pickle.load(com_0)
        with open(f"experiments/{tmp_folder}/comm_tmp_1.pkl", "rb") as com_1:
            bob_com = pickle.load(com_1)
        cost = (alice_com['bytes'] + bob_com['bytes'])/(2e6 * params['batch_size']) # convert to MB
        round_vals = (alice_com['rounds'] + bob_com['rounds'])/(2 * params['batch_size'])
        
        com_costs.append(cost)
        round_nums.append(round_vals)
        inf_times.append((end-start)/params['batch_size'])
        
        preds = non_max_suppression(
            prediction=preds,
            conf_thres=params['conf_thres'],
            iou_thres=params['iou_thres'],
            classes=params['classes'],
            agnostic=params['agnostic'], 
            max_det=params['max_det']
        )
        n_preds = np.array([len(pred) for pred in preds]).mean()
        if ret_inf: # return prediction matrix if desired
            return preds
        
        queue.put(pd.DataFrame({"n_dets":[n_preds], # should be a list
                                "com_cost":com_costs,
                                "round_nums":round_nums,
                                "inf_time":inf_times}))
    except:
        print("[ERROR]: Error in RLR worker process")
        traceback.print_exc()

    # ----- DATA READING UTILITIES ---------
  
def read_im_frame(frame, img_size, stride=32):
    '''
    reads frames for detections that require the YOLOv5 object 
    detector. all other detections / bounding box operations with 
    algorithms from cv2 are performed over standard numpy arrays.
    '''
    im = letterbox(frame, img_size, stride=stride, auto=False, scaleFill=False)[0]
    im = im.transpose((2,0,1))[::-1]
    im = np.ascontiguousarray(im)
    im_tensor = torch.from_numpy(im)/255 # recolor the pixels for inference
    
    # no need to pass the original and new image sizes simultaneously
    return im_tensor
  
def fetch_sim_vid_data(params, vid_path=None, debug=False, save_im=0):
    '''
    read video data from a specified video file using methods from 
    https://learnopencv.com/reading-and-writing-videos-using-opencv/#read-video-from-file 
    '''
    if vid_path is None:
        vid_file = "../../Fully-Automated-red-light-Violation-Detection/videos/y2.mp4"
    
    vid_capture = cv2.VideoCapture(f'{vid_file}')
    
    saved_frames = 0
    n_frames = 0
    # get video metadata
    if (vid_capture.isOpened() == False):
        print("[ERROR]: error opening the video file")
    else:
        fps = int(vid_capture.get(5))
        print("[INFO]: Frame Rate: ",fps,"frames per second")
        frame_count = vid_capture.get(7)
        print("[INFO]: Frame Count: ",frame_count)
    
    vid_data = []
    while(vid_capture.isOpened()):
        ret, frame = vid_capture.read()
        
        if frame is None: 
            break # break if the frame is None
        
        if n_frames%5 != 0: # only record every fifth frame
            n_frames += 1
            continue
        else:
            saved_frames += 1 
            n_frames += 1

        if debug:
            print(f'[DEBUG]: frame = {frame}')
            print(f'[DEBUG]: frame.shape = {frame.shape}')
            
        im_tensor = read_im_frame(frame, 
                                  params['img_size'], 
                                  stride=params['stride']).unsqueeze(0)
        if debug: print(f'[DEBUG]: torch tensor = {im_tensor}')
        
        vid_data.append(im_tensor)
        if saved_frames < save_im: 
            cv2.imwrite(f"experiments/vid_ims/{vid_file.split('.')[0]}_{saved_frames}.jpg", frame)
        if ret == False:
            break
        
        
    print("[INFO]: Frames for inference = {}".format(saved_frames))
          
    vid_data = torch.stack(vid_data)
    return vid_data
  
# ----- END DATA READING UTILITIES -----

def run_agents(
    drowsy_mod_path:str,
    rlr_mod_path:str,
    im_size=(288,288),
    im_batch_size=32,
    rlr_device='cuda',
    eeg_size=(1,384),
    run_number=1, 
    d_agents=3, 
    r_agents=3,
    model_loading_lock=None
):
    '''
    pseudocode:
    load the models and obtain the video data needed for 
    inference. We don't concern ourselves with getting multiple data
    sets for the CompactCNN, though we could easily do this as well I think
    '''
    # create a experiment folder to hold all crypten information and results
    # from the run. Each worker will create a separate folder to hold their stuff
    
    rlr_params = {
        "run_number":run_number,
        "world_size":2,
        "img_size":im_size,
        "rlr_mod":rlr_mod_path,
        "data_path":"rlr_agent_src/vid_0_batch_0.pth",    # set in the target func
        "run_label":'rlr_vid_0_gpu', # just use as a replacable tmp file
        "batch_size":im_batch_size,
        "device":rlr_device,
        "debug":False,
        'conf_thres':0.5,
        'iou_thres':0.3, 
        'classes':[2,3,7,9], 
        'agnostic':None,
        'max_det':1000,
        'load_lock':model_loading_lock
    }
    drows_params = {
        "run_number":run_number,
        'world_size':2,
        'img_size':eeg_size,
        'drowsy_mod':drowsy_mod_path,
        'data_path':'driverdrowsiness/dev_work/test_features_labels/9-drowsy_features.pth',
        'run_label':'cpu_compactcnn',
        'batch_size':314, # edit in the target funcs
        'device':'cpu', 
        'debug':False,
    }
    
    # we want to create a multiprocess calls that can handle more than the number
    # of available cores on the CPU
    # A process pool object which controls a pool of worker processes to which jobs can 
    # be submitted. It supports asynchronous results with timeouts and callbacks and has
    # a parallel map implementation
    # the Pool class allows you to choose the number of processors to use.
    
    # p = mp.Pool(processes=int(os.cpu_count()*0.75))
    # for i in range(d_agents):
    #     result = p.apply_async(cryptodrowsy, args=(i,drows_params,))
    #     drowsy_async.append(result)
    # for j in range(r_agents):
    #     result = p.apply_async(fastsec_yolo, args=(j,rlr_params,))
    #     rlr_async.append(result)
        
    # # wait for the results and then process
    # drowsy_results = [result.get() for result in drowsy_async]
    # rlr_results  = [result.get() for result in rlr_async]
    
    # # close the pool of workers
    # p.close()
    # p.join()
    
    drowsy_queue = mp.Queue()
    rlr_queue = mp.Queue()
    
    drowsy_processes = []
    rlr_processes = []
    print("[DEBUG]: checkpoint 0 - about to init processes")
    
    for i in range(d_agents):
        process = mp.Process(target=cryptodrowsy, args=(i,drows_params,drowsy_queue))
        drowsy_processes.append(process)
    for j in range(r_agents):
        process = mp.Process(target=fastsec_yolo, args=(j,rlr_params,rlr_queue))
        rlr_processes.append(process)
        
    # start and join all of the processes
    for dp in drowsy_processes: dp.start()
    for rp in rlr_processes: rp.start()
    for dp in drowsy_processes: dp.join()
    for rp in rlr_processes: rp.join()
    
    drowsy_results = [drowsy_queue.get() for _ in range(drowsy_queue.qsize())]
    rlr_results = [rlr_queue.get() for _ in range(rlr_queue.qsize())]
    
    print(len(drowsy_results))
    
    # all of the processes should return a dataframe with the relevant data 
    # for the process
    drowsy_res_df = None
    for d_df in drowsy_results:
        if drowsy_res_df is None: drowsy_res_df = d_df
        else: drowsy_res_df = pd.concat([drowsy_res_df, d_df], axis=0)
    rlr_res_df = None
    for r_df in rlr_results:
        if rlr_res_df is None: rlr_res_df = r_df
        else: rlr_res_df = pd.concat([rlr_res_df, r_df], axis=0)
        
    experiment_agg_results = pd.DataFrame({
        "exp": [run_number],
        "rlr_agents": [r_agents], 
        "drowsy_agents": [d_agents], 
        "d_num_rnds": [drowsy_res_df['round_nums'].mean() if drowsy_res_df is not None else np.nan],
        "d_com_cost": [drowsy_res_df['com_cost'].mean() if drowsy_res_df is not None else np.nan],
        "d_inf_time": [drowsy_res_df['inf_time'].mean() if drowsy_res_df is not None else np.nan],
        "r_num_rnds": [rlr_res_df['round_nums'].mean() if rlr_res_df is not None else np.nan],
        "r_com_cost": [rlr_res_df['com_cost'].mean() if rlr_res_df is not None else np.nan],
        "r_inf_time": [rlr_res_df['inf_time'].mean() if rlr_res_df is not None else np.nan], 
        "r_n_dets": [rlr_res_df['n_dets'].mean() if rlr_res_df is not None else np.nan]
    })
    
    return experiment_agg_results

def main(mp_lock, vid_path=None):
    # # load model parameters and such
    # eeg_size=(1,384)
    # drowsy_device = 'cpu'
    # dw_path = "driverdrowsiness/pretrained/sub9/model.pth"
    # drowsy_mod = CryptenCompactCNN() # set arch
    # drowsy_mod.load_state_dict(torch.load(dw_path, map_location=drowsy_device)) # fake func - load weights
    
    
    # rlr_mod = torch.hub.load("ultralytics/yolov5", "yolov5n", force_reload=True, trust_repo=True)
    # rlr_mod.to(device=rlr_device) # send to the gpu
    
    if 'rlr_agent_src' not in os.listdir():
        im_size=(288,288)
        im_batch_size=4
        rlr_device = 'cuda'
        
        os.mkdir('rlr_agent_src')
        
        vids = ['y2.mp4'] # only one video for now - maybe do more later??
        for i in range(len(vids)):
            vid_data = fetch_sim_vid_data(params={'img_size':im_size, 'stride':32}, vid_file=vids[i])
            vid_data = torch.split(vid_data,im_batch_size)
            for j in range(len(vid_data)):
                torch.save(vid_data[j], f"rlr_agent_src/vid_{i}_batch_{j}.pth")
    
    res_file = 'debug_results'
    dw_path = "driverdrowsiness/pretrained/sub9/model.pth"
    yolo_path = 'yolov5n'
    rlr_device = 'cuda'
            
    # set run values
    runs = dict()
    n_total = [1, 2, 3, 5, 8, 12, 15]
    c_prop = [0.0, 0.25, 0.5, 0.75, 1.0]
    for n in n_total:
        for p in c_prop: 
            n_comp = int(n*p)
            n_rlr = n - n_comp
            if f'{n_comp}-{n_rlr}' in runs: continue
            else: runs[f'{n_comp}-{n_rlr}'] = (n_comp, n_rlr)
    runs['4-1']=[4,1]
    
    print(f"[INFO]: combs = {runs}")
    print(f"[INFO]: num combs = {len(runs)}")
    
    final_results = []
    for comb in tqdm(runs,
                     ncols=100,
                     total=len(runs),
                     desc='process run agent sims',
                     dynamic_ncols=True,
                     leave=True):
        res = run_agents(
                drowsy_mod_path=dw_path,
                rlr_mod_path=yolo_path,
                rlr_device=rlr_device,
                run_number=comb,
                d_agents=runs[comb][0],
                r_agents=runs[comb][1],
                model_loading_lock=mp_lock,
                im_batch_size=4,
                )
        final_results.append(res)
    res_csv_file = final_results[0]
    for i in range(1,len(final_results)):
        res_csv_file = pd.concat([res_csv_file,final_results[i]],axis=0)
    res_csv_file.to_csv(f"experiments/{res_file}.csv", index=False)
    
    for item in os.listdir("experiments"):
        if "crypten_tmp" in item: os.system(f"rm -rf experiments/{item}")
    
if __name__ == "__main__":
    import sys
    mp.log_to_stderr(logging.INFO)
    mp.set_start_method("spawn")
    load_lock = mp.Lock()
    
    if len(sys.argv) < 2 and ('rlr_agent_src' not in os.listdir()):
        print("[USAGE]: python3 {} path/to/input/video.mp4".format(sys.argv[0]))
        exit()
    main(load_lock, vid=sys.argv[1])
