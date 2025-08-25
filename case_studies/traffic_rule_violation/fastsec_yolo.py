import logging
import os
from examples.multiprocess_launcher import MultiProcessLauncher
import torch

def _run_sec_model(args:dict):
    '''
    Use the args parameter as a dictionary for holding key argument 
    variables for the yolo runs
    '''
    # import function to run the independent processes
    if 'agent_mode' in args:
        if args['agent_mode'] == True: 
            from .twopc_yolo_agent import run_2pc_yolo
        else: 
            from .twopc_yolo import run_2pc_yolo
    else:
        from .twopc_yolo import run_2pc_yolo
    
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    
    # pass all of the arguments
    run_2pc_yolo(args)
    
    return 

def multiproc_gpu(run_experiment, run_val='0', args:dict=None, agent=False):
    if args is None: 
        args = {
            "world_size":2,
            "img_size":(640,640), 
            "model": torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True, trust_repo=True),
            "data_path":"source/crypten_source/walkway.pth",
            "run_label":run_val,
            "batch_size":1
            # need to generate the validation data file still
        }
    else: 
        args=args
        
    if agent: # add argument to pass to the secure functionality
        args['agent_mode'] = True
        
    # the function `run_experiment` ultimately takes the input `args`
    launcher = MultiProcessLauncher(args['world_size'], run_experiment, args, launched=True)
    launcher.start()
    launcher.join()
    launcher.terminate()