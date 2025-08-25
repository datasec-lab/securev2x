from traffic_rule_violation.fastsec_yolo import multiproc_gpu, _run_sec_model
from agent_sim import read_im_frame
from traffic_rule_violation.fastsec_yolo_val import read_img_file
from traffic_rule_violation.fastsec_yolo_detect import sample_proc_output
import os
import pickle
import torch

def get_data(plain_img_path:str, img_size, device, cryp_src):
    img_tensor, _, _ = read_img_file(plain_img_path, img_size, device=device)
    name = plain_img_path.split("/")[-1].split(".")[0]
    d_path = f'{cryp_src}/{name}.pth'
    
    if cryp_src not in os.listdir():
        os.mkdir(cryp_src)
        
    torch.save(img_tensor, d_path)
    return d_path

def main(
    image="source/bus.jpg",
    load_lock=None
):
    '''
    Use best available GPU device by entering 'cuda' as the 
    'device' parameter (params['device']='cuda'). Otherwise, use the CPU
    by entering 'cpu'
    
    Parameter Dictionary:
    - world_size --- the number of parties used for computation - will only work
      for two parties.
    - run_label ---- the label for the inference run (defaults to fastsec_yolo_inf)
    - folder ------- the folder storing results of the secure inference
    - img_size ----- the size to scale input images to for inference 
    - batch_size --- the number of images to perform inference for 
    - rlr_mod ------ the model type to be used (yolov: 5n, v5s, v5m, v5l, v5x)
    - load_lock ---- to handle multi-process loading of yolo architecture
    - debug -------- set true if you would like see debug output from the run
    '''
    ### MODIFY TO CHANGE INPUTS ###
    
    # set device based on availability
    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
    params = {
        'world_size': 2,  # 2pc inference
        'run_label': '_fastsec_yolo_inf',
        'device': device,
        'folder': 'detection',
        'img_size': (288,288),
        'batch_size': 1,
        'rlr_mod': 'yolov5n',
        'load_lock': load_lock,
        'debug': False
    }
    ### DO NOT MODIFY BELOW ########        
        
    dat_path = get_data( # format data for inference
        image, 
        params['img_size'], 
        params['device'], 
        params['folder']
    ) 
    params['data_path'] = dat_path
    print("[DEBUG]: dat_path = {}".format(dat_path))
    
    # set up folder to save results
    if 'experiments' not in os.listdir():
        os.mkdir("experiments")
    if params['folder'] not in os.listdir("experiments"):
        os.mkdir('experiments/{}'.format(params['folder']))
    
    # run 2pc secure yolov5 inference (fastsec-yolo)
    multiproc_gpu(
        run_experiment=_run_sec_model,
        args=params,
        agent=True
    )
    with open("experiments/{}/run_{}.pkl".format(params['folder'],params['run_label']), "rb") as f: # read results from pickle file
        preds, start, end = pickle.load(f)
    with open("experiments/{}/comm_tmp_0.pkl".format(params['folder']), "rb") as com_0:
        alice_com = pickle.load(com_0)
    with open("experiments/{}/comm_tmp_1.pkl".format(params['folder']), "rb") as com_1:
        bob_com = pickle.load(com_1)
        
    # map bounding boxes to the image
    sample_proc_output(
        imgsz=params['img_size'],
        folder=params['folder'],
        run_val=params['run_label'],
        img_name=image.split("/")[1],    # second item in file path
        file_source=image.split("/")[0], # first item in file path
        debug=False
    )
    
    # clean up tmp folder
    for f in os.listdir('experiments/{}'.format(params['folder'])):
        os.remove('experiments/{}/{}'.format(params['folder'],f))
    
if __name__ == "__main__":
    import sys
    import multiprocessing as mp
    mp.set_start_method('spawn')
    load_lock = mp.Lock() # set lock for loading yolo model
    if len(sys.argv) != 2:
        print("[USAGE]: python3 {} source/image.jpg".format(sys.argv[0]))
        exit()
    main(image=sys.argv[1], load_lock=load_lock)