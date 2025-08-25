# SecureV2X

SecureV2X is a research framework for securely computing V2X 
applications. Namely, we design secure protocols for private object detection and red light violation detection
with YOLOv5n, and drowsiness detection via CompactCNN. To use our
framework, please follow these steps:

1. First, install `crypten` by running 
   
   ```
   pip install crypten
   ```

   CrypTen is not currently available through Conda and so pip must
   be used to install it.
   Our framework relies upon the use of secure primitives implemented by 
   CrypTen and so this package will be necessary to run *FastSec-YOLO*
   or *CryptoDrowsy*. Additionally, please make sure to install all of
   the packages listed in `requirements.txt`. 
2. Once `crypten` is installed, run the following in python
   
   ```python
   import crypten
   crypten.__file__
   ```

   This will output the location of `../crypten/__init__.py`. Navigate
   to the `nn` folder from there, and replace the default version of  `crypten/nn/module.py` with the `module.py` file provided
   at `case_studies/module.py` from our repository. Our version provides
   the necessary protocols to run SecureV2X. **NOTE: SecureV2X will not
   work correctly if this step is not followed correctly. Please take
   care to correctly locate the original `module.py` file and replace
   it with the updated code**

Next, run `pip install requirements.txt` to install the required packages for SecureV2X. 
The most recent version of each should work with our code. 

### Scalability Simulations

We evaluate the scalability of SecureV2X to a network of clients by running multiple 
instances of CryptoDrowsy and FastSec-YOLO simultaneously. These simulations are 
conducted in `agent_sim.py` and will require the following steps to execute 
properly. 

1. Navigate to [this repository](https://github.com/AhmadYahya97/Fully-Automated-red-light-Violation-Detection)
   and download `y2.mp4` directly from the `videos` folder. Store this video file
   in `case_studies` to perform multi-client secure inference. 
2. Run `python3 agent_sim.py y2.mp4` in `case_studies` to run multi-client inference. 
3. check the output results in `experiments/agg_results.csv` where 
   + `exp` = (x-y) pairs where x is the number of cryptodrowsy clients run and y is the number of 
     fastsec-yolo clients run
   + `d_inf_time` = time required for each CryptoDrowsy inference (on average)
   + `r_inf_time` = time required for each FastSec-YOLO client inference (on average)
     from a batch of 4 video frames from `y2.mp4`

Note the output results correspond to section 5.3 from the associated paper. 

## CryptoDrowsy

Three core results are provided for CryptoDrowsy - namely running time (over the CPU) 
in addition to communication cost and rounds required per inference.
To reproduce the results we provide in the paper, please navigate to 
`case_studies` and run `python3 run_cryptodrowsy`. This will perform inference
over each of the 314 (3-second 128Hz) eeg samples corresponding to subject 9
as studied in [this paper](https://arxiv.org/abs/2106.00613).
The accuracy, communication, round complexity, and timing results will be reported
to the CLI. As was done in the prior work, we have pretrained a model for each 
of the 11 test subjects, and have included their respective labels and features
as well. `run_cryptodrowsy.py` sets subject 9 as the default. However, if you 
would like to run inference for any one of the other 10 subjects, simply 
change `sub9` to `subx` where $x$ is any number from 1 to 11 (inclusive). 
Likewise, change `9-drowsy_features.pth` and `9-drowsy_labels.pth` 
to `x-drowsy_features.pth` and `x-drowsy_labels.pth` where `x` is the same
as for `subx`. 

In order to reproduce private inference over CompactCNN using Delphi, 
first, install rust and then run the following commands:

```
sudo apt install cmake pkg-config g++ gcc libssl-dev libclang-dev
rustup install nightly
```

Next, navigate to `delphi/rust/experiments/src/validation/compactCNN/validation` and 
run `python3 run_validation.py`. If an error occurs, please carefully check to make
sure that all of the paths listed in `run_validation.py` are correct. 
Before running this functionality

+ `weights_path` - `case_studies/driverdrowsiness/pretrained/sub9/model.npy`
+ `eeg_test_data_path` - folder containing all of the .npy data samples for CompactCNN
  + default value - `delphi/rust/experiments/src/validation/compactCNN/Eeg_Samples_and_Validation`
    given as a relative path from `delphi/rust/experiments/src/inference`
+ `num_samples` - max of 314 - recommend a lower number - all 314 may take at least 
  8 hours to run all 314 samples.
+ `accuracy_results_path` - relative path from `delphi/rust/experiments/src/inference` to 
  `delphi/rust/experiments/src/validation/compactCNN/Eeg_Samples_and_Validation/Classification_ResultsX.txt`
+ `output_file` - will divert text output of delphi run to a saved text file. By default, this 
  is set as the relative path to `delphi/rust/experiments/src/validation/compactCNN/validation_runs/vlaidation_runX.txt` from `delphi/rust/experiments/src/inference/`

Don't be alarmed if a connection refused error occurs during the run. Simply restart
the process. If you intend to run all 314 samples, you will want to check 
`delphi/rust/experiments/src/inference/compactCNN/sequential_inference.rs` to adjust
values for the range of samples to run if you have already run several by the 
time a connection refused error occurs. 

In order to run CompactCNN on CrypTFlow2, you will need to clone [EzPC](https://github.com/mpc-msri/EzPC)
and use the custom compiler they provide to compile a secure version of CompactCNN. Follow the 
instructions provided for running **SCI** at the link. 

## FastSec-YOLO

FastSec-YOLO can be run by doing the following:
First, navigate to `case_studies` and run `run_fastsec_yolo.py` with a command line 
argument of the form `source/image.jpg` where `image.jpg` is the image you want to 
perform secure object detection over. If you would like to perform inference over 
your own image, please upload that image to the source folder directly. Otherwise, 
you may test this functionality by running inference over the `bus.jpg` image 
which is held within the source folder. 

Once inference is performed, you may check the bounding box detection results 
in the `experiments/img_visuals` folder which will have been generated. 
The most likely issue to occur during this step is the improper initialization
of CUDA. If this happens, please check to make sure that your cuda settings 
are properly set. ⚠️ The code should try to handle potential issues automatically, 
but please be aware 

### Experiments

First, download the coco128 data from this [link](https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip). If the link does not work for some reason, please visit ultralytics/yolov5 on github and download
coco128.zip from release v1.0. 
Make a note of the absolute path to the image and label folders on your system,
or move the image and label data into the repository so that you can 
Our experiments for FastSec-YOLO can be reproduced as follows:

1. Navigate to `case_studies/traffic_rule_violation`
2. Run `python3 fastsec_yolo_val.py benchmark <labels_folder> <images_folder>` 
   where `<labels_folder>` contains the labels for the coco128 train image 
   set, and `<images_folder>` contains the actual images for the coco128 image
   dataset. 

Image and batch size experiments can be reproduced by chnaging "benchmark" to 
either "batch_size" or "img_size" for batch and image size experiments respectively.
The results of this experiment will be output to `case_studies/traffic_rule_violation/experiments/model_type_exps`
where each file in that folder is a `.csv` file containing the results of both plaintext
and secure inference over coco128 for each model (yolov5: n, s, m, l, x).
❗Note that results may differ from those reported in the associated paper due to 
computational machinery differences. 

<!-- ## TODO: 8/5/24

1. Add the modified version of CrypTen to the repository with reproducible instructions
   to install the updated version of the code from source.
2. Clean up the crypten compactcnn implementation and convert to a script which can be run 
   more conveniently
3. Include instructions for accessing the data utilized for this work (make sure this is robust)
4. Update fully-automated RLR detection script, and move into the main v2x-delphi-2pc repo
   (the repo needs to be renamed as well to "v2x-2pc" or something)

For this code to work, we need to run all scripts from the case_studies package as a 
relative call now. This is because I have restructured everything as a package format. -->