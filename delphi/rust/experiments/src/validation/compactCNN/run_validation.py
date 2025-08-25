import os

def main():
    '''
    ⚠️Warning: all file paths are relative to the inference file (except for os.chidr()
    command at the beginning of the run)
    since this is where the rust source file for running inference is actually located
    '''
    # src/inference relative to al file paths
    run_number=9
    weights_path = "../../../../../case_studies/driverdrowsiness/pretrained/sub9/model.npy" 
    approx_layers = 0
    eeg_test_data_path = "../validation/compactCNN/Eeg_Samples_and_Validation"
    num_samples = 314 
    accuracy_results_path = "../validation/compactCNN/Eeg_Samples_and_Validation/Classification_Results{}.txt".format(run_number)
    output_file = "../validation/compactCNN/validation_runs/validation_run{}.txt".format(run_number)
    
    os.chdir("../../inference")
    if not accuracy_results_path in os.listdir("../validation/compactCNN/Eeg_Samples_and_Validation/"):
        os.system("touch {}".format(accuracy_results_path))
        
    os.system("cargo +nightly run --bin compact-cnn-sequential-inference -- --weights {} --layers {} --eeg_data {} --num_samples {} --results_file {} > {}".format(weights_path, approx_layers, eeg_test_data_path, num_samples, accuracy_results_path, output_file))

if __name__ == "__main__":
    main()
